"""Provider-format -> normalized content part detection.

Each provider has a different multimodal message shape. This module normalizes
them into a single `ContentPart` representation so the rest of the pipeline
(redactors, stores, span emitters) can operate provider-agnostically.

Supported provider shapes:
- "openai": chat completion `messages[].content` arrays with type=text|image_url|input_audio
- "anthropic": `messages[].content` arrays with type=text|image|document
- "google": Gemini `contents[].parts[]` with text|inline_data|file_data
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

PartType = Literal["text", "image", "audio", "video", "document"]
MediaSource = Literal["inline_offloaded", "external_url", "reference_only"]


@dataclass
class ContentPart:
    """Normalized representation of a single content part."""

    type: PartType
    text: Optional[str] = None
    # Raw bytes (only when type != text and we have inline data to offload)
    data: Optional[bytes] = None
    # Pre-existing URL (for image_url style references that we don't re-host)
    external_url: Optional[str] = None
    media_mime_type: Optional[str] = None
    media_byte_size: Optional[int] = None
    media_uri: Optional[str] = None
    media_source: Optional[MediaSource] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedMessage:
    role: str
    parts: List[ContentPart]


def _decode_b64(data: Any) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=False)
        except Exception:  # noqa: BLE001
            return None
    return None


def _detect_openai_part(item: Any) -> Optional[ContentPart]:
    """OpenAI / OpenAI-compatible chat content array element."""
    if isinstance(item, str):
        return ContentPart(type="text", text=item)
    if not isinstance(item, dict):
        return None

    ptype = item.get("type")
    if ptype == "text":
        return ContentPart(type="text", text=item.get("text"))
    if ptype == "image_url":
        url_obj = item.get("image_url") or {}
        url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
        if isinstance(url, str) and url.startswith("data:"):
            # data:<mime>;base64,<payload>
            try:
                header, payload = url.split(",", 1)
                mime = header.split(";", 1)[0].removeprefix("data:") or "image/png"
                return ContentPart(
                    type="image",
                    data=_decode_b64(payload),
                    media_mime_type=mime,
                )
            except Exception:  # noqa: BLE001
                logger.debug("Failed to decode data: URL")
                return ContentPart(type="image", external_url=url)
        return ContentPart(type="image", external_url=url, media_mime_type="image/*")
    if ptype == "input_audio":
        audio = item.get("input_audio") or {}
        fmt = audio.get("format") or "wav"
        return ContentPart(
            type="audio",
            data=_decode_b64(audio.get("data")),
            media_mime_type=f"audio/{fmt}",
        )
    if ptype == "input_video" or ptype == "video":
        v = item.get("input_video") or item.get("video") or item
        url = v.get("url") if isinstance(v, dict) else None
        if isinstance(url, str) and url.startswith("data:"):
            try:
                header, payload = url.split(",", 1)
                mime = header.split(";", 1)[0].removeprefix("data:") or "video/mp4"
                return ContentPart(type="video", data=_decode_b64(payload), media_mime_type=mime)
            except Exception:  # noqa: BLE001
                return ContentPart(type="video", external_url=url, media_mime_type="video/*")
        return ContentPart(
            type="video",
            data=_decode_b64(v.get("data")) if isinstance(v, dict) else None,
            external_url=url,
            media_mime_type=(v.get("mime_type") if isinstance(v, dict) else None) or "video/mp4",
        )
    if ptype in ("file", "input_file"):
        f = item.get("file") or item
        return ContentPart(
            type="document",
            data=_decode_b64(f.get("file_data") or f.get("data")),
            external_url=f.get("file_url") or f.get("url"),
            media_mime_type="application/pdf",
        )
    return None


def _detect_anthropic_part(item: Any) -> Optional[ContentPart]:
    if isinstance(item, str):
        return ContentPart(type="text", text=item)
    if not isinstance(item, dict):
        return None

    btype = item.get("type")
    if btype == "text":
        return ContentPart(type="text", text=item.get("text"))
    if btype == "image":
        src = item.get("source") or {}
        stype = src.get("type")
        mime = src.get("media_type") or "image/png"
        if stype == "base64":
            return ContentPart(
                type="image",
                data=_decode_b64(src.get("data")),
                media_mime_type=mime,
            )
        if stype == "url":
            return ContentPart(type="image", external_url=src.get("url"), media_mime_type=mime)
    if btype == "document":
        src = item.get("source") or {}
        mime = src.get("media_type") or "application/pdf"
        if src.get("type") == "base64":
            return ContentPart(
                type="document", data=_decode_b64(src.get("data")), media_mime_type=mime
            )
        if src.get("type") == "url":
            return ContentPart(type="document", external_url=src.get("url"), media_mime_type=mime)
    return None


def _detect_google_part(item: Any) -> Optional[ContentPart]:
    if isinstance(item, str):
        return ContentPart(type="text", text=item)
    if not isinstance(item, dict):
        # google-genai SDK uses pydantic models; try attribute access
        if hasattr(item, "text") and getattr(item, "text", None):
            return ContentPart(type="text", text=getattr(item, "text"))
        if hasattr(item, "inline_data"):
            inline = getattr(item, "inline_data", None)
            if inline is not None:
                mime = getattr(inline, "mime_type", None) or "application/octet-stream"
                data = _decode_b64(getattr(inline, "data", None))
                return _classify_google(mime, data=data)
        if hasattr(item, "file_data"):
            fd = getattr(item, "file_data", None)
            if fd is not None:
                mime = getattr(fd, "mime_type", None) or "application/octet-stream"
                uri = getattr(fd, "file_uri", None)
                return _classify_google(mime, external_url=uri)
        return None

    if "text" in item and item.get("text"):
        return ContentPart(type="text", text=item["text"])
    inline = item.get("inline_data") or item.get("inlineData")
    if inline:
        mime = inline.get("mime_type") or inline.get("mimeType") or "application/octet-stream"
        return _classify_google(mime, data=_decode_b64(inline.get("data")))
    fd = item.get("file_data") or item.get("fileData")
    if fd:
        mime = fd.get("mime_type") or fd.get("mimeType") or "application/octet-stream"
        return _classify_google(mime, external_url=fd.get("file_uri") or fd.get("fileUri"))
    return None


def _classify_google(
    mime: str, data: Optional[bytes] = None, external_url: Optional[str] = None
) -> ContentPart:
    if mime.startswith("image/"):
        ptype: PartType = "image"
    elif mime.startswith("audio/"):
        ptype = "audio"
    elif mime.startswith("video/"):
        ptype = "video"
    elif mime.startswith(("application/pdf", "application/", "text/")):
        ptype = "document"
    else:
        ptype = "document"
    return ContentPart(type=ptype, data=data, external_url=external_url, media_mime_type=mime)


_DETECTORS = {
    "openai": _detect_openai_part,
    "anthropic": _detect_anthropic_part,
    "google": _detect_google_part,
    "groq": _detect_openai_part,
}


def detect_parts(provider: str, content: Any, *, fallback_role: str = "user") -> List[ContentPart]:
    """Normalize a single message's content field into ContentPart objects.

    Args:
        provider: one of "openai" | "anthropic" | "google" | "groq"
        content: the raw value of `message["content"]` (str, list, or provider object)
        fallback_role: ignored here; preserved for symmetry with NormalizedMessage callers

    Returns:
        A list of ContentPart. Returns an empty list if nothing detected.
    """
    detector = _DETECTORS.get(provider, _detect_openai_part)

    if content is None:
        return []
    if isinstance(content, str):
        return [ContentPart(type="text", text=content)]
    if isinstance(content, list):
        parts: List[ContentPart] = []
        for item in content:
            cp = detector(item)
            if cp is not None:
                # populate byte_size if data present
                if cp.data is not None and cp.media_byte_size is None:
                    cp.media_byte_size = len(cp.data)
                parts.append(cp)
        return parts
    if isinstance(content, dict):
        cp = detector(content)
        return [cp] if cp else []
    # Unknown shape — best-effort string coercion
    return [ContentPart(type="text", text=str(content))]
