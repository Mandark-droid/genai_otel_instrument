"""Orchestrates the redact -> upload -> URI pipeline for content parts."""

from __future__ import annotations

import hashlib
import importlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .detector import ContentPart
from .stores.base import MediaStore

logger = logging.getLogger(__name__)


def _resolve_callable(dotted: str) -> Optional[Callable[..., bytes]]:
    if not dotted:
        return None
    try:
        module_name, attr = dotted.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to resolve media redactor '%s': %s", dotted, e)
        return None


def _key_for(prefix_template: str, trace_id: str, part: ContentPart) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = prefix_template.format(date=today, trace_id=trace_id or "untraced")
    if not prefix.endswith("/"):
        prefix += "/"
    ext = _ext_for_mime(part.media_mime_type)
    return f"{prefix}{uuid.uuid4().hex}{ext}"


def _ext_for_mime(mime: Optional[str]) -> str:
    if not mime:
        return ".bin"
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "audio/wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/ogg": ".ogg",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
        "application/pdf": ".pdf",
    }
    return mapping.get(mime.lower(), ".bin")


def offload_part(
    part: ContentPart,
    *,
    config: Any,
    store: Optional[MediaStore],
    trace_id: str = "",
) -> ContentPart:
    """Apply redaction + upload to a content part, mutating it with URI metadata.

    Behavior is gated by `config.media_capture_mode`:
    - "off": returns part unchanged (caller should not have invoked us)
    - "reference_only": records modality + mime + size only, no upload
    - "full": redact -> upload -> set media_uri + media_source=inline_offloaded

    Returns the (possibly mutated) part.
    """
    mode = getattr(config, "media_capture_mode", "off")
    if mode == "off" or part.type == "text":
        return part

    # Modality allow-list
    allowed = set(
        s.strip()
        for s in (
            getattr(config, "media_allowed_modalities", "image,audio,video,document") or ""
        ).split(",")
        if s.strip()
    )
    if part.type not in allowed:
        part.media_source = "reference_only"
        part.extra["stripped_reason"] = "modality_not_allowed"
        part.data = None
        return part

    # External URL: don't re-host, just record reference
    if part.data is None and part.external_url:
        part.media_uri = part.external_url
        part.media_source = "external_url"
        return part

    # Reference-only: record metadata, drop bytes
    if mode == "reference_only":
        part.media_source = "reference_only"
        part.data = None
        return part

    # Full mode requires bytes + a store
    if part.data is None or store is None:
        part.media_source = "reference_only"
        part.extra["stripped_reason"] = "no_data_or_store"
        return part

    # Size cap
    max_bytes = int(getattr(config, "media_max_bytes", 10 * 1024 * 1024))
    if len(part.data) > max_bytes:
        part.media_source = "reference_only"
        part.extra["stripped_reason"] = "size_exceeded"
        part.media_byte_size = len(part.data)
        part.data = None
        return part

    # Redactor hook
    redactor = _resolve_callable(getattr(config, "media_redactor", "") or "")
    payload = part.data
    if redactor is not None:
        try:
            payload = redactor(part.type, part.media_mime_type or "", payload)
        except Exception as e:  # noqa: BLE001
            logger.warning("Redactor failed; dropping bytes: %s", e)
            part.media_source = "reference_only"
            part.extra["stripped_reason"] = "redactor_error"
            part.data = None
            return part

    # Upload
    try:
        key = _key_for(
            getattr(config, "media_store_prefix", "traces/{date}/{trace_id}/"),
            trace_id,
            part,
        )
        uri = store.put(
            payload, key=key, mime_type=part.media_mime_type or "application/octet-stream"
        )
        part.media_uri = uri
        part.media_source = "inline_offloaded"
        part.media_byte_size = len(payload)
        part.extra["sha256"] = hashlib.sha256(payload).hexdigest()
        # Drop in-memory bytes after upload
        part.data = None
    except Exception as e:  # noqa: BLE001
        logger.warning("Media upload failed; dropping bytes: %s", e)
        part.media_source = "reference_only"
        part.extra["stripped_reason"] = "upload_error"
        part.data = None
    return part
