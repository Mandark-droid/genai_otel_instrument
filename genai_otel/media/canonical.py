"""Build OTel-canonical `gen_ai.input.messages` / `gen_ai.output.messages`
JSON from our normalized ContentPart list.

Maps to the upstream schemas at
https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-input-messages.json

Mapping:
- text part -> TextPart  {"type": "text", "content": ...}
- image/audio/video with offloaded bytes (media_uri set) -> UriPart
  {"type": "uri", "modality": ..., "mime_type": ..., "uri": ..., "byte_size": ...}
- image/audio/video with inline bytes (data set) -> BlobPart
  {"type": "blob", "modality": ..., "mime_type": ..., "content": <base64>, "byte_size": ...}
- document parts use modality "document" (proposed addition; see
  docs/proposals/upstream-pr-draft/) -- consumers that don't know the value
  treat it as a free-form string per the schema's `anyOf` fallback.
- reference_only / stripped parts -> StrippedPart (proposed addition)
  {"type": "stripped", "modality": ..., "stripped_reason": ...}
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List

from .detector import ContentPart


def _modality_for(part: ContentPart) -> str:
    # "document" is currently a proposed addition to the OTel Modality enum
    # (see issue #3672). The schema permits string values outside the enum via
    # `anyOf: [{$ref: Modality}, {type: string}]`, so emitting it is safe.
    return part.type


def _part_to_canonical(part: ContentPart) -> Dict[str, Any]:
    """Convert a single ContentPart into an OTel-canonical schema part dict."""
    if part.type == "text":
        return {"type": "text", "content": part.text or ""}

    # reference_only or stripped reasons -> StrippedPart (proposed)
    if part.media_source == "reference_only" or (part.extra and part.extra.get("stripped_reason")):
        out: Dict[str, Any] = {"type": "stripped", "modality": _modality_for(part)}
        reason = part.extra.get("stripped_reason") if part.extra else None
        if reason:
            out["stripped_reason"] = reason
        if part.media_mime_type:
            out["mime_type"] = part.media_mime_type
        if part.media_byte_size is not None:
            out["byte_size"] = int(part.media_byte_size)
        return out

    # External or offloaded -> UriPart
    if part.media_uri:
        out = {
            "type": "uri",
            "modality": _modality_for(part),
            "uri": part.media_uri,
        }
        if part.media_mime_type:
            out["mime_type"] = part.media_mime_type
        if part.media_byte_size is not None:
            out["byte_size"] = int(part.media_byte_size)
        return out

    # Inline bytes still present -> BlobPart
    if part.data is not None:
        out = {
            "type": "blob",
            "modality": _modality_for(part),
            "content": base64.b64encode(part.data).decode("ascii"),
        }
        if part.media_mime_type:
            out["mime_type"] = part.media_mime_type
        if part.media_byte_size is not None:
            out["byte_size"] = int(part.media_byte_size)
        return out

    # Fallback: GenericPart with our internal type name
    return {"type": part.type}


def build_canonical_messages(messages_with_parts) -> List[Dict[str, Any]]:
    """Build a list of ChatMessage dicts conforming to the upstream JSON schema.

    Args:
        messages_with_parts: iterable of (role: str, parts: List[ContentPart]).

    Returns:
        A list ready to be JSON-serialized into `gen_ai.input.messages` /
        `gen_ai.output.messages`.
    """
    out: List[Dict[str, Any]] = []
    for role, parts in messages_with_parts:
        out.append(
            {
                "role": role,
                "parts": [_part_to_canonical(p) for p in parts],
            }
        )
    return out
