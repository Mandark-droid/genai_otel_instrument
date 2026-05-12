"""Build OTel-canonical `gen_ai.input.messages` / `gen_ai.output.messages`
JSON from our normalized ContentPart list.

Maps to the gen-ai message JSON schemas in
https://github.com/open-telemetry/semantic-conventions-genai (the dedicated
GenAI semantic-conventions repo). TraceVerde's upstream PRs:
- #142 (approved): `document` value on the `Modality` enum.
- #143 (under review): optional `byte_size` on `BlobPart` / `FilePart` / `UriPart`.
- #144 (under review): make `content` / `file_id` / `uri` optional and add
  `stripped_reason` to the existing media parts, with a top-level `anyOf`
  requiring either a non-null payload or a non-null `stripped_reason`.

Mapping (post-pivot from the earlier `StrippedPart` proposal):
- text part -> TextPart  {"type": "text", "content": ...}
- image/audio/video/document with offloaded bytes (media_uri set) -> UriPart
  {"type": "uri", "modality": ..., "mime_type": ..., "uri": ..., "byte_size": ...}
- image/audio/video/document with inline bytes (data set) -> BlobPart
  {"type": "blob", "modality": ..., "mime_type": ..., "content": <base64>, "byte_size": ...}
- stripped (instrumentation observed but intentionally did not capture the
  payload bytes): emit the SAME part type as it would have been if captured,
  but omit the content-bearing field and set `stripped_reason`. Original
  `type` + `modality` + `mime_type` + `byte_size` are preserved so consumers
  can still distinguish "no media" from "media stripped".

  TraceVerde's strips originate from inline data we decided not to upload, so
  in practice they map to BlobPart shape with `content` omitted:
  {"type": "blob", "modality": ..., "mime_type": ..., "byte_size": ...,
   "stripped_reason": "size_exceeded" | "modality_not_allowed" | ...}
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List

from .detector import ContentPart


def _modality_for(part: ContentPart) -> str:
    # `document` was added to the Modality enum in upstream PR #142 (approved).
    # The schema also accepts free-form strings via `Union[Modality, str]`, so
    # any future ContentPart.type value falls through to the string branch.
    return part.type


def _part_to_canonical(part: ContentPart) -> Dict[str, Any]:
    """Convert a single ContentPart into an OTel-canonical schema part dict."""
    if part.type == "text":
        return {"type": "text", "content": part.text or ""}

    stripped_reason = (part.extra or {}).get("stripped_reason")
    is_stripped = part.media_source == "reference_only" or stripped_reason is not None

    # UriPart: emitted when we have an external or successfully-offloaded URI.
    # If the part is stripped, we still emit a UriPart shape if a URI was
    # already known at capture time (e.g. external_url case where strip
    # happened for an unrelated reason); content-bearing `uri` is omitted.
    if part.media_uri and not is_stripped:
        out: Dict[str, Any] = {
            "type": "uri",
            "modality": _modality_for(part),
            "uri": part.media_uri,
        }
        if part.media_mime_type:
            out["mime_type"] = part.media_mime_type
        if part.media_byte_size is not None:
            out["byte_size"] = int(part.media_byte_size)
        return out

    # BlobPart: inline bytes captured OR stripped (we observed inline data but
    # intentionally did not upload / capture it).
    if part.data is not None or is_stripped:
        out = {
            "type": "blob",
            "modality": _modality_for(part),
        }
        if part.media_mime_type:
            out["mime_type"] = part.media_mime_type
        if part.media_byte_size is not None:
            out["byte_size"] = int(part.media_byte_size)
        if is_stripped:
            if stripped_reason:
                out["stripped_reason"] = stripped_reason
        elif part.data is not None:
            out["content"] = base64.b64encode(part.data).decode("ascii")
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
