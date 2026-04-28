"""Multimodal content-part detection and offload for genai-otel-instrument.

Public API:
- ContentPart: normalized representation of one prompt/completion content part
- detect_parts: provider-format -> List[ContentPart]
- offload_part: redact + upload bytes, return part with media_uri set
- get_store: build a MediaStore from OTelConfig
"""

from .canonical import build_canonical_messages
from .detector import ContentPart, detect_parts
from .offload import offload_part
from .stores import get_store

__all__ = [
    "ContentPart",
    "detect_parts",
    "offload_part",
    "get_store",
    "build_canonical_messages",
]
