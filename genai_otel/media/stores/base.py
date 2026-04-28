"""MediaStore protocol shared across backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MediaStore(Protocol):
    """Protocol implemented by all media store backends.

    `put` uploads bytes and returns a stable URI that downstream consumers
    can resolve back to the original content. The URI scheme is
    backend-specific (e.g. `s3://bucket/key`, `file:///abs/path`,
    `https://gateway/uuid`).
    """

    def put(self, data: bytes, *, key: str, mime_type: str) -> str:
        """Upload `data` under `key` and return its retrieval URI."""
        ...
