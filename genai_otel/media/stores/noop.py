"""No-op store used when capture_mode=reference_only."""

from __future__ import annotations


class NoopStore:
    def put(self, data: bytes, *, key: str, mime_type: str) -> str:  # noqa: D401
        return f"noop://{key}"
