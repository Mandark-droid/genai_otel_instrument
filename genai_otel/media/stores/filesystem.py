"""Filesystem-backed media store. Useful for local dev and air-gapped tests."""

from __future__ import annotations

import os
from pathlib import Path


class FilesystemStore:
    def __init__(self, root: str, bucket: str = "genai-otel-media") -> None:
        self.root = Path(root) / bucket
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, *, key: str, mime_type: str) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.resolve().as_uri()
