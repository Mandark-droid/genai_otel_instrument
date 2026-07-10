"""Filesystem-backed media store. Useful for local dev and air-gapped tests."""

from __future__ import annotations

import os
from pathlib import Path


class FilesystemStore:
    def __init__(self, root: str, bucket: str = "genai-otel-media") -> None:
        self.root = (Path(root) / bucket).resolve()
        # Create the store root with restrictive permissions (0o700 on POSIX;
        # mode is effectively a no-op on Windows but harmless).
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _safe_target(self, key: str) -> Path:
        """Resolve ``key`` under the store root, rejecting traversal/absolute keys.

        Guards against absolute keys and ``..`` segments that would write
        outside the store root (path traversal).
        """
        root = self.root.resolve()
        key_path = Path(key)
        # Reject absolute keys and Windows drive/UNC-anchored keys outright.
        if key_path.is_absolute() or key_path.drive or key_path.anchor:
            raise ValueError(f"Unsafe media key (absolute path not allowed): {key!r}")
        target = (root / key).resolve()
        # Containment: the resolved target must live inside the store root.
        try:
            target.relative_to(root)
        except ValueError as e:
            raise ValueError(f"Unsafe media key escapes store root: {key!r}") from e
        return target

    def put(self, data: bytes, *, key: str, mime_type: str) -> str:
        target = self._safe_target(key)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Write, then restrict the file to owner read/write only (0o600).
        target.write_bytes(data)
        try:
            os.chmod(target, 0o600)
        except OSError:
            # Best-effort: some filesystems / Windows configurations reject chmod.
            pass
        return target.as_uri()
