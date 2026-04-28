"""Pluggable media-store backends for offloaded multimodal content."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import MediaStore

logger = logging.getLogger(__name__)


def get_store(config: Any) -> Optional[MediaStore]:
    """Build a MediaStore from OTelConfig, or return None if disabled."""
    backend = (getattr(config, "media_store", "none") or "none").lower()
    if backend in ("none", ""):
        return None
    if backend == "filesystem":
        from .filesystem import FilesystemStore

        return FilesystemStore(
            root=getattr(config, "media_store_endpoint", None) or "./genai-otel-media",
            bucket=getattr(config, "media_store_bucket", "genai-otel-media"),
        )
    if backend in ("s3", "minio"):
        from .s3_minio import S3MinioStore

        return S3MinioStore(
            endpoint=getattr(config, "media_store_endpoint", None),
            bucket=getattr(config, "media_store_bucket", "genai-otel-media"),
            access_key=getattr(config, "media_store_access_key", None),
            secret_key=getattr(config, "media_store_secret_key", None),
        )
    if backend == "http":
        from .http import HttpStore

        return HttpStore(
            endpoint=getattr(config, "media_store_endpoint", None),
        )
    logger.warning("Unknown media store backend '%s'; disabling capture", backend)
    return None


__all__ = ["MediaStore", "get_store"]
