"""Pluggable media-store backends for offloaded multimodal content."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import MediaStore

logger = logging.getLogger(__name__)


def _require_https(config: Any) -> bool:
    """True when network media stores must reject plaintext http:// endpoints.

    Enforced under a strict/bfsi/bank profile or when external egress is
    disabled (air-gapped / no-egress posture).
    """
    profile = (getattr(config, "profile", "") or "").lower()
    if profile in ("strict", "bfsi", "bank"):
        return True
    return getattr(config, "allow_external_egress", True) is False


def get_store(config: Any) -> Optional[MediaStore]:
    """Build a MediaStore from OTelConfig, or return None if disabled."""
    backend = (getattr(config, "media_store", "none") or "none").lower()
    if backend in ("none", ""):
        return None
    require_https = _require_https(config)
    if backend == "filesystem":
        from .filesystem import FilesystemStore

        return FilesystemStore(
            root=getattr(config, "media_store_endpoint", None) or "./genai-otel-media",
            bucket=getattr(config, "media_store_bucket", "genai-otel-media"),
        )
    if backend in ("s3", "minio", "seaweed"):
        from .s3_minio import S3MinioStore

        endpoint = getattr(config, "media_store_endpoint", None)
        if backend == "seaweed" and not endpoint:
            raise ValueError(
                "SeaweedFS media store requires GENAI_OTEL_MEDIA_STORE_ENDPOINT "
                "pointing to the SeaweedFS S3 gateway"
            )
        return S3MinioStore(
            endpoint=endpoint,
            bucket=getattr(config, "media_store_bucket", "genai-otel-media"),
            access_key=getattr(config, "media_store_access_key", None),
            secret_key=getattr(config, "media_store_secret_key", None),
            addressing_style="path" if backend == "seaweed" else "auto",
            create_bucket_if_missing=backend != "seaweed",
            require_https=require_https,
        )
    if backend == "http":
        from .http import HttpStore

        return HttpStore(
            endpoint=getattr(config, "media_store_endpoint", None),
            require_https=require_https,
        )
    logger.warning("Unknown media store backend '%s'; disabling capture", backend)
    return None


__all__ = ["MediaStore", "get_store"]
