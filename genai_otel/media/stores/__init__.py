"""Pluggable media-store backends for offloaded multimodal content."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

from .base import MediaStore

logger = logging.getLogger(__name__)


def _float(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _require_https(config: Any) -> bool:
    """True when network media stores must reject plaintext http:// endpoints.

    Enforced under a strict/bfsi/bank profile or when external egress is
    disabled (air-gapped / no-egress posture).
    """
    profile = (getattr(config, "profile", "") or "").lower()
    if profile in ("strict", "bfsi", "bank"):
        return True
    return getattr(config, "allow_external_egress", True) is False


_STORE_CONFIG_FIELDS = (
    "media_store",
    "media_store_endpoint",
    "media_store_bucket",
    "media_store_access_key",
    "media_store_secret_key",
    "media_store_connect_timeout",
    "media_store_trace_calls",
    "media_store_read_timeout",
    "profile",
    "allow_external_egress",
)
_STORE_CACHE: Dict[Tuple[Any, ...], Optional[MediaStore]] = {}
_STORE_LOCK = threading.Lock()


def clear_store_cache() -> None:
    """Forget cached stores (tests, or after reconfiguration)."""
    with _STORE_LOCK:
        _STORE_CACHE.clear()


def get_store(config: Any) -> Optional[MediaStore]:
    """Return the MediaStore for this config, built once per process per effective config.

    Returns None if disabled. A store that fails to construct raises on every
    call (nothing is cached for a failure) so misconfiguration stays visible.
    """
    key = tuple(getattr(config, f, None) for f in _STORE_CONFIG_FIELDS)
    with _STORE_LOCK:
        if key in _STORE_CACHE:
            return _STORE_CACHE[key]
        store = _build_store(config)
        _STORE_CACHE[key] = store
        return store


def _build_store(config: Any) -> Optional[MediaStore]:
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
            connect_timeout=_float(getattr(config, "media_store_connect_timeout", None), 1.0),
            read_timeout=_float(getattr(config, "media_store_read_timeout", None), 5.0),
            trace_calls=bool(getattr(config, "media_store_trace_calls", False)),
        )
    if backend == "http":
        from .http import HttpStore

        return HttpStore(
            endpoint=getattr(config, "media_store_endpoint", None),
            require_https=require_https,
        )
    logger.warning("Unknown media store backend '%s'; disabling capture", backend)
    return None


__all__ = ["MediaStore", "clear_store_cache", "get_store"]
