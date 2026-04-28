"""HTTP PUT-based media store. For self-hosted ingest gateways."""

from __future__ import annotations

import logging
import urllib.request
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class HttpStore:
    """PUTs blobs to a configured base URL. Returns the canonical URL."""

    def __init__(self, *, endpoint: Optional[str]) -> None:
        if not endpoint:
            raise ValueError("HttpStore requires GENAI_OTEL_MEDIA_STORE_ENDPOINT")
        self._endpoint = endpoint.rstrip("/")

    def put(self, data: bytes, *, key: str, mime_type: str) -> str:
        url = f"{self._endpoint}/{key or uuid.uuid4().hex}"
        req = urllib.request.Request(
            url,
            data=data,
            method="PUT",
            headers={"Content-Type": mime_type or "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(
                req, timeout=10
            ) as resp:  # nosec B310 - endpoint configured by operator
                resp.read()
        except Exception as e:  # noqa: BLE001
            logger.warning("HttpStore PUT failed: %s", e)
            raise
        return url
