"""HTTP PUT-based media store. For self-hosted ingest gateways."""

from __future__ import annotations

import logging
import urllib.request
import uuid
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


def _strip_userinfo(url: str) -> str:
    """Remove any embedded ``user:pass@`` credentials from a URL.

    The returned media_uri is placed on spans / telemetry; credentials in the
    configured endpoint must never leak into it.
    """
    try:
        parts = urlsplit(url)
    except Exception:  # noqa: BLE001
        return url
    if parts.username or parts.password:
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        parts = parts._replace(netloc=netloc)
    return urlunsplit(parts)


class HttpStore:
    """PUTs blobs to a configured base URL. Returns the canonical URL."""

    def __init__(self, *, endpoint: Optional[str], require_https: bool = False) -> None:
        if not endpoint:
            raise ValueError("HttpStore requires GENAI_OTEL_MEDIA_STORE_ENDPOINT")
        if require_https and endpoint.lower().startswith("http://"):
            raise ValueError(
                "Plaintext http:// media store endpoint is not allowed under a strict "
                "profile or when external egress is disabled; use https://"
            )
        # Full endpoint (may contain credentials) is used to perform the PUT.
        self._endpoint = endpoint.rstrip("/")
        # Credential-stripped endpoint is used to build the returned media_uri.
        self._public_endpoint = _strip_userinfo(self._endpoint)

    def put(self, data: bytes, *, key: str, mime_type: str) -> str:
        safe_key = key or uuid.uuid4().hex
        url = f"{self._endpoint}/{safe_key}"
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
        # Return a credential-free URI for telemetry.
        return f"{self._public_endpoint}/{safe_key}"
