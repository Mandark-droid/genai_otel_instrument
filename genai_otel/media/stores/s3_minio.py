"""S3-compatible media store for AWS S3, MinIO, SeaweedFS, and others."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


def _strip_userinfo(url: str) -> str:
    """Remove any embedded ``user:pass@`` credentials from a URL."""
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


class S3MinioStore:
    """Uploads to any S3-compatible object store (AWS S3, MinIO, SeaweedFS, Ceph, etc.).

    Configures via constructor args or env-var fallback:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    For MinIO, set `endpoint` to e.g. `https://minio.internal:9000`.
    For SeaweedFS, set its S3 gateway endpoint (usually port 8333) and use
    path-style addressing.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str],
        bucket: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
        addressing_style: str = "auto",
        create_bucket_if_missing: bool = True,
        require_https: bool = False,
        connect_timeout: float = 1.0,
        read_timeout: float = 5.0,
    ) -> None:
        if require_https and endpoint and endpoint.lower().startswith("http://"):
            raise ValueError(
                "Plaintext http:// S3/MinIO endpoint is not allowed under a strict profile "
                "or when external egress is disabled; use https://"
            )
        try:
            import boto3  # type: ignore
            from botocore.client import Config  # type: ignore
        except ImportError as e:
            raise ImportError(
                "S3/MinIO store requires `boto3`. Install with: "
                "pip install 'genai-otel-instrument[multimodal-s3]'"
            ) from e

        self._bucket = bucket
        self._endpoint = endpoint
        self._warned = False
        self._down_until = 0.0
        self._lock = threading.Lock()
        client_kwargs = {
            "service_name": "s3",
            "region_name": region,
            "config": Config(
                signature_version="s3v4",
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                retries={"max_attempts": 1},
                s3={"addressing_style": addressing_style},
            ),
        }
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key
        self._client = boto3.client(**client_kwargs)

        if create_bucket_if_missing:
            # Best-effort bucket creation (idempotent)
            try:
                self._client.create_bucket(Bucket=bucket)
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
                if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                    logger.debug("create_bucket: %s", e)
                else:
                    self._note_failure("create_bucket", e)

    _COOLDOWN_SECONDS = 30.0

    def _note_failure(self, op: str, exc: Exception) -> None:
        """Log ONE warning per store and start a short cool-down for uploads."""
        with self._lock:
            self._down_until = time.monotonic() + self._COOLDOWN_SECONDS
            if self._warned:
                return
            self._warned = True
        logger.warning(
            "Media store %s failed (%s); media offload is skipped while it is "
            "unreachable. Further failures are not logged at WARNING.",
            op,
            exc,
        )

    def put(self, data: bytes, *, key: str, mime_type: str) -> str:
        if time.monotonic() < self._down_until:
            raise RuntimeError("media store unreachable (cooling down)")
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=mime_type or "application/octet-stream",
            )
        except Exception as e:  # noqa: BLE001
            self._note_failure("put_object", e)
            raise
        if self._endpoint:
            # Strip any embedded credentials so they don't leak into telemetry.
            public_endpoint = _strip_userinfo(self._endpoint.rstrip("/"))
            return f"{public_endpoint}/{self._bucket}/{key}"
        return f"s3://{self._bucket}/{key}"
