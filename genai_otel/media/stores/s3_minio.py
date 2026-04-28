"""S3 / MinIO media store. Requires the `multimodal-s3` extra (boto3)."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class S3MinioStore:
    """Uploads to any S3-compatible object store (AWS S3, MinIO, Ceph, etc.).

    Configures via constructor args or env-var fallback:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    For MinIO, set `endpoint` to e.g. `http://minio.internal:9000`.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str],
        bucket: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
    ) -> None:
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
        client_kwargs = {
            "service_name": "s3",
            "region_name": region,
            "config": Config(signature_version="s3v4"),
        }
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key
        self._client = boto3.client(**client_kwargs)

        # Best-effort bucket creation (idempotent)
        try:
            self._client.create_bucket(Bucket=bucket)
        except Exception as e:  # noqa: BLE001
            # BucketAlreadyOwnedByYou / BucketAlreadyExists are expected
            logger.debug("create_bucket: %s", e)

    def put(self, data: bytes, *, key: str, mime_type: str) -> str:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=mime_type or "application/octet-stream",
        )
        if self._endpoint:
            return f"{self._endpoint.rstrip('/')}/{self._bucket}/{key}"
        return f"s3://{self._bucket}/{key}"
