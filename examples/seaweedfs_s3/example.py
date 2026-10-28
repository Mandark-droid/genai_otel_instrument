"""Trace standard boto3 S3 operations against a SeaweedFS S3 gateway.

Set SEAWEEDFS_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
GENAI_OTEL_MEDIA_STORE_ACCESS_KEY, GENAI_OTEL_MEDIA_STORE_SECRET_KEY, and
SEAWEEDFS_S3_BUCKET before running. The bucket must already exist and the
credentials need permission to put, read, and delete objects in it.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("OTEL_SERVICE_NAME", "seaweedfs-s3-example")

from genai_otel import flush_telemetry, instrument  # noqa: E402
from genai_otel.config import OTelConfig  # noqa: E402
from genai_otel.media.stores import get_store  # noqa: E402

instrument()

# Import boto3 after instrumentation initialization, as in an application that
# enables genai_otel at startup.
import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402


def main() -> None:
    endpoint = _required_env(
        "SeaweedFS S3 endpoint",
        "SEAWEEDFS_S3_ENDPOINT",
        "GENAI_OTEL_MEDIA_STORE_ENDPOINT",
        "endpoint",
    )
    bucket = _required_env(
        "SeaweedFS S3 bucket",
        "SEAWEEDFS_S3_BUCKET",
        "GENAI_OTEL_MEDIA_STORE_BUCKET",
        "bucket",
    )
    access_key = _required_env(
        "SeaweedFS S3 access key",
        "AWS_ACCESS_KEY_ID",
        "GENAI_OTEL_MEDIA_STORE_ACCESS_KEY",
        "access_key",
    )
    secret_key = _required_env(
        "SeaweedFS S3 secret key",
        "AWS_SECRET_ACCESS_KEY",
        "GENAI_OTEL_MEDIA_STORE_SECRET_KEY",
        "secret_key",
    )
    media_store = get_store(
        OTelConfig(
            media_store="seaweed",
            media_store_endpoint=endpoint,
            media_store_bucket=bucket,
            media_store_access_key=access_key,
            media_store_secret_key=secret_key,
        )
    )
    if media_store is None:
        raise RuntimeError("SeaweedFS media offload store failed to initialize")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    key = f"genai-otel-instrument/{uuid.uuid4()}.txt"
    payload = b"SeaweedFS S3 instrumentation example"
    try:
        uri = media_store.put(payload, key=key, mime_type="text/plain")
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        if body != payload:
            raise RuntimeError("SeaweedFS returned content that did not match the uploaded object")
        client.delete_object(Bucket=bucket, Key=key)
        print(f"Round-tripped and deleted {uri}")
    finally:
        flush_telemetry()


def _required_env(label: str, *names: str) -> str:
    """Return the first populated variable without printing its value."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"Set one of these environment variables for {label}: {', '.join(names)}")


if __name__ == "__main__":
    main()
