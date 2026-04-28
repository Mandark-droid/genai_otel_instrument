"""Live integration test against a MinIO instance.

Skipped unless GENAI_OTEL_MEDIA_STORE_ENDPOINT, _ACCESS_KEY, _SECRET_KEY are set.

For the TraceVerse VM:
    export GENAI_OTEL_MEDIA_STORE_ENDPOINT=http://192.168.206.129:9000
    export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=<from VM .env>
    export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=<from VM .env>
    pytest tests/integration/test_minio_offload.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not (
        os.getenv("GENAI_OTEL_MEDIA_STORE_ENDPOINT")
        and os.getenv("GENAI_OTEL_MEDIA_STORE_ACCESS_KEY")
        and os.getenv("GENAI_OTEL_MEDIA_STORE_SECRET_KEY")
    ),
    reason="MinIO credentials not provided in environment",
)


def test_minio_put_and_url():
    boto3 = pytest.importorskip("boto3")
    from genai_otel.media.stores.s3_minio import S3MinioStore

    bucket = os.getenv("GENAI_OTEL_MEDIA_STORE_BUCKET", "genai-otel-media-test")
    store = S3MinioStore(
        endpoint=os.environ["GENAI_OTEL_MEDIA_STORE_ENDPOINT"],
        bucket=bucket,
        access_key=os.environ["GENAI_OTEL_MEDIA_STORE_ACCESS_KEY"],
        secret_key=os.environ["GENAI_OTEL_MEDIA_STORE_SECRET_KEY"],
    )

    # 1x1 transparent PNG
    png = bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
        "0000000D49444154789C636400000000060005A36CFC8E0000000049454E44AE426082"
    )
    key = f"integration-tests/{uuid.uuid4().hex}.png"
    uri = store.put(png, key=key, mime_type="image/png")
    assert key in uri
    assert uri.startswith("http")
