# SeaweedFS S3 instrumentation

This example uses boto3's regular S3 client against the SeaweedFS S3 gateway.
The SDK emits client spans for S3 API operations; no SeaweedFS-specific Python
library is needed.

Install the SDK with boto3 and configure an OTLP collector:

```bash
pip install 'genai-otel-instrument[aws,multimodal-s3]'
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export SEAWEEDFS_S3_ENDPOINT=http://localhost:8333
export SEAWEEDFS_S3_BUCKET=agent-artifacts
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=your-access-key
export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=your-secret-key
python examples/seaweedfs_s3/example.py
```

The example also accepts `endpoint`, `access_key`, `secret_key`, and `bucket`
from the repository's `.env` file, for environments that already use those
variable names.

The bucket must already exist. Use HTTPS for production and keep the credentials
in your deployment's secret manager. The example uses the SDK's `seaweed` media
offload backend to upload a small temporary object, then reads and deletes it
through boto3. Path-style addressing works when the gateway has not been
configured for virtual-hosted bucket domains.

Expected spans include `s3.PutObject`, `s3.GetObject`, and `s3.DeleteObject`,
with `db.system=s3`, `db.operation`, `s3.bucket`, `s3.object`, and the endpoint
hostname. The spans do not include credentials, object bodies, or other request
parameters. For `GetObject`, the span ends when boto3 returns the response; a
later read from its streaming body is outside the span.

See the [SeaweedFS S3 API guide](https://github.com/seaweedfs/seaweedfs/wiki/Amazon-S3-API)
for gateway configuration, API coverage, and authentication setup.
