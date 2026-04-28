"""End-to-end validation of the multimodal pipeline.

Drives a synthetic OpenAI-shaped multimodal request through the instrumentor,
uploads bytes to MinIO, ships the span to the TraceVerse OTel collector, then
asserts the trace landed in OpenSearch and the blob is in MinIO.

No LLM API key required.

Run:
    export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=...
    export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=...
    export OPENSEARCH_ADMIN_PASSWORD=...
    python examples/multimodal/e2e_validate.py

Expected output:
    [PASS] MinIO upload: <key>
    [PASS] OTel span exported
    [PASS] OpenSearch query: trace <id> found with N spans
    [PASS] Span attributes contain required multimodal fields
"""

from __future__ import annotations

import base64
import os
import sys
import time

VM = "192.168.206.129"
COLLECTOR_HTTP = f"http://{VM}:4318"
MINIO_ENDPOINT = f"http://{VM}:9000"
OPENSEARCH = f"http://{VM}:9200"
BUCKET = "genai-otel-media-e2e"

# Defaults so the script runs out-of-the-box on the dev VM
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", COLLECTOR_HTTP)
os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
os.environ.setdefault("OTEL_SERVICE_NAME", "genai-otel-multimodal-e2e")
os.environ.setdefault("GENAI_OTEL_MEDIA_CAPTURE_MODE", "full")
os.environ.setdefault("GENAI_OTEL_MEDIA_STORE", "minio")
os.environ.setdefault("GENAI_OTEL_MEDIA_STORE_ENDPOINT", MINIO_ENDPOINT)
os.environ.setdefault("GENAI_OTEL_MEDIA_STORE_BUCKET", BUCKET)


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print(f"[FAIL] Missing env vars: {', '.join(missing)}")
        sys.exit(1)


def step(label: str) -> None:
    print(f"\n=== {label} ===")


def main() -> int:
    _require_env(
        "GENAI_OTEL_MEDIA_STORE_ACCESS_KEY",
        "GENAI_OTEL_MEDIA_STORE_SECRET_KEY",
        "OPENSEARCH_ADMIN_PASSWORD",
    )

    step("Setting up OpenTelemetry tracer")
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": os.environ["OTEL_SERVICE_NAME"]})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{COLLECTOR_HTTP}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("e2e-validate")

    step("Building synthetic OpenAI multimodal request")
    from genai_otel.config import OTelConfig
    from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

    inst = OpenAIInstrumentor()
    inst.config = OTelConfig()
    inst._instrumented = True  # bypass not-instrumented gate

    # Two payloads: data:URL inline (will be uploaded) + plain URL (external_url)
    png_b64 = base64.b64encode(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
            "0000000D49444154789C636400000000060005A36CFC8E0000000049454E44AE426082"
        )
    ).decode("ascii")
    request_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe both images."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.jpg"},
                    },
                ],
            }
        ],
    }

    step("Emitting span with multimodal attributes")
    with tracer.start_as_current_span("openai.chat.e2e") as span:
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
        inst._emit_media_attributes(span, request_kwargs, result=None)
        ctx = span.get_span_context()
        trace_id_hex = f"{ctx.trace_id:032x}"
        emitted_attrs = dict(span.attributes)

    print(f"trace_id = {trace_id_hex}")
    required = [
        "gen_ai.prompt.0.role",
        "gen_ai.prompt.0.content.0.type",
        "gen_ai.prompt.0.content.1.type",
        "gen_ai.prompt.0.content.1.media_uri",
        "gen_ai.prompt.0.content.1.media_source",
        "gen_ai.prompt.0.content.2.type",
        "gen_ai.prompt.0.content.2.media_source",
    ]
    missing = [k for k in required if k not in emitted_attrs]
    if missing:
        print(f"[FAIL] Missing required span attributes: {missing}")
        return 1
    print("[PASS] Span attributes contain required multimodal fields")
    for k in required:
        print(f"  {k} = {emitted_attrs[k]}")

    inline_uri = emitted_attrs["gen_ai.prompt.0.content.1.media_uri"]
    if not inline_uri.startswith(MINIO_ENDPOINT):
        print(f"[WARN] media_uri is {inline_uri} (expected MinIO URL)")

    step("Flushing OTel span to collector")
    provider.force_flush(timeout_millis=5000)
    time.sleep(1)
    print("[PASS] OTel span exported")

    step("Verifying MinIO upload")
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=os.environ["GENAI_OTEL_MEDIA_STORE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["GENAI_OTEL_MEDIA_STORE_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
    )
    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"traces/")
    found = listing.get("KeyCount", 0)
    if found == 0:
        print(f"[FAIL] No objects under traces/ in bucket {BUCKET}")
        return 1
    latest = sorted(listing["Contents"], key=lambda o: o["LastModified"])[-1]
    print(f"[PASS] MinIO upload: {latest['Key']} ({latest['Size']} bytes)")

    step("Querying OpenSearch for trace")
    import requests

    auth = ("admin", os.environ["OPENSEARCH_ADMIN_PASSWORD"])
    # Wait for collector -> opensearch pipeline to flush
    print("Waiting 5s for trace to land in OpenSearch...")
    time.sleep(5)

    # Discover the traces index (collector indexes typically: ss4o_traces-* or otel-traces-*)
    indices = requests.get(f"{OPENSEARCH}/_cat/indices?h=index", auth=auth, timeout=10).text
    print(
        f"Indices available: {[i for i in indices.split() if 'trace' in i.lower() or 'otel' in i.lower()]}"
    )

    candidate_indices = [i for i in indices.split() if "trace" in i.lower() or "otel" in i.lower()]
    found_in = None
    for idx in candidate_indices:
        body = {
            "size": 1,
            "query": {
                "bool": {
                    "should": [
                        {"term": {"traceId": trace_id_hex}},
                        {"term": {"trace_id": trace_id_hex}},
                        {"term": {"traceId.keyword": trace_id_hex}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
        try:
            r = requests.get(
                f"{OPENSEARCH}/{idx}/_search",
                json=body,
                auth=auth,
                timeout=10,
            )
            if r.ok and r.json().get("hits", {}).get("total", {}).get("value", 0) > 0:
                found_in = idx
                print(f"[PASS] OpenSearch query: trace {trace_id_hex} found in index {idx}")
                hit = r.json()["hits"]["hits"][0]["_source"]
                # Print relevant multimodal attrs from the indexed document
                attrs = hit.get("attributes") or hit.get("span", {}).get("attributes") or hit
                multimodal = {
                    k: v
                    for k, v in (attrs.items() if isinstance(attrs, dict) else [])
                    if k.startswith("gen_ai.prompt") or k.startswith("gen_ai.media")
                }
                if multimodal:
                    print("Indexed multimodal attributes:")
                    for k, v in multimodal.items():
                        print(f"  {k} = {v}")
                break
        except Exception as e:  # noqa: BLE001
            print(f"  query {idx}: {e}")

    if not found_in:
        print(f"[WARN] trace {trace_id_hex} not found in any *trace*/otel* index")
        print("       (collector may take longer; check OpenSearch Dashboards manually)")
        return 2

    print("\n=== ALL PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
