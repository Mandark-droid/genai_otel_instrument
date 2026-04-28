"""End-to-end run across all 4 modalities for downstream enterprise consumption.

Drives synthetic image, audio, video, and document requests through the genai_otel
multimodal pipeline against the production MinIO bucket. Verifies:

- Each modality produces a span with correct multimodal attributes
- Bytes land in MinIO under traces/<date>/<trace_id>/
- The OTel-canonical gen_ai.input.messages JSON is also emitted (dual-emission)

No LLM API key required.

Run:
    export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=...
    export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=...
    python examples/multimodal/e2e_all_modalities.py

Output: per-modality status + a summary of MinIO objects under
http://192.168.206.129:9000/genai-otel-media/traces/<today>/.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from datetime import datetime, timezone

VM = "192.168.206.129"
COLLECTOR_HTTP = f"http://{VM}:4318"
MINIO_ENDPOINT = f"http://{VM}:9000"
BUCKET = "genai-otel-media"

os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", COLLECTOR_HTTP)
os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
os.environ.setdefault("OTEL_SERVICE_NAME", "genai-otel-multimodal-prod")
os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai")
os.environ.setdefault("GENAI_OTEL_MEDIA_CAPTURE_MODE", "full")
os.environ.setdefault("GENAI_OTEL_MEDIA_STORE", "minio")
os.environ.setdefault("GENAI_OTEL_MEDIA_STORE_ENDPOINT", MINIO_ENDPOINT)
os.environ.setdefault("GENAI_OTEL_MEDIA_STORE_BUCKET", BUCKET)


def _require_env(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print(f"[FAIL] Missing env vars: {', '.join(missing)}")
        sys.exit(1)


# Tiny synthetic payloads — small enough to be obviously fake, big enough to round-trip.
PNG_BYTES = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
    "0000000D49444154789C636400000000060005A36CFC8E0000000049454E44AE426082"
)
WAV_BYTES = base64.b64decode("UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=")
MP4_BYTES = bytes.fromhex("0000001866747970697333366d000000016973336d6f6d703431" "0000000800000000")
PDF_BYTES = (
    b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]>>endobj\n"
    b"trailer<</Size 4/Root 1 0 R>>\n%%EOF\n"
)


def build_request(modality: str) -> dict:
    """Build a synthetic OpenAI- or Anthropic-shaped request for the modality."""
    if modality == "image":
        b64 = base64.b64encode(PNG_BYTES).decode()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
    if modality == "audio":
        b64 = base64.b64encode(WAV_BYTES).decode()
        return {
            "model": "gpt-4o-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this clip."},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"},
                        },
                    ],
                }
            ],
        }
    if modality == "video":
        b64 = base64.b64encode(MP4_BYTES).decode()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this video clip."},
                        {
                            "type": "input_video",
                            "input_video": {"url": f"data:video/mp4;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
    if modality == "document":
        b64 = base64.b64encode(PDF_BYTES).decode()
        # Anthropic-shaped: document content block
        return {
            "_provider": "anthropic",
            "model": "claude-3-5-sonnet-latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": "Extract the key fields."},
                    ],
                }
            ],
        }
    raise ValueError(modality)


def main() -> int:
    _require_env("GENAI_OTEL_MEDIA_STORE_ACCESS_KEY", "GENAI_OTEL_MEDIA_STORE_SECRET_KEY")

    # OTel setup
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": os.environ["OTEL_SERVICE_NAME"]})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{COLLECTOR_HTTP}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("e2e-all-modalities")

    from genai_otel.config import OTelConfig
    from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor
    from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

    cfg = OTelConfig()
    openai_inst = OpenAIInstrumentor()
    openai_inst.config = cfg
    openai_inst._instrumented = True
    anthropic_inst = AnthropicInstrumentor()
    anthropic_inst.config = cfg
    anthropic_inst._instrumented = True

    summary = []
    for modality in ("image", "audio", "video", "document"):
        kwargs = build_request(modality)
        inst = anthropic_inst if kwargs.pop("_provider", None) == "anthropic" else openai_inst
        span_name = f"{inst.MEDIA_PROVIDER}.chat.{modality}"
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("gen_ai.system", inst.MEDIA_PROVIDER)
            span.set_attribute("gen_ai.request.model", kwargs.get("model", ""))
            inst._emit_media_attributes(span, kwargs, result=None)
            ctx = span.get_span_context()
            trace_id = f"{ctx.trace_id:032x}"
            attrs = dict(span.attributes)

        # Verify on-span shape
        flat_type = attrs.get("gen_ai.prompt.0.content.1.type") or attrs.get(
            "gen_ai.prompt.0.content.0.type"
        )
        flat_uri = attrs.get("gen_ai.prompt.0.content.1.media_uri") or attrs.get(
            "gen_ai.prompt.0.content.0.media_uri"
        )
        flat_source = attrs.get("gen_ai.prompt.0.content.1.media_source") or attrs.get(
            "gen_ai.prompt.0.content.0.media_source"
        )
        canonical_blob = attrs.get("gen_ai.input.messages")
        canonical_parsed = json.loads(canonical_blob) if canonical_blob else None

        ok = (
            flat_type
            and flat_uri
            and flat_source == "inline_offloaded"
            and canonical_parsed is not None
        )
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {modality:8s}  trace_id={trace_id}")
        print(f"  flat: type={flat_type}  source={flat_source}")
        print(f"        uri={flat_uri}")
        if canonical_parsed:
            canonical_part = next(
                (
                    p
                    for m in canonical_parsed
                    for p in m["parts"]
                    if p.get("type") in ("uri", "blob", "stripped")
                ),
                None,
            )
            print(f"  canonical: {json.dumps(canonical_part, separators=(',', ':'))[:120]}")
        summary.append((modality, ok, trace_id, flat_uri))

    provider.force_flush(timeout_millis=5000)
    time.sleep(1)

    print("\n=== MinIO listing ===")
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=os.environ["GENAI_OTEL_MEDIA_STORE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["GENAI_OTEL_MEDIA_STORE_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
    )
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = f"traces/{today}/"
    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    objs = listing.get("Contents", [])
    matched = []
    for _modality, _ok, trace_id, _uri in summary:
        for obj in objs:
            if trace_id in obj["Key"]:
                matched.append((trace_id, obj["Key"], obj["Size"]))
    for trace_id, key, size in matched:
        print(f"  {key}  ({size} bytes)")

    print("\n=== SUMMARY ===")
    print(f"Bucket: {BUCKET} @ {MINIO_ENDPOINT}")
    print(f"Spans emitted: {len(summary)}")
    print(f"MinIO objects under {prefix} matching this run: {len(matched)}")
    failed = [m for m, ok, _, _ in summary if not ok]
    if failed:
        print(f"FAILED modalities: {failed}")
        return 1
    print("All 4 modalities (image, audio, video, document) round-tripped successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
