"""Replay one real TraceSense document-extraction turn through this package.

The input is a real PDF or image from the TraceVerse Chaos Lab.  This is an
offline telemetry replay: it validates the document content-part shape and
the OCR workflow hierarchy without pretending that a local fallback is a
provider-backed OCR result.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor


def _document_text(path: Path) -> str:
    """Extract text when a local PDF reader is available; otherwise stay honest."""

    if path.suffix.lower() != ".pdf":
        return ""
    try:
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages).strip()
    except Exception:  # noqa: BLE001
        return ""


def _media_shape(data: bytes, mime_type: str) -> tuple[str, dict[str, Any]]:
    if mime_type == "application/pdf":
        return "document", {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": base64.b64encode(data).decode("ascii"),
            },
        }
    return "image", {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def run(document_path: Path, endpoint: str | None, export: bool) -> dict[str, Any]:
    data = document_path.read_bytes()
    mime_type = mimetypes.guess_type(document_path.name)[0] or "application/octet-stream"
    if mime_type not in {"application/pdf", "image/png", "image/jpeg", "image/webp"}:
        raise ValueError(f"Unsupported OCR input type: {mime_type}")

    config = OTelConfig()
    config.media_capture_mode = "reference_only"
    config.media_store = "none"
    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "tracesense-document-ocr-real-world",
                "deployment.environment": "validation",
            }
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    if export and endpoint:
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
        )
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("examples.real_world.document_ocr")
    media = AnthropicInstrumentor()
    media.config = config
    media._instrumented = True
    modality, content_part = _media_shape(data, mime_type)
    extracted_text = _document_text(document_path)

    with tracer.start_as_current_span("tracesense.document_ocr.workflow") as root:
        root.set_attribute("document.workflow", "tracesense.extract_document")
        root.set_attribute("document.input.filename", document_path.name)
        root.set_attribute("document.input.mime_type", mime_type)
        root.set_attribute("document.input.bytes", len(data))

        with tracer.start_as_current_span("tracesense.document.ingest") as ingest:
            ingest.set_attribute("document.modality", modality)
            ingest.set_attribute("document.pages", extracted_text.count("\f") + 1)
            ingest.set_attribute("document.text_chars", len(extracted_text))

        with tracer.start_as_current_span("tracesense.document.ocr") as ocr:
            ocr.set_attribute("gen_ai.system", "ollama-compatible")
            ocr.set_attribute("gen_ai.request.model", "gemma-4-E2B-it")
            ocr.set_attribute("gen_ai.operation.name", "document_extraction")
            media._emit_media_attributes(
                ocr,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract the structured fields from this document.",
                                },
                                content_part,
                            ],
                        }
                    ]
                },
                result=None,
            )
            ocr.set_attribute("document.ocr.extracted_text_chars", len(extracted_text))

        with tracer.start_as_current_span("tracesense.document.validation") as validation:
            validation.set_attribute("document.validation.input_read", True)
            validation.set_attribute("document.validation.provider_result", False)
            validation.set_attribute("document.validation.replay_only", True)

        trace_id = f"{root.get_span_context().trace_id:032x}"

    provider.force_flush(timeout_millis=15000)
    provider.shutdown()
    spans = memory_exporter.get_finished_spans()
    ocr_span = next(span for span in spans if span.name == "tracesense.document.ocr")
    emitted_type = ocr_span.attributes.get("gen_ai.prompt.0.content.1.type")
    emitted_mime = ocr_span.attributes.get("gen_ai.prompt.0.content.1.media_mime_type")
    expected_type = "document" if mime_type == "application/pdf" else "image"
    if emitted_type != expected_type or emitted_mime != mime_type:
        raise AssertionError(
            f"Expected {expected_type}/{mime_type}, got {emitted_type!r}/{emitted_mime!r}"
        )
    result = {
        "trace_id": trace_id,
        "span_count": len(spans),
        "document": str(document_path),
        "content_type": emitted_type,
        "mime_type": emitted_mime,
        "extracted_text_chars": len(extracted_text),
        "provider_result": False,
        "replay_only": True,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    )
    parser.add_argument("--no-export", action="store_true", help="Only validate the local trace")
    args = parser.parse_args()
    if not args.document.is_file() or args.document.stat().st_size == 0:
        print(f"Document not found or empty: {args.document}", file=sys.stderr)
        return 2
    run(args.document, args.endpoint, export=not args.no_export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
