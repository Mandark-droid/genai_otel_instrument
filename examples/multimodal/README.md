# Multimodal Observability Examples

These examples demonstrate the multimodal observability features added in genai-otel-instrument v1.0.0:
images, audio, video, and document content parts captured as OpenTelemetry spans, with optional offload to
a configured blob store (MinIO / S3 / filesystem / HTTP).

## Setup

```bash
pip install -e ".[multimodal,openai,anthropic,google]"

# 1. Choose capture mode (default: off)
export GENAI_OTEL_MEDIA_CAPTURE_MODE=full        # off | reference_only | full

# 2. Pick a store backend
export GENAI_OTEL_MEDIA_STORE=minio              # none | filesystem | s3 | minio | http
export GENAI_OTEL_MEDIA_STORE_ENDPOINT=http://localhost:9000
export GENAI_OTEL_MEDIA_STORE_BUCKET=genai-otel-media
export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=...
export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=...

# 3. (Optional) Plug in a redactor before upload
export GENAI_OTEL_MEDIA_REDACTOR=genai_otel.media.redactors.exif_stripper
```

For an air-gapped local run, use the filesystem store:

```bash
export GENAI_OTEL_MEDIA_STORE=filesystem
export GENAI_OTEL_MEDIA_STORE_ENDPOINT=./.genai-otel-media
```

## Examples

| File | What it shows |
|---|---|
| `openai_vision.py` | OpenAI chat with an `image_url` content part |
| `anthropic_vision.py` | Anthropic message with a base64 image block |
| `gemini_multimodal.py` | Google Gemini `inline_data` audio part |
| `gemini_video.py` | Google Gemini `inline_data` video clip (MP4) |
| `document_extraction.py` | Anthropic document (PDF) content block |
| `redactor_face_blur.py` | Plug in `face_blur` redactor before MinIO upload |

Each example sets up a console exporter so you can see the resulting span attributes
(`gen_ai.prompt.0.content.1.type=image`, `media_uri=...`, `media_mime_type=...`).

## Capture modes

- **`off`** (default): no multimodal attributes emitted; identical behaviour to pre-1.0.0.
- **`reference_only`**: modality + MIME + byte size captured, bytes are *not* stored. Good
  for compliance-sensitive deployments that want metadata without the privacy footprint.
- **`full`**: redact (if configured) → upload to store → emit URI. Bytes never appear in
  span attributes.

## Verifying with TraceVerse / OTLP collector

When `media_capture_mode=full` and `media_store=minio` point at the TraceVerse VM
(`http://192.168.206.129:9000`), spans will reference uploaded blobs by their MinIO URL
and the TraceVerse UI's multimodal renderers can fetch them via signed URLs.
