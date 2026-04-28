# Multimodal Observability

`genai-otel-instrument` v1.0.0 adds first-class capture of multimodal content — images, audio,
video, and documents — alongside the existing text instrumentation. Multimodal payloads never
appear inline in span attributes; bytes are offloaded to a configured store and referenced by URI.

## Why this exists

OpenTelemetry GenAI semantic conventions standardize text prompts/completions but do **not**
standardize how multimodal attachments — images, audio, video, documents — are captured on spans.
This library defines that shape with an additive, OTel-compatible attribute namespace plus a
pluggable offload pipeline so binary content lives in your object store, not in span attributes.
(See `docs/proposals/otel_genai_multimodal_content_parts.md` for the upstream proposal we're
filing with `open-telemetry/semantic-conventions`.)

## Quickstart

```bash
pip install 'genai-otel-instrument[multimodal,openai,anthropic,google]'
```

```bash
export GENAI_OTEL_MEDIA_CAPTURE_MODE=full
export GENAI_OTEL_MEDIA_STORE=minio
export GENAI_OTEL_MEDIA_STORE_ENDPOINT=http://localhost:9000
export GENAI_OTEL_MEDIA_STORE_BUCKET=genai-otel-media
export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=...
export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=...
```

```python
from genai_otel import instrument
instrument()

# Any subsequent OpenAI / Anthropic / Gemini / Groq call with multimodal content
# automatically emits content-part attributes and uploads bytes to MinIO.
```

## Capture modes

| `GENAI_OTEL_MEDIA_CAPTURE_MODE` | Behaviour |
|---|---|
| `off` (default) | No multimodal attributes emitted. Identical to pre-1.0.0. |
| `reference_only` | Modality + MIME + byte size captured. Bytes NOT stored. |
| `full` | Redact (if configured) → upload to store → reference by URI. |

`off` is the default to preserve byte-identical behaviour for existing users. BFSI deployments
that want to record multimodal *presence* without storing bytes should use `reference_only`.
Production deployments that want full traceability + UI rendering use `full` plus a store.

## Store backends

Set via `GENAI_OTEL_MEDIA_STORE`:

| Value | Notes |
|---|---|
| `none` (default) | No offload. Works with `capture_mode=reference_only`. |
| `filesystem` | Local directory. `MEDIA_STORE_ENDPOINT` is the root path. |
| `s3` / `minio` | S3-compatible object storage. Requires the `multimodal-s3` extra. |
| `http` | PUTs to a configured base URL (self-hosted ingest gateways). |

Per-blob size is capped by `GENAI_OTEL_MEDIA_MAX_BYTES` (default 10 MiB). Oversized blobs are
recorded with `gen_ai.media.stripped_reason="size_exceeded"` and the bytes are dropped.

## Redactors

Redactors run on raw bytes **before** upload, per modality. Built-ins:

- `genai_otel.media.redactors.exif_stripper` — removes EXIF/metadata from images. Requires `Pillow`.
- `genai_otel.media.redactors.face_blur` — Haar-cascade face detection + Gaussian blur. Requires `opencv-python-headless`.
- `genai_otel.media.redactors.pdf_pii_redact` — regex-based PII removal in PDFs. Requires `pypdf`.

Plug one in:

```bash
export GENAI_OTEL_MEDIA_REDACTOR=genai_otel.media.redactors.face_blur
```

Or write your own:

```python
def my_redactor(modality: str, mime_type: str, data: bytes) -> bytes:
    if modality == "image":
        return strip_pii_from_image(data)
    return data
```

```bash
export GENAI_OTEL_MEDIA_REDACTOR=mypkg.my_redactor
```

If the redactor raises, the bytes are dropped and `stripped_reason=redactor_error` is recorded —
fail-closed, never fail-open with unredacted bytes.

## Span attribute reference

For every multimodal message, the instrumentor emits:

| Attribute | Example |
|---|---|
| `gen_ai.prompt.{n}.role` | `user` |
| `gen_ai.prompt.{n}.content.{m}.type` | `text` \| `image` \| `audio` \| `video` \| `document` |
| `gen_ai.prompt.{n}.content.{m}.text` | text parts only |
| `gen_ai.prompt.{n}.content.{m}.media_uri` | `s3://bucket/key` or `https://...` |
| `gen_ai.prompt.{n}.content.{m}.media_mime_type` | `image/png` |
| `gen_ai.prompt.{n}.content.{m}.media_byte_size` | `123456` |
| `gen_ai.prompt.{n}.content.{m}.media_source` | `inline_offloaded` \| `external_url` \| `reference_only` |
| `gen_ai.media.stripped_reason` | `size_exceeded`, `modality_not_allowed`, `redactor_error`, `upload_error` |

The `gen_ai.completion.*` namespace mirrors this for generated content (e.g. image generation
output, TTS responses).

## Provider coverage

| Provider | Image | Audio | Video | Document |
|---|---|---|---|---|
| OpenAI / OpenRouter / Groq | `image_url` (URL or data:) | `input_audio` | `input_video` (URL / data:) | via `file` blocks |
| Anthropic | `image` (base64 / url) | — | — | `document` (base64 / url) |
| Google Gemini | `inline_data` / `file_data` | `inline_data` | `inline_data` / `file_data` (`video/*`) | `inline_data` (PDF) |

## Configuration reference

| Env var | Default | Notes |
|---|---|---|
| `GENAI_OTEL_MEDIA_CAPTURE_MODE` | `off` | `off` \| `reference_only` \| `full` |
| `GENAI_OTEL_MEDIA_STORE` | `none` | `none` \| `filesystem` \| `s3` \| `minio` \| `http` |
| `GENAI_OTEL_MEDIA_STORE_ENDPOINT` | — | URL or local path |
| `GENAI_OTEL_MEDIA_STORE_BUCKET` | `genai-otel-media` | |
| `GENAI_OTEL_MEDIA_STORE_PREFIX` | `traces/{date}/{trace_id}/` | template |
| `GENAI_OTEL_MEDIA_STORE_ACCESS_KEY` | — | for s3/minio |
| `GENAI_OTEL_MEDIA_STORE_SECRET_KEY` | — | for s3/minio |
| `GENAI_OTEL_MEDIA_MAX_BYTES` | `10485760` | per-blob size cap |
| `GENAI_OTEL_MEDIA_ALLOWED_MODALITIES` | `image,audio,video,document` | comma-separated |
| `GENAI_OTEL_MEDIA_REDACTOR` | — | dotted path to callable |
