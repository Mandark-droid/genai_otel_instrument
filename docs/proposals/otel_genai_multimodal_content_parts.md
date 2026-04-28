# Proposal: Multimodal Content-Part Attributes for OpenTelemetry GenAI Semantic Conventions

**Status:** Draft, pre-submission
**Target repo:** `open-telemetry/semantic-conventions`
**Author:** Kshitij Thakkar, `genai-otel-instrument` maintainer
**Last updated:** April 2026

## 1. Problem

OpenTelemetry GenAI semantic conventions (as of April 2026) define structured chat-message
attributes — `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions` —
and an enum for `gen_ai.output.type` (`text`, `json`, `image`, `speech`).

What is **not** standardized:

1. How a single message's *content parts* are individually attributed.
2. How non-text content (images, audio, video, documents) is referenced when bytes have been
   offloaded to an external blob store.
3. How instrumentation should record the MIME type, byte size, and provenance of each
   multimodal part.

Multimodal LLM workloads — vision input, audio input/output, video understanding (CCTV
summarization, sports/coaching analytics, video Q&A with Gemini and similar models), and
document understanding (PDF KYC/loan-form extraction is a high-volume real-world workload) —
are increasingly common in production. Without an OTel-defined attribute shape for these parts, every observability
backend ends up inventing its own multimodal schema, fragmenting the ecosystem in exactly the
way the GenAI semantic conventions effort exists to prevent. This proposal sets the open
standard for multimodal AI observability so that traces remain portable across any
OTel-compatible backend.

## 2. Goals

- Define a vendor-neutral attribute namespace for multimodal content parts.
- Keep text-only traces byte-identical to today's behaviour (additive, opt-in).
- Allow binaries to be offloaded to user-owned blob storage and referenced by URI.
- Be implementable by every major LLM SDK (OpenAI, Anthropic, Google Gemini, AWS Bedrock,
  Azure, Cohere, etc.).

## 3. Non-Goals

- Standardizing the offload protocol itself (S3, MinIO, presigners, etc. are out of scope).
- Defining how UI consumers render media (renderer-side, not protocol-side).
- Audio diarization (speaker-aware redaction) — deferred until a follow-up proposal.

## 4. Proposed Attributes

Add to `gen_ai-spans.md` under a new section, **"Multimodal content parts"**.

### 4.1 Per-message role (already present in some emitters)

| Attribute | Type | Example |
|---|---|---|
| `gen_ai.prompt.{n}.role` | string | `user` |
| `gen_ai.completion.{n}.role` | string | `assistant` |

### 4.2 Per-content-part attributes (new)

| Attribute | Type | Required when | Example |
|---|---|---|---|
| `gen_ai.prompt.{n}.content.{m}.type` | string enum | always (if part captured) | `text` \| `image` \| `audio` \| `video` \| `document` |
| `gen_ai.prompt.{n}.content.{m}.text` | string | type=text | `"Describe this image."` |
| `gen_ai.prompt.{n}.content.{m}.media_uri` | string | type≠text and bytes captured | `s3://bucket/key`, `https://...` |
| `gen_ai.prompt.{n}.content.{m}.media_mime_type` | string | type≠text | `image/png` |
| `gen_ai.prompt.{n}.content.{m}.media_byte_size` | int | type≠text | `123456` |
| `gen_ai.prompt.{n}.content.{m}.media_source` | string enum | type≠text | `inline_offloaded` \| `external_url` \| `reference_only` |
| `gen_ai.media.stripped_reason` | string | when bytes intentionally not captured | `size_exceeded` \| `modality_not_allowed` \| `redactor_error` \| `upload_error` |

The `gen_ai.completion.*` namespace mirrors `gen_ai.prompt.*` for generated content (e.g.
image generation output, TTS).

### 4.3 Enum semantics

`type`:
- `text`: pure text. The `text` attribute carries the value.
- `image`: a static image (PNG, JPEG, WebP, etc.).
- `audio`: an audio clip (WAV, MP3, OGG, etc.).
- `video`: a video clip (MP4, WebM, MOV, etc.).
- `document`: a document (PDF, DOCX, etc.).

`media_source`:
- `inline_offloaded`: the part was sent inline by the caller (e.g. base64), the instrumentor
  uploaded it to a configured store, and `media_uri` resolves to that copy.
- `external_url`: the caller already provided a URL the model fetched directly; the instrumentor
  did not re-host it. `media_uri` is the original URL.
- `reference_only`: bytes were detected but intentionally not stored (compliance / cost). Only
  modality + MIME + size are recorded.

## 5. Backwards Compatibility

This proposal is **purely additive**. Text-only traces emit no new attributes. Existing
consumers that don't recognize `content.{m}.type=image` can degrade gracefully — the
`gen_ai.prompt.{n}.role` is still readable, and the absence of `text` parts will simply make
the message appear empty, not malformed.

## 6. Relationship to `gen_ai.input.messages`

The existing `gen_ai.input.messages` blob (opt-in, JSON-serialized) and the proposed flat
attributes are not redundant — they serve different consumer profiles:

- **Flat attributes** are queryable in any OTel backend without JSON parsing, suitable for
  dashboards and alerting (e.g. count traces with `content.{m}.type=image`).
- **JSON blob** preserves arbitrary nesting and provider-specific fields the flat attributes
  don't model (tool calls inside content, etc.).

We propose emitters MAY emit either or both, controlled by `OTEL_SEMCONV_STABILITY_OPT_IN`
(existing dual-emission pattern).

## 7. Provider Coverage Sketch

| Provider | Image | Audio | Video | Document |
|---|---|---|---|---|
| OpenAI Chat | `image_url` part → `type=image` | `input_audio` part → `type=audio` | `input_video` part / `file_data` MIME → `type=video` | `file` part → `type=document` |
| Anthropic Messages | `image` block → `type=image` | — | — | `document` block → `type=document` |
| Google Gemini | `inline_data`/`file_data` MIME → derived | `inline_data` audio MIME | `inline_data`/`file_data` `video/*` MIME | `inline_data` PDF MIME |
| AWS Bedrock Converse | `image`/`document` blocks | — | — | `document` blocks |

## 8. Reference Implementation

`genai-otel-instrument` v1.0.0 (April 2026) ships a working implementation:

- Detector module: `genai_otel/media/detector.py`
- Offload pipeline: `genai_otel/media/offload.py`
- Pluggable stores: `genai_otel/media/stores/{filesystem,s3_minio,http}.py`
- Pluggable redactors: `genai_otel/media/redactors.py`

This serves as both proof of feasibility and a public conformance test corpus once the
attributes are accepted upstream.

## 9. Draft GitHub Issue Body

> ### Feature Request: Standardize multimodal content-part attributes
>
> The current GenAI semconv defines `gen_ai.input.messages` and `gen_ai.output.type` (with
> `image`/`speech` as enum values), but does not standardize how individual non-text content
> parts are attributed when bytes are offloaded to a blob store.
>
> I'd like to propose an additive attribute namespace, with a working reference implementation
> in `genai-otel-instrument` v1.0.0. Full proposal:
> https://github.com/Mandark-droid/genai_otel_instrument/blob/main/docs/proposals/otel_genai_multimodal_content_parts.md
>
> Would the working group be open to a PR adding these attributes to `gen_ai-spans.md`?

## 10. Open Questions

1. Naming: `gen_ai.prompt.{n}.content.{m}.media_uri` vs. shorter `gen_ai.prompt.{n}.content.{m}.uri`.
   We chose `media_*` to disambiguate from any future `tool.uri` etc., but happy to defer to
   the WG.
2. Should `media_source` include a `data_url` value for cases where small images are kept inline
   as `data:` URLs? Probably yes, as a fourth enum value.
3. How does this interact with `gen_ai.tool.call.arguments` when a tool result is itself
   multimodal (e.g. browser screenshots from computer-use agents)? Likely needs a parallel
   `gen_ai.tool.call.result.content.{m}.*` mirror — TBD in a follow-up.
