# Semantic Conventions

TraceVerde follows OpenTelemetry semantic conventions for GenAI with additional custom attributes.

## Span Attributes

### GenAI Core (OTel Standard)

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.system` | string | Provider name (e.g., "openai", "anthropic") |
| `gen_ai.request.model` | string | Requested model identifier |
| `gen_ai.response.model` | string | Actual model used in response |
| `gen_ai.request.type` | string | Call type ("chat", "embedding", "completion") |
| `gen_ai.usage.input_tokens` | int | Input token count |
| `gen_ai.usage.output_tokens` | int | Output token count |
| `gen_ai.usage.total_tokens` | int | Total token count |
| `gen_ai.usage.token_count_estimated` | bool | `true` when prompt/completion token counts came from a fallback estimate (e.g. multimodal Ollama responses lacking `prompt_eval_count`, or HuggingFace vision/audio pipelines that don't surface usage). Absent on spans whose tokens come from the provider response. |
| `gen_ai.usage.image_count` | int | Number of input images counted by the instrumentor for multimodal calls (vision pipelines). |
| `gen_ai.usage.audio_seconds` | float | Total input audio duration in seconds for ASR / audio pipelines. |
| `gen_ai.usage.audio_duration_seconds` | float | Canonical total input audio duration in seconds for speech-to-text and audio pipelines. |
| `gen_ai.operation.name` | string | Operation being performed (`chat`, `embeddings`) |

!!! note "Cost lives under `gen_ai.usage.cost.*`"

    Earlier revisions of this page listed `gen_ai.cost.amount` as the cost
    attribute. Cost is reported under
    [`gen_ai.usage.cost.total`](#cost-attributes) and its breakdown. Only the
    Hyperbolic instrumentor also emits `gen_ai.cost.amount`, for backwards
    compatibility with dashboards built against it; do not query it as a
    general cost attribute, because no other provider sets it.

### Embeddings

Emitted on embedding spans (`openai.embeddings`), which are the retrieval half
of a RAG trace.

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.request.type` | string | `embedding` - the value cost lookup dispatches on |
| `gen_ai.request.input_count` | int | Number of texts in the request; a batch of chunks counts each one, pre-tokenised input counts as one |
| `gen_ai.request.dimensions` | int | Requested vector dimension, when the caller asked for one |
| `gen_ai.request.encoding_format` | string | Requested encoding format, when set |
| `gen_ai.response.embedding_count` | int | Number of vectors returned |
| `gen_ai.response.vector_size` | int | Dimension of the returned vectors |
| `embedding.model_name` | string | Embedding model; set only when content capture is enabled |
| `embedding.text` | string | The embedded text, truncated to `GENAI_CONTENT_MAX_LENGTH`; set only when content capture is enabled |
| `embedding.vector` | string | JSON-encoded vector; off unless `capture_embedding_vectors` is set, since vectors dominate span size |

Token usage and cost are recorded exactly as for chat, priced against the
`embeddings` table rather than the chat one.

### Retrieval quality

The `rag.*` attributes are additive to `retrieval.*` and `db.vector.*`. They
describe embedding provenance, score quality, corpus version, and downstream
context policy. `top_k` and result count remain `db.vector.top_k` and
`retrieval.document_count`; see the [retrieval quality guide](../guides/retrieval-quality.md).

`gen_ai.rag.context` is application-owned input for evaluation processors. The
library reads it when present but does not emit it automatically.

### Degradation events

Capability downgrades are recorded as a `gen_ai.degraded` span event rather
than an error. The event attributes are `gen_ai.degraded.component`,
`gen_ai.degraded.from`, `gen_ai.degraded.to`, `gen_ai.degraded.reason`, and
`gen_ai.degraded.recoverable`. The span status remains `OK` when the request
successfully completes through the degraded path.

### Cost Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.usage.cost.total` | float | Total cost in USD |
| `gen_ai.usage.cost.prompt` | float | Prompt token cost |
| `gen_ai.usage.cost.completion` | float | Completion token cost |
| `gen_ai.usage.cost.pricing_source` | string | Where the price came from: `table`, `estimated` or `unpriced` |
| `gen_ai.request.model.deprecated` | bool | Set only when the provider has announced the model's retirement |
| `gen_ai.request.model.deprecation_note` | string | Retirement date and migration target |

!!! warning "Read `pricing_source` before summing cost"

    A `gen_ai.usage.cost.total` of `0.0` is ambiguous on its own — it means
    either "this call really was free" or "no price could be found for this
    model". Always read `pricing_source` alongside it:

    | Value | Meaning |
    |-------|---------|
    | `table` | Matched an entry in the shipped pricing file. Billable. |
    | `estimated` | No entry; price inferred from the parameter count in the model name via the local-model size tier. **Indicative only** — do not treat as a billable figure. |
    | `unpriced` | No price could be determined. `cost.total` will be `0.0` and must not be counted as free spend. |

    A dashboard that sums `cost.total` without filtering on `pricing_source`
    will silently under-report spend for every unpriced model.

### Streaming Latency

Emitted **only on streamed calls**. On a non-streamed call these attributes are
absent, not zero.

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.server.time_to_first_token` | float | Seconds from request start to the first streamed chunk (TTFT) |
| `gen_ai.server.time_per_output_token` | float | Seconds per output token after the first (TPOT), `(duration - ttft) / max(output_tokens - 1, 1)` |
| `gen_ai.server.ttft` | float | Superseded spelling of TTFT, still emitted for consumers already reading it |
| `gen_ai.streaming.token_count` | int | Number of chunks yielded by the stream (chunks, not tokens) |
| `gen_ai.streaming.tpot_unavailable_reason` | string | Why TPOT was omitted; currently only `output_token_count_unavailable` |

!!! warning "An absent TTFT is not a zero"

    Span duration cannot substitute for TTFT: a fast-starting long answer and a
    slow-starting short one can share a duration. But a fabricated `0` is worse
    than nothing — it is indistinguishable from an instant first token and drags
    down every average it enters.

    So these attributes are omitted whenever they were not genuinely measured:

    - **Non-streamed calls** carry neither attribute.
    - **TPOT needs a real output-token count.** Some providers only send usage
      in the final chunk when asked. With OpenAI, for example, pass
      `stream_options={"include_usage": True}`; without it the span gets TTFT
      plus `gen_ai.streaming.tpot_unavailable_reason`, and no TPOT. The chunk
      count is *not* used as a stand-in — chunks are not tokens.
    - **Text-to-speech streams** report TTFT (time to first audio byte) and
      never TPOT, having no output tokens to divide by.

    Report a missing attribute as "not measured" rather than substituting zero.

### Session Tracking

| Attribute | Type | Description |
|-----------|------|-------------|
| `session.id` | string | Session identifier |
| `user.id` | string | User identifier |

### Resource Attributes

| Attribute | Description |
|-----------|-------------|
| `service.name` | Service name |
| `service.instance.id` | Instance identifier |
| `deployment.environment` | Environment name |
| `telemetry.auto.name` | "genai-otel-instrument" |
| `telemetry.auto.version` | Package version |

## Metrics

### GenAI Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `gen_ai.requests` | Counter | requests | Request count by provider/model |
| `gen_ai.client.token.usage` | Counter | tokens | Token usage (prompt/completion) |
| `gen_ai.client.operation.duration` | Histogram | seconds | Request latency |
| `gen_ai.cost` | Counter | USD | Estimated costs |
| `gen_ai.errors` | Counter | errors | Error count by type |
| `gen_ai.server.time_to_first_token` | Histogram | seconds | TTFT, recorded on streamed calls only |
| `gen_ai.server.time_per_output_token` | Histogram | seconds | TPOT, recorded when the output-token count is known |
| `gen_ai.server.ttft` | Histogram | seconds | Superseded spelling of TTFT |
| `gen_ai.server.tbt` | Histogram | seconds | Time between consecutive streamed chunks |

### GPU Metrics

See [GPU Metrics Guide](../guides/gpu-metrics.md) for the complete list.

### Evaluation Metrics

See [Evaluation Guide](../guides/evaluation.md) for detector-specific metrics.

### Multimodal Content-Part Attributes (v1.0.0)

Emitted only when `GENAI_OTEL_MEDIA_CAPTURE_MODE` is set to `reference_only` or `full`.
Default behaviour (`off`) emits nothing additional.

| Attribute | Type | Notes |
|---|---|---|
| `gen_ai.prompt.{n}.role` | string | `user`, `system`, `assistant`, `tool` |
| `gen_ai.prompt.{n}.content.{m}.type` | enum | `text` \| `image` \| `audio` \| `video` \| `document` |
| `gen_ai.prompt.{n}.content.{m}.text` | string | text parts only |
| `gen_ai.prompt.{n}.content.{m}.media_uri` | string | URI returned by the configured store |
| `gen_ai.prompt.{n}.content.{m}.media_mime_type` | string | e.g. `image/png` |
| `gen_ai.prompt.{n}.content.{m}.media_byte_size` | int | size of the captured payload |
| `gen_ai.prompt.{n}.content.{m}.media_source` | enum | `inline_offloaded` \| `external_url` \| `reference_only` |
| `gen_ai.completion.{n}.*` | — | mirror namespace for generated content |
| `gen_ai.media.stripped_reason` | enum | `size_exceeded`, `modality_not_allowed`, `redactor_error`, `upload_error` |

Not yet part of upstream OTel GenAI semconv. See
[the proposal](../proposals/otel_genai_multimodal_content_parts.md) for the upstream contribution we plan to file.
