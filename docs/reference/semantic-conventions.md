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
| `gen_ai.cost.amount` | float | Estimated cost in USD |

### Cost Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.usage.cost.total` | float | Total cost in USD |
| `gen_ai.usage.cost.prompt` | float | Prompt token cost |
| `gen_ai.usage.cost.completion` | float | Completion token cost |
| `gen_ai.usage.cost.pricing_source` | string | Where the price came from: `table`, `estimated` or `unpriced` |

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
