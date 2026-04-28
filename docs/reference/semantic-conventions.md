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
| `gen_ai.usage.prompt_tokens` | int | Input token count |
| `gen_ai.usage.completion_tokens` | int | Output token count |
| `gen_ai.usage.total_tokens` | int | Total token count |
| `gen_ai.cost.amount` | float | Estimated cost in USD |

### Cost Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.usage.cost.total` | float | Total cost in USD |
| `gen_ai.usage.cost.prompt` | float | Prompt token cost |
| `gen_ai.usage.cost.completion` | float | Completion token cost |

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
