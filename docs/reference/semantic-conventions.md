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
