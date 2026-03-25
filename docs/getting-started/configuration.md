# Configuration

TraceVerde can be configured via environment variables or programmatically.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SERVICE_NAME` | `genai-app` | Service name for traces |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTLP endpoint URL |
| `OTEL_EXPORTER_OTLP_HEADERS` | | Headers in `key1=val1,key2=val2` format |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | `10.0` | OTLP exporter timeout in seconds |
| `OTEL_SERVICE_INSTANCE_ID` | | Service instance identifier |
| `OTEL_ENVIRONMENT` | | Environment name (e.g., `production`) |
| `GENAI_ENABLE_GPU_METRICS` | `true` | Enable GPU metrics collection |
| `GENAI_ENABLE_COST_TRACKING` | `true` | Enable automatic cost calculation |
| `GENAI_ENABLE_MCP_INSTRUMENTATION` | `true` | Enable MCP tool instrumentation |
| `GENAI_FAIL_ON_ERROR` | `false` | Fail on instrumentation errors |
| `GENAI_OTEL_LOG_LEVEL` | `INFO` | Logging level |
| `GENAI_ENABLED_INSTRUMENTORS` | all defaults | Comma-separated list of instrumentors |
| `GENAI_SAMPLING_RATE` | `1.0` | Trace sampling rate (0.0-1.0) |
| `GENAI_CUSTOM_PRICING_JSON` | | Custom model pricing JSON |

## Programmatic Configuration

```python
import genai_otel

genai_otel.instrument(
    service_name="my-app",
    endpoint="http://localhost:4318",
    enable_gpu_metrics=True,
    enable_cost_tracking=True,
    enable_mcp_instrumentation=True,
    enabled_instrumentors=["openai", "anthropic", "crewai"],
    sampling_rate=0.5,
    fail_on_error=False,

    # Evaluation features
    enable_pii_detection=True,
    enable_toxicity_detection=True,
    enable_bias_detection=True,
    enable_prompt_injection_detection=True,
    enable_restricted_topics=True,
    enable_hallucination_detection=True,
)
```

## Telemetry (Opt-in)

Anonymous usage telemetry is disabled by default. To opt in:

```bash
export TRACEVERDE_TELEMETRY=true
export TRACEVERDE_TELEMETRY_URL=https://your-endpoint.example.com/v1/telemetry
```

See `genai_otel/telemetry.py` for details on what is collected.
