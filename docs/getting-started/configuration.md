# Configuration

TraceVerde is configured via environment variables or programmatically. All settings have sensible defaults - most users only need to set `OTEL_SERVICE_NAME` and `OTEL_EXPORTER_OTLP_ENDPOINT`.

A complete `sample.env` template is included in the repository.

## Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SERVICE_NAME` | `genai-app` | Service name for traces and metrics |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTLP endpoint URL. Leave empty for console output |
| `OTEL_EXPORTER_OTLP_HEADERS` | | Headers in `key1=val1,key2=val2` format |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | `60` | OTLP exporter timeout in seconds |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Protocol: `http/protobuf` (default) or `grpc` |
| `OTEL_SERVICE_INSTANCE_ID` | | Instance identifier (container ID, pod name) |
| `OTEL_ENVIRONMENT` | `dev` | Deployment environment (dev, staging, production) |
| `GENAI_OTEL_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `GENAI_FAIL_ON_ERROR` | `false` | Raise exceptions on instrumentation errors |

## Feature Toggles

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_COST_TRACKING` | `true` | Automatic cost calculation for LLM calls |
| `GENAI_ENABLE_GPU_METRICS` | `true` | GPU utilization, memory, temperature, power metrics |
| `GENAI_ENABLE_MCP_INSTRUMENTATION` | `true` | Database, cache, vector DB, queue instrumentation |
| `GENAI_ENABLE_HTTP_INSTRUMENTATION` | `false` | HTTP/API instrumentation (disabled to avoid OTLP conflicts) |
| `GENAI_ENABLE_CO2_TRACKING` | `false` | CO2 emissions tracking from GPU power consumption |
| `GENAI_ENABLE_CONTENT_CAPTURE` | `false` | Capture prompt/response content as span events |

## Instrumentor Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLED_INSTRUMENTORS` | all defaults | Comma-separated list of instrumentors to enable |

Default instrumentors: `openai`, `openrouter`, `anthropic`, `google.generativeai`, `boto3`, `azure.ai.openai`, `cohere`, `mistralai`, `together`, `groq`, `ollama`, `vertexai`, `replicate`, `anyscale`, `sambanova`, `sarvamai`, `langchain`, `langgraph`, `llama_index`, `transformers`, `autogen`, `autogen_agentchat`, `google_adk`, `pydantic_ai`, `openai_agents`, `bedrock_agents`, `crewai`, `smolagents` (3.10+), `litellm` (3.10+)

Example - enable only specific instrumentors:

```bash
export GENAI_ENABLED_INSTRUMENTORS=openai,anthropic,crewai
```

## Sampling

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_SAMPLING_RATE` | `1.0` | Trace sampling rate (0.0-1.0). 1.0 = trace everything |

Use lower values in high-traffic production to reduce telemetry volume:

```bash
export GENAI_SAMPLING_RATE=0.1  # Sample 10% of traces
```

## Cost Tracking

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_COST_TRACKING` | `true` | Enable/disable cost tracking |
| `GENAI_CUSTOM_PRICING_JSON` | | Custom model pricing (see [Cost Tracking guide](../guides/cost-tracking.md)) |

```bash
export GENAI_CUSTOM_PRICING_JSON='{"chat":{"my-model":{"promptPrice":0.001,"completionPrice":0.002}}}'
```

## GPU Metrics

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_GPU_METRICS` | `true` | Enable GPU metrics collection |
| `GENAI_GPU_COLLECTION_INTERVAL` | `5` | Collection interval in seconds |
| `GENAI_POWER_COST_PER_KWH` | `0.12` | Electricity cost in USD per kWh |

Common electricity rates: US average ~$0.12/kWh, Europe ~$0.20/kWh, Industrial ~$0.07/kWh.

## CO2 Emissions Tracking

TraceVerde can track CO2 emissions from GPU power consumption. Two modes are available:

**Manual mode**: Uses a fixed carbon intensity value you provide.
**Codecarbon mode**: Uses [codecarbon](https://github.com/mlco2/codecarbon) for automatic region-based carbon intensity lookup.

```bash
pip install genai-otel-instrument[co2]  # Install codecarbon
export GENAI_ENABLE_CO2_TRACKING=true
```

### CO2 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_CO2_TRACKING` | `false` | Enable CO2 emissions tracking |
| `GENAI_CARBON_INTENSITY` | `475.0` | Carbon intensity in gCO2e/kWh (manual fallback) |
| `GENAI_CO2_USE_MANUAL` | `false` | Force manual calculation even with codecarbon installed |
| `GENAI_CO2_COUNTRY_ISO_CODE` | | 3-letter ISO country code (e.g., `USA`, `GBR`, `DEU`, `IND`) |
| `GENAI_CO2_REGION` | | Region/state within country (e.g., `california`, `texas`) |
| `GENAI_CO2_CLOUD_PROVIDER` | | Cloud provider: `aws`, `gcp`, `azure` |
| `GENAI_CO2_CLOUD_REGION` | | Cloud region (e.g., `us-east-1`, `europe-west1`) |
| `GENAI_CO2_OFFLINE_MODE` | `true` | Run codecarbon without external API calls |
| `GENAI_CO2_TRACKING_MODE` | `machine` | `machine` (all processes) or `process` (current only) |
| `GENAI_CODECARBON_LOG_LEVEL` | `error` | Codecarbon logging verbosity |

### Example: CO2 Tracking for US West Coast

```bash
export GENAI_ENABLE_CO2_TRACKING=true
export GENAI_CO2_COUNTRY_ISO_CODE=USA
export GENAI_CO2_REGION=california
```

### Example: CO2 Tracking on AWS

```bash
export GENAI_ENABLE_CO2_TRACKING=true
export GENAI_CO2_CLOUD_PROVIDER=aws
export GENAI_CO2_CLOUD_REGION=us-east-1
```

### Example: Manual Carbon Intensity

```bash
export GENAI_ENABLE_CO2_TRACKING=true
export GENAI_CO2_USE_MANUAL=true
export GENAI_CARBON_INTENSITY=56.0   # France (mostly nuclear)
```

Reference carbon intensity values (gCO2e/kWh):
- France: ~56 (nuclear)
- UK: ~233
- Germany: ~350
- US average: ~420
- China: ~555
- India: ~700

### CO2 Metrics

When enabled, these metrics are recorded:

| Metric | Unit | Description |
|--------|------|-------------|
| `gen_ai.co2.emissions` | kgCO2e | Cumulative CO2 emissions |
| `gen_ai.power.consumption` | kWh | Cumulative power consumption |
| `gen_ai.power.cost` | USD | Cumulative electricity cost |

## Content Capture

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_CONTENT_CAPTURE` | `false` | Capture prompt/response text in spans |
| `GENAI_CONTENT_MAX_LENGTH` | `200` | Max characters to capture (0 = unlimited) |

!!! warning "Privacy"
    Content capture records full prompts and responses. This may expose sensitive data. Ensure proper data handling and access controls before enabling in production.

## Semantic Conventions

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SEMCONV_STABILITY_OPT_IN` | `gen_ai` | Semantic convention mode |

Options:

- `gen_ai` - New conventions only (default): `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`
- `gen_ai/dup` - Dual emission for migration: emits both new and old attribute names (`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`)

## Ollama Server Metrics

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_OLLAMA_SERVER_METRICS` | `true` | Poll Ollama's `/api/ps` for VRAM usage |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GENAI_OLLAMA_METRICS_INTERVAL` | `5.0` | Polling interval in seconds |
| `GENAI_OLLAMA_MAX_VRAM_GB` | auto-detected | Override GPU VRAM size in GB |

## Evaluation Features

All evaluation features are opt-in and disabled by default. See the [Evaluation guide](../guides/evaluation.md) for detailed examples.

### PII Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_PII_DETECTION` | `false` | Enable PII detection |
| `GENAI_PII_MODE` | `detect` | Mode: `detect`, `redact`, or `block` |
| `GENAI_PII_THRESHOLD` | `0.5` | Detection confidence threshold (0.0-1.0) |
| `GENAI_PII_GDPR_MODE` | `false` | Add GDPR-specific entity types (IBAN, EU passports) |
| `GENAI_PII_HIPAA_MODE` | `false` | Add HIPAA-specific entity types (medical records) |
| `GENAI_PII_PCI_DSS_MODE` | `false` | Add PCI-DSS entity types (credit cards) |

### Toxicity Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_TOXICITY_DETECTION` | `false` | Enable toxicity detection |
| `GENAI_TOXICITY_THRESHOLD` | `0.7` | Score threshold (0.0-1.0) |
| `GENAI_TOXICITY_USE_PERSPECTIVE_API` | `false` | Use Google Perspective API (cloud) instead of Detoxify (local) |
| `GENAI_TOXICITY_PERSPECTIVE_API_KEY` | | Perspective API key (required if using Perspective) |
| `GENAI_TOXICITY_BLOCK_ON_DETECTION` | `false` | Block requests with toxic content |

### Bias Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_BIAS_DETECTION` | `false` | Enable bias detection |
| `GENAI_BIAS_THRESHOLD` | `0.4` | Detection threshold (0.0-1.0) |
| `GENAI_BIAS_BLOCK_ON_DETECTION` | `false` | Block requests with detected bias |

### Prompt Injection Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_PROMPT_INJECTION_DETECTION` | `false` | Enable prompt injection detection |
| `GENAI_PROMPT_INJECTION_THRESHOLD` | `0.5` | Detection threshold (0.0-1.0) |
| `GENAI_PROMPT_INJECTION_BLOCK_ON_DETECTION` | `false` | Block detected injection attempts |

### Restricted Topics

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_RESTRICTED_TOPICS` | `false` | Enable restricted topics detection |
| `GENAI_RESTRICTED_TOPICS_THRESHOLD` | `0.5` | Detection threshold (0.0-1.0) |
| `GENAI_RESTRICTED_TOPICS_BLOCK_ON_DETECTION` | `false` | Block restricted topic requests |

### Hallucination Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_HALLUCINATION_DETECTION` | `false` | Enable hallucination detection |
| `GENAI_HALLUCINATION_THRESHOLD` | `0.6` | Detection threshold (0.0-1.0) |

## Telemetry (Opt-in)

Anonymous usage telemetry is disabled by default.

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACEVERDE_TELEMETRY` | `false` | Enable anonymous usage reporting |
| `TRACEVERDE_TELEMETRY_URL` | | Telemetry endpoint URL |

See `genai_otel/telemetry.py` for details on what is collected.

## Programmatic Configuration

All environment variables can be set programmatically:

```python
import genai_otel

genai_otel.instrument(
    # Core
    service_name="my-app",
    endpoint="http://localhost:4318",
    enabled_instrumentors=["openai", "anthropic", "crewai"],
    sampling_rate=0.5,
    fail_on_error=False,

    # Features
    enable_gpu_metrics=True,
    enable_cost_tracking=True,
    enable_mcp_instrumentation=True,
    enable_content_capture=False,

    # CO2 tracking
    enable_co2_tracking=True,
    co2_country_iso_code="USA",
    co2_region="california",

    # GPU
    gpu_collection_interval=10,
    power_cost_per_kwh=0.15,

    # Evaluation
    enable_pii_detection=True,
    pii_mode="redact",
    pii_threshold=0.5,
    pii_gdpr_mode=True,

    enable_toxicity_detection=True,
    toxicity_threshold=0.7,

    enable_bias_detection=True,
    bias_threshold=0.4,

    enable_prompt_injection_detection=True,
    prompt_injection_threshold=0.5,

    enable_restricted_topics=True,
    restricted_topics_threshold=0.5,

    enable_hallucination_detection=True,
    hallucination_threshold=0.6,
)
```

## Session and User Tracking

For programmatic session/user tracking, provide extractor callables:

```python
genai_otel.instrument(
    session_id_extractor=lambda instance, args, kwargs: kwargs.get("metadata", {}).get("session_id"),
    user_id_extractor=lambda instance, args, kwargs: kwargs.get("metadata", {}).get("user_id"),
)
```

CrewAI, LangGraph, and LangChain have built-in automatic session ID propagation. See the [Multi-Agent Frameworks guide](../guides/multi-agent-frameworks.md) for details.
