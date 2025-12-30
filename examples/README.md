# GenAI OTel Instrumentation - Examples

This directory contains examples demonstrating how to use `genai-otel-instrument` with various LLM providers and frameworks.

## Quick Start Pattern

All examples follow the same simple pattern:

```python
import genai_otel

# Step 1: Enable auto-instrumentation (do this BEFORE importing LLM libraries)
genai_otel.instrument()

# Step 2: Use your LLM library normally
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(...)  # Automatically instrumented!
```

That's it! No code changes needed in your application logic.

## Available Examples

### LLM Providers

- **[openai/](openai/)** - OpenAI GPT models (GPT-4, GPT-3.5, etc.)
- **[anthropic/](anthropic/)** - Anthropic Claude models
- **[google_ai/](google_ai/)** - Google Gemini models
- **[aws_bedrock/](aws_bedrock/)** - AWS Bedrock (Claude, Titan, etc.)
- **[azure_openai/](azure_openai/)** - Azure OpenAI Service
- **[cohere/](cohere/)** - Cohere models
- **[groq/](groq/)** - Groq LPU inference
- **[mistralai/](mistralai/)** - Mistral AI models
- **[ollama/](ollama/)** - Ollama local models
- **[replicate/](replicate/)** - Replicate API
- **[togetherai/](togetherai/)** - Together AI
- **[vertexai/](vertexai/)** - Google Cloud Vertex AI

### Frameworks

- **[langchain/](langchain/)** - LangChain framework
- **[llamaindex/](llamaindex/)** - LlamaIndex framework
- **[huggingface/](huggingface/)** - HuggingFace Transformers

### Evaluation & Safety Features

- **[pii_detection/](pii_detection/)** - PII (Personally Identifiable Information) detection
  - Basic detection, redaction, and blocking modes
  - GDPR, HIPAA, PCI-DSS compliance
  - Response detection and custom thresholds
- **[toxicity_detection/](toxicity_detection/)** - Toxicity and harmful content detection
  - Local model (Detoxify) and cloud API (Perspective) options
  - 6 toxicity categories: toxicity, severe_toxicity, identity_attack, insult, profanity, threat
  - Blocking mode and threshold configuration
- **[bias_detection/](bias_detection/)** - Bias detection (Coming Soon)
  - Gender, racial, age, and other bias categories
  - Compliance support for EEO and fair housing

### Complete Demo

- **[demo/](demo/)** - 🎯 **START HERE!** Self-contained Docker demo with Jaeger

## Common Setup

### 1. Install the package

```bash
# Install with specific provider
pip install genai-otel-instrument[openai]

# Or install with all providers
pip install genai-otel-instrument[all]
```

### 2. Set up environment variables

Each example includes a `.env.example` file. Copy it to `.env` and configure:

```bash
cd examples/openai
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start an OTLP collector

**Using Jaeger (recommended for getting started):**

```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 4318:4318 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

View traces at: http://localhost:16686

**Using OpenTelemetry Collector:**

```bash
docker run -d --name otel-collector \
  -p 4318:4318 \
  otel/opentelemetry-collector:latest
```

### 4. Run the example

```bash
python example.py
```

## Environment Variables Reference

### Required for OTLP Export

- `OTEL_SERVICE_NAME` - Your service name (default: "genai-app")
- `OTEL_EXPORTER_OTLP_ENDPOINT` - OTLP endpoint (default: "http://localhost:4318")

### Optional Configuration

- `OTEL_EXPORTER_OTLP_HEADERS` - Headers for authentication (format: "key1=val1,key2=val2")
- `OTEL_SERVICE_INSTANCE_ID` - Service instance identifier
- `OTEL_ENVIRONMENT` - Environment name (dev, staging, prod, etc.)
- `OTEL_EXPORTER_OTLP_TIMEOUT` - Timeout in seconds (default: "10.0")

### Feature Flags

- `GENAI_ENABLE_GPU_METRICS` - Enable GPU metrics collection (default: "true")
- `GENAI_ENABLE_COST_TRACKING` - Enable cost calculation (default: "true")
- `GENAI_ENABLE_MCP_INSTRUMENTATION` - Enable MCP tools instrumentation (default: "true")
- `GENAI_FAIL_ON_ERROR` - Fail on instrumentation errors (default: "false")

### Evaluation & Safety Flags

- `GENAI_ENABLE_PII_DETECTION` - Enable PII detection (default: "false")
- `GENAI_PII_MODE` - PII mode: detect, redact, or block (default: "detect")
- `GENAI_PII_THRESHOLD` - PII confidence threshold 0.0-1.0 (default: "0.7")
- `GENAI_PII_GDPR_MODE` - Enable GDPR compliance (default: "false")
- `GENAI_PII_HIPAA_MODE` - Enable HIPAA compliance (default: "false")
- `GENAI_PII_PCI_DSS_MODE` - Enable PCI-DSS compliance (default: "false")
- `GENAI_ENABLE_TOXICITY_DETECTION` - Enable toxicity detection (default: "false")
- `GENAI_TOXICITY_THRESHOLD` - Toxicity threshold 0.0-1.0 (default: "0.7")
- `GENAI_TOXICITY_USE_PERSPECTIVE_API` - Use Google Perspective API (default: "false")
- `GENAI_TOXICITY_PERSPECTIVE_API_KEY` - Perspective API key (optional)
- `GENAI_TOXICITY_BLOCK_ON_DETECTION` - Block toxic content (default: "false")

### Logging

- `GENAI_OTEL_LOG_LEVEL` - Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: "INFO")

## What Gets Instrumented?

For each LLM call, the following data is automatically captured:

### Traces (Spans)

- Operation name (e.g., "openai.chat.completions")
- Duration and timing
- Model name and provider
- Request parameters
- Response metadata
- Parent-child relationships (for chains/agents)

### Metrics

- `genai.requests` - Request counts by provider/model
- `genai.tokens` - Token usage (prompt, completion, total)
- `genai.latency` - Request latency histogram
- `genai.cost` - Estimated costs in USD
- `genai.errors` - Error counts
- `genai.gpu.*` - GPU metrics (if enabled)

### Evaluation & Safety Metrics

- `genai.evaluation.pii.detections` - PII detection events
- `genai.evaluation.pii.entities` - PII entities by type
- `genai.evaluation.pii.blocked` - Blocked PII requests
- `genai.evaluation.toxicity.detections` - Toxicity detection events
- `genai.evaluation.toxicity.categories` - Toxicity by category
- `genai.evaluation.toxicity.score` - Toxicity score distribution
- `genai.evaluation.toxicity.blocked` - Blocked toxic requests

### Cost Tracking

Automatic cost calculation for:
- ✅ OpenAI (all models)
- ✅ Anthropic Claude (all models)
- ✅ Google Gemini
- ✅ AWS Bedrock
- ✅ Azure OpenAI
- ✅ And more!

## Advanced Usage

### Custom Configuration

```python
import genai_otel

genai_otel.instrument(
    service_name="my-custom-service",
    endpoint="https://my-collector:4318",
    enable_gpu_metrics=False,
    enable_cost_tracking=True,
    fail_on_error=False
)
```

### Selective Instrumentation

```python
import genai_otel

# Only instrument specific providers
genai_otel.instrument(
    enabled_instrumentors=["openai", "anthropic"]
)
```

### Using with Existing OpenTelemetry

```python
from opentelemetry import trace
import genai_otel

# Get existing tracer provider
tracer_provider = trace.get_tracer_provider()

# Instrument using existing setup
genai_otel.instrument(tracer_provider=tracer_provider)
```

### Enabling Safety Features

```python
import genai_otel

# Enable PII detection with redaction
genai_otel.instrument(
    service_name="my-app",
    enable_pii_detection=True,
    pii_mode="redact",  # Options: detect, redact, block
    pii_threshold=0.7,
    pii_gdpr_mode=True,  # EU compliance
)

# Enable toxicity detection
genai_otel.instrument(
    service_name="my-app",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    toxicity_block_on_detection=True,
)

# Enable multiple safety features
genai_otel.instrument(
    service_name="my-app",
    enable_pii_detection=True,
    pii_mode="redact",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)
```

## Troubleshooting

### Traces not appearing?

1. Check that OTLP collector is running: `curl http://localhost:4318/v1/traces`
2. Verify environment variables are set correctly
3. Check logs with `GENAI_OTEL_LOG_LEVEL=DEBUG`

### Import errors?

Make sure you've installed the provider-specific extras:
```bash
pip install genai-otel-instrument[openai,anthropic]
```

### Cost tracking not working?

Cost tracking requires:
- Valid model names
- Up-to-date pricing data (included in package)
- `GENAI_ENABLE_COST_TRACKING=true` (default)

## Contributing Examples

Want to add an example? Please:

1. Follow the existing example structure
2. Include a `.env.example` file
3. Add a README explaining the specific features
4. Keep it simple and focused on one use case
5. Submit a PR!

## Learn More

- [Main README](../README.md)
- [Troubleshooting Guide](../TROUBLESHOOTING.md)
- [Semantic Conventions](../OTEL_SEMANTIC_COMPATIBILITY.md)
- [Demo Application](demo/) - Complete working example with Docker
