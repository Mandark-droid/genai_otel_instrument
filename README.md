# GenAI OpenTelemetry Auto-Instrumentation

Production-ready OpenTelemetry instrumentation for GenAI/LLM applications with zero-code setup.

## Features

🚀 **Zero-Code Instrumentation** - Just install and set env vars
🤖 **15+ LLM Providers** - OpenAI, Anthropic, Google, AWS, Azure, and more
🔧 **MCP Tool Support** - Auto-instrument databases, APIs, caches, vector DBs
💰 **Cost Tracking** - Automatic cost calculation per request
🎮 **GPU Metrics** - Real-time GPU utilization, memory, temperature
📊 **Complete Observability** - Traces, metrics, and rich span attributes
➕ **Service Instance ID & Environment** - Identify your services and environments
⏱️ **Configurable Exporter Timeout** - Set timeout for OTLP exporter
🔗 **OpenInference Instrumentors** - Smolagents, MCP, and LiteLLM instrumentation

## Quick Start

### Installation

```bash
pip install genai-otel-instrument
```

### Usage

**Option 1: Environment Variables (No code changes)**

```bash
export OTEL_SERVICE_NAME=my-llm-app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
python your_app.py
```

**Option 2: One line of code**

```python
import genai_otel
genai_otel.instrument()

# Your existing code works unchanged
import openai
client = openai.OpenAI()
response = client.chat.completions.create(...)
```

**Option 3: CLI wrapper**

```bash
genai-instrument python your_app.py
```

For a more comprehensive demonstration of various LLM providers and MCP tools, refer to `example_usage.py` in the project root. Note that running this example requires setting up relevant API keys and external services (e.g., databases, Redis, Pinecone).

## What Gets Instrumented?

### LLM Providers (Auto-detected)
- OpenAI, Anthropic, Google AI, AWS Bedrock, Azure OpenAI
- Cohere, Mistral AI, Together AI, Groq, Ollama
- Vertex AI, Replicate, Anyscale, HuggingFace

### Frameworks
- LangChain (chains, agents, tools)
- LlamaIndex (query engines, indices)

### MCP Tools (Model Context Protocol)
- **Databases**: PostgreSQL, MySQL, MongoDB, SQLAlchemy
- **Caching**: Redis
- **Message Queues**: Apache Kafka
- **Vector Databases**: Pinecone, Weaviate, Qdrant, ChromaDB, Milvus, FAISS
- **APIs**: HTTP/REST requests (requests, httpx)

### OpenInference (Optional - Python 3.10+ only)
- Smolagents
- MCP
- LiteLLM

**Note:** OpenInference instrumentors require Python >= 3.10. Install with:
```bash
pip install genai-otel-instrument[openinference]
```

## Collected Telemetry

### Traces
Every LLM call, database query, API request, and vector search is traced with full context propagation.

### Metrics
- `gen_ai.requests` - Request counts by provider/model
- `gen_ai.client.token.usage` - Token usage (prompt/completion)
- `gen_ai.client.operation.duration` - Request latency histogram (optimized buckets for LLM workloads)
- `gen_ai.usage.cost` - Total estimated costs in USD
- `gen_ai.usage.cost.prompt` - Prompt tokens cost (granular)
- `gen_ai.usage.cost.completion` - Completion tokens cost (granular)
- `gen_ai.usage.cost.reasoning` - Reasoning tokens cost (OpenAI o1 models)
- `gen_ai.usage.cost.cache_read` - Cache read cost (Anthropic)
- `gen_ai.usage.cost.cache_write` - Cache write cost (Anthropic)
- `gen_ai.client.errors` - Error counts by operation and type
- `gen_ai.gpu.*` - GPU utilization, memory, temperature (ObservableGauges)
- `gen_ai.co2.emissions` - CO2 emissions tracking (opt-in)

### Span Attributes
**Core Attributes:**
- `gen_ai.system` - Provider name (e.g., "openai")
- `gen_ai.operation.name` - Operation type (e.g., "chat")
- `gen_ai.request.model` - Model identifier
- `gen_ai.usage.prompt_tokens` / `gen_ai.usage.input_tokens` - Input tokens (dual emission supported)
- `gen_ai.usage.completion_tokens` / `gen_ai.usage.output_tokens` - Output tokens (dual emission supported)
- `gen_ai.usage.total_tokens` - Total tokens

**Request Parameters:**
- `gen_ai.request.temperature` - Temperature setting
- `gen_ai.request.top_p` - Top-p sampling
- `gen_ai.request.max_tokens` - Max tokens requested
- `gen_ai.request.frequency_penalty` - Frequency penalty
- `gen_ai.request.presence_penalty` - Presence penalty

**Response Attributes:**
- `gen_ai.response.id` - Response ID from provider
- `gen_ai.response.model` - Actual model used (may differ from request)
- `gen_ai.response.finish_reasons` - Array of finish reasons

**Tool/Function Calls:**
- `llm.tools` - JSON-serialized tool definitions
- `llm.output_messages.{choice}.message.tool_calls.{index}.tool_call.id` - Tool call ID
- `llm.output_messages.{choice}.message.tool_calls.{index}.tool_call.function.name` - Function name
- `llm.output_messages.{choice}.message.tool_calls.{index}.tool_call.function.arguments` - Function arguments

**Cost Attributes (granular):**
- `gen_ai.usage.cost.total` - Total cost
- `gen_ai.usage.cost.prompt` - Prompt tokens cost
- `gen_ai.usage.cost.completion` - Completion tokens cost
- `gen_ai.usage.cost.reasoning` - Reasoning tokens cost (o1 models)
- `gen_ai.usage.cost.cache_read` - Cache read cost (Anthropic)
- `gen_ai.usage.cost.cache_write` - Cache write cost (Anthropic)

**Content Events (opt-in):**
- `gen_ai.prompt.{index}` events with role and content
- `gen_ai.completion.{index}` events with role and content

**Additional:**
- Database, vector DB, and API attributes from MCP instrumentation

## Configuration

### Environment Variables

```bash
# Required
OTEL_SERVICE_NAME=my-app
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Optional
OTEL_EXPORTER_OTLP_HEADERS=x-api-key=secret
GENAI_ENABLE_GPU_METRICS=true
GENAI_ENABLE_COST_TRACKING=true
GENAI_ENABLE_MCP_INSTRUMENTATION=true
OTEL_SERVICE_INSTANCE_ID=instance-1 # Optional service instance id
OTEL_ENVIRONMENT=production # Optional environment
OTEL_EXPORTER_OTLP_TIMEOUT=10.0 # Optional timeout for OTLP exporter

# Semantic conventions (NEW)
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai  # "gen_ai" for new conventions only, "gen_ai/dup" for dual emission
GENAI_ENABLE_CONTENT_CAPTURE=false  # WARNING: May capture sensitive data. Enable with caution.

# Logging configuration
GENAI_OTEL_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL. Logs are written to 'logs/genai_otel.log' with rotation (10 files, 10MB each).

# Error handling
GENAI_FAIL_ON_ERROR=false  # true to fail fast, false to continue on errors
```

### Programmatic Configuration

```python
import genai_otel

genai_otel.instrument(
    service_name="my-app",
    endpoint="http://localhost:4318",
    enable_gpu_metrics=True,
    enable_cost_tracking=True,
    enable_mcp_instrumentation=True
)
```

### Sample Environment File (`sample.env`)

A `sample.env` file has been generated in the project root directory. This file contains commented-out examples of all supported environment variables, along with their default values or expected formats. You can copy this file to `.env` and uncomment/modify the variables to configure the instrumentation for your specific needs.

## Example: Full-Stack GenAI App

```python
import genai_otel
genai_otel.instrument()

import openai
import pinecone
import redis
import psycopg2

# All of these are automatically instrumented:

# Cache check
cache = redis.Redis().get('key')

# Vector search
pinecone_index = pinecone.Index("embeddings")
results = pinecone_index.query(vector=[...], top_k=5)

# Database query
conn = psycopg2.connect("dbname=mydb")
cursor = conn.cursor()
cursor.execute("SELECT * FROM context")

# LLM call with full context
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...]
)

# You get:
# ✓ Distributed traces across all services
# ✓ Cost tracking for the LLM call
# ✓ Performance metrics for DB, cache, vector DB
# ✓ GPU metrics if using local models
# ✓ Complete observability with zero manual instrumentation
```

## Backend Integration

Works with any OpenTelemetry-compatible backend:
- Jaeger, Zipkin
- Prometheus, Grafana
- Datadog, New Relic, Honeycomb
- AWS X-Ray, Google Cloud Trace
- Elastic APM, Splunk
- Self-hosted OTEL Collector

## Project Structure

```bash
genai-otel-instrument/
├── setup.py
├── MANIFEST.in
├── README.md
├── LICENSE
├── example_usage.py
└── genai_otel/
    ├── __init__.py
    ├── config.py
    ├── auto_instrument.py
    ├── cli.py
    ├── cost_calculator.py
    ├── gpu_metrics.py
    ├── instrumentors/
    │   ├── __init__.py
    │   ├── base.py
    │   └── (other instrumentor files)
    └── mcp_instrumentors/
        ├── __init__.py
        ├── manager.py
        └── (other mcp files)
```

## License
Apache-2.0 license