"""
# GenAI OpenTelemetry Auto-Instrumentation

Production-ready OpenTelemetry instrumentation for GenAI/LLM applications with zero-code setup.

## Features

🚀 **Zero-Code Instrumentation** - Just install and set env vars
🤖 **15+ LLM Providers** - OpenAI, Anthropic, Google, AWS, Azure, and more
🔧 **MCP Tool Support** - Auto-instrument databases, APIs, caches, vector DBs
💰 **Cost Tracking** - Automatic cost calculation per request
🎮 **GPU Metrics** - Real-time GPU utilization, memory, temperature
📊 **Complete Observability** - Traces, metrics, and rich span attributes

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

## Collected Telemetry

### Traces
Every LLM call, database query, API request, and vector search is traced with full context propagation.

### Metrics
- `genai.requests` - Request counts by provider/model
- `genai.tokens` - Token usage (prompt/completion)
- `genai.latency` - Request latency histogram
- `genai.cost` - Estimated costs in USD
- `genai.gpu.*` - GPU utilization, memory, temperature, power

### Span Attributes
- `gen_ai.system` - Provider name
- `gen_ai.request.model` - Model identifier
- `gen_ai.usage.prompt_tokens` - Input tokens
- `gen_ai.usage.completion_tokens` - Output tokens
- `gen_ai.cost.amount` - Estimated cost
- Database, vector DB, and API attributes

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
# Logging configuration
GENAI_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
GENAI_LOG_FILE=/var/log/genai-otel.log  # Optional log file

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
├── MANIFEST.in  # New
├── README.md
├── LICENSE  # New
├── example_usage.py  # Optional
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