# TraceVerde

<div align="center">
  <img src=".github/images/Logo.jpg" alt="TraceVerde - GenAI OpenTelemetry Instrumentation Logo" width="400"/>
</div>

<br/>

[![PyPI version](https://badge.fury.io/py/genai-otel-instrument.svg)](https://badge.fury.io/py/genai-otel-instrument)
[![Python Versions](https://img.shields.io/pypi/pyversions/genai-otel-instrument.svg)](https://pypi.org/project/genai-otel-instrument/)
[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Downloads](https://static.pepy.tech/badge/genai-otel-instrument)](https://pepy.tech/project/genai-otel-instrument)
[![Downloads/Month](https://static.pepy.tech/badge/genai-otel-instrument/month)](https://pepy.tech/project/genai-otel-instrument)

[![GitHub Stars](https://img.shields.io/github/stars/Mandark-droid/genai_otel_instrument?style=social)](https://github.com/Mandark-droid/genai_otel_instrument)
[![GitHub Forks](https://img.shields.io/github/forks/Mandark-droid/genai_otel_instrument?style=social)](https://github.com/Mandark-droid/genai_otel_instrument)
[![GitHub Issues](https://img.shields.io/github/issues/Mandark-droid/genai_otel_instrument)](https://github.com/Mandark-droid/genai_otel_instrument/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/Mandark-droid/genai_otel_instrument)](https://github.com/Mandark-droid/genai_otel_instrument/pulls)

[![Code Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/Mandark-droid/genai_otel_instrument)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-1.20%2B-blueviolet)](https://opentelemetry.io/)
[![Semantic Conventions](https://img.shields.io/badge/OTel%20Semconv-GenAI%20v1.28-orange)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/Mandark-droid/genai_otel_instrument/actions)

---

<div align="center">
  <img src=".github/images/Landing_Page.jpg" alt="GenAI OpenTelemetry Instrumentation Overview" width="800"/>
</div>

---

Production-ready OpenTelemetry instrumentation for GenAI/LLM applications with zero-code setup.

## Features

🚀 **Zero-Code Instrumentation** - Just install and set env vars
🤖 **15+ LLM Providers** - OpenAI, Anthropic, Google, AWS, Azure, and more
🔧 **MCP Tool Support** - Auto-instrument databases, APIs, caches, vector DBs
💰 **Cost Tracking** - Automatic cost calculation for both streaming and non-streaming requests
⚡ **Streaming Support** - Full observability for streaming responses with TTFT/TBT metrics and cost tracking
🎮 **GPU Metrics** - Real-time GPU utilization, memory, temperature, power, and electricity cost tracking
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
- **With Full Cost Tracking**: OpenAI, Anthropic, Google AI, AWS Bedrock, Azure OpenAI, Cohere, Mistral AI, Together AI, Groq, Ollama, Vertex AI
- **Hardware/Local Pricing**: Replicate (hardware-based $/second), HuggingFace (local execution with estimated costs)
  - **HuggingFace Support**: `pipeline()`, `AutoModelForCausalLM.generate()`, `AutoModelForSeq2SeqLM.generate()`, `InferenceClient` API calls
- **Other Providers**: Anyscale

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
- Smolagents - HuggingFace smolagents framework tracing
- MCP - Model Context Protocol instrumentation
- LiteLLM - Multi-provider LLM proxy

**Cost Enrichment:** OpenInference instrumentors are automatically enriched with cost tracking! When cost tracking is enabled (`GENAI_ENABLE_COST_TRACKING=true`), a custom `CostEnrichmentSpanProcessor` extracts model and token usage from OpenInference spans and adds cost attributes (`gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`) using our comprehensive pricing database of 145+ models.

The processor supports OpenInference semantic conventions:
- Model: `llm.model_name`, `embedding.model_name`
- Tokens: `llm.token_count.prompt`, `llm.token_count.completion`
- Operations: `openinference.span.kind` (LLM, EMBEDDING, CHAIN, RETRIEVER, etc.)

**Note:** OpenInference instrumentors require Python >= 3.10. Install with:
```bash
pip install genai-otel-instrument[openinference]
```

## Screenshots

See the instrumentation in action across different LLM providers and observability backends.

### OpenAI Instrumentation
Full trace capture for OpenAI API calls with token usage, costs, and latency metrics.

<div align="center">
  <img src=".github/images/Screenshots/Traces_OpenAI.png" alt="OpenAI Traces" width="900"/>
</div>

### Ollama (Local LLM) Instrumentation
Zero-code instrumentation for local models running on Ollama with comprehensive observability.

<div align="center">
  <img src=".github/images/Screenshots/Traces_Ollama.png" alt="Ollama Traces" width="900"/>
</div>

### HuggingFace Transformers
Direct instrumentation of HuggingFace Transformers with automatic token counting and cost estimation.

<div align="center">
  <img src=".github/images/Screenshots/Trace_HuggingFace_Transformer_Models.png" alt="HuggingFace Transformer Traces" width="900"/>
</div>

### SmolAgents Framework
Complete agent workflow tracing with tool calls, iterations, and cost breakdown.

<div align="center">
  <img src=".github/images/Screenshots/Traces_SmolAgent_with_tool_calls.png" alt="SmolAgent Traces with Tool Calls" width="900"/>
</div>

### GPU Metrics Collection
Real-time GPU utilization, memory, temperature, and power consumption metrics.

<div align="center">
  <img src=".github/images/Screenshots/GPU_Metrics.png" alt="GPU Metrics Dashboard" width="900"/>
</div>

### Additional Screenshots

- **[Token Cost Breakdown](.github/images/Screenshots/Traces_SmolAgent_Token_Cost_breakdown.png)** - Detailed token usage and cost analysis for SmolAgent workflows
- **[OpenSearch Dashboard](.github/images/Screenshots/GENAI_OpenSearch_output.png)** - GenAI metrics visualization in OpenSearch/Kibana

---

## Demo Video

Watch a comprehensive walkthrough of GenAI OpenTelemetry Auto-Instrumentation in action, demonstrating setup, configuration, and real-time observability across multiple LLM providers.

<div align="center">

  **🎥 [Watch Demo Video](https://youtu.be/YOUR_VIDEO_ID_HERE)**
  *(Coming Soon)*

</div>

---

## Cost Tracking Coverage

The library includes comprehensive cost tracking with pricing data for **145+ models** across **11 providers**:

### Providers with Full Token-Based Cost Tracking
- **OpenAI**: GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, o1/o3 series, embeddings, audio, vision (35+ models)
- **Anthropic**: Claude 3.5 Sonnet/Opus/Haiku, Claude 3 series (10+ models)
- **Google AI**: Gemini 1.5/2.0 Pro/Flash, PaLM 2 (12+ models)
- **AWS Bedrock**: Amazon Titan, Claude, Llama, Mistral models (20+ models)
- **Azure OpenAI**: Same as OpenAI with Azure-specific pricing
- **Cohere**: Command R/R+, Command Light, Embed v3/v2 (8+ models)
- **Mistral AI**: Mistral Large/Medium/Small, Mixtral, embeddings (8+ models)
- **Together AI**: DeepSeek-R1, Llama 3.x, Qwen, Mixtral (25+ models)
- **Groq**: Llama 3.x series, Mixtral, Gemma models (15+ models)
- **Ollama**: Local models with token tracking (pricing via cost estimation)
- **Vertex AI**: Gemini models via Google Cloud with usage metadata extraction

### Special Pricing Models
- **Replicate**: Hardware-based pricing ($/second of GPU/CPU time) - not token-based
- **HuggingFace Transformers**: Local model execution with estimated costs based on parameter count
  - Supports `pipeline()`, `AutoModelForCausalLM.generate()`, `AutoModelForSeq2SeqLM.generate()`
  - Cost estimation uses GPU/compute resource pricing tiers (tiny/small/medium/large)
  - Automatic token counting from tensor shapes

### Pricing Features
- **Differential Pricing**: Separate rates for prompt tokens vs. completion tokens
- **Reasoning Tokens**: Special pricing for OpenAI o1/o3 reasoning tokens
- **Cache Pricing**: Anthropic prompt caching costs (read/write)
- **Granular Cost Metrics**: Per-request cost breakdown by token type
- **Auto-Updated Pricing**: Pricing data maintained in `llm_pricing.json`
- **Custom Pricing**: Add pricing for custom/proprietary models via environment variable

### Adding Custom Model Pricing

For custom or proprietary models not in `llm_pricing.json`, you can provide custom pricing via the `GENAI_CUSTOM_PRICING_JSON` environment variable:

```bash
# For chat models
export GENAI_CUSTOM_PRICING_JSON='{"chat":{"my-custom-model":{"promptPrice":0.001,"completionPrice":0.002}}}'

# For embeddings
export GENAI_CUSTOM_PRICING_JSON='{"embeddings":{"my-custom-embeddings":0.00005}}'

# For multiple categories
export GENAI_CUSTOM_PRICING_JSON='{
  "chat": {
    "my-custom-chat": {"promptPrice": 0.001, "completionPrice": 0.002}
  },
  "embeddings": {
    "my-custom-embed": 0.00005
  },
  "audio": {
    "my-custom-tts": 0.02
  }
}'
```

**Pricing Format:**
- **Chat models**: `{"promptPrice": <$/1k tokens>, "completionPrice": <$/1k tokens>}`
- **Embeddings**: Single number for price per 1k tokens
- **Audio**: Price per 1k characters (TTS) or per second (STT)
- **Images**: Nested structure with quality/size pricing (see `llm_pricing.json` for examples)

**Hybrid Pricing:** Custom prices are merged with default pricing from `llm_pricing.json`. If you provide custom pricing for an existing model, the custom price overrides the default.

**Coverage Statistics**: As of v0.1.3, 89% test coverage with 415 passing tests, including comprehensive cost calculation validation and cost enrichment processor tests (supporting both GenAI and OpenInference semantic conventions).

## Collected Telemetry

### Traces
Every LLM call, database query, API request, and vector search is traced with full context propagation.

### Metrics

**GenAI Metrics:**
- `gen_ai.requests` - Request counts by provider and model
- `gen_ai.client.token.usage` - Token usage (prompt/completion)
- `gen_ai.client.operation.duration` - Request latency histogram (optimized buckets for LLM workloads)
- `gen_ai.usage.cost` - Total estimated costs in USD
- `gen_ai.usage.cost.prompt` - Prompt tokens cost (granular)
- `gen_ai.usage.cost.completion` - Completion tokens cost (granular)
- `gen_ai.usage.cost.reasoning` - Reasoning tokens cost (OpenAI o1 models)
- `gen_ai.usage.cost.cache_read` - Cache read cost (Anthropic)
- `gen_ai.usage.cost.cache_write` - Cache write cost (Anthropic)
- `gen_ai.client.errors` - Error counts by operation and type
- `gen_ai.gpu.*` - GPU utilization, memory, temperature, power (ObservableGauges)
- `gen_ai.co2.emissions` - CO2 emissions tracking (opt-in via `GENAI_ENABLE_CO2_TRACKING`)
- `gen_ai.power.cost` - Cumulative electricity cost in USD based on GPU power consumption (configurable via `GENAI_POWER_COST_PER_KWH`)
- `gen_ai.server.ttft` - Time to First Token for streaming responses (histogram, 1ms-10s buckets)
- `gen_ai.server.tbt` - Time Between Tokens for streaming responses (histogram, 10ms-2.5s buckets)

**MCP Metrics (Database Operations):**
- `mcp.requests` - Number of MCP/database requests
- `mcp.client.operation.duration` - Operation duration histogram (1ms to 10s buckets)
- `mcp.request.size` - Request payload size histogram (100B to 5MB buckets)
- `mcp.response.size` - Response payload size histogram (100B to 5MB buckets)

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

**Streaming Attributes:**
- `gen_ai.server.ttft` - Time to First Token (seconds) for streaming responses
- `gen_ai.streaming.token_count` - Total number of chunks in streaming response
- `gen_ai.usage.prompt_tokens` - Actual prompt tokens (extracted from final chunk)
- `gen_ai.usage.completion_tokens` - Actual completion tokens (extracted from final chunk)
- `gen_ai.usage.total_tokens` - Total tokens (extracted from final chunk)
- `gen_ai.usage.cost.total` - Total cost for streaming request
- `gen_ai.usage.cost.prompt` - Prompt tokens cost for streaming request
- `gen_ai.usage.cost.completion` - Completion tokens cost for streaming request
- All granular cost attributes (reasoning, cache_read, cache_write) also available for streaming

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
GENAI_GPU_COLLECTION_INTERVAL=5  # GPU metrics collection interval in seconds (default: 5)
OTEL_SERVICE_INSTANCE_ID=instance-1 # Optional service instance id
OTEL_ENVIRONMENT=production # Optional environment
OTEL_EXPORTER_OTLP_TIMEOUT=60 # Timeout for OTLP exporter in seconds (default: 60)
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf # Protocol: "http/protobuf" (default) or "grpc"

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

## Advanced Features

### Session and User Tracking

Track user sessions and identify users across multiple LLM requests for better analytics, debugging, and cost attribution.

**Configuration:**

```python
import genai_otel
from genai_otel import OTelConfig

# Define extractor functions
def extract_session_id(instance, args, kwargs):
    """Extract session ID from request metadata."""
    # Option 1: From kwargs metadata
    metadata = kwargs.get("metadata", {})
    return metadata.get("session_id")

    # Option 2: From custom headers
    # headers = kwargs.get("headers", {})
    # return headers.get("X-Session-ID")

    # Option 3: From thread-local storage
    # import threading
    # return getattr(threading.current_thread(), "session_id", None)

def extract_user_id(instance, args, kwargs):
    """Extract user ID from request metadata."""
    metadata = kwargs.get("metadata", {})
    return metadata.get("user_id")

# Configure with extractors
config = OTelConfig(
    service_name="my-rag-app",
    endpoint="http://localhost:4318",
    session_id_extractor=extract_session_id,
    user_id_extractor=extract_user_id,
)

genai_otel.instrument(config)
```

**Usage:**

```python
from openai import OpenAI

client = OpenAI()

# Pass session and user info via metadata
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is OpenTelemetry?"}],
    extra_body={"metadata": {"session_id": "sess_12345", "user_id": "user_alice"}}
)
```

**Span Attributes Added:**
- `session.id` - Unique session identifier for tracking conversations
- `user.id` - User identifier for per-user analytics and cost tracking

**Use Cases:**
- Track multi-turn conversations across requests
- Analyze usage patterns per user
- Debug session-specific issues
- Calculate per-user costs and quotas
- Build user-specific dashboards

### RAG and Embedding Attributes

Enhanced observability for Retrieval-Augmented Generation (RAG) workflows, including embedding generation and document retrieval.

**Helper Methods:**

The `BaseInstrumentor` provides helper methods to add RAG-specific attributes to your spans:

```python
from opentelemetry import trace
from genai_otel.instrumentors.base import BaseInstrumentor

# Get your instrumentor instance (or create spans manually)
tracer = trace.get_tracer(__name__)

# 1. Embedding Attributes
with tracer.start_as_current_span("embedding.create") as span:
    # Your embedding logic
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input="OpenTelemetry provides observability"
    )

    # Add embedding attributes (if using BaseInstrumentor)
    # instrumentor.add_embedding_attributes(
    #     span,
    #     model="text-embedding-3-small",
    #     input_text="OpenTelemetry provides observability",
    #     vector=embedding_response.data[0].embedding
    # )

    # Or manually set attributes
    span.set_attribute("embedding.model_name", "text-embedding-3-small")
    span.set_attribute("embedding.text", "OpenTelemetry provides observability"[:500])
    span.set_attribute("embedding.vector.dimension", len(embedding_response.data[0].embedding))

# 2. Retrieval Attributes
with tracer.start_as_current_span("retrieval.search") as span:
    # Your retrieval logic
    retrieved_docs = [
        {
            "id": "doc_001",
            "score": 0.95,
            "content": "OpenTelemetry is an observability framework...",
            "metadata": {"source": "docs.opentelemetry.io", "category": "intro"}
        },
        # ... more documents
    ]

    # Add retrieval attributes (if using BaseInstrumentor)
    # instrumentor.add_retrieval_attributes(
    #     span,
    #     documents=retrieved_docs,
    #     query="What is OpenTelemetry?",
    #     max_docs=5
    # )

    # Or manually set attributes
    span.set_attribute("retrieval.query", "What is OpenTelemetry?"[:500])
    span.set_attribute("retrieval.document_count", len(retrieved_docs))

    for i, doc in enumerate(retrieved_docs[:5]):  # Limit to 5 docs
        prefix = f"retrieval.documents.{i}.document"
        span.set_attribute(f"{prefix}.id", doc["id"])
        span.set_attribute(f"{prefix}.score", doc["score"])
        span.set_attribute(f"{prefix}.content", doc["content"][:500])

        # Add metadata
        for key, value in doc.get("metadata", {}).items():
            span.set_attribute(f"{prefix}.metadata.{key}", str(value))
```

**Embedding Attributes:**
- `embedding.model_name` - Embedding model used
- `embedding.text` - Input text (truncated to 500 chars)
- `embedding.vector` - Embedding vector (optional, if configured)
- `embedding.vector.dimension` - Vector dimensions

**Retrieval Attributes:**
- `retrieval.query` - Search query (truncated to 500 chars)
- `retrieval.document_count` - Number of documents retrieved
- `retrieval.documents.{i}.document.id` - Document ID
- `retrieval.documents.{i}.document.score` - Relevance score
- `retrieval.documents.{i}.document.content` - Document content (truncated to 500 chars)
- `retrieval.documents.{i}.document.metadata.*` - Custom metadata fields

**Safeguards:**
- Text content truncated to 500 characters to avoid span size explosion
- Document count limited to 5 by default (configurable via `max_docs`)
- Metadata values truncated to prevent excessive attribute counts

**Complete RAG Workflow Example:**

See `examples/phase4_session_rag_tracking.py` for a comprehensive demonstration of:
- Session and user tracking across RAG pipeline
- Embedding attribute capture
- Retrieval attribute capture
- End-to-end RAG workflow with full observability

**Use Cases:**
- Monitor retrieval quality and relevance scores
- Debug RAG pipeline performance
- Track embedding model usage
- Analyze document retrieval patterns
- Optimize vector search configurations

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

## Roadmap

### Next Release (v0.2.0) - Q1 2026

We're planning significant enhancements for the next major release, focusing on evaluation metrics and safety guardrails alongside completing OpenTelemetry semantic convention compliance.

#### 🎯 Evaluation & Monitoring

**LLM Output Quality Metrics**
- **Bias Detection** - Automatically detect and measure bias in LLM responses
  - Gender, racial, political, and cultural bias detection
  - Bias score metrics with configurable thresholds
  - Integration with fairness libraries (e.g., Fairlearn, AIF360)

- **Toxicity Detection** - Monitor and alert on toxic or harmful content
  - Perspective API integration for toxicity scoring
  - Custom toxicity models support
  - Real-time toxicity metrics and alerts
  - Configurable severity levels

- **Hallucination Detection** - Track factual accuracy and groundedness
  - Fact-checking against provided context
  - Citation validation for RAG applications
  - Confidence scoring for generated claims
  - Hallucination rate metrics by model and use case

**Implementation:**
```python
import genai_otel

# Enable evaluation metrics
genai_otel.instrument(
    enable_bias_detection=True,
    enable_toxicity_detection=True,
    enable_hallucination_detection=True,

    # Configure thresholds
    bias_threshold=0.7,
    toxicity_threshold=0.5,
    hallucination_threshold=0.8
)
```

**Metrics Added:**
- `gen_ai.eval.bias_score` - Bias detection scores (histogram)
- `gen_ai.eval.toxicity_score` - Toxicity scores (histogram)
- `gen_ai.eval.hallucination_score` - Hallucination probability (histogram)
- `gen_ai.eval.violations` - Count of threshold violations by type

#### 🛡️ Safety Guardrails

**Input/Output Filtering**
- **Prompt Injection Detection** - Protect against prompt injection attacks
  - Pattern-based detection (jailbreaking attempts)
  - ML-based classifier for sophisticated attacks
  - Real-time blocking with configurable policies
  - Attack attempt metrics and logging

- **Restricted Topics** - Block sensitive or inappropriate topics
  - Configurable topic blacklists (legal, medical, financial advice)
  - Industry-specific content filters
  - Topic detection with confidence scoring
  - Custom topic definition support

- **Sensitive Information Protection** - Prevent PII leakage
  - PII detection (emails, phone numbers, SSN, credit cards)
  - Automatic redaction or blocking
  - Compliance mode (GDPR, HIPAA, PCI-DSS)
  - Data leak prevention metrics

**Implementation:**
```python
import genai_otel

# Configure guardrails
genai_otel.instrument(
    enable_prompt_injection_detection=True,
    enable_restricted_topics=True,
    enable_sensitive_info_detection=True,

    # Custom configuration
    restricted_topics=["medical_advice", "legal_advice", "financial_advice"],
    pii_detection_mode="block",  # or "redact", "warn"

    # Callbacks for custom handling
    on_guardrail_violation=my_violation_handler
)
```

**Metrics Added:**
- `gen_ai.guardrail.prompt_injection_detected` - Injection attempts blocked
- `gen_ai.guardrail.restricted_topic_blocked` - Restricted topic violations
- `gen_ai.guardrail.pii_detected` - PII detection events
- `gen_ai.guardrail.violations` - Total guardrail violations by type

**Span Attributes:**
- `gen_ai.guardrail.violation_type` - Type of violation detected
- `gen_ai.guardrail.violation_severity` - Severity level (low, medium, high, critical)
- `gen_ai.guardrail.blocked` - Whether request was blocked (boolean)
- `gen_ai.eval.bias_categories` - Detected bias types (array)
- `gen_ai.eval.toxicity_categories` - Toxicity categories (array)

#### 🔄 Migration Support

**Backward Compatibility:**
- All new features are opt-in via configuration
- Existing instrumentation continues to work unchanged
- Gradual migration path for new semantic conventions

**Version Support:**
- Python 3.9+ (evaluation features require 3.10+)
- OpenTelemetry SDK 1.20.0+
- Backward compatible with existing dashboards

### Future Releases

**v0.3.0 - Advanced Analytics**
- Custom metric aggregations
- Cost optimization recommendations
- Automated performance regression detection
- A/B testing support for prompts

**v0.4.0 - Enterprise Features**
- Multi-tenancy support
- Role-based access control for telemetry
- Advanced compliance reporting
- SLA monitoring and alerting

**Community Feedback**

We welcome feedback on our roadmap! Please:
- Open issues for feature requests
- Join discussions on prioritization
- Share your use cases and requirements

See [Contributing.md](Contributing.md) for how to get involved.

## License

TraceVerde is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).

Copyright (C) 2025 Kshitij Thakkar

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See the [LICENSE](LICENSE) file for the full license text.
