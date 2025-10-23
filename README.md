# GenAI OpenTelemetry Auto-Instrumentation

[![PyPI version](https://badge.fury.io/py/genai-otel-instrument.svg)](https://badge.fury.io/py/genai-otel-instrument)
[![Python Versions](https://img.shields.io/pypi/pyversions/genai-otel-instrument.svg)](https://pypi.org/project/genai-otel-instrument/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
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

Production-ready OpenTelemetry instrumentation for GenAI/LLM applications with zero-code setup.

## Features

🚀 **Zero-Code Instrumentation** - Just install and set env vars
🤖 **15+ LLM Providers** - OpenAI, Anthropic, Google, AWS, Azure, and more
🔧 **MCP Tool Support** - Auto-instrument databases, APIs, caches, vector DBs
💰 **Cost Tracking** - Automatic cost calculation per request
🎮 **GPU Metrics** - Real-time GPU utilization, memory, temperature, power
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
- **Hardware/Local Pricing**: Replicate (hardware-based $/second), HuggingFace (local execution, free)
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
- **HuggingFace Transformers**: Local execution - no API costs

### Pricing Features
- **Differential Pricing**: Separate rates for prompt tokens vs. completion tokens
- **Reasoning Tokens**: Special pricing for OpenAI o1/o3 reasoning tokens
- **Cache Pricing**: Anthropic prompt caching costs (read/write)
- **Granular Cost Metrics**: Per-request cost breakdown by token type
- **Auto-Updated Pricing**: Pricing data maintained in `llm_pricing.json`

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
- `gen_ai.co2.emissions` - CO2 emissions tracking (opt-in)
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
- `gen_ai.streaming.token_count` - Total number of chunks/tokens in streaming response

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

#### 📊 Enhanced OpenTelemetry Compliance

Completing remaining items from [OTEL_SEMANTIC_GAP_ANALYSIS_AND_IMPLEMENTATION_PLAN.md](OTEL_SEMANTIC_GAP_ANALYSIS_AND_IMPLEMENTATION_PLAN.md):

**Phase 4: Optional Enhancements**
- ✅ Session & User Tracking - Track sessions and users across requests
  ```python
  genai_otel.instrument(
      session_id_extractor=lambda ctx: ctx.get("session_id"),
      user_id_extractor=lambda ctx: ctx.get("user_id")
  )
  ```

- ✅ RAG/Embedding Attributes - Enhanced observability for retrieval-augmented generation
  - `embedding.model_name` - Embedding model used
  - `embedding.vector_dimensions` - Vector dimensions
  - `retrieval.documents.{i}.document.id` - Retrieved document IDs
  - `retrieval.documents.{i}.document.score` - Relevance scores
  - `retrieval.documents.{i}.document.content` - Document content (truncated)

- ✅ Agent Workflow Tracking - Better support for agentic workflows
  - `agent.name` - Agent identifier
  - `agent.iteration` - Current iteration number
  - `agent.action` - Action taken
  - `agent.observation` - Observation received

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
Apache-2.0 license