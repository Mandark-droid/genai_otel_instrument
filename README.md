# TraceVerde

<div align="center">
  <img src="https://raw.githubusercontent.com/Mandark-droid/genai_otel_instrument/main/.github/images/Logo.jpg" alt="TraceVerde Logo" width="400"/>

  **The most comprehensive OpenTelemetry auto-instrumentation library for LLM/GenAI applications**

  *Trace from OpenTelemetry traces. Verde meaning green - for sustainable, transparent AI observability.*

  [Documentation](https://mandark-droid.github.io/genai_otel_instrument/) | [Examples](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples) | [Discord](https://discord.gg/6SVz6VKK) | [PyPI](https://pypi.org/project/genai-otel-instrument/)
</div>

<br/>

<div align="center">

[![PyPI version](https://badge.fury.io/py/genai-otel-instrument.svg)](https://badge.fury.io/py/genai-otel-instrument)
[![Python Versions](https://img.shields.io/pypi/pyversions/genai-otel-instrument.svg)](https://pypi.org/project/genai-otel-instrument/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/badge/genai-otel-instrument)](https://pepy.tech/project/genai-otel-instrument)
[![Downloads/Month](https://static.pepy.tech/badge/genai-otel-instrument/month)](https://pepy.tech/project/genai-otel-instrument)
[![GitHub Stars](https://img.shields.io/github/stars/Mandark-droid/genai_otel_instrument?style=social)](https://github.com/Mandark-droid/genai_otel_instrument)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-1.20%2B-blueviolet)](https://opentelemetry.io/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/Mandark-droid/genai_otel_instrument/actions)

</div>

---

## Get Started in 30 Seconds

```bash
pip install genai-otel-instrument
```

```python
import genai_otel
genai_otel.instrument()

# Your existing code works unchanged - traces, metrics, and costs are captured automatically
import openai
client = openai.OpenAI()
response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello!"}])
```

That's it. No wrappers, no decorators, no config files. Every LLM call, database query, and agent interaction is automatically traced with full cost breakdown.

## Why TraceVerde?

| Feature | TraceVerde | OpenLIT | Traceloop/OpenLLMetry | Langfuse |
|---------|-----------|---------|----------------------|----------|
| Zero-code setup | Yes | Yes | Yes | SDK required |
| LLM providers | 19+ | 25+ | 15+ | Via integrations |
| Multi-agent frameworks | 8 (CrewAI, LangGraph, ADK, AutoGen, OpenAI Agents, Pydantic AI, etc.) | Limited | Limited | Limited |
| Cost tracking | Automatic (1,050+ models) | Manual config | Manual config | Manual config |
| GPU metrics (NVIDIA + AMD) | Yes | No | No | No |
| MCP tool instrumentation | Yes (databases, caches, vector DBs, queues) | Limited | Limited | No |
| Evaluation (PII, toxicity, bias, hallucination, prompt injection) | Built-in (6 detectors) | No | No | Separate service |
| OpenTelemetry native | Yes | Yes | Yes | Partial |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT |

## What Gets Instrumented?

### LLM Providers (19+)
OpenAI, OpenRouter, Anthropic, Google AI, Google GenAI, AWS Bedrock, Azure OpenAI, Cohere, Mistral AI, Together AI, Groq, Ollama, Vertex AI, Replicate, HuggingFace, SambaNova, Sarvam AI, Hyperbolic, LiteLLM

[See all providers with examples >>](https://mandark-droid.github.io/genai_otel_instrument/guides/llm-providers/)

### Multi-Agent Frameworks (8)
CrewAI, LangGraph, Google ADK, AutoGen, AutoGen AgentChat, OpenAI Agents SDK, Pydantic AI, AWS Bedrock Agents

[See all frameworks with examples >>](https://mandark-droid.github.io/genai_otel_instrument/guides/multi-agent-frameworks/)

### MCP Tools (20+)
**Databases:** PostgreSQL, MySQL, MongoDB, SQLAlchemy, TimescaleDB, OpenSearch, Elasticsearch, FalkorDB
**Caching:** Redis | **Queues:** Kafka, RabbitMQ | **Storage:** MinIO
**Vector DBs:** Pinecone, Weaviate, Qdrant, ChromaDB, Milvus, FAISS, LanceDB

[See all MCP tools >>](https://mandark-droid.github.io/genai_otel_instrument/guides/mcp-tools/)

### Built-in Evaluation (6 Detectors)
PII Detection (GDPR/HIPAA/PCI-DSS), Toxicity Detection, Bias Detection, Prompt Injection Detection, Restricted Topics, Hallucination Detection

[See all evaluation features with examples >>](https://mandark-droid.github.io/genai_otel_instrument/guides/evaluation/)

## Screenshots

<div align="center">
  <img src="https://raw.githubusercontent.com/Mandark-droid/genai_otel_instrument/main/.github/images/Screenshots/Traces_OpenAI.png" alt="OpenAI Traces" width="800"/>
  <p><em>OpenAI traces with token usage, costs, and latency</em></p>
</div>

<details>
<summary>More screenshots</summary>

### Ollama (Local LLM)
<div align="center">
  <img src="https://raw.githubusercontent.com/Mandark-droid/genai_otel_instrument/main/.github/images/Screenshots/Traces_Ollama.png" alt="Ollama Traces" width="800"/>
</div>

### SmolAgents with Tool Calls
<div align="center">
  <img src="https://raw.githubusercontent.com/Mandark-droid/genai_otel_instrument/main/.github/images/Screenshots/Traces_SmolAgent_with_tool_calls.png" alt="SmolAgent Traces" width="800"/>
</div>

### GPU Metrics
<div align="center">
  <img src="https://raw.githubusercontent.com/Mandark-droid/genai_otel_instrument/main/.github/images/Screenshots/GPU_Metrics.png" alt="GPU Metrics" width="800"/>
</div>

### OpenSearch Dashboard
<div align="center">
  <img src="https://raw.githubusercontent.com/Mandark-droid/genai_otel_instrument/main/.github/images/Screenshots/GENAI_OpenSearch_output.png" alt="OpenSearch Dashboard" width="800"/>
</div>

</details>

## Key Features

### Automatic Cost Tracking

1,050+ models across 30+ providers with per-request cost breakdown. Supports differential pricing (prompt vs completion), reasoning tokens, cache pricing, and custom model pricing.

```python
# Cost tracking is enabled by default - just instrument and go
genai_otel.instrument()

# Or add custom pricing for proprietary models
export GENAI_CUSTOM_PRICING_JSON='{"chat":{"my-model":{"promptPrice":0.001,"completionPrice":0.002}}}'
```

[Cost tracking guide >>](https://mandark-droid.github.io/genai_otel_instrument/guides/cost-tracking/)

### GPU Metrics (NVIDIA + AMD)

Real-time monitoring of utilization, memory, temperature, power, PCIe throughput, throttling, and ECC errors. Multi-GPU aggregate metrics included.

```bash
pip install genai-otel-instrument[gpu]      # NVIDIA
pip install genai-otel-instrument[amd-gpu]  # AMD
```

[GPU metrics guide >>](https://mandark-droid.github.io/genai_otel_instrument/guides/gpu-metrics/)

### Multi-Agent Tracing

Complete span hierarchy for agent frameworks with automatic context propagation:

```
Crew Execution
  +-- Agent: Senior Researcher (gpt-4)
  |     +-- Task: Research OpenTelemetry
  |           +-- openai.chat.completions (tokens: 1250, cost: $0.03)
  +-- Agent: Technical Writer (ollama:llama2)
        +-- Task: Write blog post
              +-- ollama.chat (tokens: 890, cost: $0.00)
```

### Safety & Evaluation

```python
genai_otel.instrument(
    enable_pii_detection=True,       # GDPR/HIPAA/PCI-DSS compliance
    enable_toxicity_detection=True,  # Perspective API + Detoxify
    enable_bias_detection=True,      # 8 bias categories
    enable_prompt_injection_detection=True,
    enable_hallucination_detection=True,
    enable_restricted_topics=True,
)
```

[Evaluation guide with 50+ examples >>](https://mandark-droid.github.io/genai_otel_instrument/guides/evaluation/)

## Configuration

```bash
# Required
export OTEL_SERVICE_NAME=my-llm-app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Optional
export GENAI_ENABLE_GPU_METRICS=true
export GENAI_ENABLE_COST_TRACKING=true
export GENAI_SAMPLING_RATE=0.5                    # Reduce volume in production
export GENAI_ENABLED_INSTRUMENTORS=openai,crewai  # Select specific instrumentors
```

[Full configuration reference >>](https://mandark-droid.github.io/genai_otel_instrument/getting-started/configuration/)

## Backend Integration

Works with any OpenTelemetry-compatible backend:

Jaeger, Zipkin, Prometheus, Grafana, Datadog, New Relic, Honeycomb, AWS X-Ray, Google Cloud Trace, Elastic APM, Splunk, SigNoz, self-hosted OTel Collector

Pre-built [Grafana dashboard templates](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/dashboards) included.

## Examples

90+ ready-to-run examples covering every provider, framework, and evaluation feature:

```
examples/
+-- openai/              # OpenAI chat, embeddings
+-- anthropic/           # Anthropic + PII/toxicity detection
+-- ollama/              # Local models + all evaluation features
+-- crewai_example.py    # Multi-agent crew orchestration
+-- langgraph_example.py # Stateful graph workflows
+-- google_adk_example.py # Google Agent Development Kit
+-- autogen_example.py   # Microsoft AutoGen agents
+-- pii_detection/       # 10 PII examples (GDPR, HIPAA, PCI-DSS)
+-- toxicity_detection/  # 8 toxicity examples
+-- bias_detection/      # 8 bias examples (hiring compliance, etc.)
+-- prompt_injection/    # 6 injection defense examples
+-- hallucination/       # 4 hallucination detection examples
+-- ...                  # And many more
```

[Browse all examples >>](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples)

## Who Uses TraceVerde?

TraceVerde is used by developers and teams building production GenAI applications. If you're using TraceVerde, we'd love to hear from you!

[Add your company](https://github.com/Mandark-droid/genai_otel_instrument/issues/new?title=Add+my+company+to+users+list&labels=users) | [Join Discord](https://discord.gg/6SVz6VKK)

<!-- Add your company/project logo and link here -->

## Community

- [Documentation](https://mandark-droid.github.io/genai_otel_instrument/) - Full guides, API reference, and tutorials
- [Discord](https://discord.gg/6SVz6VKK) - Chat with the community
- [GitHub Issues](https://github.com/Mandark-droid/genai_otel_instrument/issues) - Bug reports and feature requests
- [Contributing](https://mandark-droid.github.io/genai_otel_instrument/community/contributing/) - How to contribute

## License

Copyright 2025 Kshitij Thakkar. Licensed under the [Apache License 2.0](LICENSE).
