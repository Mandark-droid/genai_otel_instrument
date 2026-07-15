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

| Feature | TraceVerde | OpenLIT | Traceloop/OpenLLMetry | Langfuse | Galileo | Arize (Phoenix) | Opik (Comet) |
|---------|-----------|---------|----------------------|----------|---------|-----------------|--------------|
| Zero-code setup | Yes | Yes | Yes | SDK required | SDK required | SDK / auto (OpenInference) | SDK required |
| LLM providers | 20+ | 25+ | 15+ | Via integrations | Via integrations | Via integrations | Via integrations |
| Multi-agent frameworks | 8 (CrewAI, LangGraph, ADK, AutoGen, OpenAI Agents, Pydantic AI, etc.) | Limited | Limited | Limited | Limited | Limited | Limited |
| Cost tracking | Automatic (1,050+ models) | Manual config | Manual config | Manual config | Yes | Yes | Yes |
| GPU metrics (NVIDIA + AMD) | Yes | No | No | No | No | No | No |
| MCP tool instrumentation | Yes (databases, caches, vector DBs, queues) | Limited | Limited | No | No | No | No |
| Evaluation (PII, toxicity, bias, hallucination, prompt injection) | Built-in (6 detectors) | No | No | Separate service | Extensive (core focus) | Built-in (Phoenix evals) | Built-in (core focus) |
| OpenTelemetry native | Yes | Yes | Yes | Partial | Partial | Yes (OpenInference) | Partial |
| Self-hosted / on-prem | Yes (fully local) | Yes | Yes | Yes | Enterprise tier | Yes (Phoenix) | Yes |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT | Proprietary | Open-source + Commercial | Apache-2.0 |

## What Gets Instrumented?

### LLM Providers (20+)
OpenAI, OpenRouter, CometAPI, Anthropic, Google AI, Google GenAI, AWS Bedrock, Azure OpenAI, Cohere, Mistral AI, Together AI, Groq, Ollama, Vertex AI, Replicate, HuggingFace, SambaNova, Sarvam AI, Hyperbolic, LiteLLM

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

### Multimodal Observability (v1.1.0)

First-class capture of image, audio, video, and document content parts on
OpenAI, Anthropic, Google Gemini, and Groq spans. Bytes are offloaded to your
configured object store (MinIO / S3 / filesystem / HTTP) and referenced from
spans by URI — they never appear inline in span attributes.

```bash
# Opt in (default is off — text-only behaviour is byte-identical to 1.0.x)
export GENAI_OTEL_MEDIA_CAPTURE_MODE=full
export GENAI_OTEL_MEDIA_STORE=minio
export GENAI_OTEL_MEDIA_STORE_ENDPOINT=http://localhost:9000
export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=...
export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=...
# Optional: plug in a redactor before upload
export GENAI_OTEL_MEDIA_REDACTOR=genai_otel.media.redactors.face_blur
```

Spans get two co-emitted representations of the same multimodal content:

- A **flat, queryable attribute namespace** — `gen_ai.prompt.{n}.content.{m}.{type, media_uri, media_mime_type, media_byte_size, media_source}` plus a `gen_ai.completion.*` mirror — for backends that index on flat attributes.
- The **upstream-canonical `gen_ai.input.messages` / `gen_ai.output.messages` JSON** conforming to the [gen-ai message schemas](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-input-messages.json) in the dedicated `semantic-conventions-genai` repo, including the `document` modality, optional `byte_size`, and `stripped_reason` shape standardised by our upstream PRs #142 / #143 / #144 (see [Standards Contributions](#opentelemetry-standards-contributions) below).

[Multimodal guide >>](https://mandark-droid.github.io/genai_otel_instrument/guides/multimodal/)

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

## OpenTelemetry Standards Contributions

TraceVerde isn't only a consumer of OpenTelemetry GenAI semantic conventions — production gaps surfaced by the library are being upstreamed back into the spec. Active proposals on [`open-telemetry/semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai):

| PR | Proposal | Status |
|---|---|---|
| [**#142**](https://github.com/open-telemetry/semantic-conventions-genai/pull/142) | Add `document` to the `Modality` enum on `BlobPart` / `FilePart` / `UriPart` — PDFs, DOCX, and other non-image/video/audio payloads currently fall through to the free-form `string` branch. BFSI KYC extraction is a high-volume real example. | Approved (@MikeGoldsmith) |
| [**#143**](https://github.com/open-telemetry/semantic-conventions-genai/pull/143) | Add optional `byte_size` on the three media-part types so consumers get a uniform handle on payload size whether the content was carried inline, by URI, or by provider file id — useful for cost-of-capture telemetry and storage planning. Pydantic `ge=0` → JSON schema `"minimum": 0`. | Under review |
| [**#144**](https://github.com/open-telemetry/semantic-conventions-genai/pull/144) | Make `content` / `file_id` / `uri` optional and add a free-form `stripped_reason` (`size_exceeded`, `modality_not_allowed`, `redactor_error`, `upload_error`, `no_store_configured`) so an instrumentation can fail-closed — record that it observed a media part but intentionally did not capture its bytes — while preserving the original part `type` and `modality`. Enforced via a top-level `anyOf` so structurally-empty parts cannot validate. | Under review |

All three were migrated from the closed [`open-telemetry/semantic-conventions#3673`](https://github.com/open-telemetry/semantic-conventions/pull/3673) after the GenAI conventions split into the dedicated repo on 2026-05-05. Each PR ships under the new repo's V2 Weaver schema with the corresponding `models.ipynb` updates and `make check-policies` / `make generate-all` validation.

TraceVerde v1.1.1 already emits the proposed shape on the wire via dual-emission (`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai`), providing the reference implementation for these conventions.

## Ecosystem & Framework Contributions

Beyond the spec, `genai_otel` is shipping **inside agent frameworks** as their OpenTelemetry observability layer — in-tree where the project accepts it, or as a standalone plugin where the project's policy keeps integrations out of the core tree — demonstrating the library powering real third-party agents, not just first-party services:

| Integration | Framework | Contribution | Status |
|---|---|---|---|
| [**hermes-otel-plugin**](https://github.com/Mandark-droid/hermes-otel-plugin) | [NousResearch Hermes](https://github.com/NousResearch/hermes-agent) | A standalone `otel` plugin (`hermes plugins install Mandark-droid/hermes-otel-plugin`) exporting Hermes turns / LLM calls / tool calls as OTel GenAI spans **and** dashboard log records, with no changes to Hermes core. When `genai-otel-instrument` is installed it additionally unlocks on-prem **GPU / energy / CO2 metrics**, **local-model cost** (parameter-size pricing for Ollama / HF / vLLM), and an inline **eval / guardrail suite** (PII, toxicity, bias, prompt-injection, restricted-topics, hallucination) scored on prompt **and** response — signals no vanilla OTel SDK or other GenAI instrumentor emits. Originally upstream [PR #48184](https://github.com/NousResearch/hermes-agent/pull/48184) — approved on code review, then republished standalone per Hermes's policy that observability backends ship as standalone plugin repos. | **Shipped** (standalone plugin, MIT) |

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
