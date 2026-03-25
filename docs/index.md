# TraceVerde

**The most comprehensive OpenTelemetry auto-instrumentation library for LLM/GenAI applications.**

TraceVerde provides production-ready, zero-code instrumentation for GenAI applications. Install, set two environment variables, and get complete observability across 19+ LLM providers, 8 multi-agent frameworks, and 20+ MCP tools.

## Key Features

- **Zero-Code Setup** - Just install and set env vars, or add one line of code
- **19+ LLM Providers** - OpenAI, Anthropic, Google AI, AWS Bedrock, Azure, Cohere, Mistral, Together AI, Groq, Ollama, and more
- **8 Multi-Agent Frameworks** - CrewAI, LangGraph, Google ADK, AutoGen, OpenAI Agents SDK, Pydantic AI, Haystack, DSPy
- **Automatic Cost Tracking** - 1,050+ model pricing database with per-request cost breakdown
- **GPU Metrics** - Real-time NVIDIA and AMD GPU monitoring (utilization, memory, temperature, power)
- **MCP Tool Instrumentation** - Databases, caches, vector DBs, message queues, object storage
- **Built-in Evaluation** - PII detection, toxicity, bias, prompt injection, restricted topics, hallucination detection
- **OpenTelemetry Native** - Works with any OTel-compatible backend (Grafana, Jaeger, Datadog, etc.)

## Quick Start

```bash
pip install genai-otel-instrument
```

```python
import genai_otel
genai_otel.instrument()

# Your existing code works unchanged
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Traces, metrics, and costs are captured automatically
```

## Next Steps

- [Installation](getting-started/installation.md) - Detailed installation options
- [Quick Start](getting-started/quickstart.md) - Get up and running in minutes
- [Configuration](getting-started/configuration.md) - Environment variables and options
- [LLM Providers](guides/llm-providers.md) - 19+ providers with code examples
- [Multi-Agent Frameworks](guides/multi-agent-frameworks.md) - CrewAI, LangGraph, Google ADK, AutoGen, and more
- [MCP Tools](guides/mcp-tools.md) - Databases, caches, vector DBs, message queues
- [Cost Tracking](guides/cost-tracking.md) - Automatic cost calculation for 1,050+ models
- [GPU Metrics](guides/gpu-metrics.md) - NVIDIA and AMD GPU monitoring
- [Evaluation & Safety](guides/evaluation.md) - PII, toxicity, bias, prompt injection, hallucination detection

## Examples

90+ ready-to-run examples for every provider, framework, and evaluation feature:

- [OpenAI](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/openai/), [Anthropic](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/anthropic/), [Google AI](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/google_ai/), [Ollama](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/ollama/), [AWS Bedrock](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/aws_bedrock/)
- [CrewAI](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/crewai_example.py), [LangGraph](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/langgraph_example.py), [Google ADK](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/google_adk_example.py), [AutoGen](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/autogen_example.py)
- [PII Detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/), [Toxicity](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/toxicity_detection/), [Bias](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bias_detection/), [Prompt Injection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/prompt_injection/)

Browse all examples in the [examples/ directory](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples).

## Community

- [GitHub](https://github.com/Mandark-droid/genai_otel_instrument)
- [Discord](https://discord.gg/6SVz6VKK)
- [PyPI](https://pypi.org/project/genai-otel-instrument/)
