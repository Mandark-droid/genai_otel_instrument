# Quick Start

## Prerequisites

- Python 3.9+ (3.10+ for evaluation features)
- An OpenTelemetry backend (Jaeger, Grafana, Datadog, etc.)
- API key for your LLM provider

## Step 1: Install

```bash
pip install genai-otel-instrument
```

## Step 2: Start an OTel Backend

The fastest way to get started is with Jaeger:

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Jaeger UI will be available at `http://localhost:16686`.

## Step 3: Instrument Your App

### Option A: One Line of Code (Recommended)

```python
import genai_otel
genai_otel.instrument()

# Your existing code works unchanged
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is OpenTelemetry?"},
    ],
    max_tokens=150,
)

print(f"Response: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}")
# Traces and metrics are automatically sent to your OTel backend
```

### Option B: Environment Variables (No Code Changes)

```bash
export OTEL_SERVICE_NAME=my-llm-app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OPENAI_API_KEY=your_key_here

python your_app.py
```

### Option C: CLI Wrapper

```bash
genai-instrument python your_app.py
```

### Option D: Programmatic Configuration

```python
import genai_otel

genai_otel.instrument(
    service_name="my-app",
    endpoint="http://localhost:4318",
    enable_cost_tracking=True,
    enable_gpu_metrics=True,
    enabled_instrumentors=["openai", "anthropic", "crewai"],
    sampling_rate=1.0,
)
```

## Step 4: View Traces

Open Jaeger at `http://localhost:16686`, select your service name, and click "Find Traces".

You should see:

- **Span name**: `openai.chat.completions` (or your provider's operation)
- **Attributes**: model name, token counts, cost, latency
- **Metrics**: request rates, token usage, cost over time

## What Gets Captured Automatically

For every LLM call:

| What | Attribute | Example Value |
|------|-----------|---------------|
| Provider | `gen_ai.system` | `openai` |
| Model | `gen_ai.request.model` | `gpt-4o-mini` |
| Prompt tokens | `gen_ai.usage.prompt_tokens` | `125` |
| Completion tokens | `gen_ai.usage.completion_tokens` | `87` |
| Total cost | `gen_ai.usage.cost.total` | `0.000315` |
| Latency | span duration | `1.23s` |

## Example: Multi-Provider App

```python
import genai_otel
genai_otel.instrument(service_name="multi-provider-app")

# OpenAI
from openai import OpenAI
openai_client = OpenAI()
openai_response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize quantum computing"}],
)

# Anthropic
import anthropic
anthropic_client = anthropic.Anthropic()
anthropic_response = anthropic_client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    messages=[{"role": "user", "content": "Explain neural networks"}],
)

# Both calls are traced with cost tracking automatically
# View them side-by-side in Jaeger with provider-specific attributes
```

## Example: CrewAI Multi-Agent

```python
import genai_otel
genai_otel.instrument(service_name="crewai-app")

from crewai import Crew, Agent, Task

researcher = Agent(
    role="Researcher",
    goal="Research AI trends",
    backstory="Expert AI researcher",
    llm="gpt-4o-mini"
)

task = Task(
    description="List top 3 AI trends in 2026",
    expected_output="Bullet point list",
    agent=researcher
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()

# Trace hierarchy: Crew -> Agent -> Task -> OpenAI calls
# Each level has its own span with rich attributes
```

## Example: With Evaluation

```python
import genai_otel

genai_otel.instrument(
    service_name="safe-llm-app",
    enable_pii_detection=True,
    pii_mode="redact",
    enable_toxicity_detection=True,
    enable_prompt_injection_detection=True,
)

from openai import OpenAI
client = OpenAI()

# PII in the prompt is automatically detected and redacted
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "My email is user@example.com, call me at 555-1234"}],
)

# Traces include evaluation attributes:
# evaluation.pii.prompt.detected = true
# evaluation.pii.prompt.entity_types = ["EMAIL_ADDRESS", "PHONE_NUMBER"]
# evaluation.pii.prompt.redacted = "My email is [REDACTED], call me at [REDACTED]"
```

## Next Steps

- [Configuration](configuration.md) - All settings and environment variables
- [LLM Providers](../guides/llm-providers.md) - Provider-specific setup and examples
- [Multi-Agent Frameworks](../guides/multi-agent-frameworks.md) - CrewAI, LangGraph, AutoGen, etc.
- [Cost Tracking](../guides/cost-tracking.md) - Custom pricing, supported models
- [Evaluation](../guides/evaluation.md) - PII, toxicity, bias, prompt injection detection
- [90+ Examples](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples) - Ready-to-run code for every feature
