# Demo Application

TraceVerde includes a demo application that showcases automatic instrumentation across multiple LLM providers in a single script.

## What It Does

The demo runs three instrumented calls back-to-back:

1. **OpenAI GPT-3.5 Turbo** - Chat completion
2. **Anthropic Claude** - Chat completion
3. **LangChain with OpenAI** - Chain execution

Each call is automatically traced with token usage, cost tracking, and latency metrics.

## Prerequisites

1. An OpenTelemetry backend (Jaeger recommended for quick start)
2. API keys for the providers you want to test

## Setup

### 1. Start Jaeger

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

### 2. Set API Keys

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key  # Optional
```

### 3. Install Dependencies

```bash
pip install genai-otel-instrument[openai,anthropic]
pip install langchain langchain-openai
```

### 4. Run the Demo

```bash
cd examples/demo
python app.py
```

## Expected Output

```
Starting GenAI OTel Demo...
Auto-instrumenting all LLM libraries...
Instrumentation enabled!

============================================================
DEMO 1: OpenAI GPT-3.5 Turbo
============================================================
OpenAI Response: OpenTelemetry is an open-source observability framework...
Tokens: 42
Estimated cost captured in metrics

============================================================
DEMO 2: Anthropic Claude
============================================================
Claude Response: Distributed tracing is a method...
Tokens: 38
Estimated cost captured in metrics

============================================================
DEMO 3: LangChain with OpenAI
============================================================
LangChain Response: Observability is the ability to...
Chain execution traced automatically

Demo Complete!
```

## View Traces

Open Jaeger at `http://localhost:16686`:

1. Select service `genai-app` from the dropdown
2. Click "Find Traces"
3. You'll see three traces - one for each provider call
4. Click any trace to see span details including:
   - `gen_ai.system` - Provider name
   - `gen_ai.request.model` - Model used
   - `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens` - Token counts
   - `gen_ai.usage.cost.total` - Estimated cost in USD

## Customizing the Demo

The demo gracefully skips providers whose API keys are not set. You can run it with just OpenAI, just Anthropic, or all three.

To add evaluation features:

```python
# Edit examples/demo/app.py, change the instrument() call:
genai_otel.instrument(
    enable_pii_detection=True,
    enable_toxicity_detection=True,
)
```

## Source Code

See [examples/demo/app.py](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/demo/app.py).
