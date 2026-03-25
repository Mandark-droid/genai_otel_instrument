# Quick Start

## Three Ways to Instrument

### Option 1: One Line of Code

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
```

### Option 2: Environment Variables (No Code Changes)

```bash
export OTEL_SERVICE_NAME=my-llm-app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
python your_app.py
```

### Option 3: CLI Wrapper

```bash
genai-instrument python your_app.py
```

## Verify It Works

1. Start an OpenTelemetry backend (e.g., Jaeger):

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

2. Run your instrumented app:

```python
import genai_otel
genai_otel.instrument(
    service_name="my-app",
    endpoint="http://localhost:4318",
)

import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is OpenTelemetry?"}]
)
print(response.choices[0].message.content)
```

3. Open Jaeger at `http://localhost:16686` and find traces for `my-app`.

## What Gets Captured

For every LLM call, TraceVerde automatically captures:

- **Span attributes**: provider, model, request type
- **Token usage**: prompt tokens, completion tokens, total tokens
- **Cost**: estimated cost in USD based on model pricing
- **Latency**: request duration in seconds
- **Errors**: exception details and error types

## Next Steps

- [Configuration](configuration.md) - All available settings
- [LLM Providers](../guides/llm-providers.md) - Provider-specific setup
- [Cost Tracking](../guides/cost-tracking.md) - Cost calculation details
