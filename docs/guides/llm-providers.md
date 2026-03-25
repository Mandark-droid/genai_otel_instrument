# LLM Providers

TraceVerde auto-instruments 19+ LLM providers. No code changes are needed - just install the provider SDK and TraceVerde handles the rest.

## Providers with Full Cost Tracking

| Provider | Models | Install Extra | Example |
|----------|--------|---------------|---------|
| OpenAI | GPT-4o, GPT-4 Turbo, GPT-5.2, o1/o3, embeddings (50+) | `[openai]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/openai/example.py) |
| OpenRouter | All models via OpenAI-compatible API | `[openrouter]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/openrouter/example.py) |
| Anthropic | Claude Sonnet 4.6, Claude 3.5/3 series (15+) | `[anthropic]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/anthropic/example.py) |
| Google AI | Gemini 2.5/2.0 Pro/Flash, PaLM 2 (30+) | `[google]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/google_ai/example.py) |
| AWS Bedrock | Amazon Titan, Claude, Llama, Mistral (25+) | `[aws]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/aws_bedrock/example.py) |
| Azure OpenAI | Same as OpenAI with Azure pricing | `[openai]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/azure_openai/example.py) |
| Cohere | Command R/R+, Embed v4/v3, rerankers (15+) | `[cohere]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/cohere/example.py) |
| Mistral AI | Large/Medium/Small, Mixtral, embeddings (20+) | `[mistral]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/mistralai/example.py) |
| Together AI | DeepSeek-R1, Llama 3.x, Qwen (25+) | `[together]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/togetherai/example.py) |
| Groq | Llama 3.x, Mixtral, Gemma, Whisper (20+) | `[groq]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/groq/example.py) |
| Ollama | All local models with token tracking | `[ollama]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/ollama/example.py) |
| Vertex AI | Gemini models via Google Cloud | `[vertexai]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/vertexai/example.py) |
| SambaNova | sarvam-m, Saarika, Bulbul (12+) | `[sambanova]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sambanova_example.py) |
| Sarvam AI | Indian language models | `[sarvamai]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/) |
| Replicate | Hardware-based pricing ($/second) | `[replicate]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/replicate/example.py) |

## Quick Example: OpenAI

```python
import genai_otel
genai_otel.instrument()

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
# Traces, metrics, and costs are automatically captured
```

## Quick Example: Anthropic

```python
import genai_otel
genai_otel.instrument()

import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
)

print(message.content[0].text)
# Cost tracking and token usage automatically captured
```

## Quick Example: Ollama (Local)

```python
import genai_otel
genai_otel.instrument()

import ollama

response = ollama.chat(
    model="llama2",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)

print(response["message"]["content"])
# Local model traces captured with token counting
```

## Special Providers

### HuggingFace Transformers

Local model execution with estimated costs based on parameter count.

```bash
pip install genai-otel-instrument[huggingface]
```

Instruments:

- `pipeline()`
- `AutoModelForCausalLM.generate()`
- `AutoModelForSeq2SeqLM.generate()`
- `InferenceClient` API calls

See examples:

- [Basic HuggingFace](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/example.py)
- [AutoModel](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/example_automodel.py)
- [With PII detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/pii_example.py)
- [With toxicity detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/toxicity_example.py)
- [With bias detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/bias_example.py)
- [Multiple evaluations](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/multiple_evaluations_example.py)

### Hyperbolic

Requires OTLP gRPC exporter due to `requests` library conflicts.

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export GENAI_ENABLED_INSTRUMENTORS="openai,anthropic,hyperbolic"
```

See [Hyperbolic example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/hyperbolic_example.py).

### Google GenAI (new SDK)

```bash
pip install genai-otel-instrument[google]
```

See [Google GenAI example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/google_genai_example.py).

### LiteLLM (Multi-Provider Proxy)

```bash
pip install genai-otel-instrument[openinference]
```

LiteLLM enables cost tracking across 100+ providers via a single proxy. See [LiteLLM example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/litellm/example.py).

### Smolagents (HuggingFace Agents)

```bash
pip install genai-otel-instrument[openinference]
```

See [Smolagents example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/smolagents/example.py).

## Captured Attributes

For every LLM call:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.system` | Provider name (e.g., "openai") |
| `gen_ai.request.model` | Requested model |
| `gen_ai.response.model` | Actual model used |
| `gen_ai.request.type` | Call type (chat, embedding) |
| `gen_ai.usage.prompt_tokens` | Input token count |
| `gen_ai.usage.completion_tokens` | Output token count |
| `gen_ai.usage.total_tokens` | Total tokens |
| `gen_ai.cost.amount` | Estimated cost in USD |

## All Examples

Browse all provider examples in the [examples/ directory](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples).
