# LLM Providers

TraceVerde auto-instruments 19+ LLM providers. No code changes are needed - just install the provider SDK and TraceVerde handles the rest.

## Providers with Full Cost Tracking

| Provider | Models | Install Extra |
|----------|--------|---------------|
| OpenAI | GPT-4o, GPT-4 Turbo, GPT-5.2, o1/o3, embeddings (50+) | `[openai]` |
| OpenRouter | All models via OpenAI-compatible API | `[openrouter]` |
| Anthropic | Claude Sonnet 4.6, Claude 3.5/3 series (15+) | `[anthropic]` |
| Google AI | Gemini 2.5/2.0 Pro/Flash, PaLM 2 (30+) | `[google]` |
| AWS Bedrock | Amazon Titan, Claude, Llama, Mistral (25+) | `[aws]` |
| Azure OpenAI | Same as OpenAI with Azure pricing | `[openai]` |
| Cohere | Command R/R+, Embed v4/v3, rerankers (15+) | `[cohere]` |
| Mistral AI | Large/Medium/Small, Mixtral, embeddings (20+) | `[mistral]` |
| Together AI | DeepSeek-R1, Llama 3.x, Qwen (25+) | `[together]` |
| Groq | Llama 3.x, Mixtral, Gemma, Whisper (20+) | `[groq]` |
| Ollama | All local models with token tracking | `[ollama]` |
| Vertex AI | Gemini models via Google Cloud | `[vertexai]` |
| SambaNova | sarvam-m, Saarika, Bulbul (12+) | `[sambanova]` |
| Sarvam AI | Indian language models | `[sarvamai]` |
| Replicate | Hardware-based pricing ($/second) | `[replicate]` |

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

### Hyperbolic

Requires OTLP gRPC exporter due to `requests` library conflicts.

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export GENAI_ENABLED_INSTRUMENTORS="openai,anthropic,hyperbolic"
```

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
