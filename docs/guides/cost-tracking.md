# Cost Tracking

TraceVerde includes automatic cost tracking with pricing data for **1,050+ models** across **30+ providers**. Every LLM call is enriched with per-request cost breakdown.

## How It Works

Cost tracking is enabled by default. For every LLM call, TraceVerde:

1. Reads the model name from the span attributes
2. Looks up pricing in the built-in `llm_pricing.json` database
3. Calculates cost from token usage (prompt + completion)
4. Adds cost attributes to the span

No configuration needed - just instrument and go.

## Cost Attributes

Every LLM span gets these attributes:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.usage.cost.total` | Total cost in USD | `0.003250` |
| `gen_ai.usage.cost.prompt` | Prompt token cost | `0.001250` |
| `gen_ai.usage.cost.completion` | Completion token cost | `0.002000` |

## Supported Providers

| Provider | Models | Pricing Type |
|----------|--------|--------------|
| OpenAI | GPT-4o, GPT-4 Turbo, GPT-5.2, o1/o3, embeddings (50+) | Per token (prompt/completion) |
| Anthropic | Claude Sonnet 4.6, Claude 3.5/3 series (15+) | Per token + cache pricing |
| Google AI | Gemini 2.5/2.0 Pro/Flash, PaLM 2 (30+) | Per token |
| AWS Bedrock | Titan, Claude, Llama, Mistral (25+) | Per token |
| Azure OpenAI | Same as OpenAI | Per token |
| Cohere | Command R/R+, Embed v4/v3, rerankers (15+) | Per token |
| Mistral AI | Large/Medium/Small, Mixtral, embeddings (20+) | Per token |
| Together AI | DeepSeek-R1, Llama 3.x, Qwen (25+) | Per token |
| Groq | Llama 3.x, Mixtral, Gemma (20+) | Per token |
| Ollama | All local models | Token tracking (free) |
| Vertex AI | Gemini models | Per token |
| Replicate | All models | Per second (hardware-based) |
| HuggingFace | Local models | Estimated (parameter-based) |
| Sarvam AI | sarvam-m, Saarika, Bulbul (12+) | Per token |
| Voyage AI | voyage-4/3.5/3 series (15+) | Per token |
| Jina AI | jina-embeddings-v3, jina-clip-v2 (5+) | Per token |
| Deepgram | Nova-3/2, Aura, Whisper (20+) | Per second/character |
| AssemblyAI | Universal-3, slam-1 (5+) | Per second |
| ElevenLabs | Multilingual v2, Turbo v2 (8+) | Per character |
| IBM Granite | Chat, vision, embeddings (10+) | Per token |
| DeepSeek | V3, R1, VL (15+) | Per token |
| Qwen/Alibaba | Qwen 3.5, VL, embeddings (25+) | Per token |
| xAI | Grok 4.20, Grok 4.1 (5+) | Per token |

## Special Pricing

- **Reasoning tokens**: OpenAI o1/o3 series have separate pricing for reasoning tokens
- **Cache pricing**: Anthropic prompt caching costs (read/write rates)
- **Batch pricing**: Some providers offer discounted batch pricing
- **Hardware pricing**: Replicate charges per second of GPU/CPU time

## Custom Model Pricing

For models not in the pricing database:

```bash
# Chat models
export GENAI_CUSTOM_PRICING_JSON='{"chat":{"my-model":{"promptPrice":0.001,"completionPrice":0.002}}}'

# Embeddings
export GENAI_CUSTOM_PRICING_JSON='{"embeddings":{"my-embed":0.00005}}'

# Multiple categories
export GENAI_CUSTOM_PRICING_JSON='{
  "chat": {
    "my-custom-chat": {"promptPrice": 0.001, "completionPrice": 0.002}
  },
  "embeddings": {
    "my-custom-embed": 0.00005
  }
}'
```

Custom prices merge with defaults. If you provide pricing for an existing model, the custom price overrides the default.

**Pricing format:**

- **Chat models**: `{"promptPrice": <$/1k tokens>, "completionPrice": <$/1k tokens>}`
- **Embeddings**: Single number for price per 1k tokens
- **Audio**: Price per 1k characters (TTS) or per second (STT)

## OpenInference Cost Enrichment

When using OpenInference instrumentors (LiteLLM, Smolagents, MCP), cost tracking is automatically applied via `CostEnrichmentSpanProcessor`. It reads OpenInference semantic conventions and adds cost attributes:

- `llm.model_name` -> model lookup
- `llm.token_count.prompt` / `llm.token_count.completion` -> cost calculation
- `openinference.span.kind` -> call type (LLM, EMBEDDING, etc.)

## Disable Cost Tracking

```bash
export GENAI_ENABLE_COST_TRACKING=false
```

Or programmatically:

```python
genai_otel.instrument(enable_cost_tracking=False)
```

## Grafana Dashboard

Import the pre-built [GenAI overview dashboard](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/dashboards/grafana/genai-overview.json) to visualize costs over time by provider and model.
