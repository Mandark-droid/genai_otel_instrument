# Cost Tracking

TraceVerde includes automatic cost tracking with pricing data for 1,050+ models across 30+ providers.

## How It Works

When cost tracking is enabled (default), every LLM call span is enriched with:

- `gen_ai.usage.cost.total` - Total cost in USD
- `gen_ai.usage.cost.prompt` - Cost for prompt/input tokens
- `gen_ai.usage.cost.completion` - Cost for completion/output tokens

Cost is calculated from token usage and model pricing stored in `llm_pricing.json`.

## Pricing Models

- **Token-based** - Most providers (separate prompt and completion rates)
- **Reasoning tokens** - OpenAI o1/o3 series
- **Cache pricing** - Anthropic prompt caching (read/write rates)
- **Hardware-based** - Replicate ($/second of GPU time)
- **Estimated** - HuggingFace local models (based on parameter count)

## Custom Model Pricing

For models not in the pricing database, provide custom pricing via environment variable:

```bash
# Chat models
export GENAI_CUSTOM_PRICING_JSON='{"chat":{"my-model":{"promptPrice":0.001,"completionPrice":0.002}}}'

# Embeddings
export GENAI_CUSTOM_PRICING_JSON='{"embeddings":{"my-embed":0.00005}}'
```

Custom prices are merged with defaults. If you provide pricing for an existing model, the custom price overrides the default.

## Disable Cost Tracking

```bash
export GENAI_ENABLE_COST_TRACKING=false
```

## OpenInference Cost Enrichment

When using OpenInference instrumentors (LiteLLM, Smolagents, MCP), cost tracking is automatically applied via `CostEnrichmentSpanProcessor`. It reads OpenInference semantic conventions (`llm.model_name`, `llm.token_count.*`) and adds cost attributes.
