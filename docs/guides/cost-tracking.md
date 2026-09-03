# Cost Tracking

TraceVerde includes automatic cost tracking with pricing data for **1,700+ models** across **40+ providers**. Every LLM call is enriched with per-request cost breakdown.

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
| `gen_ai.usage.cost.pricing_source` | Where the price came from | `table` |
| `gen_ai.request.model.deprecated` | Present only when the provider has announced the model's end | `true` |
| `gen_ai.request.model.deprecation_note` | Why, and what to migrate to | `Moonshot v1 platform sunset 2026-08-31...` |

### Spotting models that are about to stop existing

A deprecated model is not a pricing problem — it bills normally, at a real rate,
right up until the provider withdraws it. That is exactly what makes it easy to
miss: nothing in the telemetry looks wrong until the day the calls start failing,
and the only prior warning was a line in a vendor changelog.

Spans for such models carry `gen_ai.request.model.deprecated`, so the question
"what are we running that has an end date?" becomes a query instead of an audit:

```python
retiring = {s.attributes["gen_ai.request.model"]
            for s in spans if s.attributes.get("gen_ai.request.model.deprecated")}
```

The attributes are absent on active models, so this costs nothing in the normal
case. `pricing_source` stays `table` for deprecated models — conflating the two
would make a retiring model look unpriced when it is still costing money.

Deprecations are recorded in the `deprecated` map in
`genai_otel/llm_pricing.json`, keyed by pricing key so it covers every category,
including the ones whose values are bare numbers and cannot carry an inline flag.

### Audio prices declare their unit

A bare number cannot say what it is *per*. Text-to-speech bills per character,
transcription per second, and some audio models bill per token — so audio entries
in `genai_otel/llm_pricing.json` state the unit in the key:

```jsonc
"eleven_multilingual_v2":  { "per_1k_chars":  0.10     },  // synthesis
"elevenlabs/scribe_v1":    { "per_second":    6.11e-05 },  // transcription
"gpt-4o-transcribe":       { "per_1k_tokens": 0.0025   }
```

This is not tidiness. Storing both as an undifferentiated float is what allowed 42
entries to sit at a per-minute rate against a per-second contract — a 60× error
that nothing in the data contradicted. With the unit declared, billing a
per-second model by character is detected and **refused** rather than silently
producing a plausible number:

```python
calc.calculate_cost("elevenlabs/scribe_v1", {"characters": 1000}, "audio")  # -> 0.0 + warning
```

A bare number is still accepted for backwards compatibility and for custom
pricing, and keeps the old inferred-unit behaviour.

### Knowing which prices have actually been checked

Most of the pricing table is inherited from an upstream aggregate. That is a
reasonable starting point and a poor source of truth — auditing in August 2026
found rates stale by a full model generation, transposed between tiers, and off
by 2.5x, none of which looked wrong in the file.

Entries verified against the vendor's own pricing page carry the date that
happened, in the `prices_checked` map:

```python
from genai_otel.cost_calculator import CostCalculator
calc = CostCalculator()

calc.price_checked("elevenlabs/scribe_v1", "audio")   # '2026-08-12'
calc.price_checked("gpt-4o-mini")                     # None - never verified here
```

`None` means "nobody has checked this against the vendor", not "suspect". It is
the honest default for an inherited number, and it makes the audit a query
instead of a manual sweep:

```python
for key, checked in calc.stale_prices(older_than_days=180):
    print(key, checked or "never verified")
```

Pass `include_unverified=False` to see only entries that were checked once and
have since aged out.

### Telling "free" apart from "not measured"

`cost.total = 0.0` means one of two very different things, so every span also
carries `gen_ai.usage.cost.pricing_source`:

- **`table`** — matched the shipped pricing file. Billable.
- **`estimated`** — no entry for this model, so the price was inferred from the
  parameter count in its name (e.g. `...-7b`) via the local-model size tier.
  Useful for relative comparison, not for a bill.
- **`unpriced`** — no price could be determined at all.

Filter on this before aggregating. Summing `cost.total` across all spans without
it under-reports spend, because every unpriced model contributes `0.0` and looks
identical to a genuinely free call:

```python
# Spend you can defend, plus an explicit count of what could not be priced.
billable = [s for s in spans if s.attributes.get("gen_ai.usage.cost.pricing_source") == "table"]
unpriced = [s for s in spans if s.attributes.get("gen_ai.usage.cost.pricing_source") == "unpriced"]
print(f"${sum(s.attributes['gen_ai.usage.cost.total'] for s in billable):.4f}"
      f" across {len(billable)} calls; {len(unpriced)} calls could not be priced")
```

If `unpriced` is large, the model is missing from `genai_otel/llm_pricing.json` —
add it via `custom_pricing_json` or open an issue.

## Supported Providers

The pricing file carries **1400+ chat models**, covering roughly 83% of the
priced catalogue on models.dev. Coverage is deliberately limited to first-party
providers and the major clouds that resell under their own SKUs — gateway and
router listings are excluded, because they re-price the same underlying model
and the recorded cost would then depend on which aggregator was indexed first.

| Provider | Models | Pricing Type |
|----------|--------|--------------|
| OpenAI | GPT-5.5 series, GPT-4o, GPT-4 Turbo, o1/o3, embeddings (50+) | Per token (prompt/completion) |
| Anthropic | Claude Opus 5, Claude Sonnet 5, Claude Fable 5, Claude Opus 4.8, Claude Sonnet 4.6, Claude 3.5/3 series (20+) | Per token + cache pricing |
| Google AI | Gemini 3.7 Flash, Gemini 3.6 Flash, Gemini 3.5 Flash / Flash Lite, Gemini 3.1/2.5 Pro/Flash, PaLM 2 (35+) | Per token |
| AWS Bedrock | Titan, Claude, Llama, Mistral (25+) | Per token |
| Azure OpenAI | Same as OpenAI | Per token |
| Cohere | Command A/R/R+, North Mini Code, Embed v4/v3, rerankers (15+) | Per token |
| Mistral AI | Large/Medium/Small, Mixtral, embeddings (20+) | Per token |
| Moonshot AI | Kimi K3, Kimi K2.7 Code, K2.6, K2.5, Kimi Latest (20+) | Per token + cache pricing |
| Alibaba / Qwen | Qwen3.8-Max, Qwen3.8 Flash, Qwen3.8 27B, Qwen3.7 Flash/Max, Qwen3.6 series (40+) | Per token |
| Thinking Machines | Inkling, Inkling Small | Per token |
| Xiaomi | MiMo V2.5 Pro/UltraSpeed, MiMo V2 Flash/Omni/Pro (6+) | Per token + cache pricing |
| Zhipu / Z.AI | GLM-5.3, GLM-5.3-Flash, GLM-5.2, GLM-5.1, GLM-5 series | Per token + cache pricing |
| Meituan | LongCat-2.0, LongCat Flash Chat | Per token (cache free) |
| Sakana AI | Sakana Namazu, Fugu Ultra | Per token + cache pricing |
| Nvidia | Nemotron 3 Ultra, Nemotron 4/Nano/Super (15+) | Per token |
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
| xAI | Grok 4.6, Grok 4.20, Grok 4.1 (5+) | Per token |
| Upstage | Solar Pro 4, Solar Pro 2, Solar Mini | Per token |

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
