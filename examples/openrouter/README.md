# OpenRouter Examples

This directory contains examples of using genai-otel-instrument with OpenRouter.

## What is OpenRouter?

OpenRouter provides unified access to 300+ LLM models from 60+ providers through an OpenAI-compatible API. It offers:
- **Multi-provider access**: Single API for Claude, GPT, Gemini, Llama, Mistral, and more
- **Automatic fallback**: Route to alternative providers if primary fails
- **Cost optimization**: Choose the best price/performance ratio
- **No vendor lock-in**: Switch between providers seamlessly

## Prerequisites

1. Install the package with OpenRouter support:
   ```bash
   pip install genai-otel-instrument[openrouter]
   # or just use the OpenAI SDK (already installed with genai-otel-instrument)
   pip install openai
   ```

2. Get an OpenRouter API key from [openrouter.ai](https://openrouter.ai/)

3. Set your API key:
   ```bash
   export OPENROUTER_API_KEY=your_openrouter_api_key
   ```

4. Configure OpenTelemetry endpoint:
   ```bash
   export OTEL_SERVICE_NAME=my-openrouter-app
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
   ```

## Examples

### Basic Example

`example.py` - Demonstrates basic OpenRouter usage with automatic instrumentation:
- Uses Claude 3.5 Sonnet via OpenRouter
- Automatic detection of OpenRouter client via base_url
- Token usage and cost tracking
- Shows available model options

Run it:
```bash
python example.py
```

## How It Works

The OpenRouter instrumentor:
1. **Automatically detects** OpenRouter clients by checking if `base_url` contains `openrouter.ai`
2. **Captures OpenRouter-specific parameters**: `provider` (routing preferences) and `route` (fallback strategy)
3. **Tracks token usage and costs** using OpenRouter's model pricing
4. **Records response attributes**: finish_reason, response_id, model used, etc.

## Observed Telemetry

When you run an OpenRouter example, you'll see:

**Traces:**
- Span name: `openrouter.chat.completion`
- Attributes:
  - `gen_ai.system`: "openrouter"
  - `gen_ai.request.model`: Model requested (e.g., "anthropic/claude-3.5-sonnet")
  - `gen_ai.response.model`: Actual model used
  - `gen_ai.usage.prompt_tokens`: Input tokens
  - `gen_ai.usage.completion_tokens`: Output tokens
  - `gen_ai.usage.cost.total`: Estimated cost in USD
  - `openrouter.provider`: Provider preferences (if specified)
  - `openrouter.route`: Routing strategy (if specified)

**Metrics:**
- `genai.requests`: Request count
- `genai.tokens`: Token usage
- `genai.latency`: Request latency histogram
- `genai.cost`: Cost tracking

## Model Selection

OpenRouter supports 300+ models with the format `provider/model`:

**Popular Models:**
```python
# Claude models
"anthropic/claude-3-opus"
"anthropic/claude-3.5-sonnet"
"anthropic/claude-3-haiku"

# GPT models
"openai/gpt-4-turbo"
"openai/gpt-4"
"openai/gpt-3.5-turbo"

# Gemini models
"google/gemini-pro"
"google/gemini-pro-1.5"

# Llama models
"meta-llama/llama-3-70b-instruct"
"meta-llama/llama-3-8b-instruct"

# Mistral models
"mistralai/mistral-large"
"mistralai/mixtral-8x7b-instruct"

# DeepSeek models
"deepseek/deepseek-chat"
"deepseek/deepseek-coder"
```

See the full model list at [openrouter.ai/models](https://openrouter.ai/models)

## Cost Tracking

OpenRouter pricing varies by model. The instrumentor uses base provider pricing (without the 5.5% platform fee) for cost calculations. You can customize pricing using the `GENAI_CUSTOM_PRICING_JSON` environment variable if needed.

## Troubleshooting

**Issue**: "OpenAI library not installed"
- **Solution**: Install with `pip install openai>=1.0.0`

**Issue**: Authentication error
- **Solution**: Ensure `OPENROUTER_API_KEY` is set correctly

**Issue**: Instrumentation not working
- **Solution**: Ensure `genai_otel.instrument()` is called BEFORE creating the OpenAI client

**Issue**: Cost tracking shows $0.00
- **Solution**: Some models may not have pricing in our database. You can add custom pricing with `GENAI_CUSTOM_PRICING_JSON`

## Learn More

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Pricing](https://openrouter.ai/pricing)
- [OpenRouter Models](https://openrouter.ai/models)
- [genai-otel-instrument Documentation](https://github.com/Mandark-droid/genai_otel_instrument)
