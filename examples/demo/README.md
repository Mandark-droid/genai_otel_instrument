# GenAI OTel Instrumentation - Complete Demo

This is a fully self-contained Docker demo that showcases the genai-otel-instrument library with multiple LLM providers.

## What's Included

- **Jaeger**: For trace visualization and metrics
- **Demo Application**: Python app demonstrating:
  - OpenAI instrumentation
  - Anthropic Claude instrumentation
  - LangChain instrumentation
  - Automatic cost tracking
  - Token usage metrics
  - Distributed tracing

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (required)
- Anthropic API key (optional, for Claude demo)

### Setup

1. **Copy the environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys**:
   ```bash
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...  # Optional
   ```

3. **Start the demo**:
   ```bash
   docker-compose up --build
   ```

4. **View the traces**:
   - Open http://localhost:16686 in your browser
   - Select "genai-demo-app" from the Service dropdown
   - Click "Find Traces"
   - Explore the captured telemetry data!

## What You'll See

### In the Console

The demo app will run through 3 scenarios:
1. OpenAI GPT-3.5 Turbo completion
2. Anthropic Claude message
3. LangChain chain execution

Each will show:
- ✅ Success message with response
- 📊 Token usage
- 💰 Cost tracking

### In Jaeger UI (http://localhost:16686)

You'll see detailed traces including:
- **Spans**: Each LLM call as a separate span
- **Tags**: Model name, provider, token counts, costs
- **Timing**: Request duration and latency
- **Parent-Child Relationships**: LangChain chains show nested structure
- **Metadata**: All request/response attributes

### Metrics

The following metrics are automatically captured:
- `genai.requests`: Request counts by provider and model
- `genai.tokens`: Token usage (prompt, completion, total)
- `genai.latency`: Request latency histogram
- `genai.cost`: Estimated costs in USD
- `genai.errors`: Error counts

## Stopping the Demo

```bash
docker-compose down
```

To remove all data and start fresh:
```bash
docker-compose down -v
```

## Customization

### Add More Providers

Edit `app.py` to add demos for:
- Google Gemini
- AWS Bedrock
- Azure OpenAI
- Groq
- Mistral AI
- And more!

### Change Configuration

Edit `docker-compose.yml` to modify:
- Service name
- OTLP endpoint
- Feature flags (GPU metrics, cost tracking, etc.)
- Log level

### Use Different Collector

Replace Jaeger with:
- Grafana Tempo
- Elastic APM
- Honeycomb
- Datadog
- Any OTLP-compatible backend

Just update the `OTEL_EXPORTER_OTLP_ENDPOINT` in docker-compose.yml

## Troubleshooting

### "No API key found"
Make sure you've copied `.env.example` to `.env` and added your API keys.

### "Cannot connect to collector"
Ensure Jaeger container is running: `docker ps | grep jaeger`

### "Import error"
The demo includes all dependencies in `requirements.txt`. If you see import errors, try rebuilding:
```bash
docker-compose build --no-cache
```

## Next Steps

1. Try adding your own LLM calls to `app.py`
2. Explore different OpenTelemetry backends
3. Integrate into your existing applications
4. Set up alerts based on cost or error metrics
5. Create custom dashboards for your GenAI workloads

## Learn More

- [Project README](../../README.md)
- [Troubleshooting Guide](../../TROUBLESHOOTING.md)
- [Semantic Conventions](../../OTEL_SEMANTIC_COMPATIBILITY.md)
