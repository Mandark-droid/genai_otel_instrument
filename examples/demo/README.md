# GenAI OTel Instrumentation - Complete Demo

This is a fully self-contained Docker demo that showcases the genai-otel-instrument library with multiple LLM providers and advanced trace analytics.

## What's Included

- **OpenSearch**: Long-term trace storage with full-text search and aggregations
- **Jaeger with OpenSearch Backend**: Distributed tracing with persistent storage
- **OpenTelemetry Collector**: Central telemetry pipeline
- **Prometheus**: Metrics storage and querying
- **Grafana**: Pre-built dashboards for metrics and trace analytics
- **GenAI Ingest Pipeline**: Automatically extracts and flattens all GenAI semantic convention fields
- **Demo Application**: Python app demonstrating:
  - OpenAI instrumentation
  - Anthropic Claude instrumentation
  - LangChain instrumentation
  - Automatic cost tracking
  - Token usage metrics
  - GPU metrics (if available)
  - CO2 emissions tracking
  - Distributed tracing

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (required)
- Anthropic API key (optional, for Claude demo)
- **System Configuration for OpenSearch**:
  ```bash
  # Required for OpenSearch to start properly
  sudo sysctl -w vm.max_map_count=262144

  # To make it permanent, add to /etc/sysctl.conf:
  echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
  ```

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

4. **View the traces and metrics**:
   - **Grafana**: http://localhost:3000 (dashboards - START HERE!)
     - "GenAI OTel Demo Metrics" - Token usage, costs, latency, GPU metrics
     - "GenAI Traces - OpenSearch" - Advanced trace analytics
   - **Jaeger UI**: http://localhost:16686 (trace visualization)
   - **OpenSearch**: http://localhost:9200 (direct API access)
   - **Prometheus**: http://localhost:9091 (raw metrics)

5. **Explore the data**:
   - In Grafana, navigate to "GenAI Traces - OpenSearch" dashboard
   - Click on any Trace ID to jump to detailed view in Jaeger
   - Analyze costs, token usage, and performance by model
   - Track errors and slow requests

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

## OpenSearch Integration

This demo includes OpenSearch for advanced trace analytics. See [OPENSEARCH_SETUP.md](./OPENSEARCH_SETUP.md) for detailed documentation.

### What You Get

- **Persistent Storage**: Traces are stored in OpenSearch, not just in-memory
- **Advanced Analytics**: Query and aggregate trace data using OpenSearch's powerful query DSL
- **GenAI Field Extraction**: Automatic extraction of all GenAI semantic convention fields:
  - Model names, token counts, costs
  - GPU metrics, CO2 emissions
  - Error details, performance metrics
- **Pre-built Dashboard**: Grafana dashboard with cost analysis, performance metrics, and error tracking
- **Direct Queries**: Use OpenSearch REST API for custom analytics

### Quick Examples

```bash
# View all extracted GenAI fields
curl "http://localhost:9200/jaeger-span-*/_search?pretty&size=1"

# Get total cost by model
curl "http://localhost:9200/jaeger-span-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 0,
  "query": {"exists": {"field": "gen_ai_system"}},
  "aggs": {
    "by_model": {
      "terms": {"field": "gen_ai_request_model.keyword"},
      "aggs": {
        "total_cost": {"sum": {"field": "gen_ai_cost_amount"}}
      }
    }
  }
}'
```

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
