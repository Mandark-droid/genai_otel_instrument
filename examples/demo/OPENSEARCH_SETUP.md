# OpenSearch Integration for GenAI Traces

This demo now includes OpenSearch for advanced trace analytics and long-term storage of GenAI telemetry data.

## What's Included

### Infrastructure
- **OpenSearch**: Stores traces with full-text search and aggregation capabilities
- **Jaeger with OpenSearch Backend**: Writes traces to OpenSearch instead of memory
- **GenAI Ingest Pipeline**: Automatically extracts and flattens all GenAI semantic convention fields
- **Index Template**: Pre-configured mappings for optimal query performance
- **Grafana OpenSearch Dashboard**: Pre-built analytics dashboard for GenAI traces

### Extracted GenAI Fields

The ingest pipeline extracts and flattens the following fields from span tags:

#### Core GenAI Fields
- `gen_ai_system`: Provider (openai, anthropic, google, etc.)
- `gen_ai_request_model`: Model name (gpt-3.5-turbo, claude-3-5-sonnet, etc.)
- `gen_ai_request_type`: Request type (chat, embedding, completion)
- `gen_ai_operation_name`: Operation being performed

#### Token Usage
- `gen_ai_usage_prompt_tokens`: Input tokens consumed
- `gen_ai_usage_completion_tokens`: Output tokens generated
- `gen_ai_usage_total_tokens`: Total tokens used

#### Cost Tracking
- `gen_ai_cost_amount`: Estimated cost in USD
- `gen_ai_cost_currency`: Currency (USD)

#### Performance Metrics
- `gen_ai_server_ttft`: Time to first token (streaming)
- `gen_ai_server_tbt`: Time between tokens (streaming)
- `duration`: Request duration in microseconds
- `span_status`: OK, SLOW, or ERROR
- `trace_status`: Overall trace status

#### GPU Metrics (if enabled)
- `gen_ai_gpu_utilization`: GPU usage percentage
- `gen_ai_gpu_memory_used`: GPU memory consumption (MiB)
- `gen_ai_gpu_temperature`: GPU temperature (Celsius)
- `gen_ai_gpu_power`: GPU power consumption (Watts)
- `gpu_id`: GPU identifier
- `gpu_name`: GPU model name

#### Environmental Impact
- `gen_ai_co2_emissions`: CO2 emissions in grams (gCO2e)

#### Service Context
- `service_name`: Application name
- `service_instance_id`: Instance identifier
- `service_version`: Application version
- `telemetry_sdk_language`: SDK language (python, java, etc.)

#### Error Information
- `error`: Error flag (true/false)
- `exception_type`: Exception class name
- `exception_message`: Error message
- `exception_stacktrace`: Full stack trace
- `http_status_code`: HTTP response code (for API calls)

## Architecture

```
Demo App → OTel Collector → Jaeger → OpenSearch
                                ↓
                            Grafana Dashboards
```

1. **Demo App**: Instrumented with genai-otel-instrument
2. **OTel Collector**: Receives OTLP data and forwards to Jaeger
3. **Jaeger**: Processes traces and writes to OpenSearch
4. **OpenSearch**:
   - Runs ingest pipeline on incoming spans
   - Extracts and flattens GenAI fields
   - Stores in `jaeger-span-*` indices
5. **Grafana**: Queries OpenSearch for analytics

## System Requirements

OpenSearch requires the `vm.max_map_count` kernel parameter to be set:

```bash
# Check current value
sysctl vm.max_map_count

# Set to required value (temporary - resets on reboot)
sudo sysctl -w vm.max_map_count=262144

# Make it permanent
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Why is this needed?**
OpenSearch uses memory-mapped files extensively. The default limit (typically 65530) is too low and will cause OpenSearch to fail with errors like:
- `max virtual memory areas vm.max_map_count [65530] is too low`
- `bootstrap checks failed`

## Quick Start

### 1. Configure System (First Time Only)

```bash
# Set vm.max_map_count
sudo sysctl -w vm.max_map_count=262144
```

### 2. Start the Stack

```bash
cd examples/demo

# Ensure .env file exists with API keys
cp .env.example .env
# Edit .env and add your API keys

# Start all services
docker compose up --build
```

### 2. Verify OpenSearch Setup

The `opensearch-setup` container automatically creates:
- Ingest pipeline: `genai-ingest-pipeline`
- Index template: `jaeger-span-template`

Check the setup:

```bash
# Check pipeline
curl http://localhost:9200/_ingest/pipeline/genai-ingest-pipeline

# Check template
curl http://localhost:9200/_index_template/jaeger-span-template

# List indices
curl http://localhost:9200/_cat/indices/jaeger-span-*?v
```

### 3. Access the Dashboards

- **Grafana**: http://localhost:3000
  - Navigate to "GenAI Traces - OpenSearch" dashboard
- **Jaeger UI**: http://localhost:16686 (still available for trace viewing)
- **OpenSearch**: http://localhost:9200 (for direct queries)

## Using the GenAI Traces Dashboard

The pre-built Grafana dashboard includes:

### GenAI Request Overview
- **Table**: All GenAI requests with clickable trace IDs linking to Jaeger
- **Columns**: Trace ID, Timestamp, Provider, Model, Tokens, Cost, Duration, Status
- **Filters**: Automatically shows only root spans (top-level requests)

### Token Usage & Cost Analysis
- **By Model**: Total tokens, cost, and request count per model
- **By Provider**: Aggregated costs and usage by LLM provider

### Performance Analysis
- **Latency Stats**: Average, P95, and P99 duration by model
- **Identifies slow models**: Helps optimize model selection

### Error Analysis
- **Error Table**: All failed GenAI requests with error details
- **Columns**: Trace ID, Provider, Model, Error Type, Error Message, HTTP Status

## Example Queries

### Direct OpenSearch Queries

```bash
# Get all GenAI spans
curl "http://localhost:9200/jaeger-span-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "query": {
    "exists": {
      "field": "gen_ai_system"
    }
  },
  "size": 10
}'

# Aggregate cost by model
curl "http://localhost:9200/jaeger-span-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 0,
  "query": {
    "exists": {
      "field": "gen_ai_system"
    }
  },
  "aggs": {
    "by_model": {
      "terms": {
        "field": "gen_ai_request_model.keyword"
      },
      "aggs": {
        "total_cost": {
          "sum": {
            "field": "gen_ai_cost_amount"
          }
        },
        "total_tokens": {
          "sum": {
            "field": "gen_ai_usage_total_tokens"
          }
        }
      }
    }
  }
}'

# Find slow requests (>10 seconds)
curl "http://localhost:9200/jaeger-span-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "query": {
    "bool": {
      "must": [
        {
          "exists": {
            "field": "gen_ai_system"
          }
        },
        {
          "range": {
            "duration": {
              "gte": 10000000
            }
          }
        }
      ]
    }
  }
}'

# Get errors with details
curl "http://localhost:9200/jaeger-span-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "query": {
    "bool": {
      "must": [
        {
          "exists": {
            "field": "gen_ai_system"
          }
        },
        {
          "term": {
            "span_status": "ERROR"
          }
        }
      ]
    }
  }
}'
```

## Customizing the Pipeline

To modify the ingest pipeline, edit `opensearch-setup.sh` and update the pipeline definition. Then restart the setup:

```bash
docker compose up -d opensearch-setup
```

Or update it manually:

```bash
curl -X PUT "http://localhost:9200/_ingest/pipeline/genai-ingest-pipeline" \
  -H 'Content-Type: application/json' \
  -d @your-pipeline.json
```

## Index Management

### View Index Stats

```bash
# Get index sizes
curl "http://localhost:9200/_cat/indices/jaeger-span-*?v&h=index,docs.count,store.size"

# Get mapping
curl "http://localhost:9200/jaeger-span-*/_mapping?pretty"
```

### Clean Up Old Data

```bash
# Delete indices older than 7 days (manual cleanup)
curl -X DELETE "http://localhost:9200/jaeger-span-2025-01-01"
```

For production, consider using Index State Management (ISM) policies to automatically:
- Roll over indices daily
- Delete old indices after retention period
- Optimize replica count based on age

## Troubleshooting

### Pipeline Not Applied

Check if the pipeline is attached to the index:

```bash
curl "http://localhost:9200/jaeger-span-*/_settings?pretty" | grep pipeline
```

If not, the template may not have been applied before index creation. Delete and recreate indices:

```bash
curl -X DELETE "http://localhost:9200/jaeger-span-*"
# Restart demo app to regenerate spans
docker compose restart demo-app
```

### Fields Not Extracted

Verify the pipeline is working:

```bash
# Simulate pipeline execution
curl -X POST "http://localhost:9200/_ingest/pipeline/genai-ingest-pipeline/_simulate" \
  -H 'Content-Type: application/json' \
  -d '{
  "docs": [
    {
      "_source": {
        "tags": [
          {"key": "gen_ai.system", "value": "openai"},
          {"key": "gen_ai.request.model", "value": "gpt-3.5-turbo"}
        ]
      }
    }
  ]
}'
```

### OpenSearch Memory Issues

If OpenSearch crashes with memory errors, increase heap size in `docker-compose.yml`:

```yaml
environment:
  - "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g"  # Increase from 512m
```

## Performance Tuning

### Optimize for Write Performance

```bash
curl -X PUT "http://localhost:9200/jaeger-span-*/_settings" \
  -H 'Content-Type: application/json' \
  -d '{
  "index": {
    "refresh_interval": "30s",
    "number_of_replicas": 0
  }
}'
```

### Optimize for Query Performance

Add more replicas once data is stable:

```bash
curl -X PUT "http://localhost:9200/jaeger-span-*/_settings" \
  -H 'Content-Type: application/json' \
  -d '{
  "index": {
    "number_of_replicas": 1
  }
}'
```

## Next Steps

1. **Create Custom Dashboards**: Use the extracted fields to build custom analytics
2. **Set Up Alerts**: Configure Grafana alerts for high costs, errors, or slow requests
3. **Index Lifecycle Management**: Implement ISM policies for data retention
4. **Scale**: Add more OpenSearch nodes for production workloads
5. **Security**: Enable OpenSearch security plugin for production deployments

## References

- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [Jaeger OpenSearch Backend](https://www.jaegertracing.io/docs/latest/deployment/#opensearch)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Grafana OpenSearch Data Source](https://grafana.com/docs/grafana/latest/datasources/opensearch/)
