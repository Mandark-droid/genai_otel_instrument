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
- `gen_ai_usage_cost_total` / `_prompt` / `_completion`: Cost breakdown in USD
- `gen_ai_usage_cost_reasoning`: Cost of reasoning tokens
- `gen_ai_usage_cost_cache_read` / `_cache_write`: Prompt-cache cost split
- `gen_ai_usage_cost_pricing_source`: `table` when the model was priced from
  `llm_pricing.json`, `estimated` when it fell back to a parameter-size estimate

#### Prompt Cache and Reasoning Tokens
- `gen_ai_usage_cache_read_input_tokens`: Tokens served from a prompt cache
- `gen_ai_usage_cache_write_input_tokens`: Tokens written to a prompt cache.
  This is the current convention spelling; `gen_ai_usage_cache_creation_input_tokens`
  is the superseded name and is emitted alongside it under the default
  `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup`
- `gen_ai_usage_reasoning_output_tokens`: Reasoning tokens. Billed as output but
  producing no visible text, so they are worth charting separately

#### Per-Modality Token Usage
From [semantic-conventions-genai#440](https://github.com/open-telemetry/semantic-conventions-genai/pull/440):

- `gen_ai_usage_{text,image,audio}_input_tokens`
- `gen_ai_usage_{text,image,audio}_output_tokens`
- `gen_ai_usage_{text,image,audio}_cache_read_input_tokens`

**These are subsets, not additions.** Each is already included in
`gen_ai_usage_input_tokens` / `gen_ai_usage_output_tokens`, so summing a modality
field alongside the total double counts. A modality the provider did not report
is absent rather than zero, which keeps "no audio in this request" distinct from
"this provider does not break usage down".

#### Agent Budget Governance
From [semantic-conventions-genai#425](https://github.com/open-telemetry/semantic-conventions-genai/issues/425):

- `gen_ai_agent_name`: Agent identifier, the natural grouping key
- `gen_ai_agent_token_budget` / `_consumed`: Configured token cap and actual use
- `gen_ai_agent_iteration_budget` / `_consumed`: Configured loop cap and actual use

Most frameworks expose only an iteration budget; a token budget appears only where
the framework configures one and is never synthesised from the two.

#### Inference Engine Latency (self-hosted engines)
From [semantic-conventions-genai#408](https://github.com/open-telemetry/semantic-conventions-genai/issues/408):

- `gen_ai_latency_time_in_queue`, `_time_to_first_token`, `_e2e`
- `gen_ai_latency_time_in_model_prefill`, `_time_in_model_decode`, `_time_in_model_inference`

**Expect these absent on current vLLM.** The V1 engine sets
`RequestOutput.metrics` to `None` and exposes no per-request timing on its Python
API, so nothing is emitted rather than a wall-clock guess being substituted. They
populate on engines that do report them.

#### Endpoint and Request Parameters
- `server_address` / `server_port`: The endpoint a call actually reached, so
  self-hosted, proxied and gateway traffic is distinguishable from a vendor's
  public API. Absent when the SDK exposes no base URL
- `gen_ai_request_seed`, `_top_k`, `_choice_count`, `_stream`: Recorded only when
  the caller passed them, so an absent value means "not set" rather than a
  provider default
- `gen_ai_output_type`: `json` or `text`, derived from `response_format`
- `gen_ai_request_id`: Engine-assigned request id (vLLM / SGLang)
- `gen_ai_embeddings_dimension_count`, `gen_ai_request_encoding_formats`

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

#### Host and Process Context
These arrive as **process tags** rather than span tags, so they are lifted from a
separate block in the pipeline:

- `host_name`, `host_arch`: Machine identity and architecture
- `os_type`, `os_version`: Operating system
- `process_pid`: Process id, typed `integer` so it can be aggregated
- `process_runtime_name`, `process_runtime_version`: Python runtime
- `telemetry_distro_name` / `_version`, `telemetry_auto_version`: Which
  instrumentation build produced the span

### How a field becomes usable

Adding an attribute to the library is not enough for it to appear here. Each one
needs **both** halves:

1. **Promotion** in the ingest pipeline, lifting it from `tags` (or
   `process.tags`) to a top-level `ctx.*` field.
2. **An explicit type** in the index template's `properties` block.

Anything missing the second half falls through to the `strings_as_keyword`
dynamic template, which sets `"index": false` - leaving the field neither
searchable nor aggregatable. Numeric fields must be declared `integer`, `long`,
`double` or `float`; left as `tag.*` keywords they can be displayed but never
summed, averaged or charted.

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

### Prompt Cache & Reasoning
- **Cache Effectiveness by Model**: Cache reads against cache writes and total
  input tokens, plus cache-read cost. A working prompt-cache setup shows reads
  far exceeding writes; the reverse means the cache is being rewritten rather
  than hit.
- **Reasoning Tokens & Cost by Model**: Reasoning tokens are billed as output but
  produce no text the user sees, so comparing them to total output tokens shows
  how much of the spend is invisible.

### Token Usage by Modality
- **Modality Token Split by Model**: The per-modality breakdown from
  [semantic-conventions-genai#440](https://github.com/open-telemetry/semantic-conventions-genai/pull/440).
- **These values are subsets of the totals.** Adding a modality column to
  `gen_ai_usage_input_tokens` double counts. The panel description repeats this,
  because it is the easy mistake to make when building a derived panel.

### Agent Budget Governance
- **Budget Utilisation by Agent**: Configured cap against actual consumption, per
  [semantic-conventions-genai#425](https://github.com/open-telemetry/semantic-conventions-genai/issues/425).
- Consumption is accumulated on the agent span rather than summed from child
  spans, so the numbers survive head sampling that drops the child inference
  spans - which is exactly the situation a runaway agent produces.
- **Requests by Endpoint**: Groups traffic by `server_address`, so self-hosted,
  proxied and gateway calls are distinguishable from a vendor's public API.

### Inference Engine Latency
- **Engine Latency Breakdown by Model**: Queue, prefill, decode, time-to-first-token
  and end-to-end, using the keys from
  [semantic-conventions-genai#408](https://github.com/open-telemetry/semantic-conventions-genai/issues/408).
- **Expect this panel empty on current vLLM.** The V1 engine sets
  `RequestOutput.metrics` to `None` and exposes no per-request timing, so the
  library emits nothing rather than substituting a wall-clock guess. An empty
  panel here means "the engine did not report it", not a broken dashboard.

> **Two dashboard files, one uid.** Both
> `genai-opensearch-traces-dashboard.json` and the timestamped
> `GenAI Traces - OpenSearch-*.json` export carry the uid
> `genai-opensearch-traces`, and provisioning loads the whole directory. Whichever
> Grafana reads last wins, so edits must be applied to both copies until one is
> removed or given a distinct uid.

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
