# TraceVerde Dashboard Templates

Pre-built dashboard templates for visualizing GenAI traces and metrics collected by TraceVerde.

## Grafana Dashboards

### GenAI Overview (`grafana/genai-overview.json`)

Comprehensive LLM observability dashboard with:
- Request rate by provider (time series)
- Token usage: prompt vs completion (stacked time series)
- Cost tracking over time by provider
- Latency distribution (histogram)
- Error rate by provider and error type
- Top 10 models by usage (table)
- Summary stats: total cost, requests, tokens, avg latency (24h)

### GPU Metrics (`grafana/gpu-metrics.json`)

Real-time GPU monitoring dashboard with:
- GPU utilization gauge (per GPU)
- Memory utilization gauge (per GPU)
- Temperature gauge with thresholds
- Power consumption stat
- Time series for utilization, memory, temperature, power
- Multi-GPU aggregate metrics (mean utilization, total memory, total power, max temp)

### Import Instructions

1. Open Grafana and navigate to Dashboards > Import
2. Click "Upload dashboard JSON file"
3. Select the desired JSON file from this directory
4. Select your Prometheus datasource when prompted
5. Click "Import"

**Prerequisites:**
- Prometheus must be configured to scrape your OpenTelemetry Collector
- The OTel Collector must be configured with a Prometheus exporter
- TraceVerde must be running with metrics enabled

Example OTel Collector config:
```yaml
exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    metrics:
      exporters: [prometheus]
```

## OpenSearch Dashboards

### GenAI Overview (`opensearch/genai-overview.ndjson`)

Basic saved objects for OpenSearch Dashboards including index pattern and visualizations for GenAI trace data.

### Import Instructions

1. Open OpenSearch Dashboards
2. Go to Management > Saved Objects
3. Click "Import"
4. Select the `.ndjson` file
5. Click "Import" and resolve any conflicts
