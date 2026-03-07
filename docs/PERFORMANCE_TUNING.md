# Performance Tuning Guide

This guide covers configuration options for optimizing genai-otel-instrument in production.

## Import Time

The library uses lazy imports. `import genai_otel` loads in ~20ms. Heavy modules (OTel SDK, instrumentors, GPU metrics) are only loaded when you call `genai_otel.instrument()` or access specific classes.

## Sampling Rate

Reduce telemetry volume by sampling a fraction of traces:

```bash
# Only trace 10% of requests
export GENAI_SAMPLING_RATE=0.1
```

```python
from genai_otel import instrument
instrument(sampling_rate=0.1)  # 10% of traces
```

A sampling rate of `1.0` (default) traces everything. Set to `0.0` to disable tracing entirely.

## Batch Export Configuration

The library uses `BatchSpanProcessor` and `PeriodicExportingMetricReader` for non-blocking export. OpenTelemetry environment variables control batching:

```bash
# Spans: max batch size and export interval
export OTEL_BSP_MAX_EXPORT_BATCH_SIZE=512    # default: 512
export OTEL_BSP_SCHEDULE_DELAY_MILLIS=5000   # default: 5000ms
export OTEL_BSP_MAX_QUEUE_SIZE=2048          # default: 2048

# Metrics: export interval
export OTEL_METRIC_EXPORT_INTERVAL=60000     # default: 60000ms (1 min)
```

For high-throughput applications, increase `MAX_QUEUE_SIZE` and `MAX_EXPORT_BATCH_SIZE`.

## Content Capture

Capturing prompt/response content adds overhead to span size and export bandwidth:

```bash
# Disable content capture for lower overhead
export GENAI_ENABLE_CONTENT_CAPTURE=false

# Or limit content length (default: 1000 chars)
export GENAI_CONTENT_MAX_LENGTH=200
```

## GPU Metrics

GPU metrics collection runs in a background thread. Disable if not needed:

```bash
export GENAI_ENABLE_GPU_METRICS=false
```

Or adjust collection interval:

```bash
export GENAI_GPU_METRICS_INTERVAL=30  # seconds (default: 15)
```

## Cost Tracking

Cost calculation adds minimal overhead but can be disabled:

```bash
export GENAI_ENABLE_COST_TRACKING=false
```

## MCP Instrumentation

Disable database/cache/API instrumentation if not needed:

```bash
export GENAI_ENABLE_MCP_INSTRUMENTATION=false
```

## Selective Instrumentors

Only enable the instrumentors you need:

```bash
# Only instrument OpenAI and Anthropic
export GENAI_ENABLED_INSTRUMENTORS=openai,anthropic
```

## OTLP Protocol

HTTP (default) works for most setups. gRPC may offer better performance for high-throughput:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## OTLP Timeout

Increase timeout for slow networks or reduce for faster failure:

```bash
export OTEL_EXPORTER_OTLP_TIMEOUT=5  # seconds (default: 10)
```

## Production Checklist

1. Set `GENAI_SAMPLING_RATE` to 0.1-0.5 for high-traffic services
2. Disable `GENAI_ENABLE_CONTENT_CAPTURE` unless needed for debugging
3. Set `GENAI_ENABLE_GPU_METRICS=false` if no GPUs are used
4. Use `GENAI_ENABLED_INSTRUMENTORS` to limit to providers you actually use
5. Set `GENAI_FAIL_ON_ERROR=false` (default) so instrumentation issues don't crash your app
6. Configure `OTEL_BSP_MAX_QUEUE_SIZE` based on your request volume
