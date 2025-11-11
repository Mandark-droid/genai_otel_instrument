# Server Metrics - Manual Instrumentation Guide

This guide explains how to use the **Server Metrics Collector** for manual instrumentation of server-side LLM serving metrics, including KV cache usage and request queue depth.

## Overview

Server metrics are designed for production LLM serving scenarios where you have access to server-side statistics from frameworks like:
- **vLLM** - High-throughput LLM serving
- **TGI** (Text Generation Inference) - HuggingFace's serving solution
- **Custom serving infrastructure** - Your own LLM servers

These metrics complement the automatic client-side instrumentation and provide insights into server capacity, memory usage, and request processing.

## Available Metrics

### KV Cache Metrics

**`gen_ai.server.kv_cache.usage`** (Gauge)
- GPU KV-cache usage percentage (0-100)
- **Attributes**: `model` (model name)
- **Use case**: Monitor memory pressure, detect cache exhaustion

### Request Queue Metrics

**`gen_ai.server.requests.running`** (Gauge)
- Number of requests currently executing on GPU
- **Use case**: Track active request load

**`gen_ai.server.requests.waiting`** (Gauge)
- Number of requests queued for processing
- **Use case**: Detect request queueing, identify bottlenecks

**`gen_ai.server.requests.max`** (Gauge)
- Maximum concurrent request capacity
- **Use case**: Track capacity limits, plan scaling

## Quick Start

### 1. Initialize Instrumentation

```python
import genai_otel

# Initialize with automatic instrumentation
genai_otel.instrument()

# Get the server metrics collector
server_metrics = genai_otel.get_server_metrics()
```

### 2. Set KV Cache Usage

```python
# Update KV cache usage for a model (0-100%)
server_metrics.set_kv_cache_usage("gpt-4", 75.5)
server_metrics.set_kv_cache_usage("llama-2-70b", 42.0)
```

### 3. Track Request Queue

```python
# Set maximum capacity
server_metrics.set_requests_max(50)

# Update current state
server_metrics.set_requests_running(5)   # 5 active requests
server_metrics.set_requests_waiting(12)  # 12 queued requests
```

### 4. Use Increment/Decrement Helpers

```python
# When a request starts
server_metrics.increment_requests_running()

# When a request completes
server_metrics.decrement_requests_running()

# For queue management
server_metrics.increment_requests_waiting()
server_metrics.decrement_requests_waiting()
```

## Integration Patterns

### vLLM Integration

```python
from vllm import AsyncLLMEngine
import asyncio
import genai_otel

genai_otel.instrument()
server_metrics = genai_otel.get_server_metrics()

async def update_server_metrics(engine: AsyncLLMEngine):
    """Periodically update server metrics from vLLM engine."""
    while True:
        try:
            # Get KV cache statistics
            for model_name in engine.model_names:
                cache_stats = await engine.get_model_cache_stats(model_name)
                cache_usage_pct = (
                    cache_stats["used_blocks"] / cache_stats["total_blocks"]
                ) * 100
                server_metrics.set_kv_cache_usage(model_name, cache_usage_pct)

            # Get request queue statistics
            queue_stats = await engine.get_request_queue_stats()
            server_metrics.set_requests_running(queue_stats["running"])
            server_metrics.set_requests_waiting(queue_stats["waiting"])

        except Exception as e:
            print(f"Error updating metrics: {e}")

        await asyncio.sleep(1)  # Update every second

# Start metrics updater as background task
asyncio.create_task(update_server_metrics(engine))
```

### TGI (Text Generation Inference) Integration

```python
import requests
import genai_otel
from threading import Thread
import time

genai_otel.instrument()
server_metrics = genai_otel.get_server_metrics()

def poll_tgi_metrics(tgi_metrics_url: str, interval: float = 1.0):
    """Poll TGI metrics endpoint and update server metrics."""
    while True:
        try:
            response = requests.get(f"{tgi_metrics_url}/metrics")
            metrics = parse_prometheus_metrics(response.text)

            # Extract relevant metrics (adjust based on TGI version)
            server_metrics.set_requests_running(
                metrics.get("tgi_request_inference_count", 0)
            )
            server_metrics.set_requests_waiting(
                metrics.get("tgi_queue_size", 0)
            )

        except Exception as e:
            print(f"Error polling TGI metrics: {e}")

        time.sleep(interval)

# Start metrics poller in background thread
Thread(target=poll_tgi_metrics, args=("http://localhost:8080",), daemon=True).start()
```

### Custom Request Handler Integration

```python
import genai_otel
from contextlib import contextmanager

genai_otel.instrument()
server_metrics = genai_otel.get_server_metrics()

@contextmanager
def track_request():
    """Context manager to track request lifecycle."""
    server_metrics.increment_requests_running()
    try:
        yield
    finally:
        server_metrics.decrement_requests_running()

class LLMRequestHandler:
    def __init__(self):
        # Set max capacity
        server_metrics.set_requests_max(100)

    async def handle_request(self, request):
        # Track request processing
        with track_request():
            # Your request processing logic
            result = await self.process_llm_request(request)
            return result

    def update_cache_metrics(self, model: str, cache_usage: float):
        """Update KV cache metrics (called from your cache manager)."""
        server_metrics.set_kv_cache_usage(model, cache_usage)
```

## API Reference

### ServerMetricsCollector

#### KV Cache Methods

```python
set_kv_cache_usage(model_name: str, usage_percent: float)
```
Set KV cache usage for a model (clamped to 0-100 range).

#### Request Queue Methods

```python
set_requests_running(count: int)
```
Set number of currently running requests.

```python
set_requests_waiting(count: int)
```
Set number of requests waiting in queue.

```python
set_requests_max(count: int)
```
Set maximum concurrent request capacity.

#### Helper Methods

```python
increment_requests_running(delta: int = 1)
decrement_requests_running(delta: int = 1)
increment_requests_waiting(delta: int = 1)
decrement_requests_waiting(delta: int = 1)
```
Increment/decrement request counters atomically.

### Global Access

```python
from genai_otel import get_server_metrics

# Get the global server metrics collector instance
server_metrics = get_server_metrics()

# Returns None if instrumentation not initialized
if server_metrics:
    server_metrics.set_kv_cache_usage("model", 50.0)
```

## Prometheus Queries

Once metrics are exported, you can query them in Prometheus:

```promql
# KV cache usage by model
gen_ai_server_kv_cache_usage{model="gpt-4"}

# Average KV cache usage across all models
avg(gen_ai_server_kv_cache_usage)

# Request queue depth
gen_ai_server_requests_waiting

# Capacity utilization
gen_ai_server_requests_running / gen_ai_server_requests_max * 100

# Alert on high queue depth
gen_ai_server_requests_waiting > 50
```

## Grafana Dashboards

Example dashboard panels:

**KV Cache Usage**
```
Metric: gen_ai_server_kv_cache_usage
Type: Time series (line graph)
Legend: {{model}}
```

**Request Queue Depth**
```
Metric: gen_ai_server_requests_waiting
Type: Time series (area graph)
Alert: > 100 requests
```

**Request Throughput**
```
Metric: rate(gen_ai_requests[1m])
Type: Time series (line graph)
Combined with: gen_ai_server_requests_running
```

## Thread Safety

All server metrics operations are **thread-safe** and can be called from:
- Multiple threads
- Async coroutines
- Background workers
- Request handlers

The `ServerMetricsCollector` uses internal locks to ensure atomic updates.

## Best Practices

1. **Update Frequency**: Update KV cache metrics every 1-5 seconds for accurate monitoring
2. **Request Tracking**: Use increment/decrement methods in try-finally blocks
3. **Model Names**: Use consistent model naming across your stack
4. **Capacity Planning**: Set `requests_max` based on your GPU memory and model size
5. **Alerting**: Set up alerts for high cache usage (>90%) and queue depth

## Troubleshooting

**Metrics not appearing in Prometheus:**
- Ensure `genai_otel.instrument()` is called before using metrics
- Check OTLP endpoint configuration
- Verify metrics are being set (use debug logging)

**KV cache always shows 0:**
- Server metrics require **manual instrumentation**
- Must call `set_kv_cache_usage()` explicitly
- Check integration with your serving framework

**Request counts incorrect:**
- Use try-finally to ensure decrements always happen
- Check for exceptions in request handlers
- Verify no requests are bypassing tracking

## Examples

See:
- `examples/server_metrics_example.py` - Complete simulation
- `examples/huggingface/example_automodel.py` - Basic usage
- `examples/vllm_integration.py` - Production vLLM pattern (if available)

## Related Documentation

- [NVIDIA NIM Observability](https://docs.nvidia.com/nim/large-language-models/latest/observability.html)
- [OpenTelemetry Metrics API](https://opentelemetry.io/docs/specs/otel/metrics/api/)
- [Prometheus Metric Types](https://prometheus.io/docs/concepts/metric_types/)
