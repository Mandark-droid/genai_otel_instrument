# Server Metrics - Manual Instrumentation Guide

This guide explains how to use the **Server Metrics Collector** for manual instrumentation of server-side LLM serving metrics, including KV cache usage and request queue depth.

## Overview

Server metrics are designed for production LLM serving scenarios where you have access to server-side statistics from frameworks like:
- **Ollama** - Local LLM serving with automatic metrics collection (see Automatic Integration section)
- **vLLM** - High-throughput LLM serving
- **TGI** (Text Generation Inference) - HuggingFace's serving solution
- **Custom serving infrastructure** - Your own LLM servers

These metrics complement the automatic client-side instrumentation and provide insights into server capacity, memory usage, and request processing.

> **Note**: Ollama integration is **automatic** - no manual instrumentation needed! See the [Ollama Automatic Integration](#ollama-automatic-integration) section below.

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

## Ollama Automatic Integration

**Ollama server metrics are collected automatically** - no manual instrumentation required!

> **Note**: This feature requires **Python 3.11 or higher**. On Python 3.9 and 3.10, the feature is automatically skipped and a debug message is logged. Basic Ollama instrumentation (tracing and metrics) still works on all supported Python versions.

When you enable Ollama instrumentation, the library automatically:
1. Polls Ollama's `/api/ps` endpoint every 5 seconds (configurable)
2. Extracts VRAM usage per model
3. Updates `gen_ai.server.kv_cache.usage` metrics
4. Tracks number of models loaded in memory

### Quick Start with Ollama

```python
import genai_otel

# Initialize instrumentation (Ollama server metrics enabled by default)
genai_otel.instrument()

import ollama

# Make requests - metrics are collected automatically in the background
response = ollama.generate(model="llama2", prompt="Hello!")
```

That's it! The metrics will appear automatically.

### Configuration

Control Ollama server metrics via environment variables:

```bash
# Enable/disable automatic Ollama server metrics (default: true)
export GENAI_ENABLE_OLLAMA_SERVER_METRICS=true

# Ollama server URL (default: http://localhost:11434)
export OLLAMA_BASE_URL=http://localhost:11434

# Polling interval in seconds (default: 5.0)
export GENAI_OLLAMA_METRICS_INTERVAL=5.0

# OPTIONAL: Your GPU's VRAM in GB for accurate percentage calculation
# If not set, VRAM is auto-detected using nvidia-ml-py or nvidia-smi
# Only set this if auto-detection fails or you want to override
# Example: 24 for RTX 4090, 16 for RTX 4080, 12 for RTX 3080
# export GENAI_OLLAMA_MAX_VRAM_GB=24
```

**GPU VRAM Auto-Detection:**
The poller automatically detects your GPU's VRAM using:
1. **nvidia-ml-py** (preferred) - Requires `pip install genai-otel-instrument[gpu]`
2. **nvidia-smi** (fallback) - Uses the command-line tool if available
3. **Manual override** - Set `GENAI_OLLAMA_MAX_VRAM_GB` to bypass auto-detection

Auto-detection logs:
```
INFO - Auto-detected GPU VRAM: 24.0GB  # Success via nvidia-ml-py
INFO - Auto-detected GPU VRAM: 16.0GB  # Success via nvidia-smi
INFO - GPU VRAM not detected, using heuristic-based percentages  # Both failed
```

### What Gets Collected

The automatic poller queries `/api/ps` and collects:

**Per-Model Metrics:**
- `gen_ai.server.kv_cache.usage{model="llama2"}` - VRAM usage percentage
- Calculated from model's `size_vram` field

**Example Output:**
```json
{
  "name": "gen_ai.server.kv_cache.usage",
  "description": "GPU KV-cache usage percentage",
  "unit": "%",
  "data": {
    "attributes": {"model": "llama2:latest"},
    "value": 45.2
  }
}
```

### Disabling Automatic Collection

To disable (and use manual instrumentation instead):

```bash
export GENAI_ENABLE_OLLAMA_SERVER_METRICS=false
```

### Example with Configuration

See `examples/ollama/example_with_server_metrics.py` for a complete demonstration.

```python
import os
import genai_otel

# Configure before instrumentation
os.environ["GENAI_OLLAMA_METRICS_INTERVAL"] = "3.0"  # Poll every 3 seconds
os.environ["GENAI_OLLAMA_MAX_VRAM_GB"] = "24"  # 24GB GPU

genai_otel.instrument()

import ollama

# Use Ollama normally - metrics collected automatically
response = ollama.chat(
    model="llama2",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Check what metrics were collected
from genai_otel import get_server_metrics
server_metrics = get_server_metrics()

# View KV cache usage
for model, usage in server_metrics._kv_cache_usage.items():
    print(f"{model}: {usage:.2f}% VRAM")
```

### How It Works

The `OllamaServerMetricsPoller`:
1. Runs in a background daemon thread
2. Queries `http://localhost:11434/api/ps` every N seconds
3. Parses the JSON response to extract model VRAM usage
4. Calculates percentage based on either:
   - `GENAI_OLLAMA_MAX_VRAM_GB` (if configured) - accurate percentage
   - Model size heuristic (if not configured) - approximation
5. Updates the global `ServerMetricsCollector` instance

The poller gracefully handles:
- Ollama server not running (logs debug message)
- Connection timeouts (2 second timeout per request)
- Parse errors (logs warnings)
- No models loaded (metrics show 0)

## Integration Patterns (Manual Instrumentation)

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
