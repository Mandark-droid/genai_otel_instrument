# Troubleshooting Guide

## Common Errors and Solutions

### Error: "resp.ok has no attribute" or "NoneType has no attribute 'ok'"

**Cause**: The OTLP exporter is trying to connect to a collector that isn't running or reachable.

**Solution 1: Use Console Exporter (for testing)**

Set `fail_on_error=False` in your configuration:

```python
from genai_otel import instrument

instrument(
    service_name="my-app",
    endpoint="http://localhost:4318",
    fail_on_error=False  # Will fall back to console exporter
)
```

Or via environment variable:
```bash
export GENAI_FAIL_ON_ERROR=false
```

**Solution 2: Start an OTLP Collector**

Using Jaeger (easiest):
```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 4318:4318 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

View traces at: http://localhost:16686

Using OpenTelemetry Collector:
```bash
docker run -d --name otel-collector \
  -p 4318:4318 \
  otel/opentelemetry-collector:latest
```

**Solution 3: Check Your Endpoint**

Make sure your endpoint is correctly formatted:
```python
# ✓ Correct
endpoint="http://localhost:4318"

# ✗ Wrong
endpoint="http://localhost:4318/v1/traces"  # Don't include path
endpoint="localhost:4318"  # Missing http://
```

---

### Error: "ModuleNotFoundError: No module named 'groq'" (or other LLM library)

**Cause**: You're trying to use an LLM library that isn't installed.

**Solution**: The library will be skipped automatically. This is NOT an error if you don't plan to use that library.

If you DO want to use it, install it:
```bash
# For specific libraries
pip install openai
pip install anthropic
pip install groq

# Or install all supported libraries
pip install genai-otel-instrument[all]
```

---

### Error: "Failed to load pricing data"

**Cause**: The `llm_pricing.json` file isn't being found.

**Solution**:

1. **If using pip install**: The file should be included automatically. Reinstall:
   ```bash
   pip uninstall genai-otel-instrument
   pip install genai-otel-instrument
   ```

2. **If using development mode**:
   ```bash
   pip install -e .
   ```

3. **Verify the file exists**:
   ```python
   import genai_otel
   import os
   package_dir = os.path.dirname(genai_otel.__file__)
   pricing_file = os.path.join(package_dir, 'llm_pricing.json')
   print(f"Pricing file exists: {os.path.exists(pricing_file)}")
   ```

---

### Error: "GPU metrics collection failed"

**Cause**: GPU metrics are enabled but:
- No NVIDIA GPU present
- `nvidia-ml-py` not installed
- GPU drivers not properly configured

**Solution**:

1. **Disable GPU metrics**:
   ```python
   instrument(enable_gpu_metrics=False)
   ```

   Or via environment:
   ```bash
   export GENAI_ENABLE_GPU_METRICS=false
   ```

2. **Install GPU monitoring dependencies**:
   ```bash
   pip install genai-otel-instrument[gpu]
   ```

3. **Check GPU availability**:
   ```python
   try:
       import pynvml  # nvidia-ml-py package
       pynvml.nvmlInit()
       print(f"GPUs available: {pynvml.nvmlDeviceGetCount()}")
       pynvml.nvmlShutdown()
   except:
       print("No GPU or nvidia-ml-py not available")
   ```

---

### Error: "Span export failed" or timeout errors

**Cause**: Network issues or slow collector.

**Solution**:

1. **Increase timeout** (update `auto_instrument.py`):
   ```python
   OTLPSpanExporter(
       endpoint=endpoint,
       headers=headers,
       timeout=30  # Increase from 10 to 30 seconds
   )
   ```

2. **Check collector is running**:
   ```bash
   curl http://localhost:4318/v1/traces
   # Should return 405 Method Not Allowed (means it's listening)
   ```

3. **Use console exporter for debugging**:
   ```python
   from opentelemetry.sdk.trace.export import ConsoleSpanExporter
   # This will print spans to console instead of sending to collector
   ```

---

### Error: "Cost tracking not working"

**Cause**: Pricing data not loaded or model name not recognized.

**Solution**:

1. **Check if pricing data loaded**:
   ```python
   from genai_otel import CostCalculator
   calc = CostCalculator()
   print(f"Models with pricing: {list(calc.pricing_data.keys())}")
   ```

2. **Check model name normalization**:
   ```python
   calc = CostCalculator()
   model_name = "gpt-4-0613"
   normalized = calc._normalize_model_name(model_name)
   print(f"'{model_name}' -> '{normalized}'")
   print(f"Has pricing: {normalized in calc.pricing_data}")
   ```

3. **Add custom pricing** (create your own `llm_pricing.json`):
   ```json
   {
     "models": {
       "your-model-name": {
         "prompt": 1.0,
         "completion": 2.0
       }
     }
   }
   ```

```bash
pip uninstall pynvml
pip install genai-otel-instrument[gpu]
```

---

### Issue: Metrics have changed names after upgrade

**Cause**: Version 2.0+ uses OpenTelemetry semantic conventions for metric names.

**Solution**:

This is a **BREAKING CHANGE** in the metric naming to comply with OpenTelemetry standards.

**Metric Name Changes:**
| Old Name (v1.x) | New Name (v2.0+) |
|-----------------|------------------|
| `genai.requests` | `gen_ai.requests` |
| `genai.tokens` | `gen_ai.client.token.usage` |
| `genai.latency` | `gen_ai.client.operation.duration` |
| `genai.cost` | `gen_ai.usage.cost` |
| `genai.errors` | `gen_ai.client.errors` |
| `genai.gpu.*` | `gen_ai.gpu.*` |
| `genai.co-2.emissions` | `gen_ai.co2.emissions` |

**Action Required:**
1. **Update Dashboard Queries** - Replace metric names in all dashboard queries
2. **Update Alerting Rules** - Update alert metric names
3. **Update Aggregations** - Update any metric aggregation rules

**Example Dashboard Update (Prometheus/Grafana):**
```promql
# OLD (v1.x)
sum(rate(genai_requests_total[5m])) by (operation)

# NEW (v2.0+)
sum(rate(gen_ai_requests_total[5m])) by (operation)
```

---

### Issue: Token attributes have changed

**Cause**: OpenTelemetry semantic conventions for token attributes have evolved.

**Solution**:

We now support **both old and new conventions** with the `OTEL_SEMCONV_STABILITY_OPT_IN` environment variable:

**Option 1: New conventions only (default)**
```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai
```
Emits: `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`

**Option 2: Dual emission (migration mode)**
```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup
```
Emits both:
- New: `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`
- Old: `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`

Use dual emission during migration to ensure compatibility with systems expecting different attribute names.

---

### Issue: GPU metrics showing as monotonic counters instead of gauges

**Cause**: Fixed in v2.0+. Earlier versions incorrectly used Counter for GPU utilization.

**Solution**:

Upgrade to v2.0+. GPU metrics now correctly use ObservableGauge:
- `gen_ai.gpu.utilization` - Gauge (0-100%)
- `gen_ai.gpu.memory.used` - Gauge (MiB)
- `gen_ai.gpu.temperature` - Gauge (Celsius)
- `gen_ai.co2.emissions` - Counter (cumulative gCO2e)

If you're on v2.0+ and still seeing issues, check that NVML is properly initialized:
```python
import pynvml
pynvml.nvmlInit()
print(f"GPU count: {pynvml.nvmlDeviceGetCount()}")
pynvml.nvmlShutdown()
```

---

### Issue: Content capture not working

**Cause**: Content capture is disabled by default for security reasons.

**Solution**:

Enable content capture with environment variable:
```bash
export GENAI_ENABLE_CONTENT_CAPTURE=true
```

Or programmatically:
```python
from genai_otel import instrument
instrument(enable_content_capture=True)
```

**WARNING**: Content capture records full prompts and completions as span events. This may expose sensitive data. Only enable in trusted environments with proper data handling policies.

**What gets captured:**
- Prompt events: `gen_ai.prompt.{index}` with attributes:
  - `gen_ai.prompt.role`
  - `gen_ai.prompt.content`
- Completion events: `gen_ai.completion.{index}` with attributes:
  - `gen_ai.completion.role`
  - `gen_ai.completion.content`

---

### Issue: Histogram buckets not optimized for LLM latencies

**Cause**: Fixed in v2.0+. Earlier versions used default OTel buckets.

**Solution**:

Upgrade to v2.0+. Histogram buckets are now properly configured via OpenTelemetry Views:
- Operation duration uses buckets: `[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]` seconds
- Optimized for LLM response times (10ms to 81s)

You can verify buckets in your metrics backend or via console exporter.

---

### Logging Configuration

**Issue**: You want to adjust the logging level or manage log file output.

**Solution**:

1.  **Configure Log Level**:
    *   Set the `GENAI_OTEL_LOG_LEVEL` environment variable to your desired level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). If not set, it defaults to `INFO`.
    ```bash
    export GENAI_OTEL_LOG_LEVEL=DEBUG
    ```
2.  **Log File Location and Rotation**:
    *   By default, logs are written to `logs/genai_otel.log` within your project directory.
    *   The logging system is configured for rotation, keeping up to 10 log files, each with a maximum size of 10MB. When a log file reaches 10MB, it's rotated, and a new file is created. The oldest log file is removed when the limit of 10 files is reached.
    *   Ensure the `logs` directory has write permissions. If the directory does not exist, it will be created automatically.

---

### Warning: "pynvml package is deprecated"

**Cause**: You have the old `pynvml` package installed instead of `nvidia-ml-py`.

**Solution**:
```bash
