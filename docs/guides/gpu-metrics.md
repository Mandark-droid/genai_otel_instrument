# GPU Metrics and CO2 Tracking

TraceVerde collects real-time GPU metrics for both NVIDIA and AMD GPUs, with optional CO2 emissions and electricity cost tracking.

## Installation

```bash
# NVIDIA GPUs
pip install genai-otel-instrument[gpu]

# AMD GPUs
pip install genai-otel-instrument[amd-gpu]

# Both
pip install genai-otel-instrument[all-gpu]

# CO2 tracking (adds codecarbon)
pip install genai-otel-instrument[co2]
```

## GPU Metrics

### Per-GPU Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `gen_ai.gpu.utilization` | % | GPU compute utilization |
| `gen_ai.gpu.memory.used` | MiB | GPU memory used |
| `gen_ai.gpu.memory.total` | MiB | Total GPU memory |
| `gen_ai.gpu.memory.utilization` | % | Memory controller utilization |
| `gen_ai.gpu.temperature` | Celsius | GPU temperature |
| `gen_ai.gpu.power` | Watts | Power consumption |
| `gen_ai.gpu.power.limit` | Watts | Power limit |
| `gen_ai.gpu.clock.sm` | MHz | SM clock speed |
| `gen_ai.gpu.clock.memory` | MHz | Memory clock speed |
| `gen_ai.gpu.fan.speed` | % | Fan speed |
| `gen_ai.gpu.performance.state` | 0-15 | P-state (0=P0 highest, 15=P15 lowest) |
| `gen_ai.gpu.pcie.tx` | KB/s | PCIe TX throughput |
| `gen_ai.gpu.pcie.rx` | KB/s | PCIe RX throughput |
| `gen_ai.gpu.throttle.thermal` | 0/1 | Thermal throttling active |
| `gen_ai.gpu.throttle.power` | 0/1 | Power throttling active |
| `gen_ai.gpu.throttle.hw_slowdown` | 0/1 | Hardware slowdown active |
| `gen_ai.gpu.ecc.errors.corrected` | count | ECC corrected errors |
| `gen_ai.gpu.ecc.errors.uncorrected` | count | ECC uncorrected errors |

### Aggregate Metrics (Multi-GPU)

| Metric | Unit | Description |
|--------|------|-------------|
| `gen_ai.gpu.aggregate.mean_utilization` | % | Mean utilization across all GPUs |
| `gen_ai.gpu.aggregate.total_memory_used` | GiB | Total memory used across all GPUs |
| `gen_ai.gpu.aggregate.total_power` | W | Total power across all GPUs |
| `gen_ai.gpu.aggregate.max_temperature` | Celsius | Maximum temperature across all GPUs |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_GPU_METRICS` | `true` | Enable GPU metrics collection |
| `GENAI_GPU_COLLECTION_INTERVAL` | `5` | Collection interval in seconds |
| `GENAI_POWER_COST_PER_KWH` | `0.12` | Electricity cost in USD per kWh |

Common electricity rates:
- US average: ~$0.12/kWh
- Europe average: ~$0.20/kWh
- Industrial/datacenter: ~$0.07/kWh

The collector runs in a background daemon thread and gracefully handles systems without GPUs.

## CO2 Emissions Tracking

Track the carbon footprint of your GPU workloads. Two calculation modes are available:

### Codecarbon Mode (Recommended)

Uses [codecarbon](https://github.com/mlco2/codecarbon) for automatic region-specific carbon intensity lookup.

```bash
pip install genai-otel-instrument[co2]
```

```python
import genai_otel

genai_otel.instrument(
    enable_co2_tracking=True,
    co2_country_iso_code="USA",    # 3-letter ISO code
    co2_region="california",       # Optional: state/region
)
```

Or for cloud environments:

```python
genai_otel.instrument(
    enable_co2_tracking=True,
    co2_cloud_provider="aws",
    co2_cloud_region="us-east-1",
)
```

### Manual Mode

Use a fixed carbon intensity value without codecarbon:

```python
genai_otel.instrument(
    enable_co2_tracking=True,
    co2_use_manual=True,
    carbon_intensity=56.0,  # gCO2e/kWh (France - mostly nuclear)
)
```

Reference carbon intensity values (gCO2e/kWh):

| Country/Region | gCO2e/kWh | Notes |
|----------------|-----------|-------|
| France | ~56 | Mostly nuclear |
| Sweden | ~13 | Mostly hydro/wind |
| UK | ~233 | Mix |
| Germany | ~350 | Mix with coal |
| US average | ~420 | Varies by state |
| US (California) | ~210 | Renewables-heavy |
| US (West Virginia) | ~860 | Coal-heavy |
| China | ~555 | Coal-heavy |
| India | ~700 | Coal-heavy |

### CO2 Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_CO2_TRACKING` | `false` | Enable CO2 tracking |
| `GENAI_CARBON_INTENSITY` | `475.0` | Manual carbon intensity (gCO2e/kWh) |
| `GENAI_CO2_USE_MANUAL` | `false` | Force manual mode even with codecarbon |
| `GENAI_CO2_COUNTRY_ISO_CODE` | | ISO 3166-1 alpha-3 code (e.g., `USA`, `GBR`, `DEU`, `IND`) |
| `GENAI_CO2_REGION` | | Region/state (e.g., `california`, `texas`) |
| `GENAI_CO2_CLOUD_PROVIDER` | | Cloud provider: `aws`, `gcp`, `azure` |
| `GENAI_CO2_CLOUD_REGION` | | Cloud region (e.g., `us-east-1`, `europe-west1`) |
| `GENAI_CO2_OFFLINE_MODE` | `true` | No external API calls (uses local data) |
| `GENAI_CO2_TRACKING_MODE` | `machine` | `machine` (all processes) or `process` (current only) |
| `GENAI_CODECARBON_LOG_LEVEL` | `error` | Codecarbon verbosity |

### CO2 Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `gen_ai.co2.emissions` | kgCO2e | Cumulative CO2 emissions |
| `gen_ai.power.consumption` | kWh | Cumulative power consumption |
| `gen_ai.power.cost` | USD | Cumulative electricity cost (based on `GENAI_POWER_COST_PER_KWH`) |

### Example: Full GPU + CO2 Setup

```bash
export GENAI_ENABLE_GPU_METRICS=true
export GENAI_GPU_COLLECTION_INTERVAL=10
export GENAI_ENABLE_CO2_TRACKING=true
export GENAI_CO2_COUNTRY_ISO_CODE=USA
export GENAI_CO2_REGION=california
export GENAI_POWER_COST_PER_KWH=0.18
```

```python
import genai_otel

genai_otel.instrument(
    service_name="gpu-workload",
    enable_gpu_metrics=True,
    gpu_collection_interval=10,
    enable_co2_tracking=True,
    co2_country_iso_code="USA",
    co2_region="california",
    power_cost_per_kwh=0.18,
)

# Run your GPU workload
import transformers
pipe = transformers.pipeline("text-generation", model="gpt2", device=0)
result = pipe("Hello world", max_length=50)

# GPU metrics, CO2 emissions, and electricity costs are all tracked automatically
```

## Grafana Dashboard

Import the pre-built GPU dashboard from [dashboards/grafana/gpu-metrics.json](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/dashboards/grafana/gpu-metrics.json). Includes gauges for utilization, temperature, and power, plus time-series for historical trends.

## Ollama Server Metrics

When using Ollama, TraceVerde automatically polls the server for VRAM usage:

| Variable | Default | Description |
|----------|---------|-------------|
| `GENAI_ENABLE_OLLAMA_SERVER_METRICS` | `true` | Poll Ollama `/api/ps` endpoint |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GENAI_OLLAMA_METRICS_INTERVAL` | `5.0` | Polling interval in seconds |
| `GENAI_OLLAMA_MAX_VRAM_GB` | auto-detected | Override GPU VRAM size (GB) |

See [Server Metrics reference](../reference/server-metrics.md) for details.
