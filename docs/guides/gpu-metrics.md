# GPU Metrics

TraceVerde collects real-time GPU metrics for both NVIDIA and AMD GPUs.

## Installation

```bash
# NVIDIA GPUs
pip install genai-otel-instrument[gpu]

# AMD GPUs
pip install genai-otel-instrument[amd-gpu]

# Both
pip install genai-otel-instrument[all-gpu]
```

## Collected Metrics

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
| `gen_ai.gpu.performance.state` | 0-15 | P-state (0=highest) |
| `gen_ai.gpu.pcie.tx` | KB/s | PCIe TX throughput |
| `gen_ai.gpu.pcie.rx` | KB/s | PCIe RX throughput |
| `gen_ai.gpu.throttle.thermal` | 0/1 | Thermal throttling |
| `gen_ai.gpu.throttle.power` | 0/1 | Power throttling |
| `gen_ai.gpu.ecc.errors.corrected` | count | ECC corrected errors |
| `gen_ai.gpu.ecc.errors.uncorrected` | count | ECC uncorrected errors |

## Aggregate Metrics (Multi-GPU)

| Metric | Description |
|--------|-------------|
| `gen_ai.gpu.aggregate.mean_utilization` | Mean utilization across all GPUs (%) |
| `gen_ai.gpu.aggregate.total_memory_used` | Total memory used across all GPUs (GiB) |
| `gen_ai.gpu.aggregate.total_power` | Total power across all GPUs (W) |
| `gen_ai.gpu.aggregate.max_temperature` | Maximum temperature across all GPUs (C) |

## Configuration

GPU metrics are enabled by default. To disable:

```bash
export GENAI_ENABLE_GPU_METRICS=false
```

The collector runs in a background daemon thread with a configurable collection interval. It gracefully handles systems without GPUs.

## Dashboard

Import the Grafana GPU dashboard from `dashboards/grafana/gpu-metrics.json` for visualization.
