# Benchmarks

Performance benchmarks and results for `genai-otel-instrument`. The dominant cost
this library adds to a host application is the **per-LLM-call span-wrapper
overhead**, so that is what is measured here.

## Tooling

| Script | Measures |
|---|---|
| [`bench_wrapper_overhead.py`](bench_wrapper_overhead.py) | Per-call overhead of `BaseInstrumentor.create_span_wrapper` (single-thread + thread-scaling), for a priced and an unpriced model, with content capture ON. |

### Running

```bash
pip install -e .          # package must be importable
python benchmarks/bench_wrapper_overhead.py           # single-thread per-call
python benchmarks/bench_wrapper_overhead.py threads   # thread-scaling sweep
```

### Methodology

- Isolates the library's own work: an SDK `TracerProvider` with **no span
  processor** and an SDK `MeterProvider` with an **in-memory reader**, so numbers
  reflect the wrapper + in-process OTel SDK aggregation, with no exporter/network
  cost.
- Wraps a no-op function returning a fake response with a usage block. Warms up,
  then times 5 reps of 20,000 calls and reports the **best** (min) rep. Micro-
  benchmarks are load-sensitive: treat the best of several runs as the floor and
  expect variance under machine load.
- Realistic BFSI/audit config: `GENAI_ENABLE_CONTENT_CAPTURE=true`,
  `GENAI_ENABLE_COST_TRACKING=true`.
- Two model cases, because model-name -> pricing resolution is a material cost:
  - **priced** = a model in `llm_pricing.json` (fast dict hit);
  - **unpriced** = an internal/custom model not in the table (resolver fall-
    through, memoized after first lookup).

## Results

Best-of-N, content capture ON. Reference environment: Windows 11, Python 3.11,
`opentelemetry-sdk` 1.42.x.

### Per-call overhead

| Case | Before | After | Reduction |
|---|---:|---:|---:|
| Priced model (`gpt-4o`) | ~61 us | **~42 us** (best ~38) | ~30% |
| Unpriced / internal model | ~182 us | **~31 us** | ~83% |
| Unwrapped baseline (reference) | ~0.04-0.2 us | unchanged | - |

Both cases are comfortably under the 50 us/call target used as the ship gate for
high-throughput deployments.

### Thread scaling (single process)

| Threads | Before | After |
|---:|---:|---:|
| 1 | ~15.5k calls/s | **~23k calls/s** |
| 8 | ~10k calls/s | ~13k calls/s |

Single-thread throughput improved substantially. The 1 -> 8 thread slope stays
**negative** because instrumentation is CPU-bound Python and therefore bound by
the CPython GIL; this is a language characteristic, not a library defect. Scale
out with **multiple processes** (~23k calls/s per process x N workers) rather
than threads.

## What produced the improvement

- **Cost lookup**: replaced an O(n) scan of ~850 model keys plus a per-call
  `sorted()` with precomputed lowercase exact-match + length-sorted substring
  indices, memoized per model name (including misses). This is the ~182 -> ~31 us
  win for unpriced/internal models.
- **Shared `CostCalculator`**: one process-wide instance parses `llm_pricing.json`
  once instead of ~29 times.
- **Metric fan-out gated**: full per-request detail always goes on span
  attributes (audit unaffected); the redundant aggregated-metric instruments
  (token histograms, granular cost counters, finish counters) are opt-in via
  `GENAI_METRICS_PROFILE=full`.
- **Exemplar sampling defaults OFF** (`OTEL_METRICS_EXEMPLAR_FILTER=always_off`,
  overridable) - a per-measurement RNG draw that was ~14% of hot-path cost.
- Removed per-call debug f-strings that were formatted regardless of log level.

Set `OTEL_METRICS_EXEMPLAR_FILTER=trace_based` or `GENAI_METRICS_PROFILE=full`
when running the benchmark to see the cost of the fuller telemetry configuration.
