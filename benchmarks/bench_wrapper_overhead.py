"""Micro-benchmark for BaseInstrumentor.create_span_wrapper per-call overhead.

Measures the cost the library adds around a no-op "LLM call" that returns a fake
response carrying a usage block, under the realistic configuration used for
BFSI/audit deployments (content capture ON, cost tracking ON). It isolates the
library's own work by using an SDK TracerProvider with NO span processor and an
SDK MeterProvider with an in-memory reader, so numbers reflect the wrapper +
in-process OTel SDK aggregation, with no exporter/network cost.

Two model cases are reported because model-name -> pricing resolution is a
material cost:
  - "priced":   a model present in llm_pricing.json (fast dict hit).
  - "unpriced": an internal/custom model not in the table (must fall through the
                resolver; memoized after the first lookup).

Usage:
    python benchmarks/bench_wrapper_overhead.py            # single-thread per-call
    python benchmarks/bench_wrapper_overhead.py threads    # thread-scaling sweep

Notes:
- Run from a checkout with the package importable (`pip install -e .`).
- The library defaults metric exemplar sampling OFF
  (OTEL_METRICS_EXEMPLAR_FILTER=always_off); this script sets the same default so
  it measures the real hot path. Set the env var to `trace_based` to see the
  exemplar-on cost.
- Micro-benchmarks are sensitive to machine load; treat the *best* of several
  runs as the floor and expect run-to-run variance under load.
"""

import os
import sys
import time

os.environ.setdefault("GENAI_ENABLE_GPU_METRICS", "false")
os.environ.setdefault("GENAI_ENABLE_MCP_INSTRUMENTATION", "false")
os.environ.setdefault("GENAI_ENABLE_CONTENT_CAPTURE", "true")
os.environ.setdefault("GENAI_ENABLE_COST_TRACKING", "true")
# Match the library default set by setup_auto_instrumentation().
os.environ.setdefault("OTEL_METRICS_EXEMPLAR_FILTER", "always_off")
# Ensure a stray shell restriction does not disable instrumentors.
os.environ.pop("GENAI_ENABLED_INSTRUMENTORS", None)

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

trace.set_tracer_provider(TracerProvider())  # no span processor
metrics.set_meter_provider(MeterProvider(metric_readers=[InMemoryMetricReader()]))

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor

PRICED_MODEL = "gpt-4o"
UNPRICED_MODEL = "internal-fraud-model-v3"
ITERS = 20_000


class _Resp:
    class _Msg:
        content = "The account balance is 12345.67 as of today."

    class _Choice:
        def __init__(self):
            self.message = _Resp._Msg()
            self.finish_reason = "stop"

    class _Usage:
        prompt_tokens = 42
        completion_tokens = 18
        total_tokens = 60

    def __init__(self):
        self.choices = [_Resp._Choice()]
        self.usage = _Resp._Usage()


class _Bench(BaseInstrumentor):
    def instrument(self, config=None):
        self._instrumented = True

    def _extract_usage(self, result):
        u = result.usage
        return {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }

    def _extract_response_attributes(self, result):
        return {"gen_ai.response": result.choices[0].message.content}


def _make_wrapped(model):
    inst = _Bench()
    inst._instrumented = True
    inst.config = OTelConfig(service_name="bench")
    inst.tracer = trace.get_tracer("bench")

    def extract(instance, args, kwargs):
        return {"gen_ai.system": "bench", "gen_ai.request.model": model}

    resp = _Resp()

    def call(*a, **k):
        return resp

    return inst.create_span_wrapper("bench.chat", extract_attributes=extract)(call)


def _time_us_per_call(fn, kwargs, iters):
    for _ in range(2000):  # warmup
        fn(**kwargs)
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(**kwargs)
        reps.append((time.perf_counter() - t0) / iters)
    return min(reps) * 1e6


def single_thread():
    msgs = [{"role": "user", "content": "what is my balance?"}]
    priced = _make_wrapped(PRICED_MODEL)
    unpriced = _make_wrapped(UNPRICED_MODEL)

    us_priced = _time_us_per_call(priced, dict(model=PRICED_MODEL, messages=msgs), ITERS)
    us_unpriced = _time_us_per_call(unpriced, dict(model=UNPRICED_MODEL, messages=msgs), ITERS)

    r = _Resp()

    def raw(*a, **k):
        return r

    t0 = time.perf_counter()
    for _ in range(ITERS):
        raw(model=PRICED_MODEL, messages=msgs)
    us_raw = (time.perf_counter() - t0) / ITERS * 1e6

    print(f"content_capture=ON  priced({PRICED_MODEL})      : {us_priced:8.2f} us/call")
    print(f"content_capture=ON  unpriced(internal)  : {us_unpriced:8.2f} us/call")
    print(f"unwrapped baseline                      : {us_raw:8.4f} us/call")


def threads():
    import threading

    msgs = [{"role": "user", "content": "what is my balance?"}]
    priced = _make_wrapped(PRICED_MODEL)
    for _ in range(2000):
        priced(model=PRICED_MODEL, messages=msgs)

    def run(n):
        for _ in range(n):
            priced(model=PRICED_MODEL, messages=msgs)

    for nthreads in (1, 2, 4, 8):
        per = 40_000
        ths = [threading.Thread(target=run, args=(per,)) for _ in range(nthreads)]
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        print(f"threads={nthreads}: {per * nthreads / wall:10.0f} calls/s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "threads":
        threads()
    else:
        single_thread()
