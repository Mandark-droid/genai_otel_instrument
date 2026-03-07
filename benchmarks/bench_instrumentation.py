"""Performance benchmarks for genai-otel-instrument (G-P2).

Run with: python benchmarks/bench_instrumentation.py

These benchmarks measure the overhead of instrumentation on mock LLM calls.
"""

import statistics
import subprocess  # nosec B404
import sys
import time
from unittest.mock import MagicMock


def _time_fn(fn, iterations=100):
    """Time a function over N iterations, return (mean_ms, p50_ms, p99_ms)."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    mean = statistics.mean(times)
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    return mean, p50, p99


def bench_import_time():
    """Measure bare import time."""
    code = (
        "import time; s=time.perf_counter(); import genai_otel; "
        "print(f'{(time.perf_counter()-s)*1000:.1f}')"
    )
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    ms = float(result.stdout.strip())
    print(f"  import genai_otel: {ms:.1f}ms")
    return ms


def bench_span_wrapper_overhead():
    """Measure overhead of a span wrapper around a no-op function.

    Uses a real TracerProvider set globally so spans are properly created.
    """
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({"service.name": "bench"}))
    trace_api.set_tracer_provider(provider)
    tracer = provider.get_tracer("bench")

    from genai_otel.instrumentors.base import BaseInstrumentor

    class BenchInstrumentor(BaseInstrumentor):
        def instrument(self, config):
            pass

        def _extract_usage(self, result):
            return None

    instrumentor = BenchInstrumentor()
    instrumentor._tracer = tracer
    instrumentor._instrumented = True
    # Set up metrics with mocks to avoid meter provider issues
    instrumentor.request_counter = MagicMock()
    instrumentor.error_counter = MagicMock()
    instrumentor.latency_histogram = MagicMock()
    instrumentor.token_counter = MagicMock()
    instrumentor.cost_counter = MagicMock()

    wrapper = instrumentor.create_span_wrapper("test_op")

    # Baseline: direct function call
    def noop(*args, **kwargs):
        return MagicMock(choices=[MagicMock()])

    mean_base, p50_base, p99_base = _time_fn(noop, 1000)

    # Wrapped call
    wrapped = wrapper(noop)

    def call_wrapped():
        return wrapped(model="test")

    mean_wrap, p50_wrap, p99_wrap = _time_fn(call_wrapped, 1000)

    overhead = mean_wrap - mean_base
    print(f"  Direct call:  mean={mean_base:.3f}ms  p50={p50_base:.3f}ms  p99={p99_base:.3f}ms")
    print(f"  Wrapped call: mean={mean_wrap:.3f}ms  p50={p50_wrap:.3f}ms  p99={p99_wrap:.3f}ms")
    print(f"  Overhead:     mean={overhead:.3f}ms")

    provider.shutdown()
    trace_api.set_tracer_provider(trace_api.NoOpTracerProvider())
    return overhead


def bench_cost_calculation():
    """Measure cost calculation overhead."""
    from genai_otel.cost_calculator import CostCalculator

    calc = CostCalculator()

    def calc_cost():
        calc.calculate_cost(
            "gpt-4o",
            {"prompt_tokens": 1000, "completion_tokens": 500},
            "chat",
        )

    mean, p50, p99 = _time_fn(calc_cost, 1000)
    print(f"  Cost calc:    mean={mean:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms")
    return mean


if __name__ == "__main__":
    print("=" * 60)
    print("genai-otel-instrument Performance Benchmarks")
    print("=" * 60)

    print("\n1. Import Time:")
    bench_import_time()

    print("\n2. Span Wrapper Overhead:")
    bench_span_wrapper_overhead()

    print("\n3. Cost Calculation:")
    bench_cost_calculation()

    print("\n" + "=" * 60)
    print("Done.")
