"""Regression tests for hot-path hardening in BaseInstrumentor / CostCalculator.

Covers:
- The wrapped (business) call must execute exactly once, even when the span is
  sampled out (NonRecordingSpan) or when the call raises. Previously the outer
  fallback re-invoked the wrapped function, double-executing LLM calls (double
  API spend + duplicated side effects).
- CostCalculator memoization returns results identical to a fresh calculation
  and caches both hits and misses.
- Metric-verbosity config defaults are lean (aggregated histograms / granular
  cost counters / finish counters are opt-in; span attributes stay full).
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, TraceIdRatioBased

from genai_otel.config import OTelConfig
from genai_otel.cost_calculator import CostCalculator
from genai_otel.instrumentors.base import BaseInstrumentor


class _Inst(BaseInstrumentor):
    def instrument(self, config=None):
        self._instrumented = True

    def _extract_usage(self, result):
        return None


def _make_wrapped(fn, sampler):
    provider = TracerProvider(sampler=sampler)
    inst = _Inst()
    inst._instrumented = True
    inst.tracer = provider.get_tracer("test")
    return inst.create_span_wrapper("test.chat")(fn)


def test_sampled_out_span_does_not_double_execute():
    """A span dropped by sampling must not cause the wrapped call to run twice."""
    calls = []

    def fn(*args, **kwargs):
        calls.append(1)
        return {"ok": True}

    wrapped = _make_wrapped(fn, TraceIdRatioBased(0.0))
    result = wrapped(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert result == {"ok": True}
    assert len(calls) == 1, "sampled-out span double-executed the wrapped call"


def test_errored_call_does_not_double_execute():
    """An errored call must run once and propagate the original exception."""
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise ValueError("api failure")

    wrapped = _make_wrapped(boom, ALWAYS_ON)
    with pytest.raises(ValueError, match="api failure"):
        wrapped(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert len(calls) == 1, "errored call double-executed the wrapped function"


def test_recording_call_executes_once():
    """Baseline: a normally-recorded successful call runs exactly once."""
    calls = []

    def fn(*args, **kwargs):
        calls.append(1)
        return {"ok": True}

    wrapped = _make_wrapped(fn, ALWAYS_ON)
    result = wrapped(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert result == {"ok": True}
    assert len(calls) == 1


def test_cost_calculator_memoizes_and_matches():
    """Memoized pricing resolution matches a fresh calc and caches hits + misses."""
    cc = CostCalculator()
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}

    r1 = cc.calculate_granular_cost("gpt-4o", usage, "chat")
    r2 = cc.calculate_granular_cost("gpt-4o", usage, "chat")
    assert r1 == r2
    assert r1["total"] > 0
    assert "gpt-4o" in cc._chat_pricing_cache  # hit memoized

    unknown = "zzz-internal-model-qqq"
    r3 = cc.calculate_granular_cost(unknown, usage, "chat")
    assert r3["total"] == 0.0
    assert unknown in cc._chat_pricing_cache  # miss memoized too


def test_normalize_model_name_uses_cache():
    """_normalize_model_name is memoized including negative results."""
    cc = CostCalculator()
    assert cc._normalize_model_name("gpt-4o", "chat") == cc._normalize_model_name("gpt-4o", "chat")
    # Negative result is cached (not re-scanned every call).
    cc._normalize_model_name("no-such-model-xyzzy", "chat")
    assert ("no-such-model-xyzzy", "chat") in cc._norm_cache


class _BlockResult:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


class _FakeDetector:
    def __init__(self, **flags):
        self._flags = flags

    def detect(self, text, language="en"):
        return _BlockResult(**self._flags)


def _reset_block_state():
    BaseInstrumentor._pii_detector = None
    BaseInstrumentor._blocking_policies = frozenset()
    BaseInstrumentor._blocking_active = False


def test_block_mode_prevents_the_call():
    """A block-mode policy triggering on the prompt raises PolicyViolationError
    and the wrapped LLM call is never executed."""
    from genai_otel.exceptions import PolicyViolationError

    calls = []

    def fn(*args, **kwargs):
        calls.append(1)
        return {"ok": True}

    BaseInstrumentor._pii_detector = _FakeDetector(has_pii=True)
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True
    try:
        wrapped = _make_wrapped(fn, ALWAYS_ON)
        with pytest.raises(PolicyViolationError):
            wrapped(model="gpt-4o", messages=[{"role": "user", "content": "SSN 123-45-6789"}])
        assert calls == [], "blocked request still executed the LLM call"
    finally:
        _reset_block_state()


def test_block_mode_allows_clean_prompt():
    """When the blocking detector does not trigger, the call proceeds normally."""
    calls = []

    def fn(*args, **kwargs):
        calls.append(1)
        return {"ok": True}

    BaseInstrumentor._pii_detector = _FakeDetector(has_pii=False)
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True
    try:
        wrapped = _make_wrapped(fn, ALWAYS_ON)
        result = wrapped(model="gpt-4o", messages=[{"role": "user", "content": "hello"}])
        assert result == {"ok": True}
        assert calls == [1]
    finally:
        _reset_block_state()


def test_no_blocking_configured_is_zero_cost_path():
    """With no blocking policies, the pre-call hook is skipped entirely."""
    _reset_block_state()
    calls = []

    def fn(*args, **kwargs):
        calls.append(1)
        return {"ok": True}

    wrapped = _make_wrapped(fn, ALWAYS_ON)
    assert wrapped(model="gpt-4o", messages=[{"role": "user", "content": "hi"}]) == {"ok": True}
    assert calls == [1]


def test_metrics_profile_defaults_are_lean(monkeypatch):
    """Aggregated metric fan-out is opt-in by default; profile defaults to standard."""
    for var in (
        "GENAI_METRICS_PROFILE",
        "GENAI_RECORD_TOKEN_HISTOGRAMS",
        "GENAI_RECORD_GRANULAR_COST_METRICS",
        "GENAI_RECORD_FINISH_METRICS",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = OTelConfig()
    assert cfg.metrics_profile == "standard"
    assert cfg.record_token_histograms is False
    assert cfg.record_granular_cost_metrics is False
    assert cfg.record_finish_metrics is False
    assert cfg.enable_concurrency_metrics is True
