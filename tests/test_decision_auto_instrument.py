"""instrument() sets up the decision HTTP matcher and points the manual helper at its config."""

from unittest.mock import MagicMock, patch

import pytest

from genai_otel import decision as manual
from genai_otel.auto_instrument import setup_auto_instrumentation
from genai_otel.config import OTelConfig

_PATCHES = (
    "genai_otel.auto_instrument.OTLPMetricExporter",
    "genai_otel.auto_instrument.OTLPSpanExporter",
    "genai_otel.auto_instrument.PeriodicExportingMetricReader",
    "genai_otel.auto_instrument.BatchSpanProcessor",
    "genai_otel.auto_instrument.MeterProvider",
    "genai_otel.auto_instrument.TracerProvider",
    "genai_otel.auto_instrument.Resource",
    "genai_otel.auto_instrument.trace",
    "genai_otel.auto_instrument.metrics",
)


def _setup(config, matcher):
    patchers = [patch(target) for target in _PATCHES]
    for patcher in patchers:
        patcher.start()
    try:
        with patch("genai_otel.instrumentors.decision_http.DecisionHTTPInstrumentor", matcher):
            setup_auto_instrumentation(config)
    finally:
        for patcher in patchers:
            patcher.stop()


def _config(**kwargs):
    return OTelConfig(
        service_name="t",
        endpoint=None,
        enabled_instrumentors=[],
        enable_gpu_metrics=False,
        enable_mcp_instrumentation=False,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    import genai_otel.auto_instrument as auto_instrument_module

    monkeypatch.setattr(auto_instrument_module, "_INSTRUMENTATION_INITIALIZED", False)
    monkeypatch.setattr(manual, "_INSTRUMENTOR", None)
    monkeypatch.setattr(manual, "_CONFIG", None)


def test_matcher_on_by_default_and_helper_configured():
    matcher = MagicMock()
    config = _config()
    _setup(config, matcher)
    matcher.return_value.instrument.assert_called_once_with(config)
    assert manual._CONFIG is config


def test_matcher_off():
    matcher = MagicMock()
    _setup(_config(decision_http_matcher=False), matcher)
    matcher.assert_not_called()


def test_a_failing_matcher_does_not_stop_instrumentation():
    matcher = MagicMock()
    matcher.return_value.instrument.side_effect = RuntimeError("boom")
    _setup(_config(), matcher)


def test_a_failing_matcher_raises_under_fail_on_error():
    matcher = MagicMock()
    matcher.return_value.instrument.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        _setup(_config(fail_on_error=True), matcher)
