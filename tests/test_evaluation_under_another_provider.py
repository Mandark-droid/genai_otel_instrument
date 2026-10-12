"""Detectors run inline when the application owns the tracer provider.

OpenTelemetry keeps the first tracer provider that is set. An application that
installs its own before ``genai_otel.instrument()`` keeps it, and flushes and
shuts it down itself. Spans whose detectors run in the background are exported
by a worker thread some time after the call; that provider does not wait for
them, and they were lost. So in that case the detectors run inline, where a span
has ended by the time its call returns.
"""

import logging
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

import genai_otel.auto_instrument as auto_instrument
from genai_otel.auto_instrument import setup_auto_instrumentation
from genai_otel.config import OTelConfig
from genai_otel.instrumentors import base


@pytest.fixture
def instrument(monkeypatch, caplog):
    """Run the set-up against stand-ins; ``ours`` says whose provider is in use."""
    monkeypatch.setattr(auto_instrument, "_INSTRUMENTATION_INITIALIZED", False)
    monkeypatch.setattr(auto_instrument, "_active_tracer_provider", None)
    monkeypatch.setattr(auto_instrument, "_active_meter_provider", None)

    def run(ours, **settings):
        detectors = {
            "enable_pii_detection": False,
            "enable_toxicity_detection": False,
            "enable_bias_detection": False,
            "enable_prompt_injection_detection": False,
            "enable_restricted_topics": False,
            "enable_hallucination_detection": False,
        }
        detectors.update(settings)
        config = OTelConfig(
            service_name="t",
            endpoint="",
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
            enabled_instrumentors=[],
            **detectors,
        )
        with ExitStack() as stack:
            provider_class = stack.enter_context(patch.object(auto_instrument, "TracerProvider"))
            trace = stack.enter_context(patch.object(auto_instrument, "trace"))
            trace.get_tracer_provider.return_value = (
                provider_class.return_value if ours else MagicMock(name="the application's own")
            )
            stack.enter_context(patch.object(auto_instrument, "metrics"))
            stack.enter_context(patch.object(auto_instrument, "Resource"))
            stack.enter_context(patch("genai_otel.auto_instrument._check_openinference"))
            stack.enter_context(patch.object(base, "register_evaluation_exit_hook"))
            # A stand-in: the real one registers detectors for the whole process.
            stack.enter_context(
                patch("genai_otel.evaluation.span_processor.EvaluationSpanProcessor")
            )
            with caplog.at_level(logging.WARNING, logger="genai_otel.auto_instrument"):
                setup_auto_instrumentation(config)
            monkeypatch.setattr(auto_instrument, "_INSTRUMENTATION_INITIALIZED", False)
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING and "inline" in record.getMessage()
        ]
        return config, warnings

    return run


def test_under_the_application_s_provider_detectors_run_inline(instrument):
    config, warnings = instrument(ours=False, enable_bias_detection=True)

    assert config.evaluation_mode == "inline"
    assert len(warnings) == 1
    # It says why, and how to get the background mode.
    assert "already installed" in warnings[0]
    assert "does not wait" in warnings[0]
    assert "call it before anything else sets one" in warnings[0]


def test_under_this_library_s_provider_detectors_run_in_the_background(instrument):
    config, warnings = instrument(ours=True, enable_bias_detection=True)

    assert config.evaluation_mode == "background"
    assert warnings == []


def test_nothing_is_said_when_no_detector_is_enabled(instrument):
    config, warnings = instrument(ours=False)

    assert config.evaluation_mode == "background"
    assert warnings == []


def test_nothing_is_said_when_inline_was_asked_for_anyway(instrument):
    config, warnings = instrument(ours=False, enable_bias_detection=True, evaluation_mode="inline")

    assert config.evaluation_mode == "inline"
    assert warnings == []


def test_an_instrumentor_given_that_configuration_does_not_use_the_queue(instrument, monkeypatch):
    """The decision is made once, at set-up; a call only reads it."""
    config, _ = instrument(ours=False, enable_bias_detection=True)
    queue = base._EvaluationQueue()
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", queue)
    monkeypatch.setattr(base.BaseInstrumentor, "_bias_detector", MagicMock())

    class Instrumentor(base.BaseInstrumentor):
        def instrument(self, config):
            self.config = config

        def _extract_usage(self, result):
            return None

    instrumentor = Instrumentor()
    instrumentor.config = config
    span = MagicMock()
    span.is_recording.return_value = True
    span.attributes = {}

    assert instrumentor._evaluate_or_defer(span, (), {"messages": []}, None) is False
    assert queue.submitted == 0
