import logging
import os
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from genai_otel.config import OTelConfig, setup_tracing


@pytest.fixture(autouse=True)
def reset_tracer(monkeypatch):
    mock_tracer_provider = MagicMock(spec=TracerProvider)
    mock_tracer = MagicMock(spec=trace.Tracer)
    mock_tracer_provider.get_tracer.return_value = mock_tracer

    monkeypatch.setattr(trace, "set_tracer_provider", mock_tracer_provider)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: mock_tracer_provider)
    yield
    # Ensure the global tracer provider is reset to NoOp after tests
    trace.set_tracer_provider(trace.NoOpTracerProvider())


def test_setup_tracing_with_otlp():
    config = OTelConfig(service_name="test-service", endpoint="http://localhost:4317")
    tracer = setup_tracing(config, "test-tracer")
    assert tracer is not None
    assert isinstance(tracer, trace.Tracer)
    # Add more assertions to check if the OTLP exporter is configured correctly
    # For example, check if the span processor is an instance of BatchSpanProcessor
    # or SimpleSpanProcessor, depending on the disable_batch parameter.


def test_setup_tracing_with_console():
    # Test with no endpoint, should use console exporter
    config = OTelConfig(service_name="test-service", endpoint="")
    tracer = setup_tracing(config, "test-tracer")
    assert tracer is not None
    assert isinstance(tracer, trace.Tracer)
    # Add more assertions to check if the console exporter is configured correctly


def test_enabled_instrumentors_from_env(monkeypatch):
    """Test that enabled_instrumentors can be loaded from environment variable."""
    monkeypatch.setenv("GENAI_ENABLED_INSTRUMENTORS", "openai, anthropic, cohere")
    config = OTelConfig()
    assert config.enabled_instrumentors == ["openai", "anthropic", "cohere"]


def test_grpc_exporter_import(monkeypatch):
    """Test that grpc exporter is imported when OTEL_EXPORTER_OTLP_PROTOCOL is grpc."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

    # Force reimport to trigger the grpc path
    import importlib

    import genai_otel.config

    importlib.reload(genai_otel.config)

    # Should import successfully
    from genai_otel.config import setup_tracing

    config = OTelConfig(service_name="test-service", endpoint="http://localhost:4317")
    tracer = setup_tracing(config, "test-tracer")
    assert tracer is not None


def test_setup_tracing_exception_handling():
    """Test that setup_tracing handles exceptions gracefully."""
    config = OTelConfig(service_name="test-service", endpoint="http://localhost:4317")

    # Save original function
    original_set_tracer_provider = trace.set_tracer_provider

    try:
        # Mock trace.set_tracer_provider to raise an exception
        def mock_set_tracer_provider(*args, **kwargs):
            raise RuntimeError("Failed to set tracer provider")

        trace.set_tracer_provider = mock_set_tracer_provider

        # Should return None instead of raising
        tracer = setup_tracing(config, "test-tracer")
        assert tracer is None
    finally:
        # Restore original function
        trace.set_tracer_provider = original_set_tracer_provider


def test_sampling_rate_default():
    """Test that sampling_rate defaults to 1.0."""
    config = OTelConfig()
    assert config.sampling_rate == 1.0


def test_sampling_rate_from_env(monkeypatch):
    """Test that sampling_rate loads from GENAI_SAMPLING_RATE env var."""
    monkeypatch.setenv("GENAI_SAMPLING_RATE", "0.25")
    config = OTelConfig()
    assert config.sampling_rate == 0.25


def test_sampling_rate_kwarg():
    """Test that sampling_rate can be set via kwargs."""
    config = OTelConfig(sampling_rate=0.5)
    assert config.sampling_rate == 0.5


# ---------------------------------------------------------------------------
# OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
#
# The cross-library spelling for the content-capture switch, defined by
# opentelemetry-util-genai and used by the reference scenarios in
# open-telemetry/semantic-conventions-genai. An application migrating from
# another GenAI instrumentation sets this one; before it was honoured, such an
# application got silence and no indication why.
# ---------------------------------------------------------------------------


class TestStandardContentCaptureEnvVar:
    _STD = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    _OURS = "GENAI_ENABLE_CONTENT_CAPTURE"

    @pytest.fixture(autouse=True)
    def _clear(self, monkeypatch):
        monkeypatch.delenv(self._STD, raising=False)
        monkeypatch.delenv(self._OURS, raising=False)

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("SPAN_ONLY", True),
            ("SPAN_AND_EVENT", True),
            ("NO_CONTENT", False),
            # This library has one boolean capture switch, not separate span and
            # event sinks, so event-only capture means no capture here rather
            # than writing content somewhere the operator did not ask for.
            ("EVENT_ONLY", False),
        ],
    )
    def test_recognised_modes(self, monkeypatch, mode, expected):
        monkeypatch.setenv(self._STD, mode)
        assert OTelConfig().enable_content_capture is expected

    def test_mode_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv(self._STD, "span_and_event")
        assert OTelConfig().enable_content_capture is True

    def test_standard_var_overrides_ours(self, monkeypatch):
        monkeypatch.setenv(self._OURS, "true")
        monkeypatch.setenv(self._STD, "NO_CONTENT")
        assert OTelConfig().enable_content_capture is False

    def test_falls_back_to_our_var_when_unset(self, monkeypatch):
        monkeypatch.setenv(self._OURS, "true")
        assert OTelConfig().enable_content_capture is True

    def test_invalid_value_does_not_capture(self, monkeypatch, caplog):
        """An unrecognised value warns and fails closed.

        Reading through to our own variable here would capture content for an
        operator whose typo was an attempt to turn capture off.
        """
        monkeypatch.setenv(self._OURS, "true")
        monkeypatch.setenv(self._STD, "SPAN")
        with caplog.at_level(logging.WARNING):
            assert OTelConfig().enable_content_capture is False
        assert "not a valid option" in caplog.text
