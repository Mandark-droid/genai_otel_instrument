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
