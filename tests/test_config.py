import os

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from genai_otel.config import OTelConfig, setup_tracing


@pytest.fixture(autouse=True)
def reset_tracer():
    # Reset the tracer provider before each test
    trace.set_tracer_provider(TracerProvider())
    yield
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
