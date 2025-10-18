import os

import pytest
from opentelemetry import metrics, trace
from opentelemetry.metrics import NoOpMeterProvider, set_meter_provider
from opentelemetry.trace import NoOpTracerProvider, set_tracer_provider

import genai_otel
from genai_otel.metrics import get_meter, get_meter_provider


@pytest.fixture(autouse=True)
def reset_otel():
    """Reset OpenTelemetry for each test."""
    env_vars_to_clear = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SERVICE_NAME",
    ]
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]

    # Reset the global providers to None to allow new providers to be set
    # This is necessary because OTel doesn't allow overriding existing providers
    metrics._METER_PROVIDER = None  # pylint: disable=protected-access
    trace._TRACER_PROVIDER = None  # pylint: disable=protected-access

    yield

    # Clean up after test - set back to NoOp
    set_meter_provider(NoOpMeterProvider())
    set_tracer_provider(NoOpTracerProvider())


def test_basic_instrumentation():
    """Basic smoke test for instrumentation."""
    # This should not raise any exceptions
    genai_otel.instrument(service_name="test-service")

    # We should have non-NoOp providers
    meter_provider = metrics.get_meter_provider()
    tracer_provider = trace.get_tracer_provider()

    assert meter_provider is not None
    assert tracer_provider is not None
    assert not isinstance(meter_provider, NoOpMeterProvider)
    assert not isinstance(tracer_provider, NoOpTracerProvider)


def test_metrics_functions():
    """Test that metrics module functions work."""
    genai_otel.instrument(service_name="test-service")

    # Test get_meter_provider
    provider = get_meter_provider()
    assert provider is not None
    assert provider is metrics.get_meter_provider()

    # Test get_meter
    meter = get_meter()
    assert meter is not None

    # Test that meter can create instruments
    counter = meter.create_counter("test_counter")
    assert counter is not None


def test_instrumentation_with_environment():
    """Test instrumentation with environment variables."""
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    os.environ["OTEL_SERVICE_NAME"] = "env-service"

    genai_otel.instrument()

    # Should work without errors
    provider = get_meter_provider()
    assert provider is not None
    assert not isinstance(provider, NoOpMeterProvider)


def test_grpc_protocol_import():
    """Test that grpc protocol import path is covered."""
    # Set environment variable before importing the module
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"

    # Force reimport to trigger the grpc path
    import importlib

    import genai_otel.metrics

    importlib.reload(genai_otel.metrics)

    # Should import successfully
    from genai_otel.metrics import get_meter

    meter = get_meter()
    assert meter is not None

    # Clean up
    del os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"]
