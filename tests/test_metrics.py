import os
from unittest.mock import DEFAULT, patch

import pytest
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

from genai_otel.metrics import setup_meter


# Mock the global meter provider to prevent actual initialization during tests
# This is crucial for testing the setup_meter function in isolation.
@pytest.fixture(autouse=True)
def mock_meter_provider(monkeypatch):
    # Create a mock MeterProvider instance
    mock_provider = MeterProvider()
    # Patch the get_meter_provider function to return our mock instance
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: mock_provider)
    # Patch set_meter_provider to set our mock provider, allowing setup_meter to interact with it
    monkeypatch.setattr(metrics, "set_meter_provider", lambda provider: None)
    # Clear any existing metric readers from previous tests
    mock_provider._all_metric_readers.clear()
    # Return the mock provider for potential use in tests if needed
    return mock_provider


def test_setup_meter_with_otlp():
    """Test that setup_meter correctly configures OTLP exporter."""
    app_name = "test-service"
    env = os.getenv("ENVIRONMENT", "dev")
    otlp_endpoint = "http://localhost:4317"
    otlp_headers = {"Authorization": "Bearer test-token"}

    metrics_dict, meter = setup_meter(app_name, env, otlp_endpoint, otlp_headers)

    assert meter is not None
    assert metrics_dict is not None

    # Verify that the correct exporter is configured
    meter_provider = metrics.get_meter_provider()  # This will return our mock provider
    assert meter_provider is not None
    # Accessing _all_metric_readers correctly by converting to a list
    readers = list(meter_provider._all_metric_readers)
    assert len(readers) == 1
    assert isinstance(readers[0]._exporter, OTLPMetricExporter)


def test_setup_meter_with_console():
    """Test that setup_meter correctly configures Console exporter when no endpoint is provided."""
    app_name = "test-service"
    env = os.getenv("ENVIRONMENT", "dev")
    otlp_endpoint = ""
    otlp_headers = None

    metrics_dict, meter = setup_meter(app_name, env, otlp_endpoint, otlp_headers)

    assert meter is not None
    assert metrics_dict is not None

    # Verify that the correct exporter is configured
    meter_provider = metrics.get_meter_provider()  # This will return our mock provider
    assert meter_provider is not None
    # Accessing _all_metric_readers correctly by converting to a list
    readers = list(meter_provider._all_metric_readers)
    assert len(readers) == 1
    assert isinstance(readers[0]._exporter, ConsoleMetricExporter)


def test_setup_meter_singleton():
    """Test that setup_meter acts as a singleton for MeterProvider initialization."""
    app_name = "singleton-test"
    env = "test"
    otlp_endpoint = "http://localhost:4318"

    # First call should initialize
    metrics_dict1, meter1 = setup_meter(app_name, env, otlp_endpoint, None)
    provider1 = metrics.get_meter_provider()

    # Second call should use the existing provider
    metrics_dict2, meter2 = setup_meter(app_name, env, otlp_endpoint, None)
    provider2 = metrics.get_meter_provider()

    assert meter1 is not None
    assert meter2 is not None
    assert provider1 is provider2  # Ensure the same provider instance is used


def test_setup_meter_no_endpoint_and_headers():
    """Test setup_meter with no endpoint and no headers."""
    app_name = "test-service"
    env = os.getenv("ENVIRONMENT", "dev")
    otlp_endpoint = None
    otlp_headers = None

    metrics_dict, meter = setup_meter(app_name, env, otlp_endpoint, otlp_headers)

    assert meter is not None
    assert metrics_dict is not None

    meter_provider = metrics.get_meter_provider()
    assert meter_provider is not None
    # Accessing _all_metric_readers correctly by converting to a list
    readers = list(meter_provider._all_metric_readers)
    assert len(readers) == 1
    assert isinstance(readers[0]._exporter, ConsoleMetricExporter)


def test_setup_meter_with_otlp_headers_only():
    """Test setup_meter with OTLP endpoint and headers."""
    app_name = "test-service"
    env = os.getenv("ENVIRONMENT", "dev")
    otlp_endpoint = "http://localhost:4317"
    otlp_headers = {"Authorization": "Bearer test-token"}

    metrics_dict, meter = setup_meter(app_name, env, otlp_endpoint, otlp_headers)

    assert meter is not None
    assert metrics_dict is not None

    meter_provider = metrics.get_meter_provider()
    assert meter_provider is not None
    # Accessing _all_metric_readers correctly by converting to a list
    readers = list(meter_provider._all_metric_readers)
    assert len(readers) == 1
    assert isinstance(readers[0]._exporter, OTLPMetricExporter)


def test_setup_meter_with_invalid_endpoint():
    """Test setup_meter with an invalid OTLP endpoint."""
    app_name = "test-service"
    env = os.getenv("ENVIRONMENT", "dev")
    otlp_endpoint = "invalid-endpoint"
    otlp_headers = None

    metrics_dict, meter = setup_meter(app_name, env, otlp_endpoint, otlp_headers)

    # Expecting None for meter if initialization fails due to invalid endpoint
    assert meter is None
    assert metrics_dict is not None


def test_setup_meter_with_invalid_headers():
    """Test setup_meter with invalid OTLP headers."""
    app_name = "test-service"
    env = os.getenv("ENVIRONMENT", "dev")
    otlp_endpoint = "http://localhost:4317"
    otlp_headers = "invalid-headers"

    metrics_dict, meter = setup_meter(app_name, env, otlp_endpoint, otlp_headers)

    # Expecting None for meter if initialization fails due to invalid headers
    assert meter is None
    assert metrics_dict is not None
