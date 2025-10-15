import os
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider

import genai_otel
from genai_otel.config import OTelConfig
from genai_otel.metrics import get_meter, get_meter_provider


@pytest.fixture(autouse=True)
def reset_otel_metrics():
    """Resets OpenTelemetry metrics and tracing to ensure clean state for each test."""
    # Unset environment variables that might influence OTelConfig
    env_vars_to_clear = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_TRACER_PROVIDER",
        "OTEL_METRIC_READER",
        "OTEL_SERVICE_NAME",
        "GENAI_ENABLE_COST_TRACKING",
        "GENAI_ENABLE_GPU_METRICS",
        "GENAI_FAIL_ON_ERROR",
    ]
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]

    # Reset the global MeterProvider and TracerProvider to NoOp providers
    yield


def test_instrument_sets_global_meter_provider():
    """
    Test that genai_otel.instrument() sets a global MeterProvider.
    """
    with patch("genai_otel.auto_instrument.TracerProvider") as mock_tracer_provider_class:
        with patch("genai_otel.auto_instrument.MeterProvider") as mock_meter_provider_class:
            with patch("genai_otel.auto_instrument.OTLPSpanExporter") as mock_otlp_span_exporter:
                with patch(
                    "genai_otel.auto_instrument.OTLPMetricExporter"
                ) as mock_otlp_metric_exporter:
                    with patch(
                        "genai_otel.auto_instrument.PeriodicExportingMetricReader"
                    ) as mock_periodic_exporting_metric_reader:
                        with patch(
                            "genai_otel.auto_instrument.BatchSpanProcessor"
                        ) as mock_batch_span_processor:
                            with patch(
                                "genai_otel.auto_instrument.ConsoleSpanExporter"
                            ) as mock_console_span_exporter:
                                with patch(
                                    "genai_otel.auto_instrument.ConsoleMetricExporter"
                                ) as mock_console_metric_exporter:
                                    genai_otel.instrument(service_name="test-app")

                                    # Verify that the MeterProvider was instantiated and set
                                    mock_meter_provider_class.assert_called_once()
                                    mock_tracer_provider_class.assert_called_once()

                                    # Get the actual global providers (which will be instances of our mocks)
                                    provider = metrics.get_meter_provider()
                                    tracer_provider = trace.get_tracer_provider()

                                    assert isinstance(provider, MeterProvider)
                                    assert isinstance(tracer_provider, TracerProvider)

                                    # Check that OTLP exporters were called (since no endpoint is explicitly set, it defaults to console)
                                    mock_otlp_span_exporter.assert_not_called()
                                    mock_otlp_metric_exporter.assert_not_called()

                                    # Check that Console exporters were called
                                    mock_console_span_exporter.assert_called_once()
                                    mock_console_metric_exporter.assert_called_once()

                                    # Check that readers and processors were set up
                                    mock_periodic_exporting_metric_reader.assert_called_once()
                                    mock_batch_span_processor.assert_called_once()


def test_get_meter_returns_valid_meter():
    """
    Test that get_meter() returns a valid Meter instance after instrumentation.
    """
    with patch("genai_otel.auto_instrument.TracerProvider") as mock_tracer_provider_class:
        with patch("genai_otel.auto_instrument.MeterProvider") as mock_meter_provider_class:
            with patch("genai_otel.auto_instrument.OTLPSpanExporter") as mock_otlp_span_exporter:
                with patch(
                    "genai_otel.auto_instrument.OTLPMetricExporter"
                ) as mock_otlp_metric_exporter:
                    with patch(
                        "genai_otel.auto_instrument.PeriodicExportingMetricReader"
                    ) as mock_periodic_exporting_metric_reader:
                        with patch(
                            "genai_otel.auto_instrument.BatchSpanProcessor"
                        ) as mock_batch_span_processor:
                            with patch(
                                "genai_otel.auto_instrument.ConsoleSpanExporter"
                            ) as mock_console_span_exporter:
                                with patch(
                                    "genai_otel.auto_instrument.ConsoleMetricExporter"
                                ) as mock_console_metric_exporter:
                                    genai_otel.instrument(service_name="test-app")

                                    # Verify that the MeterProvider was instantiated and set
                                    mock_meter_provider_class.assert_called_once()
                                    mock_tracer_provider_class.assert_called_once()

                                    # Get the actual global providers (which will be instances of our mocks)
                                    provider = metrics.get_meter_provider()
                                    tracer_provider = trace.get_tracer_provider()

                                    assert isinstance(provider, MeterProvider)
                                    assert isinstance(tracer_provider, TracerProvider)

                                    # Check that OTLP exporters were called (since no endpoint is explicitly set, it defaults to console)
                                    mock_otlp_span_exporter.assert_not_called()
                                    mock_otlp_metric_exporter.assert_not_called()

                                    # Check that Console exporters were called
                                    mock_console_span_exporter.assert_called_once()
                                    mock_console_metric_exporter.assert_called_once()

                                    # Check that readers and processors were set up
                                    mock_periodic_exporting_metric_reader.assert_called_once()
                                    mock_batch_span_processor.assert_called_once()

                                    meter = get_meter()
                                    assert meter is not None
                                    assert isinstance(meter, metrics.Meter)


def test_otlp_exporter_configured_when_endpoint_set():
    """
    Test that OTLPMetricExporter is configured when OTEL_EXPORTER_OTLP_ENDPOINT is set.
    """
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    with patch("genai_otel.auto_instrument.TracerProvider") as mock_tracer_provider_class:
        with patch("genai_otel.auto_instrument.MeterProvider") as mock_meter_provider_class:
            with patch("genai_otel.auto_instrument.OTLPSpanExporter") as mock_otlp_span_exporter:
                with patch(
                    "genai_otel.auto_instrument.OTLPMetricExporter"
                ) as mock_otlp_metric_exporter:
                    with patch(
                        "genai_otel.auto_instrument.PeriodicExportingMetricReader"
                    ) as mock_periodic_exporting_metric_reader:
                        with patch(
                            "genai_otel.auto_instrument.BatchSpanProcessor"
                        ) as mock_batch_span_processor:
                            with patch(
                                "genai_otel.auto_instrument.ConsoleSpanExporter"
                            ) as mock_console_span_exporter:
                                with patch(
                                    "genai_otel.auto_instrument.ConsoleMetricExporter"
                                ) as mock_console_metric_exporter:
                                    genai_otel.instrument(service_name="test-app")

                                    # Verify that the MeterProvider was instantiated and set
                                    mock_meter_provider_class.assert_called_once()
                                    mock_tracer_provider_class.assert_called_once()

                                    # Get the actual global providers (which will be instances of our mocks)
                                    provider = metrics.get_meter_provider()
                                    tracer_provider = trace.get_tracer_provider()

                                    assert isinstance(provider, MeterProvider)
                                    assert isinstance(tracer_provider, TracerProvider)

                                    # Check that OTLP exporters were called
                                    mock_otlp_span_exporter.assert_called_once()
                                    mock_otlp_metric_exporter.assert_called_once()

                                    # Check that Console exporters were not called
                                    mock_console_span_exporter.assert_not_called()
                                    mock_console_metric_exporter.assert_not_called()

                                    # Check that readers and processors were set up
                                    mock_periodic_exporting_metric_reader.assert_called_once()
                                    mock_batch_span_processor.assert_called_once()

                                    readers = list(provider._all_metric_readers)
                                    assert len(readers) == 1
                                    assert isinstance(readers[0], PeriodicExportingMetricReader)
                                    assert isinstance(readers[0]._exporter, OTLPMetricExporter)


def test_console_exporter_configured_when_no_endpoint():
    """
    Test that ConsoleMetricExporter is configured when no OTLP endpoint is set.
    """
    # Ensure no OTLP endpoint is set
    if "OTEL_EXPORTER_OTLP_ENDPOINT" in os.environ:
        del os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]

    with patch("genai_otel.auto_instrument.TracerProvider") as mock_tracer_provider_class:
        with patch("genai_otel.auto_instrument.MeterProvider") as mock_meter_provider_class:
            with patch("genai_otel.auto_instrument.OTLPSpanExporter") as mock_otlp_span_exporter:
                with patch(
                    "genai_otel.auto_instrument.OTLPMetricExporter"
                ) as mock_otlp_metric_exporter:
                    with patch(
                        "genai_otel.auto_instrument.PeriodicExportingMetricReader"
                    ) as mock_periodic_exporting_metric_reader:
                        with patch(
                            "genai_otel.auto_instrument.BatchSpanProcessor"
                        ) as mock_batch_span_processor:
                            with patch(
                                "genai_otel.auto_instrument.ConsoleSpanExporter"
                            ) as mock_console_span_exporter:
                                with patch(
                                    "genai_otel.auto_instrument.ConsoleMetricExporter"
                                ) as mock_console_metric_exporter:
                                    genai_otel.instrument(service_name="test-app", endpoint="")

                                    # Verify that the MeterProvider was instantiated and set
                                    mock_meter_provider_class.assert_called_once()
                                    mock_tracer_provider_class.assert_called_once()

                                    # Get the actual global providers (which will be instances of our mocks)
                                    provider = metrics.get_meter_provider()
                                    tracer_provider = trace.get_tracer_provider()

                                    assert isinstance(provider, MeterProvider)
                                    assert isinstance(tracer_provider, TracerProvider)

                                    # Check that OTLP exporters were not called
                                    mock_otlp_span_exporter.assert_not_called()
                                    mock_otlp_metric_exporter.assert_not_called()

                                    # Check that Console exporters were called
                                    mock_console_span_exporter.assert_called_once()
                                    mock_console_metric_exporter.assert_called_once()

                                    # Check that readers and processors were set up
                                    mock_periodic_exporting_metric_reader.assert_called_once()
                                    mock_batch_span_processor.assert_called_once()

                                    readers = list(provider._all_metric_readers)
                                    assert len(readers) == 1
                                    assert isinstance(readers[0], PeriodicExportingMetricReader)
                                    assert isinstance(readers[0]._exporter, ConsoleMetricExporter)


def test_otlp_exporter_fallback_on_error():
    """
    Test that if OTLP exporter setup fails, it falls back to NoOp behavior.
    """
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

    # Mock OTLPMetricExporter to raise an exception during initialization
    with patch("genai_otel.auto_instrument.OTLPMetricExporter") as mock_otlp_exporter:
        mock_otlp_exporter.side_effect = Exception("Mock OTLP Exporter Error")
        genai_otel.instrument(service_name="test-app")

        provider = get_meter_provider()
        # Expecting no metric readers to be added if OTLP setup fails and falls back to NoOp
        assert len(list(provider._all_metric_readers)) == 0
