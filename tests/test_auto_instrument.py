# tests/test_auto_instrument.py
import logging
import os
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from genai_otel.auto_instrument import INSTRUMENTORS, setup_auto_instrumentation
from genai_otel.config import OTelConfig
from genai_otel.exceptions import InstrumentationError

# Mock instrumentors for testing
# Each key should map to a mock class, and its return_value should be a mock instance
# with an 'instrument' method.
MOCK_INSTRUMENTORS = {
    "openai": MagicMock(),
    "anthropic": MagicMock(),
    "google.generativeai": MagicMock(),
    "boto3": MagicMock(),
    "cohere": MagicMock(),
    "mistralai": MagicMock(),
}

# Create mock instances for the instrumentors to be used in tests
# This is necessary because the code under test instantiates these classes.
mock_openai_instance = MagicMock()
mock_anthropic_instance = MagicMock()
mock_google_instance = MagicMock()
mock_boto3_instance = MagicMock()
mock_cohere_instance = MagicMock()
mock_mistralai_instance = MagicMock()

# Configure MOCK_INSTRUMENTORS to hold mock classes
MOCK_INSTRUMENTORS["openai"].return_value = mock_openai_instance
MOCK_INSTRUMENTORS["anthropic"].return_value = mock_anthropic_instance
MOCK_INSTRUMENTORS["google.generativeai"].return_value = mock_google_instance
MOCK_INSTRUMENTORS["boto3"].return_value = mock_boto3_instance
MOCK_INSTRUMENTORS["cohere"].return_value = mock_cohere_instance
MOCK_INSTRUMENTORS["mistralai"].return_value = mock_mistralai_instance


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    # Reset the mock instances
    mock_openai_instance.reset_mock()
    mock_anthropic_instance.reset_mock()
    mock_google_instance.reset_mock()
    mock_boto3_instance.reset_mock()
    mock_cohere_instance.reset_mock()
    mock_mistralai_instance.reset_mock()

    # Reset the mock classes themselves
    MOCK_INSTRUMENTORS["openai"].reset_mock()
    MOCK_INSTRUMENTORS["anthropic"].reset_mock()
    MOCK_INSTRUMENTORS["google.generativeai"].reset_mock()
    MOCK_INSTRUMENTORS["boto3"].reset_mock()
    MOCK_INSTRUMENTORS["cohere"].reset_mock()
    MOCK_INSTRUMENTORS["mistralai"].reset_mock()

    # Re-assign return_value for mock classes
    MOCK_INSTRUMENTORS["openai"].return_value = mock_openai_instance
    MOCK_INSTRUMENTORS["anthropic"].return_value = mock_anthropic_instance
    MOCK_INSTRUMENTORS["google.generativeai"].return_value = mock_google_instance
    MOCK_INSTRUMENTORS["boto3"].return_value = mock_boto3_instance
    MOCK_INSTRUMENTORS["cohere"].return_value = mock_cohere_instance
    MOCK_INSTRUMENTORS["mistralai"].return_value = mock_mistralai_instance


class TestAutoInstrumentation:
    """Test suite for auto_instrumentation functionality"""

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.GPUMetricsCollector")
    @patch("genai_otel.auto_instrument.MCPInstrumentorManager")
    @patch("genai_otel.auto_instrument.OTLPMetricExporter")
    @patch("genai_otel.auto_instrument.OTLPSpanExporter")
    @patch("genai_otel.auto_instrument.PeriodicExportingMetricReader")
    @patch("genai_otel.auto_instrument.BatchSpanProcessor")
    @patch("genai_otel.auto_instrument.MeterProvider")
    @patch("genai_otel.auto_instrument.TracerProvider")
    @patch("genai_otel.auto_instrument.Resource")
    def test_setup_enables_all_components(
        self,
        mock_resource,
        mock_tracer_provider_class,
        mock_meter_provider_class,
        mock_batch_span_processor,
        mock_periodic_exporting_metric_reader,
        mock_otlp_span_exporter,
        mock_otlp_metric_exporter,
        mock_mcp_manager,
        mock_gpu_collector,
    ):
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enable_gpu_metrics=True,
            enable_mcp_instrumentation=True,
            enabled_instrumentors=["openai", "anthropic"],
        )
        # Mock instances
        mock_resource_instance = MagicMock()
        mock_resource.create.return_value = mock_resource_instance
        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider_class.return_value = mock_tracer_provider_instance
        mock_meter_provider_instance = MagicMock()
        mock_meter_provider_class.return_value = mock_meter_provider_instance
        mock_meter = MagicMock()
        mock_meter_provider_instance.get_meter.return_value = mock_meter
        mock_span_exporter_instance = MagicMock()
        mock_otlp_span_exporter.return_value = mock_span_exporter_instance
        mock_metric_exporter_instance = MagicMock()
        mock_otlp_metric_exporter.return_value = mock_metric_exporter_instance
        mock_metric_reader_instance = MagicMock()
        mock_periodic_exporting_metric_reader.return_value = mock_metric_reader_instance
        mock_span_processor_instance = MagicMock()
        mock_batch_span_processor.return_value = mock_span_processor_instance
        with patch.dict("os.environ", {"OTEL_EXPORTER_OTLP_TIMEOUT": "10.0"}):
            # Act
            with patch("genai_otel.auto_instrument.logger") as mock_logger:
                with patch("genai_otel.auto_instrument.trace") as mock_trace:
                    with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                        mock_metrics.get_meter_provider.return_value = mock_meter_provider_instance
                        setup_auto_instrumentation(config)
                        # Assertions
                        mock_resource.create.assert_called_once_with(
                            {"service.name": "test-service"}
                        )
                        mock_tracer_provider_class.assert_called_once_with(
                            resource=mock_resource_instance
                        )
                        mock_trace.set_tracer_provider.assert_called_once_with(
                            mock_tracer_provider_instance
                        )
                        # OTLP exporters now read endpoint from environment variable
                        # They should be called WITHOUT endpoint parameter
                        mock_otlp_span_exporter.assert_called_once_with(
                            headers=config.headers,
                        )
                        mock_batch_span_processor.assert_called_once_with(
                            mock_span_exporter_instance
                        )
                        # Should add 2 processors: CostEnrichmentSpanProcessor + BatchSpanProcessor
                        assert mock_tracer_provider_instance.add_span_processor.call_count == 2
                        # Verify the BatchSpanProcessor was added (second call)
                        calls = mock_tracer_provider_instance.add_span_processor.call_args_list
                        assert calls[1][0][0] == mock_span_processor_instance
                        mock_otlp_metric_exporter.assert_called_once_with(
                            headers=config.headers,
                        )
                        mock_periodic_exporting_metric_reader.assert_called_once_with(
                            exporter=mock_metric_exporter_instance
                        )
                        # MeterProvider should be called with views parameter for histogram buckets
                        assert mock_meter_provider_class.call_count == 1
                        call_kwargs = mock_meter_provider_class.call_args.kwargs
                        assert call_kwargs["resource"] == mock_resource_instance
                        assert call_kwargs["metric_readers"] == [mock_metric_reader_instance]
                        assert "views" in call_kwargs
                        assert (
                            len(call_kwargs["views"]) == 8
                        )  # GenAI duration + 3 MCP + 2 streaming + 2 token histograms
                        mock_metrics.set_meter_provider.assert_called_once_with(
                            mock_meter_provider_instance
                        )
                        mock_openai_instance.instrument.assert_called_once_with(config=config)
                        mock_anthropic_instance.instrument.assert_called_once_with(config=config)
                        mock_google_instance.assert_not_called()
                        mock_gpu_collector.assert_called_once_with(
                            mock_meter, config, interval=config.gpu_collection_interval
                        )
                        mock_gpu_collector.return_value.start.assert_called_once()
                        mock_mcp_manager.assert_called_once_with(config)
                        mock_mcp_manager.return_value.instrument_all.assert_called_once_with(
                            config.fail_on_error
                        )
                        # Check log messages
                        mock_logger.info.assert_any_call("Starting auto-instrumentation setup...")
                        # Log messages now include the exporter's _endpoint attribute
                        # We can't predict the exact value since it's read from the mock
                        # So we'll check that the log was called with a string containing the key parts
                        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                        assert any(
                            "OpenTelemetry tracing configured with OTLP endpoint:" in msg
                            for msg in log_calls
                        )
                        assert any(
                            "OpenTelemetry metrics configured with OTLP endpoint:" in msg
                            for msg in log_calls
                        )
                        mock_logger.info.assert_any_call("openai instrumentation enabled")
                        mock_logger.info.assert_any_call("anthropic instrumentation enabled")
                        mock_logger.info.assert_any_call(
                            "MCP tools instrumentation enabled and set up."
                        )
                        mock_logger.info.assert_any_call(
                            f"GPU metrics collection started (interval: {config.gpu_collection_interval}s)."
                        )
                        mock_logger.info.assert_any_call("Auto-instrumentation setup complete")

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.OTLPMetricExporter")
    @patch("genai_otel.auto_instrument.OTLPSpanExporter")
    @patch("genai_otel.auto_instrument.PeriodicExportingMetricReader")
    @patch("genai_otel.auto_instrument.BatchSpanProcessor")
    @patch("genai_otel.auto_instrument.MeterProvider")
    @patch("genai_otel.auto_instrument.TracerProvider")
    @patch("genai_otel.auto_instrument.Resource")
    def test_setup_auto_instrumentation_default_config(
        self,
        mock_resource,
        mock_tracer_provider_class,
        mock_meter_provider_class,
        mock_batch_span_processor,
        mock_periodic_exporting_metric_reader,
        mock_otlp_span_exporter,
        mock_otlp_metric_exporter,
    ):
        config = OTelConfig(
            service_name="test-service",
            endpoint=None,  # Explicitly set to None
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )
        # Mock instances
        mock_resource_instance = MagicMock()
        mock_resource.create.return_value = mock_resource_instance
        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider_class.return_value = mock_tracer_provider_instance
        mock_meter_provider_instance = MagicMock()
        mock_meter_provider_class.return_value = mock_meter_provider_instance
        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.trace") as mock_trace:
                with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                    setup_auto_instrumentation(config)
                    # Assertions
                    mock_tracer_provider_class.assert_called_once()
                    mock_meter_provider_class.assert_called_once()
                    mock_otlp_span_exporter.assert_not_called()
                    mock_otlp_metric_exporter.assert_not_called()
                    mock_logger.info.assert_any_call("Starting auto-instrumentation setup...")
                    mock_logger.info.assert_any_call("Auto-instrumentation setup complete")
                    mock_logger.info.assert_any_call(
                        "No OTLP endpoint configured, traces will be exported to console."
                    )
                    mock_logger.info.assert_any_call(
                        "No OTLP endpoint configured, metrics will be exported to console."
                    )

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.OTLPMetricExporter")
    @patch("genai_otel.auto_instrument.OTLPSpanExporter")
    @patch("genai_otel.auto_instrument.PeriodicExportingMetricReader")
    @patch("genai_otel.auto_instrument.BatchSpanProcessor")
    @patch("genai_otel.auto_instrument.MeterProvider")
    @patch("genai_otel.auto_instrument.TracerProvider")
    @patch("genai_otel.auto_instrument.Resource")
    def test_setup_auto_instrumentation_with_llm_instrumentors(
        self,
        mock_resource,
        mock_tracer_provider_class,
        mock_meter_provider_class,
        mock_batch_span_processor,
        mock_periodic_exporting_metric_reader,
        mock_otlp_span_exporter,
        mock_otlp_metric_exporter,
    ):
        """Test setup_auto_instrumentation with specific LLM instrumentors enabled."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=["openai", "anthropic", "cohere"],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        # Mock instances
        mock_resource_instance = MagicMock()
        mock_resource.create.return_value = mock_resource_instance

        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider_class.return_value = mock_tracer_provider_instance

        mock_meter_provider_instance = MagicMock()
        mock_meter_provider_class.return_value = mock_meter_provider_instance

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.trace") as mock_trace:
                with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                    setup_auto_instrumentation(config)

                    # Check that only the enabled instrumentors were called
                    # Use the mock instances directly
                    mock_openai_instance.instrument.assert_called_once_with(config=config)
                    mock_anthropic_instance.instrument.assert_called_once_with(config=config)
                    mock_cohere_instance.instrument.assert_called_once_with(config=config)
                    mock_google_instance.assert_not_called()
                    mock_mistralai_instance.assert_not_called()

                    # Check log messages
                    mock_logger.info.assert_any_call("openai instrumentation enabled")
                    mock_logger.info.assert_any_call("anthropic instrumentation enabled")
                    mock_logger.info.assert_any_call("cohere instrumentation enabled")

    @patch("genai_otel.auto_instrument.OTLPMetricExporter")
    @patch("genai_otel.auto_instrument.OTLPSpanExporter")
    @patch("genai_otel.auto_instrument.PeriodicExportingMetricReader")
    @patch("genai_otel.auto_instrument.BatchSpanProcessor")
    @patch("genai_otel.auto_instrument.MeterProvider")
    @patch("genai_otel.auto_instrument.TracerProvider")
    @patch("genai_otel.auto_instrument.Resource")
    def test_setup_auto_instrumentation_with_mcp_instrumentation(
        self,
        mock_resource,
        mock_tracer_provider_class,
        mock_meter_provider_class,
        mock_batch_span_processor,
        mock_periodic_exporting_metric_reader,
        mock_otlp_span_exporter,
        mock_otlp_metric_exporter,
    ):
        """Test setup_auto_instrumentation with MCP instrumentation enabled."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enable_mcp_instrumentation=True,
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
        )

        # Mock instances
        mock_resource_instance = MagicMock()
        mock_resource.create.return_value = mock_resource_instance

        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider_class.return_value = mock_tracer_provider_instance

        mock_meter_provider_instance = MagicMock()
        mock_meter_provider_class.return_value = mock_meter_provider_instance

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.trace") as mock_trace:
                with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                    with patch(
                        "genai_otel.auto_instrument.MCPInstrumentorManager"
                    ) as mock_mcp_manager:
                        mock_mcp_instance = MagicMock()
                        mock_mcp_manager.return_value = mock_mcp_instance

                        setup_auto_instrumentation(config)

                        mock_mcp_manager.assert_called_once_with(config)
                        mock_mcp_instance.instrument_all.assert_called_once_with(
                            config.fail_on_error
                        )
                        mock_logger.info.assert_any_call(
                            "MCP tools instrumentation enabled and set up."
                        )

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.OTLPMetricExporter")
    @patch("genai_otel.auto_instrument.OTLPSpanExporter")
    @patch("genai_otel.auto_instrument.PeriodicExportingMetricReader")
    @patch("genai_otel.auto_instrument.BatchSpanProcessor")
    @patch("genai_otel.auto_instrument.MeterProvider")
    @patch("genai_otel.auto_instrument.TracerProvider")
    @patch("genai_otel.auto_instrument.Resource")
    def test_setup_auto_instrumentation_with_gpu_metrics(
        self,
        mock_resource,
        mock_tracer_provider_class,
        mock_meter_provider_class,
        mock_batch_span_processor,
        mock_periodic_exporting_metric_reader,
        mock_otlp_span_exporter,
        mock_otlp_metric_exporter,
    ):
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enable_gpu_metrics=True,
            enabled_instrumentors=[],
            enable_mcp_instrumentation=False,
        )
        # Mock instances
        mock_resource_instance = MagicMock()
        mock_resource.create.return_value = mock_resource_instance
        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider_class.return_value = mock_tracer_provider_instance
        mock_meter_provider_instance = MagicMock()
        mock_meter_provider_class.return_value = mock_meter_provider_instance
        mock_meter = MagicMock()
        mock_meter_provider_instance.get_meter.return_value = mock_meter
        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.trace") as mock_trace:
                with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                    mock_metrics.get_meter_provider.return_value = mock_meter_provider_instance
                    with patch(
                        "genai_otel.auto_instrument.GPUMetricsCollector"
                    ) as mock_gpu_collector:
                        mock_gpu_instance = MagicMock()
                        mock_gpu_collector.return_value = mock_gpu_instance
                        setup_auto_instrumentation(config)
                        # Check that get_meter was called for both GPU and server metrics
                        assert mock_meter_provider_instance.get_meter.call_count == 2
                        mock_meter_provider_instance.get_meter.assert_any_call("genai.gpu")
                        mock_meter_provider_instance.get_meter.assert_any_call("genai.server")
                        mock_gpu_collector.assert_called_once_with(
                            mock_meter, config, interval=config.gpu_collection_interval
                        )
                        mock_gpu_instance.start.assert_called_once()
                        mock_logger.info.assert_any_call(
                            f"GPU metrics collection started (interval: {config.gpu_collection_interval}s)."
                        )

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    def test_setup_auto_instrumentation_llm_instrumentor_failure_no_fail_on_error(self):
        """Test LLM instrumentor failure when fail_on_error is False."""
        # Make one instrumentor fail
        mock_failing_instrumentor_instance = MagicMock()
        mock_failing_instrumentor_instance.instrument.side_effect = Exception(
            "LLM instrumentor error"
        )
        # Replace the mock instance for openai with the failing one
        MOCK_INSTRUMENTORS["openai"].return_value = mock_failing_instrumentor_instance

        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=["openai", "anthropic"],
            fail_on_error=False,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            setup_auto_instrumentation(config)

            # Both instrumentors should be attempted, but only one fails
            mock_failing_instrumentor_instance.instrument.assert_called_once_with(config=config)
            mock_anthropic_instance.instrument.assert_called_once_with(config=config)

            # Error should be logged but not raised
            mock_logger.error.assert_called_once()
            assert (
                "Failed to instrument openai: LLM instrumentor error"
                in mock_logger.error.call_args[0][0]
            )

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    def test_setup_auto_instrumentation_llm_instrumentor_failure_with_fail_on_error(self):
        """Test LLM instrumentor failure when fail_on_error is True."""
        # Make one instrumentor fail with InstrumentationError
        mock_failing_instrumentor_instance = MagicMock()
        mock_failing_instrumentor_instance.instrument.side_effect = InstrumentationError(
            "LLM instrumentor error"
        )
        MOCK_INSTRUMENTORS["openai"].return_value = mock_failing_instrumentor_instance

        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=["openai"],
            fail_on_error=True,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            with pytest.raises(
                                InstrumentationError, match="LLM instrumentor error"
                            ):
                                setup_auto_instrumentation(config)

            mock_logger.error.assert_called_once()

    def test_setup_auto_instrumentation_mcp_instrumentation_failure_no_fail_on_error(self):
        """Test MCP instrumentation failure when fail_on_error is False."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enable_mcp_instrumentation=True,
            fail_on_error=False,
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            with patch(
                                "genai_otel.auto_instrument.MCPInstrumentorManager"
                            ) as mock_mcp_manager:
                                mock_mcp_instance = MagicMock()
                                mock_mcp_manager.return_value = mock_mcp_instance
                                mock_mcp_instance.instrument_all.side_effect = Exception(
                                    "MCP error"
                                )

                                setup_auto_instrumentation(config)

                                mock_logger.error.assert_called_once()
                                error_message = mock_logger.error.call_args[0][0]
                                assert (
                                    "Failed to set up MCP tools instrumentation: MCP error"
                                    in error_message
                                )

    def test_setup_auto_instrumentation_mcp_instrumentation_failure_with_fail_on_error(self):
        """Test MCP instrumentation failure when fail_on_error is True."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enable_mcp_instrumentation=True,
            fail_on_error=True,
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            with patch(
                                "genai_otel.auto_instrument.MCPInstrumentorManager"
                            ) as mock_mcp_manager:
                                mock_mcp_instance = MagicMock()
                                mock_mcp_manager.return_value = mock_mcp_instance
                                mock_mcp_instance.instrument_all.side_effect = InstrumentationError(
                                    "MCP error"
                                )

                                with pytest.raises(InstrumentationError, match="MCP error"):
                                    setup_auto_instrumentation(config)

                                mock_logger.error.assert_called_once()

    def test_setup_auto_instrumentation_gpu_metrics_failure_no_fail_on_error(self):
        """Test GPU metrics failure when fail_on_error is False."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enable_gpu_metrics=True,
            fail_on_error=False,
            enabled_instrumentors=[],
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics") as mock_metrics_module:
                            mock_meter_provider = MagicMock()
                            mock_metrics_module.get_meter_provider.return_value = (
                                mock_meter_provider
                            )
                            mock_meter = MagicMock()
                            mock_meter_provider.get_meter.return_value = mock_meter

                            with patch(
                                "genai_otel.auto_instrument.GPUMetricsCollector"
                            ) as mock_gpu_collector:
                                mock_gpu_collector.side_effect = Exception("GPU error")

                                setup_auto_instrumentation(config)

                                mock_logger.error.assert_called_once()
                                error_message = mock_logger.error.call_args[0][0]
                                assert (
                                    "Failed to start GPU metrics collection: GPU error"
                                    in error_message
                                )

    def test_setup_auto_instrumentation_gpu_metrics_failure_with_fail_on_error(self):
        """Test GPU metrics failure when fail_on_error is True."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enable_gpu_metrics=True,
            fail_on_error=True,
            enabled_instrumentors=[],
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics") as mock_metrics_module:
                            mock_meter_provider = MagicMock()
                            mock_metrics_module.get_meter_provider.return_value = (
                                mock_meter_provider
                            )
                            mock_meter = MagicMock()
                            mock_meter_provider.get_meter.return_value = mock_meter

                            with patch(
                                "genai_otel.auto_instrument.GPUMetricsCollector"
                            ) as mock_gpu_collector:
                                mock_gpu_collector.side_effect = InstrumentationError("GPU error")

                                with pytest.raises(InstrumentationError, match="GPU error"):
                                    setup_auto_instrumentation(config)

                                mock_logger.error.assert_called_once()

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.OTLPMetricExporter")
    @patch("genai_otel.auto_instrument.OTLPSpanExporter")
    @patch("genai_otel.auto_instrument.PeriodicExportingMetricReader")
    @patch("genai_otel.auto_instrument.BatchSpanProcessor")
    @patch("genai_otel.auto_instrument.MeterProvider")
    @patch("genai_otel.auto_instrument.TracerProvider")
    @patch("genai_otel.auto_instrument.Resource")
    def test_setup_auto_instrumentation_unknown_instrumentor(
        self,
        mock_resource,
        mock_tracer_provider_class,
        mock_meter_provider_class,
        mock_batch_span_processor,
        mock_periodic_exporting_metric_reader,
        mock_otlp_span_exporter,
        mock_otlp_metric_exporter,
    ):
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=["unknown_llm", "openai"],
            fail_on_error=False,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )
        # Mock instances
        mock_resource_instance = MagicMock()
        mock_resource.create.return_value = mock_resource_instance
        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider_class.return_value = mock_tracer_provider_instance
        mock_meter_provider_instance = MagicMock()
        mock_meter_provider_class.return_value = mock_meter_provider_instance
        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.trace") as mock_trace:
                with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                    setup_auto_instrumentation(config)
                    # Verify that the OpenAI instrumentor's instrument method was called
                    mock_openai_instance.instrument.assert_called_once_with(config=config)
                    # Verify that a warning was logged for the unknown instrumentor
                    mock_logger.warning.assert_called_once_with(
                        "Unknown instrumentor 'unknown_llm' requested."
                    )

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.OTLPMetricExporter")
    @patch("genai_otel.auto_instrument.OTLPSpanExporter")
    @patch("genai_otel.auto_instrument.PeriodicExportingMetricReader")
    @patch("genai_otel.auto_instrument.BatchSpanProcessor")
    @patch("genai_otel.auto_instrument.MeterProvider")
    @patch("genai_otel.auto_instrument.TracerProvider")
    @patch("genai_otel.auto_instrument.Resource")
    def test_setup_auto_instrumentation_multiple_unknown_instrumentors(
        self,
        mock_resource,
        mock_tracer_provider_class,
        mock_meter_provider_class,
        mock_batch_span_processor,
        mock_periodic_exporting_metric_reader,
        mock_otlp_span_exporter,
        mock_otlp_metric_exporter,
    ):
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=["unknown1", "unknown2", "openai"],
            fail_on_error=False,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )
        # Mock instances
        mock_resource_instance = MagicMock()
        mock_resource.create.return_value = mock_resource_instance
        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider_class.return_value = mock_tracer_provider_instance
        mock_meter_provider_instance = MagicMock()
        mock_meter_provider_class.return_value = mock_meter_provider_instance
        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.trace") as mock_trace:
                with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                    setup_auto_instrumentation(config)
                    # Verify that the OpenAI instrumentor's instrument method was called
                    mock_openai_instance.instrument.assert_called_once_with(config=config)
                    # Verify that warnings were logged for the unknown instrumentors
                    warning_calls = [
                        call("Unknown instrumentor 'unknown1' requested."),
                        call("Unknown instrumentor 'unknown2' requested."),
                    ]
                    mock_logger.warning.assert_has_calls(warning_calls, any_order=True)
                    assert mock_logger.warning.call_count == 2


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_setup_with_none_config(self):
        """Test that setup handles None config gracefully."""
        with pytest.raises(Exception):
            setup_auto_instrumentation(None)

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", {})
    def test_setup_with_empty_instrumentors_dict(self):
        """Test setup when INSTRUMENTORS dict is empty."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=["openai"],
            fail_on_error=False,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            setup_auto_instrumentation(config)

            # Should log warning for unknown instrumentor since INSTRUMENTORS is empty
            mock_logger.warning.assert_called_once_with("Unknown instrumentor 'openai' requested.")

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    def test_setup_auto_instrumentation_empty_instrumentor_list(self):
        """Test setup with empty instrumentor list."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            setup_auto_instrumentation(config)

            # No instrumentors should be called
            mock_openai_instance.assert_not_called()
            mock_anthropic_instance.assert_not_called()
            mock_google_instance.assert_not_called()
            mock_boto3_instance.assert_not_called()
            mock_cohere_instance.assert_not_called()
            mock_mistralai_instance.assert_not_called()

            mock_logger.info.assert_any_call("Auto-instrumentation setup complete")

    def test_setup_auto_instrumentation_no_endpoint(self):
        """Test setup when no OTLP endpoint is provided."""
        config = OTelConfig(
            service_name="test-service",
            endpoint=None,  # No endpoint
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # Mock all the OpenTelemetry setup to avoid side effects
            with patch("genai_otel.auto_instrument.TracerProvider") as mock_tracer_provider_class:
                with patch("genai_otel.auto_instrument.MeterProvider") as mock_meter_provider_class:
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            with patch(
                                "genai_otel.auto_instrument.OTLPSpanExporter"
                            ) as mock_span_exporter:
                                with patch(
                                    "genai_otel.auto_instrument.OTLPMetricExporter"
                                ) as mock_metric_exporter:
                                    setup_auto_instrumentation(config)

                                    # Should not call exporters when no endpoint
                                    mock_span_exporter.assert_not_called()
                                    mock_metric_exporter.assert_not_called()

                                    # Should still set up providers
                                    mock_tracer_provider_class.assert_called_once()
                                    mock_meter_provider_class.assert_called_once()

                            # Should log warnings about no endpoint
                            info_logs = [args[0][0] for args in mock_logger.info.call_args_list]
                            assert any(
                                "No OTLP endpoint configured, traces will be exported to console."
                                in log
                                for log in info_logs
                            )
                            assert any(
                                "No OTLP endpoint configured, metrics will be exported to console."
                                in log
                                for log in info_logs
                            )

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    def test_setup_with_service_instance_id(self):
        """Test that SERVICE_INSTANCE_ID environment variable is used."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch.dict("os.environ", {"OTEL_SERVICE_INSTANCE_ID": "instance-123"}):
            with patch("genai_otel.auto_instrument.TracerProvider") as mock_tracer_provider:
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            with patch("genai_otel.auto_instrument.Resource") as mock_resource:
                                setup_auto_instrumentation(config)

                                # Verify Resource.create was called with service.instance.id
                                resource_attrs = mock_resource.create.call_args[0][0]
                                assert resource_attrs["service.name"] == "test-service"
                                assert resource_attrs["service.instance.id"] == "instance-123"

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    def test_setup_with_environment_attribute(self):
        """Test that OTEL_ENVIRONMENT environment variable is used."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://localhost:4318",
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch.dict("os.environ", {"OTEL_ENVIRONMENT": "production"}):
            with patch("genai_otel.auto_instrument.TracerProvider"):
                with patch("genai_otel.auto_instrument.MeterProvider"):
                    with patch("genai_otel.auto_instrument.trace"):
                        with patch("genai_otel.auto_instrument.metrics"):
                            with patch("genai_otel.auto_instrument.Resource") as mock_resource:
                                setup_auto_instrumentation(config)

                                # Verify Resource.create was called with environment
                                resource_attrs = mock_resource.create.call_args[0][0]
                                assert resource_attrs["service.name"] == "test-service"
                                assert resource_attrs["environment"] == "production"

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    def test_setup_with_invalid_timeout(self):
        """Test timeout handling when OTEL_EXPORTER_OTLP_TIMEOUT is invalid."""
        with patch.dict("os.environ", {"OTEL_EXPORTER_OTLP_TIMEOUT": "invalid"}):
            with patch("genai_otel.config.logger") as mock_logger:
                # Create config AFTER patching env var so warning is logged
                config = OTelConfig(
                    service_name="test-service",
                    endpoint="http://localhost:4318",
                    enabled_instrumentors=[],
                    enable_gpu_metrics=False,
                    enable_mcp_instrumentation=False,
                )

                # Verify warning was logged about invalid timeout
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "Invalid timeout value 'invalid'" in warning_msg

                # Verify config uses default timeout
                assert config.exporter_timeout == 60


def test_instrument_wrapper_function():
    """Test the instrument() convenience wrapper function."""
    from genai_otel.auto_instrument import instrument

    with patch("genai_otel.auto_instrument.setup_auto_instrumentation") as mock_setup:
        with patch("genai_otel.auto_instrument.OTelConfig") as mock_config_class:
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance

            # Call instrument with kwargs
            instrument(service_name="test-service", endpoint="http://localhost:4318")

            # Verify OTelConfig was instantiated with the kwargs
            mock_config_class.assert_called_once_with(
                service_name="test-service", endpoint="http://localhost:4318"
            )

            # Verify setup_auto_instrumentation was called with the config
            mock_setup.assert_called_once_with(mock_config_instance)
