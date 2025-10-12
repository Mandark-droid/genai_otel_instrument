# tests/test_auto_instrument.py
import logging
from unittest.mock import MagicMock, patch, call
import pytest

from genai_otel.auto_instrument import INSTRUMENTORS, setup_auto_instrumentation
from genai_otel.config import OTelConfig
from genai_otel.exceptions import InstrumentationError

# Mock instrumentors for testing
MOCK_INSTRUMENTORS = {
    "openai": MagicMock(),
    "anthropic": MagicMock(),
    "google": MagicMock(),
    "cohere": MagicMock(),
    "mistralai": MagicMock(),
}


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    for mock_instrumentor in MOCK_INSTRUMENTORS.values():
        mock_instrumentor.reset_mock()


class TestAutoInstrumentation:
    """Test suite for auto_instrumentation functionality"""

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.GPUMetricsCollector")
    @patch("genai_otel.auto_instrument.MCPInstrumentorManager")
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_enables_all_components(
        self, mock_configure_otel, mock_mcp_manager, mock_gpu_collector
    ):
        """
        Verify that setup_auto_instrumentation initializes and enables all
        components when the configuration specifies them.
        """
        # Arrange: Create a config where everything is enabled
        config = OTelConfig(
            enable_gpu_metrics=True,
            enable_mcp_instrumentation=True,
            enabled_instrumentors=["openai", "anthropic"],
        )

        # Act
        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            setup_auto_instrumentation(config)

            # Assert
            # 1. Check that OpenTelemetry was configured
            mock_configure_otel.assert_called_once_with(config)

            # 2. Check that the correct LLM instrumentors were initialized and called
            mock_openai = MOCK_INSTRUMENTORS["openai"]
            mock_anthropic = MOCK_INSTRUMENTORS["anthropic"]
            mock_google = MOCK_INSTRUMENTORS["google"]

            mock_openai.return_value.instrument.assert_called_once_with(config=config)
            mock_anthropic.return_value.instrument.assert_called_once_with(config=config)
            mock_google.assert_not_called()

            # 3. Check that the GPU collector was initialized and started
            mock_gpu_collector.assert_called_once()
            mock_gpu_collector.return_value.start.assert_called_once()

            # 4. Check that the MCP manager was initialized and instrument_all was called
            mock_mcp_manager.assert_called_once_with(config)
            mock_mcp_manager.return_value.instrument_all.assert_called_once_with(
                config.fail_on_error
            )

            # 5. Check log messages
            mock_logger.info.assert_any_call("Starting auto-instrumentation setup...")
            mock_logger.info.assert_any_call("openai instrumentation enabled")
            mock_logger.info.assert_any_call("anthropic instrumentation enabled")
            mock_logger.info.assert_any_call("MCP tools instrumentation enabled and set up.")
            mock_logger.info.assert_any_call("GPU metrics collection started.")
            mock_logger.info.assert_any_call("Auto-instrumentation setup complete")

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_default_config(self, mock_configure_otel):
        """Test setup_auto_instrumentation with default config (no specific instrumentors, no GPU, no MCP)."""
        # Create config with everything explicitly disabled
        config = OTelConfig(
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.MCPInstrumentorManager") as mock_mcp_manager:
                with patch("genai_otel.auto_instrument.GPUMetricsCollector") as mock_gpu_collector:
                    setup_auto_instrumentation(config)

                    mock_configure_otel.assert_called_once_with(config)
                    mock_mcp_manager.assert_not_called()
                    mock_gpu_collector.assert_not_called()
                    mock_logger.info.assert_any_call("Starting auto-instrumentation setup...")
                    mock_logger.info.assert_any_call("Auto-instrumentation setup complete")

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_with_llm_instrumentors(self, mock_configure_otel):
        """Test setup_auto_instrumentation with specific LLM instrumentors enabled."""
        config = OTelConfig(
            enabled_instrumentors=["openai", "anthropic", "cohere"],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            setup_auto_instrumentation(config)

            # Check that only the enabled instrumentors were called
            MOCK_INSTRUMENTORS["openai"].return_value.instrument.assert_called_once_with(config=config)
            MOCK_INSTRUMENTORS["anthropic"].return_value.instrument.assert_called_once_with(config=config)
            MOCK_INSTRUMENTORS["cohere"].return_value.instrument.assert_called_once_with(config=config)
            MOCK_INSTRUMENTORS["google"].assert_not_called()
            MOCK_INSTRUMENTORS["mistralai"].assert_not_called()

            # Check log messages
            mock_logger.info.assert_any_call("openai instrumentation enabled")
            mock_logger.info.assert_any_call("anthropic instrumentation enabled")
            mock_logger.info.assert_any_call("cohere instrumentation enabled")

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_with_mcp_instrumentation(self, mock_configure_otel):
        """Test setup_auto_instrumentation with MCP instrumentation enabled."""
        config = OTelConfig(
            enable_mcp_instrumentation=True,
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.MCPInstrumentorManager") as mock_mcp_manager:
                mock_mcp_instance = MagicMock()
                mock_mcp_manager.return_value = mock_mcp_instance

                setup_auto_instrumentation(config)

                mock_mcp_manager.assert_called_once_with(config)
                mock_mcp_instance.instrument_all.assert_called_once_with(config.fail_on_error)
                mock_logger.info.assert_any_call("MCP tools instrumentation enabled and set up.")

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_with_gpu_metrics(self, mock_configure_otel):
        """Test setup_auto_instrumentation with GPU metrics enabled."""
        config = OTelConfig(
            enable_gpu_metrics=True,
            enabled_instrumentors=[],
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                with patch("genai_otel.auto_instrument.GPUMetricsCollector") as mock_gpu_collector:
                    # Mock the meter provider chain
                    mock_meter_provider = MagicMock()
                    mock_meter = MagicMock()
                    mock_metrics.get_meter_provider.return_value = mock_meter_provider
                    mock_meter_provider.get_meter.return_value = mock_meter
                    mock_gpu_instance = MagicMock()
                    mock_gpu_collector.return_value = mock_gpu_instance

                    setup_auto_instrumentation(config)

                    mock_metrics.get_meter_provider.assert_called_once()
                    mock_meter_provider.get_meter.assert_called_once_with("genai.gpu")
                    mock_gpu_collector.assert_called_once_with(mock_meter, config)
                    mock_gpu_instance.start.assert_called_once()
                    mock_logger.info.assert_any_call("GPU metrics collection started.")

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_configure_opentelemetry_failure(self, mock_configure_otel):
        """Test setup_auto_instrumentation when configure_opentelemetry fails."""
        mock_configure_otel.side_effect = Exception("OTel config error")
        config = OTelConfig(
            fail_on_error=False,
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            # This should not raise because fail_on_error=False
            setup_auto_instrumentation(config)

            # Should log error about the failure
            mock_logger.error.assert_called_once()
            error_message = mock_logger.error.call_args[0][0]
            assert "Failed to initialize instrumentation: OTel config error" in error_message

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_configure_opentelemetry_failure_with_fail_on_error(
        self, mock_configure_otel
    ):
        """Test setup_auto_instrumentation when configure_opentelemetry fails with fail_on_error=True."""
        mock_configure_otel.side_effect = Exception("OTel config error")
        config = OTelConfig(
            fail_on_error=True,
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with pytest.raises(Exception, match="OTel config error"):
                setup_auto_instrumentation(config)

            # Should log error before raising
            mock_logger.error.assert_called_once()

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_llm_instrumentor_failure_no_fail_on_error(
        self, mock_configure_otel
    ):
        """Test LLM instrumentor failure when fail_on_error is False."""
        # Make one instrumentor fail
        mock_failing_instrumentor = MagicMock()
        mock_failing_instrumentor.instrument.side_effect = Exception("LLM instrumentor error")
        MOCK_INSTRUMENTORS["openai"].return_value = mock_failing_instrumentor

        config = OTelConfig(
            enabled_instrumentors=["openai", "anthropic"],
            fail_on_error=False,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            setup_auto_instrumentation(config)

            # Both instrumentors should be attempted, but only one fails
            mock_failing_instrumentor.instrument.assert_called_once_with(config=config)
            MOCK_INSTRUMENTORS["anthropic"].return_value.instrument.assert_called_once_with(config=config)

            # Error should be logged but not raised
            mock_logger.error.assert_called_once()
            assert "Failed to instrument openai: LLM instrumentor error" in mock_logger.error.call_args[0][0]

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_llm_instrumentor_failure_with_fail_on_error(
        self, mock_configure_otel
    ):
        """Test LLM instrumentor failure when fail_on_error is True."""
        # Make one instrumentor fail with InstrumentationError
        mock_failing_instrumentor = MagicMock()
        mock_failing_instrumentor.instrument.side_effect = InstrumentationError("LLM instrumentor error")
        MOCK_INSTRUMENTORS["openai"].return_value = mock_failing_instrumentor

        config = OTelConfig(
            enabled_instrumentors=["openai"],
            fail_on_error=True,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with pytest.raises(InstrumentationError, match="LLM instrumentor error"):
                setup_auto_instrumentation(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_mcp_instrumentation_failure_no_fail_on_error(
        self, mock_configure_otel
    ):
        """Test MCP instrumentation failure when fail_on_error is False."""
        config = OTelConfig(
            enable_mcp_instrumentation=True,
            fail_on_error=False,
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.MCPInstrumentorManager") as mock_mcp_manager:
                mock_mcp_instance = MagicMock()
                mock_mcp_manager.return_value = mock_mcp_instance
                mock_mcp_instance.instrument_all.side_effect = Exception("MCP error")

                setup_auto_instrumentation(config)

                mock_logger.error.assert_called_once()
                error_message = mock_logger.error.call_args[0][0]
                assert "Failed to set up MCP tools instrumentation: MCP error" in error_message

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_mcp_instrumentation_failure_with_fail_on_error(
        self, mock_configure_otel
    ):
        """Test MCP instrumentation failure when fail_on_error is True."""
        config = OTelConfig(
            enable_mcp_instrumentation=True,
            fail_on_error=True,
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.MCPInstrumentorManager") as mock_mcp_manager:
                mock_mcp_instance = MagicMock()
                mock_mcp_manager.return_value = mock_mcp_instance
                mock_mcp_instance.instrument_all.side_effect = InstrumentationError("MCP error")

                with pytest.raises(InstrumentationError, match="MCP error"):
                    setup_auto_instrumentation(config)

                mock_logger.error.assert_called_once()

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_gpu_metrics_failure_no_fail_on_error(
        self, mock_configure_otel
    ):
        """Test GPU metrics failure when fail_on_error is False."""
        config = OTelConfig(
            enable_gpu_metrics=True,
            fail_on_error=False,
            enabled_instrumentors=[],
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                mock_metrics.get_meter_provider.side_effect = Exception("GPU error")

                setup_auto_instrumentation(config)

                mock_logger.error.assert_called_once()
                error_message = mock_logger.error.call_args[0][0]
                assert "Failed to start GPU metrics collection: GPU error" in error_message

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_gpu_metrics_failure_with_fail_on_error(
        self, mock_configure_otel
    ):
        """Test GPU metrics failure when fail_on_error is True."""
        config = OTelConfig(
            enable_gpu_metrics=True,
            fail_on_error=True,
            enabled_instrumentors=[],
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                mock_metrics.get_meter_provider.side_effect = InstrumentationError("GPU error")

                with pytest.raises(InstrumentationError, match="GPU error"):
                    setup_auto_instrumentation(config)

                mock_logger.error.assert_called_once()

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_unknown_instrumentor(self, mock_configure_otel):
        """Test that a warning is logged for an unknown instrumentor."""
        config = OTelConfig(
            enabled_instrumentors=["unknown_llm", "openai"],
            fail_on_error=False,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            setup_auto_instrumentation(config)

            # Should still call the known instrumentor
            MOCK_INSTRUMENTORS["openai"].return_value.instrument.assert_called_once_with(config=config)

            # Should log warning for unknown instrumentor
            mock_logger.warning.assert_called_once_with("Unknown instrumentor 'unknown_llm' requested.")

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_multiple_unknown_instrumentors(self, mock_configure_otel):
        """Test that warnings are logged for multiple unknown instrumentors."""
        config = OTelConfig(
            enabled_instrumentors=["unknown1", "unknown2", "openai"],
            fail_on_error=False,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            setup_auto_instrumentation(config)

            # Should still call the known instrumentor
            MOCK_INSTRUMENTORS["openai"].return_value.instrument.assert_called_once_with(config=config)

            # Should log warnings for both unknown instrumentors
            warning_calls = [
                call("Unknown instrumentor 'unknown1' requested."),
                call("Unknown instrumentor 'unknown2' requested."),
            ]
            mock_logger.warning.assert_has_calls(warning_calls, any_order=True)
            assert mock_logger.warning.call_count == 2

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_no_instrumentors(self, mock_configure_otel):
        """Test setup with no instrumentors enabled."""
        config = OTelConfig(
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.MCPInstrumentorManager") as mock_mcp_manager:
                with patch("genai_otel.auto_instrument.GPUMetricsCollector") as mock_gpu_collector:
                    setup_auto_instrumentation(config)

                    # Only OpenTelemetry should be configured
                    mock_configure_otel.assert_called_once_with(config)
                    mock_mcp_manager.assert_not_called()
                    mock_gpu_collector.assert_not_called()
                    mock_logger.info.assert_any_call("Auto-instrumentation setup complete")

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_all_components_disabled(self, mock_configure_otel):
        """Test setup with all components explicitly disabled."""
        config = OTelConfig(
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.MCPInstrumentorManager") as mock_mcp_manager:
                with patch("genai_otel.auto_instrument.GPUMetricsCollector") as mock_gpu_collector:
                    setup_auto_instrumentation(config)

                    # Only OpenTelemetry should be configured
                    mock_configure_otel.assert_called_once_with(config)
                    mock_mcp_manager.assert_not_called()
                    mock_gpu_collector.assert_not_called()

                    # No instrumentors should be called
                    for instrumentor in MOCK_INSTRUMENTORS.values():
                        instrumentor.assert_not_called()

                    mock_logger.info.assert_any_call("Auto-instrumentation setup complete")


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_with_none_config(self, mock_configure_otel):
        """Test that setup handles None config gracefully."""
        with pytest.raises(Exception):
            setup_auto_instrumentation(None)

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", {})
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_with_empty_instrumentors_dict(self, mock_configure_otel):
        """Test setup when INSTRUMENTORS dict is empty."""
        config = OTelConfig(
            enabled_instrumentors=["openai"],
            fail_on_error=False,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            setup_auto_instrumentation(config)

            # Should log warning for unknown instrumentor since INSTRUMENTORS is empty
            mock_logger.warning.assert_called_once_with("Unknown instrumentor 'openai' requested.")

    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_with_gpu_metrics_but_no_meter_provider(self, mock_configure_otel):
        """Test GPU metrics when meter provider is not available."""
        config = OTelConfig(
            enable_gpu_metrics=True,
            enabled_instrumentors=[],
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                # Simulate no meter provider available
                mock_metrics.get_meter_provider.return_value = None

                setup_auto_instrumentation(config)

                # Should log error
                mock_logger.error.assert_called_once()
                error_message = mock_logger.error.call_args[0][0]
                assert "Failed to start GPU metrics collection" in error_message

    @patch("genai_otel.auto_instrument.INSTRUMENTORS", MOCK_INSTRUMENTORS)
    @patch("genai_otel.auto_instrument.configure_opentelemetry")
    def test_setup_auto_instrumentation_empty_instrumentor_list(self, mock_configure_otel):
        """Test setup with empty instrumentor list."""
        config = OTelConfig(
            enabled_instrumentors=[],
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )

        with patch("genai_otel.auto_instrument.logger") as mock_logger:
            setup_auto_instrumentation(config)

            # No instrumentors should be called
            for instrumentor in MOCK_INSTRUMENTORS.values():
                instrumentor.assert_not_called()

            mock_logger.info.assert_any_call("Auto-instrumentation setup complete")