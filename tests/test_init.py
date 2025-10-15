import os
from unittest.mock import MagicMock, patch

import pytest

from genai_otel import instrument
from genai_otel.auto_instrument import setup_auto_instrumentation
from genai_otel.config import OTelConfig


def test_instrument_success():
    """Test successful instrumentation initialization."""
    # Mock the actual implementation paths, not the re-exported names
    with patch("genai_otel.OTelConfig") as mock_otel_config:
        with patch("genai_otel.setup_auto_instrumentation") as mock_setup_auto_instrumentation:
            with patch("genai_otel.logger") as mock_logger:
                mock_config_instance = MagicMock(spec=OTelConfig)
                mock_otel_config.return_value = mock_config_instance
                # Mock the config attributes that setup_auto_instrumentation uses
                mock_config_instance.service_name = "test-service"
                mock_config_instance.endpoint = "http://test-endpoint"
                mock_config_instance.headers = {}

                instrument()

                mock_otel_config.assert_called_once_with()
                mock_setup_auto_instrumentation.assert_called_once_with(mock_config_instance)
                mock_logger.info.assert_called_once_with(
                    "GenAI OpenTelemetry instrumentation initialized successfully"
                )


def test_instrument_with_kwargs():
    """Test instrumentation with custom keyword arguments."""
    with patch("genai_otel.OTelConfig") as mock_otel_config:
        with patch("genai_otel.setup_auto_instrumentation") as mock_setup_auto_instrumentation:
            with patch("genai_otel.logger") as mock_logger:
                mock_config_instance = MagicMock(spec=OTelConfig)
                mock_otel_config.return_value = mock_config_instance
                # Mock the config attributes that setup_auto_instrumentation uses
                mock_config_instance.service_name = "my-test-app"
                mock_config_instance.endpoint = "http://localhost:8080"
                mock_config_instance.headers = {}

                instrument(service_name="my-test-app", endpoint="http://localhost:8080")

                mock_otel_config.assert_called_once_with(
                    service_name="my-test-app", endpoint="http://localhost:8080"
                )
                mock_setup_auto_instrumentation.assert_called_once_with(mock_config_instance)
                mock_logger.info.assert_called_once_with(
                    "GenAI OpenTelemetry instrumentation initialized successfully"
                )


def test_instrument_failure_no_fail_on_error():
    """Test instrumentation failure when fail_on_error is False (default)."""
    with patch("genai_otel.OTelConfig") as mock_otel_config:
        with patch("genai_otel.setup_auto_instrumentation") as mock_setup_auto_instrumentation:
            with patch("genai_otel.logger") as mock_logger:
                mock_config_instance = MagicMock(spec=OTelConfig)
                mock_otel_config.return_value = mock_config_instance
                mock_setup_auto_instrumentation.side_effect = Exception("Test error")
                # Still need to mock config attributes even if setup_auto_instrumentation fails
                mock_config_instance.service_name = "test-service"
                mock_config_instance.endpoint = "http://test-endpoint"
                mock_config_instance.headers = {}

                # Should not raise when fail_on_error is False (default)
                instrument()

                mock_logger.error.assert_called_once()
                # Check the actual call arguments - the error message is the first positional argument
                error_call_args = mock_logger.error.call_args[0]
                assert len(error_call_args) >= 1
                assert "Failed to initialize instrumentation" in error_call_args[0]
                # Check that exc_info is passed as a keyword argument
                call_kwargs = mock_logger.error.call_args[1]
                assert call_kwargs.get("exc_info") is True


def test_instrument_failure_with_fail_on_error_kwarg():
    """Test instrumentation failure when fail_on_error is True via kwarg."""
    with patch("genai_otel.OTelConfig") as mock_otel_config:
        with patch("genai_otel.setup_auto_instrumentation") as mock_setup_auto_instrumentation:
            with patch("genai_otel.logger") as mock_logger:
                mock_config_instance = MagicMock(spec=OTelConfig)
                mock_otel_config.return_value = mock_config_instance
                mock_setup_auto_instrumentation.side_effect = Exception("Test error")
                # Still need to mock config attributes even if setup_auto_instrumentation fails
                mock_config_instance.service_name = "test-service"
                mock_config_instance.endpoint = "http://test-endpoint"
                mock_config_instance.headers = {}

                with pytest.raises(Exception, match="Test error"):
                    instrument(fail_on_error=True)

                mock_logger.error.assert_called_once()
                # Check the actual call arguments - the error message is the first positional argument
                error_call_args = mock_logger.error.call_args[0]
                assert len(error_call_args) >= 1
                assert "Failed to initialize instrumentation" in error_call_args[0]
                # Check that exc_info is passed as a keyword argument
                call_kwargs = mock_logger.error.call_args[1]
                assert call_kwargs.get("exc_info") is True


def test_instrument_failure_with_fail_on_error_env_var(monkeypatch):
    """Test instrumentation failure when GENAI_FAIL_ON_ERROR env var is True."""
    monkeypatch.setenv("GENAI_FAIL_ON_ERROR", "true")

    with patch("genai_otel.OTelConfig") as mock_otel_config:
        with patch("genai_otel.setup_auto_instrumentation") as mock_setup_auto_instrumentation:
            with patch("genai_otel.logger") as mock_logger:
                mock_config_instance = MagicMock(spec=OTelConfig)
                mock_otel_config.return_value = mock_config_instance
                mock_setup_auto_instrumentation.side_effect = Exception("Test error")
                # Still need to mock config attributes even if setup_auto_instrumentation fails
                mock_config_instance.service_name = "test-service"
                mock_config_instance.endpoint = "http://test-endpoint"
                mock_config_instance.headers = {}

                with pytest.raises(Exception, match="Test error"):
                    instrument()

                mock_logger.error.assert_called_once()
                # Check the actual call arguments - the error message is the first positional argument
                error_call_args = mock_logger.error.call_args[0]
                assert len(error_call_args) >= 1
                assert "Failed to initialize instrumentation" in error_call_args[0]
                # Check that exc_info is passed as a keyword argument
                call_kwargs = mock_logger.error.call_args[1]
                assert call_kwargs.get("exc_info") is True


def test_instrument_kwargs_override_env_vars(monkeypatch):
    """Test that kwargs take precedence over environment variables."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-endpoint")

    with patch("genai_otel.OTelConfig") as mock_otel_config:
        with patch("genai_otel.setup_auto_instrumentation"):
            with patch("genai_otel.logger"):
                mock_config_instance = MagicMock(spec=OTelConfig)
                mock_otel_config.return_value = mock_config_instance
                # Mock the config attributes that setup_auto_instrumentation uses
                mock_config_instance.service_name = "kwarg-service"
                mock_config_instance.endpoint = "http://kwarg-endpoint"
                mock_config_instance.headers = {}

                # Call with kwargs that should override env vars
                instrument(service_name="kwarg-service", endpoint="http://kwarg-endpoint")

                # Verify OTelConfig was called with kwargs (overriding env vars)
                mock_otel_config.assert_called_once_with(
                    service_name="kwarg-service", endpoint="http://kwarg-endpoint"
                )


def test_instrument_metrics_setup_failure():
    """Test instrumentation when metrics setup fails but doesn't break everything."""
    with patch("genai_otel.OTelConfig") as mock_otel_config:
        with patch("genai_otel.setup_auto_instrumentation") as mock_setup_auto_instrumentation:
            with patch("genai_otel.logger") as mock_logger:
                mock_config_instance = MagicMock(spec=OTelConfig)
                mock_otel_config.return_value = mock_config_instance
                # Simulate a failure during metrics setup within setup_auto_instrumentation
                mock_setup_auto_instrumentation.side_effect = Exception("Metrics setup error")
                # Mock the config attributes
                mock_config_instance.service_name = "test-service"
                mock_config_instance.endpoint = "http://test-endpoint"
                mock_config_instance.headers = {}

                instrument()

                mock_otel_config.assert_called_once_with()
                mock_setup_auto_instrumentation.assert_called_once_with(mock_config_instance)
                # Check that the overall instrumentation failure was logged
                mock_logger.error.assert_called_once()
                error_call_args = mock_logger.error.call_args[0]
                assert len(error_call_args) >= 1
                assert "Failed to initialize instrumentation" in error_call_args[0]
                call_kwargs = mock_logger.error.call_args[1]
                assert call_kwargs.get("exc_info") is True


def test_top_level_exports():
    """Test that key components are correctly re-exported at the top level."""
    from genai_otel import (
        AnthropicInstrumentor,
        CostCalculator,
        GPUMetricsCollector,
        MCPInstrumentorManager,
        OpenAIInstrumentor,
        OTelConfig,
        __author__,
        __email__,
        __license__,
        __version__,
        setup_auto_instrumentation,
    )

    # Basic assertions to ensure they are imported and not just placeholders
    assert isinstance(__version__, str)
    assert isinstance(__author__, str)
    assert isinstance(__email__, str)
    assert isinstance(__license__, str)
    assert callable(setup_auto_instrumentation)
    assert OTelConfig is not None
    assert CostCalculator is not None
    assert GPUMetricsCollector is not None
    assert OpenAIInstrumentor is not None
    assert AnthropicInstrumentor is not None
    assert MCPInstrumentorManager is not None
