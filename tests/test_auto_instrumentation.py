import logging
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.auto_instrument import INSTRUMENTORS, setup_auto_instrumentation
from genai_otel.config import OTelConfig
from genai_otel.exceptions import InstrumentationError
from genai_otel.mcp_instrumentors import MCPInstrumentorManager


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Fixture to mock external dependencies for auto_instrumentation tests."""
        with patch("genai_otel.otel_setup.configure_opentelemetry") as mock_configure_opentelemetry:
            with patch("genai_otel.auto_instrument.logger") as mock_logger:
                with patch(
                    "genai_otel.auto_instrument.MCPInstrumentorManager"
                ) as MockMCPInstrumentorManager:
                    with patch("genai_otel.auto_instrument.metrics") as mock_metrics:
                        with patch(
                            "genai_otel.auto_instrument.GPUMetricsCollector"
                        ) as MockGPUMetricsCollector:
    
                            # Mock MCPInstrumentorManager instance and its methods
                            mock_mcp_manager_instance = MagicMock(spec=MCPInstrumentorManager)
                            MockMCPInstrumentorManager.return_value = mock_mcp_manager_instance
    
                            # Mock GPU metrics collector instance and its methods
                            mock_gpu_collector_instance = MagicMock()
                            MockGPUMetricsCollector.return_value = mock_gpu_collector_instance
    
                            yield {
                                "mock_configure_opentelemetry": mock_configure_opentelemetry,
                                "mock_logger": mock_logger,
                                "MockMCPInstrumentorManager": MockMCPInstrumentorManager,
                                "mock_mcp_manager_instance": mock_mcp_manager_instance,
                                "mock_metrics": mock_metrics,
                                "MockGPUMetricsCollector": MockGPUMetricsCollector,
                                "mock_gpu_collector_instance": mock_gpu_collector_instance,
                            }

def test_setup_auto_instrumentation_default_config(mock_dependencies):
    """Test setup_auto_instrumentation with default config (no specific instrumentors, no GPU, no MCP)."""
    config = OTelConfig(enabled_instrumentors=[])
    setup_auto_instrumentation(config)

    mock_dependencies["mock_configure_opentelemetry"].assert_called_once_with(config)
    mock_dependencies["mock_logger"].info.assert_any_call(
        "Starting auto-instrumentation setup..."
    )
    mock_dependencies["mock_logger"].info.assert_any_call("Auto-instrumentation setup complete")
    mock_dependencies["MockMCPInstrumentorManager"].assert_not_called()
    mock_dependencies["MockGPUMetricsCollector"].assert_not_called()


def test_setup_auto_instrumentation_with_llm_instrumentors(mock_dependencies):
    """Test setup_auto_instrumentation with specific LLM instrumentors enabled."""
    mock_openai_instrumentor = MagicMock()
    mock_anthropic_instrumentor = MagicMock()

    with patch.dict(
        INSTRUMENTORS,
        {
            "openai": MagicMock(return_value=mock_openai_instrumentor),
            "anthropic": MagicMock(return_value=mock_anthropic_instrumentor),
        },
    ):
        config = OTelConfig(enabled_instrumentors=["openai", "anthropic"])
        setup_auto_instrumentation(config)

        mock_openai_instrumentor.instrument.assert_called_once_with(config=config)
        mock_anthropic_instrumentor.instrument.assert_called_once_with(config=config)
        mock_dependencies["mock_logger"].info.assert_any_call("openai instrumentation enabled")
        mock_dependencies["mock_logger"].info.assert_any_call("anthropic instrumentation enabled")


def test_setup_auto_instrumentation_with_mcp_instrumentation(mock_dependencies):
    """Test setup_auto_instrumentation with MCP instrumentation enabled."""
    config = OTelConfig(enable_mcp_instrumentation=True, enabled_instrumentors=[])
    setup_auto_instrumentation(config)

    mock_dependencies["MockMCPInstrumentorManager"].assert_called_once_with(config)
    mock_dependencies["mock_mcp_manager_instance"].instrument_all.assert_called_once_with(
        config.fail_on_error
    )
    mock_dependencies["mock_logger"].info.assert_any_call(
        "MCP tools instrumentation enabled and set up."
    )


def test_setup_auto_instrumentation_with_gpu_metrics(mock_dependencies):
    """Test setup_auto_instrumentation with GPU metrics enabled."""
    config = OTelConfig(enable_gpu_metrics=True, enabled_instrumentors=[])
    setup_auto_instrumentation(config)

    mock_dependencies["mock_metrics"].get_meter_provider.assert_called_once()
    mock_dependencies[
        "mock_metrics"
    ].get_meter_provider.return_value.get_meter.assert_called_once_with("genai.gpu")
    mock_dependencies["MockGPUMetricsCollector"].assert_called_once_with(
        mock_dependencies["mock_metrics"].get_meter_provider.return_value.get_meter.return_value,
        config,
    )
    mock_dependencies["mock_gpu_collector_instance"].start.assert_called_once()
    mock_dependencies["mock_logger"].info.assert_any_call("GPU metrics collection started.")


def test_setup_auto_instrumentation_configure_opentelemetry_failure(mock_dependencies):
    """Test setup_auto_instrumentation when configure_opentelemetry fails."""
    with patch("genai_otel.otel_setup.configure_opentelemetry") as mock_configure_opentelemetry:
        mock_configure_opentelemetry.side_effect = Exception("OTel config error")
        config = OTelConfig(fail_on_error=False, enabled_instrumentors=[])

        setup_auto_instrumentation(config)

        mock_dependencies["mock_logger"].error.assert_called_once()
        assert (
            "Failed to initialize instrumentation: OTel config error"
            in mock_dependencies["mock_logger"].error.call_args[0][0]
        )


def test_setup_auto_instrumentation_llm_instrumentor_failure_no_fail_on_error(mock_dependencies):
    """Test LLM instrumentor failure when fail_on_error is False."""
    mock_failing_instrumentor = MagicMock()
    mock_failing_instrumentor.instrument.side_effect = Exception("LLM instrumentor error")

    with patch.dict(
        INSTRUMENTORS, {"failing_llm": MagicMock(return_value=mock_failing_instrumentor)}
    ):
        config = OTelConfig(enabled_instrumentors=["failing_llm"], fail_on_error=False)
        setup_auto_instrumentation(config)

        mock_dependencies["mock_logger"].error.assert_called_once()
        assert (
            "Failed to instrument failing_llm: LLM instrumentor error"
            in mock_dependencies["mock_logger"].error.call_args[0][0]
        )


def test_setup_auto_instrumentation_llm_instrumentor_failure_with_fail_on_error(mock_dependencies):
    """Test LLM instrumentor failure when fail_on_error is True."""
    mock_failing_instrumentor = MagicMock()
    mock_failing_instrumentor.instrument.side_effect = InstrumentationError(
        "LLM instrumentor error"
    )

    with patch.dict(
        INSTRUMENTORS, {"failing_llm": MagicMock(return_value=mock_failing_instrumentor)}
    ):
        config = OTelConfig(enabled_instrumentors=["failing_llm"], fail_on_error=True)

        with pytest.raises(InstrumentationError, match="LLM instrumentor error"):
            setup_auto_instrumentation(config)

        mock_dependencies["mock_logger"].error.assert_called_once()


def test_setup_auto_instrumentation_mcp_instrumentation_failure_no_fail_on_error(mock_dependencies):
    """Test MCP instrumentation failure when fail_on_error is False."""
    mock_dependencies["mock_mcp_manager_instance"].instrument_all.side_effect = Exception(
        "MCP error"
    )
    config = OTelConfig(
        enable_mcp_instrumentation=True, fail_on_error=False, enabled_instrumentors=[]
    )

    setup_auto_instrumentation(config)

    mock_dependencies["mock_logger"].error.assert_called_once()
    assert (
        "Failed to set up MCP tools instrumentation: {e}"
        in mock_dependencies["mock_logger"].error.call_args[0][0]
    )


def test_setup_auto_instrumentation_mcp_instrumentation_failure_with_fail_on_error(
    mock_dependencies,
):
    """Test MCP instrumentation failure when fail_on_error is True."""
    mock_dependencies["mock_mcp_manager_instance"].instrument_all.side_effect = (
        InstrumentationError("MCP error")
    )
    config = OTelConfig(
        enable_mcp_instrumentation=True, fail_on_error=True, enabled_instrumentors=[]
    )

    with pytest.raises(InstrumentationError, match="MCP error"):
        setup_auto_instrumentation(config)

    mock_dependencies["mock_logger"].error.assert_called_once()


def test_setup_auto_instrumentation_gpu_metrics_failure_no_fail_on_error(mock_dependencies):
    """Test GPU metrics failure when fail_on_error is False."""
    mock_dependencies["mock_metrics"].get_meter_provider.side_effect = Exception("GPU error")
    config = OTelConfig(enable_gpu_metrics=True, fail_on_error=False, enabled_instrumentors=[])

    setup_auto_instrumentation(config)

    mock_dependencies["mock_logger"].error.assert_called_once()
    assert (
        "Failed to start GPU metrics collection: GPU error"
        in mock_dependencies["mock_logger"].error.call_args[0][0]
    )


def test_setup_auto_instrumentation_gpu_metrics_failure_with_fail_on_error(mock_dependencies):
    """Test GPU metrics failure when fail_on_error is True."""
    mock_dependencies["mock_metrics"].get_meter_provider.side_effect = InstrumentationError(
        "GPU error"
    )
    config = OTelConfig(enable_gpu_metrics=True, fail_on_error=True, enabled_instrumentors=[])

    with pytest.raises(InstrumentationError, match="GPU error"):
        setup_auto_instrumentation(config)

    mock_dependencies["mock_logger"].error.assert_called_once()


def test_setup_auto_instrumentation_unknown_instrumentor(mock_dependencies):
    """Test that a warning is logged for an unknown instrumentor."""
    config = OTelConfig(enabled_instrumentors=["unknown_llm"], fail_on_error=False)
    setup_auto_instrumentation(config)

    mock_dependencies["mock_logger"].warning.assert_called_once_with(
        "Unknown instrumentor 'unknown_llm' requested."
    )
