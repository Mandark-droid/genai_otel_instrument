# tests/instrumentors/test_mcp_instrumentor.py
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig

# Check if openinference is available
try:
    import openinference.instrumentation.mcp

    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False


# Mock the actual openinference instrumentor
@pytest.mark.skipif(
    not OPENINFERENCE_AVAILABLE, reason="openinference-instrumentation-mcp not installed"
)
@patch("openinference.instrumentation.mcp.MCPInstrumentor")
def test_mcp_instrumentor_integration(MockMCPInstrumentor):
    """
    Test that MCPInstrumentor is correctly integrated and its instrument method is called.
    Note: This test assumes MCPInstrumentor is used directly via the INSTRUMENTORS dict.
    If MCPInstrumentorManager is the sole mechanism, this test might need adjustment.
    """
    mock_instrumentor_instance = MockMCPInstrumentor.return_value
    mock_instrumentor_instance.instrument.return_value = None

    # For this test, we enable mcp via enabled_instrumentors.
    # We also need to mock MCPInstrumentorManager to prevent it from running if it interferes,
    # as the setup_auto_instrumentation logic might instrument MCP in two ways.
    # We set enable_mcp_instrumentation=False to focus on the INSTRUMENTORS dict path.
    config = OTelConfig(
        service_name="test-mcp", enabled_instrumentors=["mcp"], enable_mcp_instrumentation=False
    )

    from genai_otel.auto_instrument import INSTRUMENTORS, setup_auto_instrumentation

    with patch.dict(INSTRUMENTORS, {"mcp": MockMCPInstrumentor}, clear=True):
        with (
            patch("genai_otel.auto_instrument.OTLPSpanExporter"),
            patch("os.getenv", return_value="10.0"),
            patch("genai_otel.auto_instrument.Resource"),
            patch("genai_otel.auto_instrument.TracerProvider"),
            patch("genai_otel.auto_instrument.BatchSpanProcessor"),
            patch("opentelemetry.trace.propagation.tracecontext.TraceContextTextMapPropagator"),
            patch("genai_otel.auto_instrument.OTLPMetricExporter"),
            patch("genai_otel.auto_instrument.PeriodicExportingMetricReader"),
            patch("genai_otel.auto_instrument.MeterProvider"),
            patch("genai_otel.auto_instrument.GPUMetricsCollector"),
            patch("genai_otel.auto_instrument.MCPInstrumentorManager"),
        ):  # Mock the manager

            setup_auto_instrumentation(config)

    # OpenInference instrumentors don't take config parameter (see auto_instrument.py:208-211)
    mock_instrumentor_instance.instrument.assert_called_once_with()
