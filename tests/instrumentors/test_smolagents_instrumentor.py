# tests/instrumentors/test_smolagents_instrumentor.py
from unittest.mock import MagicMock, patch

import pytest

# Assuming OTelConfig is available at genai_otel.config
from genai_otel.config import OTelConfig

# Check if openinference is available
try:
    import openinference.instrumentation.smolagents

    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False

# Assuming the actual instrumentor class is from openinference
# We will mock this class and test genai_otel's integration logic.


# Mock the actual openinference instrumentor
@pytest.mark.skipif(
    not OPENINFERENCE_AVAILABLE, reason="openinference-instrumentation-smolagents not installed"
)
@patch("openinference.instrumentation.smolagents.SmolagentsInstrumentor")
def test_smolagents_instrumentor_integration(MockSmolagentsInstrumentor):
    """
    Test that SmolagentsInstrumentor is correctly integrated and its instrument method is called.
    """
    # Mock the instrument method of the instrumentor instance
    mock_instrumentor_instance = MockSmolagentsInstrumentor.return_value
    mock_instrumentor_instance.instrument.return_value = None

    # Create a dummy config that enables smolagents
    config = OTelConfig(service_name="test-smolagents", enabled_instrumentors=["smolagents"])

    # Import the function that orchestrates instrumentation
    from genai_otel.auto_instrument import INSTRUMENTORS, setup_auto_instrumentation

    # Patch the global INSTRUMENTORS dictionary to isolate the test
    # and ensure only our mocked smolagents instrumentor is considered.
    with patch.dict(INSTRUMENTORS, {"smolagents": MockSmolagentsInstrumentor}, clear=True):
        # Mock other dependencies of setup_auto_instrumentation to avoid actual initialization
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
        ):  # Mock the MCP manager as well

            setup_auto_instrumentation(config)

    # OpenInference instrumentors don't take config parameter (see auto_instrument.py:208-211)
    mock_instrumentor_instance.instrument.assert_called_once_with()
