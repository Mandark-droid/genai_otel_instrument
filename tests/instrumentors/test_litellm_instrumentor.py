# tests/instrumentors/test_litellm_instrumentor.py
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig

# Check if openinference is available
try:
    import openinference.instrumentation.litellm

    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False


# Mock the actual openinference instrumentor
@pytest.mark.skipif(
    not OPENINFERENCE_AVAILABLE, reason="openinference-instrumentation-litellm not installed"
)
@patch("openinference.instrumentation.litellm.LiteLLMInstrumentor")
def test_litellm_instrumentor_integration(MockLiteLLMInstrumentor):
    """
    Test that LiteLLMInstrumentor is correctly integrated and its instrument method is called.
    """
    mock_instrumentor_instance = MockLiteLLMInstrumentor.return_value
    mock_instrumentor_instance.instrument.return_value = None

    config = OTelConfig(service_name="test-litellm", enabled_instrumentors=["litellm"])

    from genai_otel.auto_instrument import INSTRUMENTORS, setup_auto_instrumentation

    with patch.dict(INSTRUMENTORS, {"litellm": MockLiteLLMInstrumentor}, clear=True):
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
        ):

            setup_auto_instrumentation(config)

    # OpenInference instrumentors don't take config parameter (see auto_instrument.py:208-211)
    mock_instrumentor_instance.instrument.assert_called_once_with()
