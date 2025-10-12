from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig

# Assuming BaseInstrumentor is imported from here:
from genai_otel.instrumentors.base import BaseInstrumentor

# --- ConcreteInstrumentor (Helper Class for Testing) ---


class ConcreteInstrumentor(BaseInstrumentor):
    """A concrete implementation of BaseInstrumentor for testing."""

    def instrument(self, config):
        self._instrumented = True

    def _extract_usage(self, result):
        return result.get("usage")

    # Note: We won't use this method directly in the test;
    # instead, we'll pass a lambda to avoid signature issues.
    def _extract_attributes(self, *args, **kwargs):
        return {"test.attribute": "test_value"}


# --- Fixture ---


@pytest.fixture
def instrumentor():
    """Fixture to provide a clean instrumentor instance for each test."""
    # Create fresh mocks
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_span_ctx = MagicMock()
    mock_span_ctx.__enter__.return_value = mock_span
    mock_span_ctx.__exit__.return_value = None
    mock_tracer.start_as_current_span.return_value = mock_span_ctx

    mock_cost_calculator = MagicMock()
    mock_cost_calculator.calculate_cost.return_value = 0.01

    # Patch OpenTelemetry's get_tracer during instrumentor creation
    with patch("genai_otel.instrumentors.base.trace.get_tracer", return_value=mock_tracer):
        inst = ConcreteInstrumentor()
        inst.tracer = mock_tracer
        inst.cost_calculator = mock_cost_calculator
        inst._ensure_shared_metrics_created()

        # Instrument with default config
        config = OTelConfig()
        inst.instrument(config)

        yield inst


# --- Test Function (Your original, correct test) ---


def test_create_span_wrapper_creates_span(instrumentor):
    """Verify that the wrapper creates a span with correct attributes and calls the original function."""
    original_function = MagicMock(return_value={"usage": None})

    # Setting enable_cost_tracking=False is what triggers the bug in your implementation
    config = OTelConfig(enable_cost_tracking=False)
    instrumentor.config = config

    # Use a lambda to provide attributes — avoids method signature issues
    wrapped_function = instrumentor.create_span_wrapper(
        span_name="test.span",
        extract_attributes=lambda *args, **kwargs: {"test.attribute": "test_value"},
    )(original_function)

    # Act
    result = wrapped_function("arg1", kwarg1="kwarg_value")

    ## Assert - This is the assertion that currently fails due to the missing 'attributes'
    instrumentor.tracer.start_as_current_span.assert_called_once_with(
        "test.span", attributes={"test.attribute": "test_value"}
    )
    original_function.assert_called_once_with("arg1", kwarg1="kwarg_value")
    assert result == {"usage": None}

    # Verify span context management
    mock_span_ctx = instrumentor.tracer.start_as_current_span.return_value
    mock_span_ctx.__enter__.assert_called_once()
    mock_span_ctx.__exit__.assert_called_once()
