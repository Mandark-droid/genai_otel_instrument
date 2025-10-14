import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

import genai_otel.instrumentors.base as base
from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor


# --- ConcreteInstrumentor (Helper Class for Testing) ---
class ConcreteInstrumentor(BaseInstrumentor):
    """A concrete implementation of BaseInstrumentor for testing."""

    def instrument(self, config):
        self._instrumented = True
        self.config = config

    def _extract_usage(self, result):
        return result.get("usage")


# --- Fixtures ---
@pytest.fixture(autouse=True)
def reset_shared_metrics():
    """Reset shared metrics state before/after each test."""
    BaseInstrumentor._shared_request_counter = None
    BaseInstrumentor._shared_token_counter = None
    BaseInstrumentor._shared_latency_histogram = None
    BaseInstrumentor._shared_cost_counter = None
    BaseInstrumentor._shared_error_counter = None
    base._SHARED_METRICS_CREATED = False
    yield


@pytest.fixture
def instrumentor():
    """Fixture to provide a clean instrumentor instance with mocked dependencies."""
    with patch("genai_otel.instrumentors.base.trace.get_tracer") as mock_get_tracer, patch(
        "genai_otel.instrumentors.base.metrics.get_meter"
    ) as mock_get_meter:
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_span = MagicMock()
        mock_span.name = "test.span"  # Set a default name for the span
        # FIX: Mock the attributes.get method of the mock_span
        mock_span.attributes.get.return_value = (
            "test_model"  # This should make it available when _record_result_metrics is called
        )
        mock_span_ctx = MagicMock()
        mock_span_ctx.__enter__.return_value = mock_span
        mock_span_ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_span_ctx
        inst = ConcreteInstrumentor()
        inst.instrument(OTelConfig())

        # Mock dependencies
        inst.tracer = mock_tracer
        inst.cost_calculator = MagicMock()
        inst.cost_calculator.calculate_cost.return_value = 0.01
        inst.request_counter = MagicMock()
        inst.token_counter = MagicMock()
        inst.latency_histogram = MagicMock()
        inst.error_counter = MagicMock()

        yield inst, mock_span, mock_span_ctx


# --- Tests for _ensure_shared_metrics_created ---
def test_ensure_shared_metrics_created_success():
    """Test that shared metrics are created only once."""
    inst = ConcreteInstrumentor()
    assert base._SHARED_METRICS_CREATED is True
    assert inst._shared_request_counter is not None


def test_ensure_shared_metrics_created_thread_safety():
    """Test that shared metrics creation is thread-safe."""

    def create_instrumentor():
        inst = ConcreteInstrumentor()
        inst._ensure_shared_metrics_created()
        return inst

    threads = []
    for _ in range(5):
        t = threading.Thread(target=create_instrumentor)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert base._SHARED_METRICS_CREATED is True


def test_ensure_shared_metrics_created_failure(caplog):
    """Test that shared metrics creation failure is handled gracefully."""
    with patch("genai_otel.instrumentors.base.metrics.get_meter") as mock_get_meter:
        mock_meter_instance = MagicMock()
        mock_get_meter.return_value = mock_meter_instance
        mock_meter_instance.create_counter.side_effect = ValueError("Mock error")
        inst = ConcreteInstrumentor()
        # The _ensure_shared_metrics_created is called in __init__, so we don't need to call it again
        assert inst._shared_request_counter is None
        assert "Failed to create shared metrics" in caplog.text


# --- Tests for create_span_wrapper ---
def test_create_span_wrapper_creates_span(instrumentor):
    """Test that the wrapper creates a span with correct attributes."""
    inst, mock_span, mock_span_ctx = instrumentor
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper(
        span_name="test.span",
        extract_attributes=lambda *args, **kwargs: {"test.attribute": "test_value"},
    )(original_function)

    result = wrapped("arg1", kwarg1="kwarg_value")

    inst.tracer.start_as_current_span.assert_called_once_with(
        "test.span", attributes={"test.attribute": "test_value"}
    )
    original_function.assert_called_once_with("arg1", kwarg1="kwarg_value")
    assert result == {"usage": None}


def test_create_span_wrapper_handles_extract_attributes_error(instrumentor, caplog):
    """Test that the wrapper handles errors in extract_attributes."""
    inst, mock_span, mock_span_ctx = instrumentor
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper(
        span_name="test.span", extract_attributes=lambda *args, **kwargs: 1 / 0  # Force error
    )(original_function)

    result = wrapped("arg1", kwarg1="kwarg_value")

    inst.tracer.start_as_current_span.assert_called_once_with("test.span", attributes={})
    assert "Failed to extract attributes" in caplog.text
    assert result == {"usage": None}


def test_create_span_wrapper_handles_function_error(instrumentor):
    """Test that the wrapper handles errors in the wrapped function."""
    inst, mock_span, mock_span_ctx = instrumentor
    original_function = MagicMock(side_effect=ValueError("Test error"))
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    with pytest.raises(ValueError):
        wrapped()

    assert mock_span.set_status.call_args[0][0].status_code == base.StatusCode.ERROR
    mock_span.record_exception.assert_called_once()


def test_create_span_wrapper_records_metrics(instrumentor):
    """Test that the wrapper records metrics for successful execution."""
    inst, mock_span, mock_span_ctx = instrumentor
    mock_span.attributes.get.return_value = "test_model"
    original_function = MagicMock(
        return_value={"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    )
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped()

    inst.request_counter.add.assert_called_once_with(1, {"operation": "test.span"})
    inst.token_counter.add.assert_has_calls(
        [
            call(10, {"token_type": "prompt", "operation": "test.span"}),
            call(20, {"token_type": "completion", "operation": "test.span"}),
        ]
    )
    inst.cost_counter.add.assert_called_once_with(0.01, {"model": "test_model"})
    inst.latency_histogram.record.assert_called_once()
    assert result == {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def test_create_span_wrapper_records_metrics_without_usage(instrumentor):
    """Test that the wrapper handles missing usage data."""
    inst, mock_span, mock_span_ctx = instrumentor
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped()

    inst.request_counter.add.assert_called_once_with(1, {"operation": "test.span"})
    inst.token_counter.add.assert_not_called()
    inst.cost_counter.add.assert_not_called()
    inst.latency_histogram.record.assert_called_once()
    assert result == {"usage": None}


def test_create_span_wrapper_with_cost_tracking_disabled(instrumentor):
    """Test that cost tracking is skipped when disabled."""
    inst, mock_span, mock_span_ctx = instrumentor
    inst.config.enable_cost_tracking = False
    original_function = MagicMock(
        return_value={"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    )
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped()

    inst.request_counter.add.assert_called_once_with(1, {"operation": "test.span"})
    inst.token_counter.add.assert_has_calls(
        [
            call(10, {"token_type": "prompt", "operation": "test.span"}),
            call(20, {"token_type": "completion", "operation": "test.span"}),
        ]
    )
    inst.cost_counter.add.assert_not_called()
    assert result == {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}


# --- Tests for _record_result_metrics ---
def test_record_result_metrics_success(instrumentor):
    """Test that metrics are recorded correctly for a successful result."""
    inst, mock_span, mock_span_ctx = instrumentor
    mock_span.attributes.get.return_value = "test_model"
    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    inst.token_counter.add.assert_has_calls(
        [
            call(10, {"token_type": "prompt", "operation": "test.span"}),
            call(20, {"token_type": "completion", "operation": "test.span"}),
        ]
    )
    inst.cost_counter.add.assert_called_once_with(0.01, {"model": "test_model"})
    inst.latency_histogram.record.assert_called_once()
    assert mock_span.set_attribute.call_count == 3


def test_record_result_metrics_with_errors(instrumentor, caplog):
    """Test that errors in metric recording are logged but not raised."""
    inst, mock_span, mock_span_ctx = instrumentor
    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}

    inst.token_counter.add.side_effect = ValueError("Mock error")
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    assert "Failed to extract or record usage metrics" in caplog.text


# --- Tests for instrumentation disabled ---
def test_create_span_wrapper_with_instrumentation_disabled(instrumentor):
    """Test that the wrapper bypasses instrumentation when disabled."""
    inst, mock_span, mock_span_ctx = instrumentor
    inst._instrumented = False
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped("arg1", kwarg1="kwarg_value")

    inst.tracer.start_as_current_span.assert_not_called()
    original_function.assert_called_once_with("arg1", kwarg1="kwarg_value")
    assert result == {"usage": None}
