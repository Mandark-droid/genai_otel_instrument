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
def mock_meter():
    with patch("genai_otel.instrumentors.base.metrics.get_meter") as mock_get_meter:
        mock_meter = MagicMock()
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram
        yield mock_meter, mock_counter, mock_histogram


@pytest.fixture
def mock_tracer():
    with patch("genai_otel.instrumentors.base.trace.get_tracer") as mock_get_tracer:
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_span = MagicMock()
        mock_span_ctx = MagicMock()
        mock_span_ctx.__enter__.return_value = mock_span
        mock_span_ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_span_ctx
        yield mock_tracer, mock_span, mock_span_ctx


@pytest.fixture
def instrumentor(mock_meter, mock_tracer):
    """Fixture to provide a clean instrumentor instance for each test."""
    mock_meter_instance, mock_counter, mock_histogram = mock_meter
    mock_tracer_instance, mock_span, mock_span_ctx = mock_tracer

    mock_cost_calculator = MagicMock()
    mock_cost_calculator.calculate_cost.return_value = 0.01

    inst = ConcreteInstrumentor()
    inst.tracer = mock_tracer_instance
    inst.cost_calculator = mock_cost_calculator
    # Explicitly set shared metrics to mock objects for testing
    BaseInstrumentor._shared_request_counter = mock_counter
    BaseInstrumentor._shared_token_counter = mock_counter
    BaseInstrumentor._shared_latency_histogram = mock_histogram
    BaseInstrumentor._shared_cost_counter = mock_counter
    BaseInstrumentor._shared_error_counter = mock_counter
    inst._ensure_shared_metrics_created()
    inst.instrument(OTelConfig())
    yield inst, mock_span, mock_span_ctx, mock_counter, mock_histogram


# --- Tests for _ensure_shared_metrics_created ---
def test_ensure_shared_metrics_created_success(instrumentor):
    """Test that shared metrics are created only once."""
    inst, _, _, _, _ = instrumentor
    assert base._SHARED_METRICS_CREATED is True
    assert ConcreteInstrumentor._shared_request_counter is not None


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
    inst, mock_span, mock_span_ctx, _, _ = instrumentor
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
    inst, mock_span, mock_span_ctx, _, _ = instrumentor
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
    inst, mock_span, mock_span_ctx, _, _ = instrumentor
    original_function = MagicMock(side_effect=ValueError("Test error"))
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    with pytest.raises(ValueError):
        wrapped()

    assert mock_span.set_status.call_args[0][0].status_code == base.StatusCode.ERROR
    mock_span.record_exception.assert_called_once()


def test_create_span_wrapper_records_metrics(instrumentor):
    """Test that the wrapper records metrics for successful execution."""
    inst, mock_span, mock_span_ctx, mock_counter, mock_histogram = instrumentor
    original_function = MagicMock(
        return_value={"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    )
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped()

    assert mock_counter.add.call_count == 4  # request, prompt tokens, completion tokens, cost
    mock_histogram.record.assert_called_once()
    mock_counter.add.assert_any_call(0.01, {"model": "unknown"})
    assert result == {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def test_create_span_wrapper_records_metrics_without_usage(instrumentor):
    """Test that the wrapper handles missing usage data."""
    inst, mock_span, mock_span_ctx, mock_counter, mock_histogram = instrumentor
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped()

    assert mock_counter.add.call_count == 1  # only request counter
    mock_histogram.record.assert_called_once()
    assert result == {"usage": None}


def test_create_span_wrapper_with_cost_tracking_disabled(instrumentor):
    """Test that cost tracking is skipped when disabled."""
    inst, mock_span, mock_span_ctx, mock_counter, mock_histogram = instrumentor
    inst.config.enable_cost_tracking = False
    original_function = MagicMock(
        return_value={"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    )
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped()

    # Only request and token counters should be called, not cost
    assert mock_counter.add.call_count == 3  # request, prompt, completion
    assert call(0.01, {"model": "unknown"}) not in mock_counter.add.call_args_list
    assert result == {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}


# --- Tests for _record_result_metrics ---
def test_record_result_metrics_success(instrumentor):
    """Test that metrics are recorded correctly for a successful result."""
    inst, mock_span, mock_span_ctx, mock_counter, mock_histogram = instrumentor
    mock_span.name = "test.span"
    mock_span.attributes = MagicMock()
    mock_span.attributes.get.return_value = "unknown"
    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    assert mock_counter.add.call_count == 3  # prompt, completion, cost
    mock_histogram.record.assert_called_once()
    mock_counter.add.assert_any_call(0.01, {"model": "unknown"})
    assert mock_span.set_attribute.call_count == 3


def test_record_result_metrics_with_errors(instrumentor, caplog):
    """Test that errors in metric recording are logged but not raised."""
    inst, mock_span, mock_span_ctx, mock_counter, mock_histogram = instrumentor
    mock_span.name = "test.span"
    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}

    mock_counter.add.side_effect = ValueError("Mock error")
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    assert "Failed to extract or record usage metrics" in caplog.text


# --- Tests for instrumentation disabled ---
def test_create_span_wrapper_with_instrumentation_disabled(instrumentor):
    """Test that the wrapper bypasses instrumentation when disabled."""
    inst, mock_span, mock_span_ctx, _, _ = instrumentor
    inst._instrumented = False
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped("arg1", kwarg1="kwarg_value")

    inst.tracer.start_as_current_span.assert_not_called()
    original_function.assert_called_once_with("arg1", kwarg1="kwarg_value")
    assert result == {"usage": None}
