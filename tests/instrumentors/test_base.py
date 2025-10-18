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
def instrumentor(monkeypatch):
    """Fixture to provide a clean instrumentor instance with mocked dependencies."""
    with (
        patch("genai_otel.instrumentors.base.trace.get_tracer") as mock_get_tracer,
        patch("genai_otel.instrumentors.base.metrics.get_meter") as mock_get_meter,
    ):
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_span = MagicMock()
        mock_span.name = "test.span"
        mock_span.attributes.get.return_value = "test_model"
        mock_span_ctx = MagicMock()
        mock_span_ctx.__enter__.return_value = mock_span
        mock_span_ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_span_ctx

        # Create mocks for ALL metrics *before* instantiating ConcreteInstrumentor
        mock_request_counter = MagicMock()
        mock_token_counter = MagicMock()
        mock_latency_histogram = MagicMock()
        mock_cost_counter = MagicMock()
        mock_error_counter = MagicMock()

        # Configure mock_get_meter to return a meter instance that provides distinct mocks for each counter
        mock_meter_instance = MagicMock()
        mock_get_meter.return_value = mock_meter_instance
        mock_meter_instance.create_counter.side_effect = [
            mock_request_counter,
            mock_token_counter,
            mock_latency_histogram,
            mock_cost_counter,
            mock_error_counter,
        ]

        # Patch the class-level shared metrics with mocks
        monkeypatch.setattr(BaseInstrumentor, "_shared_request_counter", mock_request_counter)
        monkeypatch.setattr(BaseInstrumentor, "_shared_token_counter", mock_token_counter)
        monkeypatch.setattr(BaseInstrumentor, "_shared_latency_histogram", mock_latency_histogram)
        monkeypatch.setattr(BaseInstrumentor, "_shared_cost_counter", mock_cost_counter)
        monkeypatch.setattr(BaseInstrumentor, "_shared_error_counter", mock_error_counter)

        # Create instrumentor with cost tracking ENABLED
        config = OTelConfig()
        config.enable_cost_tracking = True  # Explicitly enable cost tracking

        inst = ConcreteInstrumentor()
        inst.instrument(config)  # Pass the config with cost tracking enabled

        # Mock cost calculator to return a positive cost
        inst.cost_calculator = MagicMock()
        inst.cost_calculator.calculate_cost.return_value = 0.01  # Positive cost

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


def test_extract_attributes_with_non_primitive_value(instrumentor):
    """Test that non-primitive attribute values are converted to strings."""
    inst, mock_span, mock_span_ctx = instrumentor
    original_function = MagicMock(return_value={"usage": None})

    # Create an extract_attributes function that returns a non-primitive value
    def extract_attrs(instance, args, kwargs):
        return {
            "string_attr": "test",
            "int_attr": 42,
            "list_attr": [1, 2, 3],  # Non-primitive - should be converted to string
            "dict_attr": {"key": "value"},  # Non-primitive - should be converted to string
        }

    wrapped = inst.create_span_wrapper("test.span", extract_attributes=extract_attrs)(
        original_function
    )

    result = wrapped()

    # Verify that start_as_current_span was called with attributes including stringified non-primitives
    call_args = inst.tracer.start_as_current_span.call_args
    attributes = call_args[1]["attributes"]
    assert attributes["string_attr"] == "test"
    assert attributes["int_attr"] == 42
    assert attributes["list_attr"] == "[1, 2, 3]"
    assert attributes["dict_attr"] == "{'key': 'value'}"


def test_record_result_metrics_exception_in_wrapper(instrumentor, caplog):
    """Test that exceptions in _record_result_metrics call are caught and logged."""
    inst, mock_span, mock_span_ctx = instrumentor
    original_function = MagicMock(return_value={"usage": {"prompt_tokens": 10}})

    # Make _record_result_metrics raise an exception
    with patch.object(inst, "_record_result_metrics", side_effect=RuntimeError("Test error")):
        wrapped = inst.create_span_wrapper("test.span")(original_function)
        result = wrapped()

        # Should still return the result and not crash
        assert result == {"usage": {"prompt_tokens": 10}}
        assert "Failed to record metrics for span 'test.span'" in caplog.text


def test_error_counter_exception_handling(instrumentor):
    """Test that exceptions in error_counter.add are silently caught."""
    inst, mock_span, mock_span_ctx = instrumentor
    original_function = MagicMock(side_effect=ValueError("Test error"))

    # Make error_counter.add raise an exception
    inst.error_counter.add.side_effect = RuntimeError("Counter error")

    wrapped = inst.create_span_wrapper("test.span")(original_function)

    # Should still raise the original exception, not the counter error
    with pytest.raises(ValueError, match="Test error"):
        wrapped()

    # Verify error_counter.add was called (before it raised)
    inst.error_counter.add.assert_called_once()


def test_latency_histogram_exception_handling(instrumentor, caplog):
    """Test that exceptions in latency_histogram.record are caught and logged."""
    inst, mock_span, mock_span_ctx = instrumentor
    original_function = MagicMock(return_value={"usage": None})

    # Make latency_histogram.record raise an exception
    inst.latency_histogram.record.side_effect = RuntimeError("Histogram error")

    wrapped = inst.create_span_wrapper("test.span")(original_function)
    result = wrapped()

    # Should still return the result
    assert result == {"usage": None}
    assert "Failed to record latency for span 'test.span'" in caplog.text


def test_cost_calculation_exception_handling(instrumentor, caplog):
    """Test that exceptions in cost calculation are caught and logged."""
    inst, mock_span, mock_span_ctx = instrumentor
    mock_span.attributes.get.return_value = "test_model"
    original_function = MagicMock(
        return_value={"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    )

    # Make cost_calculator.calculate_cost raise an exception
    inst.cost_calculator.calculate_cost.side_effect = RuntimeError("Cost calculation error")

    wrapped = inst.create_span_wrapper("test.span")(original_function)
    result = wrapped()

    # Should still return the result
    assert result == {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    assert "Failed to calculate cost for span 'test.span'" in caplog.text
