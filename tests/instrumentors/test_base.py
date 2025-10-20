import threading
import time
import unittest.mock
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
    BaseInstrumentor._shared_prompt_cost_counter = None
    BaseInstrumentor._shared_completion_cost_counter = None
    BaseInstrumentor._shared_reasoning_cost_counter = None
    BaseInstrumentor._shared_cache_read_cost_counter = None
    BaseInstrumentor._shared_cache_write_cost_counter = None
    BaseInstrumentor._shared_error_counter = None
    # Phase 3.4: Streaming metrics
    BaseInstrumentor._shared_ttft_histogram = None
    BaseInstrumentor._shared_tbt_histogram = None
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
        # Changed from start_as_current_span to start_span (Phase 3.4)
        mock_tracer.start_span.return_value = mock_span

        # Create mocks for ALL metrics *before* instantiating ConcreteInstrumentor
        mock_request_counter = MagicMock()
        mock_token_counter = MagicMock()
        mock_latency_histogram = MagicMock()
        mock_cost_counter = MagicMock()
        mock_prompt_cost_counter = MagicMock()
        mock_completion_cost_counter = MagicMock()
        mock_reasoning_cost_counter = MagicMock()
        mock_cache_read_cost_counter = MagicMock()
        mock_cache_write_cost_counter = MagicMock()
        mock_error_counter = MagicMock()
        # Phase 3.4: Streaming metrics
        mock_ttft_histogram = MagicMock()
        mock_tbt_histogram = MagicMock()

        # Configure mock_get_meter to return a meter instance that provides distinct mocks for each counter
        mock_meter_instance = MagicMock()
        mock_get_meter.return_value = mock_meter_instance
        mock_meter_instance.create_counter.side_effect = [
            mock_request_counter,
            mock_token_counter,
            mock_cost_counter,
            mock_prompt_cost_counter,
            mock_completion_cost_counter,
            mock_reasoning_cost_counter,
            mock_cache_read_cost_counter,
            mock_cache_write_cost_counter,
            mock_error_counter,
        ]
        mock_meter_instance.create_histogram.side_effect = [
            mock_latency_histogram,
            mock_ttft_histogram,
            mock_tbt_histogram,
        ]

        # Patch the class-level shared metrics with mocks
        monkeypatch.setattr(BaseInstrumentor, "_shared_request_counter", mock_request_counter)
        monkeypatch.setattr(BaseInstrumentor, "_shared_token_counter", mock_token_counter)
        monkeypatch.setattr(BaseInstrumentor, "_shared_latency_histogram", mock_latency_histogram)
        monkeypatch.setattr(BaseInstrumentor, "_shared_cost_counter", mock_cost_counter)
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_prompt_cost_counter", mock_prompt_cost_counter
        )
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_completion_cost_counter", mock_completion_cost_counter
        )
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_reasoning_cost_counter", mock_reasoning_cost_counter
        )
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_cache_read_cost_counter", mock_cache_read_cost_counter
        )
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_cache_write_cost_counter", mock_cache_write_cost_counter
        )
        monkeypatch.setattr(BaseInstrumentor, "_shared_error_counter", mock_error_counter)
        # Phase 3.4: Streaming metrics
        monkeypatch.setattr(BaseInstrumentor, "_shared_ttft_histogram", mock_ttft_histogram)
        monkeypatch.setattr(BaseInstrumentor, "_shared_tbt_histogram", mock_tbt_histogram)

        # Create instrumentor with cost tracking ENABLED
        config = OTelConfig()
        config.enable_cost_tracking = True  # Explicitly enable cost tracking

        inst = ConcreteInstrumentor()
        inst.instrument(config)  # Pass the config with cost tracking enabled

        # Mock cost calculator to return a positive cost
        inst.cost_calculator = MagicMock()
        inst.cost_calculator.calculate_cost.return_value = 0.01  # Positive cost

        # Phase 3.4: No longer need mock_span_ctx since we use start_span instead of start_as_current_span
        yield inst, mock_span


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
    inst, mock_span = instrumentor
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper(
        span_name="test.span",
        extract_attributes=lambda *args, **kwargs: {"test.attribute": "test_value"},
    )(original_function)

    result = wrapped("arg1", kwarg1="kwarg_value")

    # Changed from start_as_current_span to start_span (Phase 3.4)
    inst.tracer.start_span.assert_called_once_with(
        "test.span", attributes={"test.attribute": "test_value"}
    )
    original_function.assert_called_once_with("arg1", kwarg1="kwarg_value")
    assert result == {"usage": None}


def test_create_span_wrapper_handles_extract_attributes_error(instrumentor, caplog):
    """Test that the wrapper handles errors in extract_attributes."""
    inst, mock_span = instrumentor
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper(
        span_name="test.span", extract_attributes=lambda *args, **kwargs: 1 / 0  # Force error
    )(original_function)

    result = wrapped("arg1", kwarg1="kwarg_value")

    # Changed from start_as_current_span to start_span (Phase 3.4)
    inst.tracer.start_span.assert_called_once_with("test.span", attributes={})
    assert "Failed to extract attributes" in caplog.text
    assert result == {"usage": None}


def test_create_span_wrapper_handles_function_error(instrumentor):
    """Test that the wrapper handles errors in the wrapped function."""
    inst, mock_span = instrumentor
    original_function = MagicMock(side_effect=ValueError("Test error"))
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    with pytest.raises(ValueError):
        wrapped()

    assert mock_span.set_status.call_args[0][0].status_code == base.StatusCode.ERROR
    mock_span.record_exception.assert_called_once()


def test_create_span_wrapper_records_metrics(instrumentor):
    """Test that the wrapper records metrics for successful execution."""
    inst, mock_span = instrumentor
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
    inst, mock_span = instrumentor
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
    inst, mock_span = instrumentor
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
    inst, mock_span = instrumentor
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
    inst, mock_span = instrumentor
    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}

    inst.token_counter.add.side_effect = ValueError("Mock error")
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    assert "Failed to extract or record usage metrics" in caplog.text


# --- Tests for instrumentation disabled ---
def test_create_span_wrapper_with_instrumentation_disabled(instrumentor):
    """Test that the wrapper bypasses instrumentation when disabled."""
    inst, mock_span = instrumentor
    inst._instrumented = False
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper("test.span")(original_function)

    result = wrapped("arg1", kwarg1="kwarg_value")

    inst.tracer.start_as_current_span.assert_not_called()
    original_function.assert_called_once_with("arg1", kwarg1="kwarg_value")
    assert result == {"usage": None}


def test_extract_attributes_with_non_primitive_value(instrumentor):
    """Test that non-primitive attribute values are converted to strings."""
    inst, mock_span = instrumentor
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

    # Verify that start_span was called with attributes including stringified non-primitives (Phase 3.4)
    call_args = inst.tracer.start_span.call_args
    attributes = call_args[1]["attributes"]
    assert attributes["string_attr"] == "test"
    assert attributes["int_attr"] == 42
    assert attributes["list_attr"] == "[1, 2, 3]"
    assert attributes["dict_attr"] == "{'key': 'value'}"


def test_record_result_metrics_exception_in_wrapper(instrumentor, caplog):
    """Test that exceptions in _record_result_metrics call are caught and logged."""
    inst, mock_span = instrumentor
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
    inst, mock_span = instrumentor
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
    inst, mock_span = instrumentor
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
    inst, mock_span = instrumentor
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


def test_dual_token_attribute_emission(instrumentor):
    """Test that both old and new token attributes are emitted when semconv_stability_opt_in=gen_ai/dup."""
    inst, mock_span = instrumentor
    # Enable dual emission
    inst.config.semconv_stability_opt_in = "gen_ai/dup"

    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify both new and old token attributes are set
    set_attribute_calls = mock_span.set_attribute.call_args_list
    attributes_set = {call[0][0]: call[0][1] for call in set_attribute_calls}

    # New semantic conventions
    assert attributes_set.get("gen_ai.usage.prompt_tokens") == 10
    assert attributes_set.get("gen_ai.usage.completion_tokens") == 20
    assert attributes_set.get("gen_ai.usage.total_tokens") == 30

    # Old semantic conventions
    assert attributes_set.get("gen_ai.usage.input_tokens") == 10
    assert attributes_set.get("gen_ai.usage.output_tokens") == 20


def test_single_token_attribute_emission(instrumentor):
    """Test that only new token attributes are emitted when semconv_stability_opt_in=gen_ai."""
    inst, mock_span = instrumentor
    # Default is gen_ai (new conventions only)
    inst.config.semconv_stability_opt_in = "gen_ai"

    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify only new token attributes are set
    set_attribute_calls = mock_span.set_attribute.call_args_list
    attributes_set = {call[0][0]: call[0][1] for call in set_attribute_calls}

    # New semantic conventions
    assert attributes_set.get("gen_ai.usage.prompt_tokens") == 10
    assert attributes_set.get("gen_ai.usage.completion_tokens") == 20
    assert attributes_set.get("gen_ai.usage.total_tokens") == 30

    # Old semantic conventions should NOT be set
    assert "gen_ai.usage.input_tokens" not in attributes_set
    assert "gen_ai.usage.output_tokens" not in attributes_set


# --- Tests for Streaming Metrics (Phase 3.4) ---
def test_streaming_response_wrapper(instrumentor):
    """Test that streaming responses are properly wrapped with TTFT/TBT metrics."""
    inst, mock_span = instrumentor

    # Create a mock streaming response
    def mock_stream_generator():
        yield "chunk1"
        yield "chunk2"
        yield "chunk3"

    # Wrap the stream
    wrapped_stream = inst._wrap_streaming_response(
        stream=mock_stream_generator(), span=mock_span, start_time=1000.0, model="gpt-4"
    )

    # Consume the stream
    chunks = list(wrapped_stream)

    # Verify chunks were yielded
    assert chunks == ["chunk1", "chunk2", "chunk3"]

    # Verify TTFT was recorded
    mock_span.set_attribute.assert_any_call("gen_ai.server.ttft", unittest.mock.ANY)

    # Verify streaming token count was set
    mock_span.set_attribute.assert_any_call("gen_ai.streaming.token_count", 3)

    # Verify span was ended
    mock_span.end.assert_called_once()

    # Verify span status was set to OK
    assert mock_span.set_status.called


def test_streaming_detection_in_wrapper(instrumentor):
    """Test that create_span_wrapper detects streaming and wraps response."""
    inst, mock_span = instrumentor

    # Create a mock function that returns an iterator
    def mock_streaming_function(*args, **kwargs):
        for i in range(3):
            yield f"chunk{i}"

    # Wrap the function with stream=True in kwargs
    wrapped = inst.create_span_wrapper(
        span_name="test.streaming",
        extract_attributes=lambda *args, **kwargs: {"gen_ai.request.model": "gpt-4"},
    )(mock_streaming_function)

    # Call with stream=True
    result = wrapped(stream=True, model="gpt-4")

    # Result should be a generator (the wrapped stream)
    assert hasattr(result, "__iter__")

    # Consume the generator
    chunks = list(result)
    assert len(chunks) == 3

    # Verify span was created
    inst.tracer.start_span.assert_called_once()
