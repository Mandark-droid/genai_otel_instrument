import asyncio
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
    BaseInstrumentor._shared_time_to_first_token_histogram = None
    BaseInstrumentor._shared_time_per_output_token_histogram = None
    # Token distribution histograms
    BaseInstrumentor._shared_prompt_tokens_histogram = None
    BaseInstrumentor._shared_completion_tokens_histogram = None
    # Finish reason tracking counters
    BaseInstrumentor._shared_request_finish_counter = None
    BaseInstrumentor._shared_request_success_counter = None
    BaseInstrumentor._shared_request_failure_counter = None
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
        mock_time_to_first_token_histogram = MagicMock()
        mock_time_per_output_token_histogram = MagicMock()
        # Token distribution histograms
        mock_prompt_tokens_histogram = MagicMock()
        mock_completion_tokens_histogram = MagicMock()
        # Finish reason tracking counters
        mock_request_finish_counter = MagicMock()
        mock_request_success_counter = MagicMock()
        mock_request_failure_counter = MagicMock()

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
            mock_request_finish_counter,
            mock_request_success_counter,
            mock_request_failure_counter,
        ]
        mock_meter_instance.create_histogram.side_effect = [
            mock_latency_histogram,
            mock_ttft_histogram,
            mock_tbt_histogram,
            mock_time_to_first_token_histogram,
            mock_time_per_output_token_histogram,
            mock_prompt_tokens_histogram,
            mock_completion_tokens_histogram,
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
        monkeypatch.setattr(
            BaseInstrumentor,
            "_shared_time_to_first_token_histogram",
            mock_time_to_first_token_histogram,
        )
        monkeypatch.setattr(
            BaseInstrumentor,
            "_shared_time_per_output_token_histogram",
            mock_time_per_output_token_histogram,
        )
        # Token distribution histograms
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_prompt_tokens_histogram", mock_prompt_tokens_histogram
        )
        monkeypatch.setattr(
            BaseInstrumentor,
            "_shared_completion_tokens_histogram",
            mock_completion_tokens_histogram,
        )
        # Finish reason tracking counters
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_request_finish_counter", mock_request_finish_counter
        )
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_request_success_counter", mock_request_success_counter
        )
        monkeypatch.setattr(
            BaseInstrumentor, "_shared_request_failure_counter", mock_request_failure_counter
        )

        # Create instrumentor with cost tracking ENABLED and full metric verbosity
        # so the granular cost counters / token histograms / finish counters
        # (opt-in by default for hot-path performance) are exercised here.
        config = OTelConfig(metrics_profile="full")
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
    # 6 attributes: input_tokens + output_tokens (current names), their two
    # superseded spellings, total_tokens, and cost.total. The superseded pair is
    # present because dual emission is the DEFAULT — see
    # genai_otel.semconv.genai_semconv_modes. Under an explicit "gen_ai"
    # (current-only) this drops back to 4; that case is covered by
    # test_token_attributes_current_only_when_explicitly_requested.
    assert mock_span.set_attribute.call_count == 6


def test_record_result_metrics_with_errors(instrumentor, caplog):
    """Test that errors in metric recording are logged but not raised."""
    inst, mock_span = instrumentor
    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}

    inst.token_counter.add.side_effect = ValueError("Mock error")
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    assert "Failed to extract or record usage metrics" in caplog.text


def test_record_result_metrics_emits_cache_and_reasoning_tokens(instrumentor):
    """Anthropic cache_read/cache_creation tokens (top-level) and OpenAI
    reasoning_tokens (nested under completion_tokens_details) surface as
    `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`,
    and `gen_ai.usage.reasoning_tokens` per upstream
    semantic-conventions-genai#76.
    """
    inst, mock_span = instrumentor
    mock_span.attributes.get.return_value = "claude-sonnet-4"
    result = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 200,
            # Anthropic top-level cache attribution
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
            # OpenAI o1/o3-style reasoning attribution
            "completion_tokens_details": {"reasoning_tokens": 15},
        }
    }
    inst._record_result_metrics(mock_span, result, time.time() - 0.1)

    set_attr_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
    assert set_attr_calls["gen_ai.usage.cache_read.input_tokens"] == 30
    assert set_attr_calls["gen_ai.usage.cache_creation.input_tokens"] == 20
    assert set_attr_calls["gen_ai.usage.reasoning.output_tokens"] == 15


def test_record_result_metrics_skips_zero_cache_and_reasoning(instrumentor):
    """Zero / missing cache and reasoning fields must NOT be emitted as
    span attributes (avoid noisy zero-valued attrs on every span).
    """
    inst, mock_span = instrumentor
    mock_span.attributes.get.return_value = "gpt-4o-mini"
    result = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            # Zero / missing cache + reasoning
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    }
    inst._record_result_metrics(mock_span, result, time.time() - 0.1)

    attrs_set = {c.args[0] for c in mock_span.set_attribute.call_args_list}
    assert "gen_ai.usage.cache_read.input_tokens" not in attrs_set
    assert "gen_ai.usage.cache_creation.input_tokens" not in attrs_set
    assert "gen_ai.usage.reasoning.output_tokens" not in attrs_set


def test_record_result_metrics_non_chat_sets_cost_attribute(instrumentor):
    """Non-chat call types (image, audio, embedding) must also set
    `gen_ai.usage.cost.total` on the span — previously cost was added to the
    counter but never surfaced as a span attribute, so backends couldn't
    aggregate cost for image-gen / audio / embedding spans.
    """
    inst, mock_span = instrumentor
    # Make the span report a non-chat call type.
    mock_span.attributes.get.side_effect = lambda key, default=None: {
        "gen_ai.request.model": "dall-e-3",
        "gen_ai.request.type": "image",
    }.get(key, default)

    inst.cost_calculator = MagicMock()
    inst.cost_calculator.calculate_cost.return_value = 0.04
    inst.cost_calculator.calculate_granular_cost.return_value = {
        "total": 0.0,
        "prompt": 0.0,
        "completion": 0.0,
        "reasoning": 0.0,
        "cache_read": 0.0,
        "cache_write": 0.0,
    }

    # Need positive token counts so the usage block runs.
    result = {"usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}}
    inst._record_result_metrics(mock_span, result, time.time() - 0.1)

    set_attr_calls = [
        c
        for c in mock_span.set_attribute.call_args_list
        if c.args and c.args[0] == "gen_ai.usage.cost.total"
    ]
    assert set_attr_calls, "expected gen_ai.usage.cost.total to be set for non-chat call_type"
    assert set_attr_calls[-1].args[1] == 0.04
    inst.cost_counter.add.assert_called_with(0.04, {"model": "dall-e-3"})


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

    # Current semantic conventions (semantic-conventions#1200, v1.27.0)
    assert attributes_set.get("gen_ai.usage.input_tokens") == 10
    assert attributes_set.get("gen_ai.usage.output_tokens") == 20
    assert attributes_set.get("gen_ai.usage.total_tokens") == 30

    # Superseded names, emitted only because dual emission is on
    assert attributes_set.get("gen_ai.usage.prompt_tokens") == 10
    assert attributes_set.get("gen_ai.usage.completion_tokens") == 20


def test_single_token_attribute_emission(instrumentor):
    """Only the current token attributes are emitted when semconv_stability_opt_in=gen_ai."""
    inst, mock_span = instrumentor
    # Default is gen_ai (current conventions only)
    inst.config.semconv_stability_opt_in = "gen_ai"

    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify only the current token attributes are set
    set_attribute_calls = mock_span.set_attribute.call_args_list
    attributes_set = {call[0][0]: call[0][1] for call in set_attribute_calls}

    # Current semantic conventions
    assert attributes_set.get("gen_ai.usage.input_tokens") == 10
    assert attributes_set.get("gen_ai.usage.output_tokens") == 20
    assert attributes_set.get("gen_ai.usage.total_tokens") == 30

    # Superseded names must NOT be set without the dup opt-in. A backend that
    # reads the current conventions (e.g. Arize AX -> llm.token_count.*) sees
    # zero tokens if these are all we emit.
    assert "gen_ai.usage.prompt_tokens" not in attributes_set
    assert "gen_ai.usage.completion_tokens" not in attributes_set


def test_set_token_usage_attributes_defaults_to_current_semconv(instrumentor):
    """Without the dup opt-in, only the current input/output names are emitted.

    Regression guard for the polarity of the two conventions: v1.27.0 renamed
    prompt_tokens -> input_tokens and completion_tokens -> output_tokens, so
    the input/output pair is what a standards-native backend reads.
    """
    inst, mock_span = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai"

    inst._set_token_usage_attributes(mock_span, prompt_tokens=7, completion_tokens=11)

    attrs = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 11
    assert "gen_ai.usage.prompt_tokens" not in attrs
    assert "gen_ai.usage.completion_tokens" not in attrs


def test_set_token_usage_attributes_dup_emits_both_conventions(instrumentor):
    """gen_ai/dup emits the current names alongside the superseded ones."""
    inst, mock_span = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai/dup"

    inst._set_token_usage_attributes(mock_span, prompt_tokens=7, completion_tokens=11)

    attrs = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 11
    assert attrs["gen_ai.usage.prompt_tokens"] == 7
    assert attrs["gen_ai.usage.completion_tokens"] == 11


@pytest.mark.parametrize("value", [0, -1, None, "12", object()])
def test_set_token_usage_attributes_skips_non_positive_and_non_numeric(instrumentor, value):
    """Zero, negative and non-numeric counts emit no token attributes at all."""
    inst, mock_span = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai/dup"

    inst._set_token_usage_attributes(mock_span, prompt_tokens=value, completion_tokens=value)

    names = {call[0][0] for call in mock_span.set_attribute.call_args_list}
    assert not [n for n in names if n.startswith("gen_ai.usage.")]


def test_set_token_usage_attributes_without_config(instrumentor):
    """A missing config must not raise, and must fall back to the SAFE default.

    Previously a missing config meant no dual emission. That is the wrong way to
    fail: an instrumentor that could not resolve its config would silently drop
    the attribute names a consumer is reading, which reads downstream as zero
    tokens rather than as a configuration problem. Absent config now resolves the
    same way an unset env var does — emit both.
    """
    inst, mock_span = instrumentor
    inst.config = None

    inst._set_token_usage_attributes(mock_span, prompt_tokens=3, completion_tokens=4)

    attrs = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert attrs["gen_ai.usage.output_tokens"] == 4
    assert attrs["gen_ai.usage.prompt_tokens"] == 3
    assert attrs["gen_ai.usage.completion_tokens"] == 4


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


# --- Tests for streaming latency semantic conventions (issue #21) ---
def _span_attrs(mock_span):
    """Collapse a mock span's set_attribute calls into a dict."""
    return {
        c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list if len(c.args) == 2
    }


def _fake_clock(monkeypatch, values):
    """Pin base.time.time() to `values`, holding the last value once exhausted."""
    seq = list(values)

    def _now():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(
        base,
        "time",
        unittest.mock.MagicMock(time=_now, sleep=time.sleep, monotonic=time.monotonic),
    )


def test_streaming_emits_semconv_ttft_and_tpot(instrumentor, monkeypatch):
    """Streamed calls carry the semconv TTFT/TPOT attributes, not just the legacy ttft."""
    inst, mock_span = instrumentor
    # chunk1 @ +0.5s, chunk2 @ +1.0s, chunk3 @ +2.0s, then the end-of-stream read.
    _fake_clock(monkeypatch, [1000.5, 1001.0, 1002.0])

    def stream():
        yield {"delta": "a"}
        yield {"delta": "b"}
        yield {"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}

    chunks = list(inst._wrap_streaming_response(stream(), mock_span, 1000.0, "gpt-4o"))
    assert len(chunks) == 3

    attrs = _span_attrs(mock_span)
    assert attrs["gen_ai.server.time_to_first_token"] == pytest.approx(0.5)
    # The pre-existing bespoke name stays for consumers already reading it.
    assert attrs["gen_ai.server.ttft"] == pytest.approx(0.5)
    # (total - ttft) / max(output_tokens - 1, 1) == (2.0 - 0.5) / 4
    assert attrs["gen_ai.server.time_per_output_token"] == pytest.approx(0.375)
    assert "gen_ai.streaming.tpot_unavailable_reason" not in attrs

    inst.time_to_first_token_histogram.record.assert_called_once()
    inst.time_per_output_token_histogram.record.assert_called_once()


def test_streaming_omits_tpot_when_output_token_count_unknown(instrumentor, monkeypatch):
    """No usage in the stream means no TPOT at all - never a fabricated zero."""
    inst, mock_span = instrumentor
    _fake_clock(monkeypatch, [1000.5, 1001.0, 1002.0])

    def stream():
        yield {"delta": "a"}
        yield {"delta": "b"}

    list(inst._wrap_streaming_response(stream(), mock_span, 1000.0, "gpt-4o"))

    attrs = _span_attrs(mock_span)
    assert attrs["gen_ai.server.time_to_first_token"] == pytest.approx(0.5)
    assert "gen_ai.server.time_per_output_token" not in attrs
    assert attrs["gen_ai.streaming.tpot_unavailable_reason"] == "output_token_count_unavailable"
    inst.time_per_output_token_histogram.record.assert_not_called()


def test_non_streaming_call_omits_streaming_latency_attributes(instrumentor):
    """A non-streamed call must not carry TTFT/TPOT - absent beats zero."""
    inst, mock_span = instrumentor

    def create(**kwargs):
        return {"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}

    wrapped = inst.create_span_wrapper(span_name="chat")(create)
    wrapped(model="gpt-4o")

    attrs = _span_attrs(mock_span)
    assert "gen_ai.server.time_to_first_token" not in attrs
    assert "gen_ai.server.time_per_output_token" not in attrs
    assert "gen_ai.server.ttft" not in attrs
    assert "gen_ai.streaming.tpot_unavailable_reason" not in attrs


def test_streaming_generator_result_is_measured(instrumentor):
    """A provider returning a bare generator for stream=True is still measured."""
    inst, mock_span = instrumentor

    def create(**kwargs):
        yield {"delta": "a"}
        yield {"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}

    wrapped = inst.create_span_wrapper(span_name="chat")(create)
    chunks = list(wrapped(model="llama3", stream=True))

    assert len(chunks) == 2
    attrs = _span_attrs(mock_span)
    assert "gen_ai.server.time_to_first_token" in attrs
    assert "gen_ai.server.time_per_output_token" in attrs
    assert attrs["gen_ai.streaming.token_count"] == 2
    mock_span.end.assert_called_once()


def test_async_streaming_defers_span_end_and_records_latency(instrumentor):
    """An async stream must keep the span open until the caller finishes iterating."""
    inst, mock_span = instrumentor

    class FakeAsyncStream:
        def __aiter__(self):
            return self._chunks()

        async def _chunks(self):
            yield {"delta": "a"}
            yield {"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}

    async def create(**kwargs):
        return FakeAsyncStream()

    wrapped = inst.create_span_wrapper(span_name="chat")(create)

    async def run():
        stream = await wrapped(model="gpt-4o", stream=True)
        # Awaiting only hands back the iterator; the model has produced nothing
        # yet, so closing the span here would time the handshake and lose usage.
        assert not mock_span.end.called
        return [chunk async for chunk in stream]

    chunks = asyncio.run(run())

    assert len(chunks) == 2
    mock_span.end.assert_called_once()
    attrs = _span_attrs(mock_span)
    assert attrs["gen_ai.server.time_to_first_token"] >= 0
    assert "gen_ai.server.time_per_output_token" in attrs
    assert attrs["gen_ai.usage.total_tokens"] == 5


def test_async_streaming_error_ends_span_once(instrumentor):
    """A stream that raises mid-iteration still closes its span exactly once."""
    inst, mock_span = instrumentor

    class ExplodingAsyncStream:
        def __aiter__(self):
            return self._chunks()

        async def _chunks(self):
            yield {"delta": "a"}
            raise RuntimeError("connection reset")

    async def create(**kwargs):
        return ExplodingAsyncStream()

    wrapped = inst.create_span_wrapper(span_name="chat")(create)

    async def run():
        stream = await wrapped(model="gpt-4o", stream=True)
        async for _ in stream:
            pass

    with pytest.raises(RuntimeError, match="connection reset"):
        asyncio.run(run())

    mock_span.end.assert_called_once()
    assert mock_span.record_exception.called


# --- Tests for Granular Cost Tracking (Phase 3.2) ---
def test_granular_cost_tracking_with_all_cost_types(instrumentor):
    """Test granular cost tracking with prompt, completion, reasoning, and cache costs."""
    inst, mock_span = instrumentor

    # Set up mock span to return appropriate attributes
    def mock_get_attribute(key, default=None):
        if key == "gen_ai.request.model":
            return "claude-3-5-sonnet-20241022"
        elif key == "gen_ai.request.type":
            return "chat"
        return default

    mock_span.attributes.get.side_effect = mock_get_attribute

    # Mock the cost calculator to return granular costs
    inst.cost_calculator.calculate_granular_cost.return_value = {
        "total": 0.05,
        "prompt": 0.01,
        "completion": 0.02,
        "reasoning": 0.005,
        "cache_read": 0.001,
        "cache_write": 0.014,
    }

    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 25,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 140,
    }
    result = {"usage": usage}

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify all granular cost counters were called
    inst._shared_cost_counter.add.assert_called_once_with(
        0.05, {"model": "claude-3-5-sonnet-20241022"}
    )
    inst._shared_prompt_cost_counter.add.assert_called_once_with(
        0.01, {"model": "claude-3-5-sonnet-20241022"}
    )
    inst._shared_completion_cost_counter.add.assert_called_once_with(
        0.02, {"model": "claude-3-5-sonnet-20241022"}
    )
    inst._shared_reasoning_cost_counter.add.assert_called_once_with(
        0.005, {"model": "claude-3-5-sonnet-20241022"}
    )
    inst._shared_cache_read_cost_counter.add.assert_called_once_with(
        0.001, {"model": "claude-3-5-sonnet-20241022"}
    )
    inst._shared_cache_write_cost_counter.add.assert_called_once_with(
        0.014, {"model": "claude-3-5-sonnet-20241022"}
    )

    # Verify span attributes were set
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.total", 0.05)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.prompt", 0.01)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.completion", 0.02)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.reasoning", 0.005)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.cache_read", 0.001)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.cache_write", 0.014)


def test_granular_cost_tracking_with_zero_costs(instrumentor):
    """Test that zero costs are not recorded to granular counters."""
    inst, mock_span = instrumentor

    # Set up mock span to return appropriate attributes
    def mock_get_attribute(key, default=None):
        if key == "gen_ai.request.model":
            return "gpt-4"
        elif key == "gen_ai.request.type":
            return "chat"
        return default

    mock_span.attributes.get.side_effect = mock_get_attribute

    # Mock the cost calculator to return costs with zeros
    inst.cost_calculator.calculate_granular_cost.return_value = {
        "total": 0.03,
        "prompt": 0.01,
        "completion": 0.02,
        "reasoning": 0.0,  # Zero - should not be recorded
        "cache_read": 0.0,  # Zero - should not be recorded
        "cache_write": 0.0,  # Zero - should not be recorded
    }

    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    result = {"usage": usage}

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify only non-zero costs were recorded
    inst._shared_cost_counter.add.assert_called_once_with(0.03, {"model": "gpt-4"})
    inst._shared_prompt_cost_counter.add.assert_called_once_with(0.01, {"model": "gpt-4"})
    inst._shared_completion_cost_counter.add.assert_called_once_with(0.02, {"model": "gpt-4"})

    # Verify zero costs were NOT recorded
    inst._shared_reasoning_cost_counter.add.assert_not_called()
    inst._shared_cache_read_cost_counter.add.assert_not_called()
    inst._shared_cache_write_cost_counter.add.assert_not_called()

    # Verify span attributes - zero costs should not set attributes
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.total", 0.03)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.prompt", 0.01)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.completion", 0.02)


def test_granular_cost_tracking_only_prompt_cost(instrumentor):
    """Test granular cost tracking with only prompt cost (embedding call)."""
    inst, mock_span = instrumentor

    # Set up mock span to return appropriate attributes
    def mock_get_attribute(key, default=None):
        if key == "gen_ai.request.model":
            return "text-embedding-3-small"
        elif key == "gen_ai.request.type":
            return "chat"
        return default

    mock_span.attributes.get.side_effect = mock_get_attribute

    # Mock the cost calculator to return only prompt cost
    inst.cost_calculator.calculate_granular_cost.return_value = {
        "total": 0.001,
        "prompt": 0.001,
        "completion": 0.0,
        "reasoning": 0.0,
        "cache_read": 0.0,
        "cache_write": 0.0,
    }

    usage = {"prompt_tokens": 500}
    result = {"usage": usage}

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify only total and prompt costs were recorded
    inst._shared_cost_counter.add.assert_called_once_with(
        0.001, {"model": "text-embedding-3-small"}
    )
    inst._shared_prompt_cost_counter.add.assert_called_once_with(
        0.001, {"model": "text-embedding-3-small"}
    )

    # Verify other costs were NOT recorded
    inst._shared_completion_cost_counter.add.assert_not_called()
    inst._shared_reasoning_cost_counter.add.assert_not_called()
    inst._shared_cache_read_cost_counter.add.assert_not_called()
    inst._shared_cache_write_cost_counter.add.assert_not_called()


def test_token_histograms_recorded(instrumentor):
    """Test that token distribution histograms are recorded alongside counters."""
    inst, mock_span = instrumentor
    mock_span.attributes = {"gen_ai.request.model": "gpt-4"}

    # Create mock usage data
    result = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }

    # Record metrics
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify counter metrics are recorded
    assert inst._shared_token_counter.add.call_count == 2
    inst._shared_token_counter.add.assert_any_call(
        100, {"token_type": "prompt", "operation": "test.span"}
    )
    inst._shared_token_counter.add.assert_any_call(
        50, {"token_type": "completion", "operation": "test.span"}
    )

    # Verify histogram metrics are recorded
    assert inst._shared_prompt_tokens_histogram.record.call_count == 1
    inst._shared_prompt_tokens_histogram.record.assert_called_once_with(
        100, {"model": "gpt-4", "operation": "test.span"}
    )

    assert inst._shared_completion_tokens_histogram.record.call_count == 1
    inst._shared_completion_tokens_histogram.record.assert_called_once_with(
        50, {"model": "gpt-4", "operation": "test.span"}
    )

    # Verify span attributes are set
    assert mock_span.set_attribute.call_count >= 3
    mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)


def test_token_histograms_with_zero_tokens(instrumentor):
    """Test that histograms are not recorded for zero token counts."""
    inst, mock_span = instrumentor
    mock_span.attributes = {"gen_ai.request.model": "gpt-4"}

    # Create mock usage data with zero tokens
    result = {
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    }

    # Record metrics
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify histograms are NOT recorded for zero values
    inst._shared_prompt_tokens_histogram.record.assert_not_called()
    inst._shared_completion_tokens_histogram.record.assert_not_called()

    # Verify counters are also not called for zero values
    inst._shared_token_counter.add.assert_not_called()


def test_token_histograms_handle_missing_model(instrumentor):
    """Test that histograms handle missing model attribute gracefully."""
    inst, mock_span = instrumentor
    # Mock attributes.get to return "unknown" as default when model key is not found
    mock_attributes = MagicMock()
    mock_attributes.get = MagicMock(
        side_effect=lambda key, default=None: (
            default if key == "gen_ai.request.model" else "test_value"
        )
    )
    mock_span.attributes = mock_attributes

    result = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }

    # Record metrics
    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify histograms are recorded with "unknown" model
    inst._shared_prompt_tokens_histogram.record.assert_called_once_with(
        100, {"model": "unknown", "operation": "test.span"}
    )
    inst._shared_completion_tokens_histogram.record.assert_called_once_with(
        50, {"model": "unknown", "operation": "test.span"}
    )


def test_finish_reason_success_recorded(instrumentor):
    """Test that finish reasons are recorded and success is tracked."""
    inst, mock_span = instrumentor
    mock_span.attributes = {"gen_ai.request.model": "gpt-4"}

    # Add _extract_finish_reason method to instance
    inst._extract_finish_reason = lambda result: result.get("finish_reason")

    result = {
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify finish reason counter was called
    inst._shared_request_finish_counter.add.assert_called_once_with(
        1, {"finish_reason": "stop", "model": "gpt-4"}
    )

    # Verify success counter was called (stop is a success reason)
    inst._shared_request_success_counter.add.assert_called_once_with(1, {"model": "gpt-4"})

    # Verify failure counter was NOT called
    inst._shared_request_failure_counter.add.assert_not_called()

    # Verify span attribute was set
    mock_span.set_attribute.assert_any_call("gen_ai.response.finish_reason", "stop")


def test_finish_reason_failure_recorded(instrumentor):
    """Test that failure finish reasons are tracked separately."""
    inst, mock_span = instrumentor
    mock_span.attributes = {"gen_ai.request.model": "gpt-4"}

    # Add _extract_finish_reason method to instance
    inst._extract_finish_reason = lambda result: result.get("finish_reason")

    result = {
        "finish_reason": "content_filter",
        "usage": {"prompt_tokens": 100, "completion_tokens": 0},
    }

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify finish reason counter was called
    inst._shared_request_finish_counter.add.assert_called_once_with(
        1, {"finish_reason": "content_filter", "model": "gpt-4"}
    )

    # Verify failure counter was called (content_filter is a failure reason)
    inst._shared_request_failure_counter.add.assert_called_once_with(
        1, {"finish_reason": "content_filter", "model": "gpt-4"}
    )

    # Verify success counter was NOT called
    inst._shared_request_success_counter.add.assert_not_called()


def test_finish_reason_not_recorded_when_missing(instrumentor):
    """Test that finish reason metrics are not recorded when unavailable."""
    inst, mock_span = instrumentor
    mock_span.attributes = {"gen_ai.request.model": "gpt-4"}

    # Instrumentor without _extract_finish_reason method
    result = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify finish reason counters were NOT called
    inst._shared_request_finish_counter.add.assert_not_called()
    inst._shared_request_success_counter.add.assert_not_called()
    inst._shared_request_failure_counter.add.assert_not_called()


def test_finish_reason_ambiguous_not_categorized(instrumentor):
    """Test that ambiguous finish reasons are recorded but not categorized as success/failure."""
    inst, mock_span = instrumentor
    mock_span.attributes = {"gen_ai.request.model": "gpt-4"}

    # Add _extract_finish_reason method to instance
    inst._extract_finish_reason = lambda result: result.get("finish_reason")

    result = {
        "finish_reason": "custom_stop",  # Not in success or failure lists
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    inst._record_result_metrics(mock_span, result, time.time() - 1)

    # Verify finish reason counter was called
    inst._shared_request_finish_counter.add.assert_called_once_with(
        1, {"finish_reason": "custom_stop", "model": "gpt-4"}
    )

    # Verify neither success nor failure counters were called
    inst._shared_request_success_counter.add.assert_not_called()
    inst._shared_request_failure_counter.add.assert_not_called()


# --- Tests for Async Function Support ---
@pytest.mark.asyncio
async def test_async_wrapper_awaits_coroutine(instrumentor):
    """Test that create_span_wrapper properly awaits async functions."""
    inst, mock_span = instrumentor

    async def async_func(*args, **kwargs):
        await asyncio.sleep(0)
        return {"usage": None}

    wrapped = inst.create_span_wrapper("test.async_span")(async_func)
    result = await wrapped()

    assert result == {"usage": None}
    mock_span.set_status.assert_called_once()
    mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_async_wrapper_records_metrics(instrumentor):
    """Test that async wrapper records metrics from the actual result."""
    inst, mock_span = instrumentor
    mock_span.attributes.get.return_value = "test_model"

    async def async_func(*args, **kwargs):
        await asyncio.sleep(0)
        return {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}

    wrapped = inst.create_span_wrapper("test.async_span")(async_func)
    result = await wrapped()

    assert result == {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    # The operation key uses span.name which is "test.span" from the fixture mock
    inst.request_counter.add.assert_called_once_with(1, {"operation": "test.span"})
    inst.latency_histogram.record.assert_called_once()


@pytest.mark.asyncio
async def test_async_wrapper_handles_exception(instrumentor):
    """Test that async wrapper properly records errors from async functions."""
    inst, mock_span = instrumentor

    async def async_func(*args, **kwargs):
        await asyncio.sleep(0)
        raise ValueError("Async error")

    wrapped = inst.create_span_wrapper("test.async_span")(async_func)

    with pytest.raises(ValueError, match="Async error"):
        await wrapped()

    assert mock_span.set_status.call_args[0][0].status_code == base.StatusCode.ERROR
    mock_span.record_exception.assert_called_once()
    mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_async_wrapper_detaches_context(instrumentor):
    """Test that async wrapper detaches OTel context after completion."""
    inst, mock_span = instrumentor

    async def async_func(*args, **kwargs):
        await asyncio.sleep(0)
        return {"usage": None}

    with patch("genai_otel.instrumentors.base.otel_context") as mock_ctx:
        mock_ctx.attach.return_value = "test_token"
        wrapped = inst.create_span_wrapper("test.async_span")(async_func)
        await wrapped()

        mock_ctx.detach.assert_called_once_with("test_token")


@pytest.mark.asyncio
async def test_async_wrapper_detaches_context_on_error(instrumentor):
    """Test that async wrapper detaches OTel context even on error."""
    inst, mock_span = instrumentor

    async def async_func(*args, **kwargs):
        raise RuntimeError("boom")

    with patch("genai_otel.instrumentors.base.otel_context") as mock_ctx:
        mock_ctx.attach.return_value = "test_token"
        wrapped = inst.create_span_wrapper("test.async_span")(async_func)

        with pytest.raises(RuntimeError):
            await wrapped()

        mock_ctx.detach.assert_called_once_with("test_token")


@pytest.mark.asyncio
async def test_async_wrapper_keeps_span_open_during_execution(instrumentor):
    """Test that span stays open while async function executes (not ended instantly)."""
    inst, mock_span = instrumentor
    span_ended_during_execution = False

    async def async_func(*args, **kwargs):
        nonlocal span_ended_during_execution
        # Check if span.end was called before our async work completes
        span_ended_during_execution = mock_span.end.called
        await asyncio.sleep(0)
        return {"usage": None}

    wrapped = inst.create_span_wrapper("test.async_span")(async_func)
    await wrapped()

    # Span should NOT have been ended before the async function completed
    assert span_ended_during_execution is False
    # But it should be ended after
    mock_span.end.assert_called_once()


def test_sync_wrapper_still_works(instrumentor):
    """Test that sync functions continue to work after adding async support."""
    inst, mock_span = instrumentor
    original_function = MagicMock(return_value={"usage": None})
    wrapped = inst.create_span_wrapper("test.sync_span")(original_function)

    result = wrapped("arg1")

    original_function.assert_called_once_with("arg1")
    assert result == {"usage": None}
    mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_async_wrapper_runs_evaluation_checks(instrumentor):
    """Test that async wrapper runs evaluation checks on the actual result."""
    inst, mock_span = instrumentor

    async def async_func(*args, **kwargs):
        return {"usage": None}

    with patch.object(inst, "_run_evaluation_checks") as mock_eval:
        wrapped = inst.create_span_wrapper("test.async_span")(async_func)
        await wrapped()

        mock_eval.assert_called_once()
        # Verify it was called with the actual result, not the coroutine
        call_args = mock_eval.call_args
        assert call_args[0][3] == {"usage": None}


@pytest.mark.asyncio
async def test_async_wrapper_decrements_server_metrics(instrumentor):
    """Test that async wrapper decrements server metrics on completion."""
    inst, mock_span = instrumentor

    async def async_func(*args, **kwargs):
        return {"usage": None}

    mock_server_metrics = MagicMock()
    with patch(
        "genai_otel.instrumentors.base.get_server_metrics", return_value=mock_server_metrics
    ):
        wrapped = inst.create_span_wrapper("test.async_span")(async_func)
        await wrapped()

    mock_server_metrics.decrement_requests_running.assert_called()


@pytest.mark.asyncio
async def test_async_wrapper_decrements_server_metrics_on_error(instrumentor):
    """Test that async wrapper decrements server metrics even on error."""
    inst, mock_span = instrumentor

    async def async_func(*args, **kwargs):
        raise RuntimeError("boom")

    mock_server_metrics = MagicMock()
    with patch(
        "genai_otel.instrumentors.base.get_server_metrics", return_value=mock_server_metrics
    ):
        wrapped = inst.create_span_wrapper("test.async_span")(async_func)
        with pytest.raises(RuntimeError):
            await wrapped()

    mock_server_metrics.decrement_requests_running.assert_called()


# ---------------------------------------------------------------------------
# Prompt-cache token breakdown: `cache_write` is the current spelling
# (semantic-conventions-genai#440 renamed `cache_creation`), `cache_creation`
# is superseded and follows the same dual-emission policy as prompt/completion.
# ---------------------------------------------------------------------------


def test_cache_tokens_dual_emission(instrumentor):
    """Both cache_write and the superseded cache_creation are emitted under gen_ai/dup."""
    inst, mock_span = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai/dup"

    result = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 7,
        }
    }
    inst._record_result_metrics(mock_span, result, time.time() - 1)
    attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}

    assert attrs.get("gen_ai.usage.cache_read.input_tokens") == 40
    assert attrs.get("gen_ai.usage.cache_write.input_tokens") == 7
    assert attrs.get("gen_ai.usage.cache_creation.input_tokens") == 7


def test_cache_tokens_current_name_only(instrumentor):
    """Under gen_ai, only the current cache_write spelling is emitted."""
    inst, mock_span = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai"

    result = {"usage": {"cache_read_input_tokens": 40, "cache_creation_input_tokens": 7}}
    inst._record_result_metrics(mock_span, result, time.time() - 1)
    attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}

    assert attrs.get("gen_ai.usage.cache_write.input_tokens") == 7
    assert "gen_ai.usage.cache_creation.input_tokens" not in attrs
    # cache_read was never renamed, so it is emitted at both tiers.
    assert attrs.get("gen_ai.usage.cache_read.input_tokens") == 40


def test_cache_tokens_absent_when_provider_reports_none(instrumentor):
    """No cache attributes at all when the provider does not report caching."""
    inst, mock_span = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai/dup"

    inst._record_result_metrics(mock_span, {"usage": {"prompt_tokens": 5}}, time.time() - 1)
    attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}

    assert not [k for k in attrs if "cache" in k]


# ---------------------------------------------------------------------------
# server.address / server.port derivation
# ---------------------------------------------------------------------------


class _Node:
    """Minimal stand-in for an SDK object graph."""


def _client_with(base_url, depth=1):
    """Build a sub-resource whose client chain ends in ``base_url``."""
    client = _Node()
    client.base_url = base_url
    for _ in range(depth):
        parent = _Node()
        parent._client = client
        client = parent
    return client


def test_server_attributes_from_sdk_client():
    """The common `sub_resource._client.base_url` shape yields host and port."""
    from genai_otel.instrumentors.base import _server_attributes

    assert _server_attributes(_client_with("https://api.openai.com/v1")) == {
        "server.address": "api.openai.com",
        "server.port": 443,
    }


def test_server_attributes_explicit_port_wins():
    """A self-hosted endpoint keeps its explicit port rather than a scheme default."""
    from genai_otel.instrumentors.base import _server_attributes

    assert _server_attributes(_client_with("http://localhost:11434")) == {
        "server.address": "localhost",
        "server.port": 11434,
    }


def test_server_attributes_walks_wrapped_transport():
    """SDKs that wrap a transport client of their own are still resolved."""
    from genai_otel.instrumentors.base import _server_attributes

    assert _server_attributes(_client_with("https://api.anthropic.com", depth=2)) == {
        "server.address": "api.anthropic.com",
        "server.port": 443,
    }


def test_server_attributes_absent_rather_than_guessed():
    """No base URL, or an unparseable one, yields no attributes at all.

    An absent attribute reads as "endpoint unknown"; a guessed vendor host would
    silently misattribute self-hosted and proxied traffic.
    """
    from genai_otel.instrumentors.base import _server_attributes

    assert _server_attributes(object()) == {}
    assert _server_attributes(_client_with("not a url")) == {}


# ---------------------------------------------------------------------------
# Per-modality token breakdown (semantic-conventions-genai#440)
# ---------------------------------------------------------------------------


def test_modality_token_attributes_emitted(instrumentor):
    """Modality counts a provider reports land under gen_ai.usage.<modality>.*."""
    inst, mock_span = instrumentor
    result = {
        "usage": {
            "prompt_tokens": 300,
            "completion_tokens": 180,
            "text_input_tokens": 100,
            "image_input_tokens": 200,
            "text_output_tokens": 180,
            "text_cache_read_input_tokens": 40,
        }
    }
    inst._record_result_metrics(mock_span, result, time.time() - 1)
    attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}

    assert attrs.get("gen_ai.usage.text.input_tokens") == 100
    assert attrs.get("gen_ai.usage.image.input_tokens") == 200
    assert attrs.get("gen_ai.usage.text.output_tokens") == 180
    assert attrs.get("gen_ai.usage.text.cache_read.input_tokens") == 40
    # Totals remain the totals; modality values are subsets of them.
    assert attrs.get("gen_ai.usage.input_tokens") == 300


def test_modality_absent_rather_than_zero(instrumentor):
    """A provider that reports no breakdown gets no modality attributes.

    Emitting zeros would claim the provider said "no audio tokens", when it
    actually said nothing about modality at all.
    """
    inst, mock_span = instrumentor
    inst._record_result_metrics(
        mock_span, {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}, time.time() - 1
    )
    attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}

    assert not [k for k in attrs if k.startswith("gen_ai.usage.audio.")]
    assert not [k for k in attrs if k.startswith("gen_ai.usage.image.")]


# ---------------------------------------------------------------------------
# Request parameters derived centrally from call kwargs
# ---------------------------------------------------------------------------


def test_request_parameter_attributes_mapped():
    from genai_otel.instrumentors.base import _request_parameter_attributes

    attrs = _request_parameter_attributes(
        {
            "seed": 42,
            "stream": True,
            "top_k": 40,
            "n": 3,
            "response_format": {"type": "json_object"},
        }
    )
    assert attrs["gen_ai.request.seed"] == 42
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.request.top_k"] == 40
    assert attrs["gen_ai.request.choice.count"] == 3
    assert attrs["gen_ai.output.type"] == "json"


def test_request_parameters_google_spelling():
    """Google's candidate_count maps onto the same choice.count attribute."""
    from genai_otel.instrumentors.base import _request_parameter_attributes

    assert _request_parameter_attributes({"candidate_count": 2})["gen_ai.request.choice.count"] == 2


def test_request_parameters_omitted_when_not_passed():
    """Unset parameters are absent, not defaulted to a provider's value."""
    from genai_otel.instrumentors.base import _request_parameter_attributes

    assert _request_parameter_attributes({"model": "gpt-4o"}) == {}
    # stream=False is a real caller choice and is recorded as such.
    assert _request_parameter_attributes({"stream": False}) == {"gen_ai.request.stream": False}


def test_request_parameters_reject_bool_as_number():
    """A bool must not be recorded as a seed or top_k."""
    from genai_otel.instrumentors.base import _request_parameter_attributes

    attrs = _request_parameter_attributes({"seed": True, "top_k": False})
    assert "gen_ai.request.seed" not in attrs
    assert "gen_ai.request.top_k" not in attrs


# ---------------------------------------------------------------------------
# Embeddings request shape uses the registry spellings
# ---------------------------------------------------------------------------


def test_embedding_request_attributes_dual_emission(instrumentor):
    inst, _ = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai/dup"

    attrs = inst.add_embedding_request_attributes({}, dimensions=512, encoding_format="float")

    assert attrs["gen_ai.embeddings.dimension.count"] == 512
    assert attrs["gen_ai.request.encoding_formats"] == ["float"]
    assert attrs["gen_ai.request.dimensions"] == 512
    assert attrs["gen_ai.request.encoding_format"] == "float"


def test_embedding_request_attributes_current_only(instrumentor):
    inst, _ = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai"

    attrs = inst.add_embedding_request_attributes({}, dimensions=512, encoding_format=["float"])

    assert attrs["gen_ai.embeddings.dimension.count"] == 512
    assert attrs["gen_ai.request.encoding_formats"] == ["float"]
    assert "gen_ai.request.dimensions" not in attrs
    assert "gen_ai.request.encoding_format" not in attrs


# ---------------------------------------------------------------------------
# finish_reasons is an array in the conventions
# ---------------------------------------------------------------------------


def test_finish_reasons_array_emitted(instrumentor):
    """The singular value is also published under the plural array name."""
    inst, mock_span = instrumentor
    inst.config.semconv_stability_opt_in = "gen_ai/dup"
    inst._extract_finish_reason = lambda result: "stop"
    mock_span.attributes = {}

    inst._record_result_metrics(mock_span, {"usage": {}}, time.time() - 1)
    attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}

    assert attrs.get("gen_ai.response.finish_reasons") == ["stop"]
    assert attrs.get("gen_ai.response.finish_reason") == "stop"


def test_finish_reasons_does_not_clobber_instrumentor_array(instrumentor):
    """An instrumentor that already extracted a real array keeps it."""
    inst, mock_span = instrumentor
    inst._extract_finish_reason = lambda result: "stop"
    mock_span.attributes = {"gen_ai.response.finish_reasons": ["length", "stop"]}

    inst._record_result_metrics(mock_span, {"usage": {}}, time.time() - 1)
    attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}

    assert "gen_ai.response.finish_reasons" not in attrs
