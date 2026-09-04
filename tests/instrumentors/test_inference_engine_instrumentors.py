"""Tests for the self-hosted inference-engine instrumentors.

vLLM, SGLang and llama.cpp are instrumented at the in-process Python API rather
than over HTTP, so these tests exercise the shapes those APIs actually return.
Response objects use ``SimpleNamespace`` deliberately: a bare ``MagicMock``
auto-creates every attribute, which defeats the ``getattr``-based shape
detection the extractors rely on and would make them look like they work when
they do not.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig
from genai_otel.engine_latency import llamacpp_latency_attributes, vllm_latency_attributes
from genai_otel.instrumentors.llamacpp_instrumentor import LlamaCppInstrumentor
from genai_otel.instrumentors.vllm_instrumentor import VLLMInstrumentor

# ---------------------------------------------------------------------------
# Latency derivation (semantic-conventions-genai#408 key set)
# ---------------------------------------------------------------------------


class TestVLLMLatencyDerivation:
    def _metrics(self, **overrides):
        base = dict(
            arrival_time=100.0,
            first_scheduled_time=100.5,
            first_token_time=101.0,
            last_token_time=103.0,
            finished_time=103.2,
            time_in_queue=0.5,
            scheduler_time=0.02,
            model_forward_time=250.0,  # milliseconds, per vLLM
            model_execute_time=0.4,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_phase_durations(self):
        attrs = vllm_latency_attributes(self._metrics())

        assert attrs["gen_ai.latency.time_in_queue"] == 0.5
        assert attrs["gen_ai.latency.time_to_first_token"] == pytest.approx(1.0)
        assert attrs["gen_ai.latency.e2e"] == pytest.approx(3.2)
        assert attrs["gen_ai.latency.time_in_model_prefill"] == pytest.approx(0.5)
        assert attrs["gen_ai.latency.time_in_model_decode"] == pytest.approx(2.0)
        assert attrs["gen_ai.latency.time_in_model_inference"] == pytest.approx(2.5)

    def test_model_forward_converted_from_milliseconds(self):
        """vLLM reports this one field in ms; the attribute set is all seconds."""
        attrs = vllm_latency_attributes(self._metrics())
        assert attrs["gen_ai.latency.time_in_model_forward"] == pytest.approx(0.25)
        assert attrs["gen_ai.latency.time_in_model_execute"] == pytest.approx(0.4)

    def test_unfinished_request_omits_derived_durations(self):
        """A request still streaming has no finished_time, so no e2e."""
        attrs = vllm_latency_attributes(self._metrics(finished_time=None))
        assert "gen_ai.latency.e2e" not in attrs
        # Phases that are already measurable still appear.
        assert "gen_ai.latency.time_to_first_token" in attrs

    def test_negative_duration_dropped(self):
        """Clock skew must not produce a negative duration on a span."""
        attrs = vllm_latency_attributes(self._metrics(first_token_time=99.0))
        assert "gen_ai.latency.time_to_first_token" not in attrs

    def test_no_metrics_object(self):
        assert vllm_latency_attributes(None) == {}


class TestLlamaCppLatencyDerivation:
    def test_timings_converted_from_milliseconds(self):
        attrs = llamacpp_latency_attributes({"prompt_ms": 120.0, "predicted_ms": 880.0})
        assert attrs["gen_ai.latency.time_in_model_prefill"] == pytest.approx(0.12)
        assert attrs["gen_ai.latency.time_in_model_decode"] == pytest.approx(0.88)
        assert attrs["gen_ai.latency.time_in_model_inference"] == pytest.approx(1.0)
        assert attrs["gen_ai.latency.e2e"] == pytest.approx(1.0)

    def test_absent_timings_produce_nothing(self):
        assert llamacpp_latency_attributes(None) == {}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,module,attr",
    [
        (VLLMInstrumentor, "vllm", "_vllm_available"),
        (LlamaCppInstrumentor, "llama_cpp", "_llamacpp_available"),
    ],
)
def test_unavailable_when_library_missing(cls, module, attr):
    with patch.dict(sys.modules, {module: None}):
        assert getattr(cls(), attr) is False


@pytest.mark.parametrize(
    "cls,module,class_name,attr",
    [
        (VLLMInstrumentor, "vllm", "LLM", "_vllm_available"),
        (LlamaCppInstrumentor, "llama_cpp", "Llama", "_llamacpp_available"),
    ],
)
def test_available_when_library_present(cls, module, class_name, attr):
    fake = MagicMock()
    setattr(fake, class_name, type(class_name, (), {}))
    with patch.dict(sys.modules, {module: fake}):
        assert getattr(cls(), attr) is True


def test_import_error_from_missing_gpu_runtime_is_survivable():
    """Importing vLLM without a GPU raises; setup must not fall over.

    A machine with no CUDA device is a perfectly normal place to run this
    library, so an engine that cannot initialise there is skipped, not fatal.
    """
    broken = MagicMock()
    type(broken).LLM = property(lambda self: (_ for _ in ()).throw(RuntimeError("no CUDA")))
    with patch.dict(sys.modules, {"vllm": broken}):
        assert VLLMInstrumentor()._vllm_available is False


def test_instrument_is_noop_when_unavailable():
    inst = VLLMInstrumentor()
    inst._vllm_available = False
    inst.instrument(OTelConfig(service_name="test"))
    assert inst._instrumented is False


# ---------------------------------------------------------------------------
# vLLM extraction
# ---------------------------------------------------------------------------


def _vllm_output(prompt_ids=(1, 2, 3), token_ids=(4, 5), **metrics):
    return SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(prompt_ids),
        outputs=[SimpleNamespace(token_ids=list(token_ids), finish_reason="stop")],
        metrics=SimpleNamespace(
            arrival_time=100.0,
            first_scheduled_time=100.5,
            first_token_time=101.0,
            last_token_time=103.0,
            finished_time=metrics.get("finished_time", 103.2),
            time_in_queue=0.5,
            scheduler_time=None,
            model_forward_time=None,
            model_execute_time=None,
        ),
    )


class TestVLLMExtraction:
    def test_usage_summed_across_batch(self):
        """generate() returns one RequestOutput per prompt; totals are the batch."""
        inst = VLLMInstrumentor()
        usage = inst._extract_usage([_vllm_output(), _vllm_output()])
        assert usage == {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}

    def test_usage_none_when_no_tokens(self):
        assert VLLMInstrumentor()._extract_usage([]) is None

    def test_response_attributes_include_latency(self):
        attrs = VLLMInstrumentor()._extract_response_attributes([_vllm_output()])
        assert attrs["gen_ai.request.id"] == "req-1"
        assert attrs["gen_ai.latency.time_in_queue"] == 0.5

    def test_batch_reports_slowest_request(self):
        """A batch is as fast as its tail, so the straggler's metrics win."""
        fast = _vllm_output(finished_time=101.0)
        slow = _vllm_output(finished_time=110.0)
        attrs = VLLMInstrumentor()._extract_response_attributes([fast, slow])
        assert attrs["gen_ai.latency.e2e"] == pytest.approx(10.0)
        assert attrs["gen_ai.response.output_count"] == 2

    def test_finish_reason(self):
        assert VLLMInstrumentor()._extract_finish_reason([_vllm_output()]) == "stop"

    def test_prompt_count_treats_string_as_one_prompt(self):
        inst = VLLMInstrumentor()
        assert inst._prompt_count("a single prompt") == 1
        assert inst._prompt_count(["a", "b", "c"]) == 3
        assert inst._prompt_count(None) is None

    def test_model_name_from_either_config_layout(self):
        inst = VLLMInstrumentor()
        old = SimpleNamespace(llm_engine=SimpleNamespace(model_config=SimpleNamespace(model="m1")))
        new = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(model_config=SimpleNamespace(model="m2"))
            )
        )
        assert inst._model_name(old) == "m1"
        assert inst._model_name(new) == "m2"
        assert inst._model_name(SimpleNamespace()) is None

    def test_sampling_params_become_request_attributes(self):
        """Sampling settings live on an object, not kwargs, so base.py cannot see them."""
        inst = VLLMInstrumentor()
        instance = SimpleNamespace(
            llm_engine=SimpleNamespace(model_config=SimpleNamespace(model="m1"))
        )
        sampling = SimpleNamespace(max_tokens=128, temperature=0.7, top_p=0.9, top_k=40, n=2)
        attrs = inst._extract_generate_attributes(instance, (["p"],), {"sampling_params": sampling})
        assert attrs["gen_ai.request.model"] == "m1"
        assert attrs["gen_ai.request.max_tokens"] == 128
        assert attrs["gen_ai.request.top_k"] == 40
        assert attrs["gen_ai.request.choice.count"] == 2
        assert attrs["gen_ai.system"] == "vllm"


# ---------------------------------------------------------------------------
# llama.cpp extraction
# ---------------------------------------------------------------------------


def _llamacpp_result(**extra):
    result = {
        "id": "cmpl-1",
        "model": "/models/llama-3-8b.gguf",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
    }
    result.update(extra)
    return result


class TestLlamaCppExtraction:
    def test_usage(self):
        usage = LlamaCppInstrumentor()._extract_usage(_llamacpp_result())
        assert usage == {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}

    def test_usage_none_without_usage_block(self):
        assert LlamaCppInstrumentor()._extract_usage({"id": "x"}) is None

    def test_response_attributes_with_timings(self):
        attrs = LlamaCppInstrumentor()._extract_response_attributes(
            _llamacpp_result(timings={"prompt_ms": 100.0, "predicted_ms": 400.0})
        )
        assert attrs["gen_ai.response.id"] == "cmpl-1"
        assert attrs["gen_ai.latency.time_in_model_prefill"] == pytest.approx(0.1)

    def test_response_attributes_without_timings(self):
        """A build with no timing support gets no phase attributes, not zeros."""
        attrs = LlamaCppInstrumentor()._extract_response_attributes(_llamacpp_result())
        assert not [k for k in attrs if k.startswith("gen_ai.latency.")]

    def test_finish_reason(self):
        assert LlamaCppInstrumentor()._extract_finish_reason(_llamacpp_result()) == "stop"

    def test_model_name_is_gguf_path(self):
        inst = LlamaCppInstrumentor()
        attrs = inst._extract_chat_attributes(
            SimpleNamespace(model_path="/models/x.gguf"), ([{"role": "user"}],), {}
        )
        assert attrs["gen_ai.request.model"] == "/models/x.gguf"
        assert attrs["gen_ai.request.input_count"] == 1

    def test_embedding_request_type_is_singular(self):
        """CostCalculator dispatches on the singular; the plural bills as chat."""
        inst = LlamaCppInstrumentor()
        attrs = inst._extract_embedding_attributes(SimpleNamespace(), (), {})
        assert attrs["gen_ai.request.type"] == "embedding"
        assert attrs["gen_ai.operation.name"] == "embeddings"


# ---------------------------------------------------------------------------
# Instrumentation wiring
# ---------------------------------------------------------------------------


def test_vllm_instrument_wraps_and_is_idempotent():
    class FakeLLM:
        def generate(self, prompts, sampling_params=None):
            return []

        def chat(self, messages, sampling_params=None):
            return []

    fake_module = MagicMock()
    fake_module.LLM = FakeLLM
    original_generate = FakeLLM.generate

    with patch.dict(sys.modules, {"vllm": fake_module}):
        inst = VLLMInstrumentor()
        inst.instrument(OTelConfig(service_name="test"))
        assert inst._instrumented is True
        assert FakeLLM.generate is not original_generate

        second = VLLMInstrumentor()
        second.instrument(OTelConfig(service_name="test"))
        # Wrapping twice would double-count every token and cost. wrapt hands
        # back a fresh bound wrapper on each attribute access, so identity of
        # `FakeLLM.generate` proves nothing; what matters is that only one
        # wrapper sits between the class and the original function.
        assert FakeLLM.generate.__wrapped__ is original_generate


def test_instrument_error_respects_fail_on_error():
    fake_module = MagicMock()
    fake_module.LLM = MagicMock()
    with patch.dict(sys.modules, {"vllm": fake_module}):
        inst = VLLMInstrumentor()
        inst._vllm_available = True
        inst._vllm_module = SimpleNamespace()  # no LLM attribute -> AttributeError
        config = OTelConfig(service_name="test")
        config.fail_on_error = False
        inst.instrument(config)
        assert inst._instrumented is False

        config.fail_on_error = True
        with pytest.raises(AttributeError):
            inst.instrument(config)


# ---------------------------------------------------------------------------
# vLLM V1 engine reality
#
# Verified live against vLLM 0.24 and 0.27: the V1 engine sets
# `RequestOutput.metrics` to None and exposes no per-request timing on the
# Python API, but DOES populate `num_cached_tokens`. Unit tests built from
# hand-made RequestMetrics objects cannot see this, which is why the live run
# was a release gate.
# ---------------------------------------------------------------------------


def _v1_output(cached=0):
    """A RequestOutput as the V1 engine actually returns it."""
    return SimpleNamespace(
        request_id="0",
        prompt_token_ids=[1, 2, 3, 4],
        outputs=[SimpleNamespace(token_ids=[5, 6, 7], finish_reason="stop")],
        metrics=None,
        num_cached_tokens=cached,
    )


class TestVLLMV1Engine:
    def test_no_latency_attributes_when_metrics_is_none(self):
        """Absent beats invented: the engine reported no timings, so we emit none."""
        attrs = VLLMInstrumentor()._extract_response_attributes([_v1_output()])
        assert not [k for k in attrs if k.startswith("gen_ai.latency.")]
        # The rest of the span is unaffected.
        assert attrs["gen_ai.request.id"] == "0"

    def test_tokens_still_extracted_without_metrics(self):
        usage = VLLMInstrumentor()._extract_usage([_v1_output()])
        assert usage["prompt_tokens"] == 4
        assert usage["completion_tokens"] == 3

    def test_num_cached_tokens_maps_to_cache_read(self):
        """V1 does populate this, and it is the conventions' cache_read concept."""
        usage = VLLMInstrumentor()._extract_usage([_v1_output(cached=128)])
        assert usage["cache_read_input_tokens"] == 128

    def test_cache_read_summed_across_batch(self):
        usage = VLLMInstrumentor()._extract_usage([_v1_output(cached=64), _v1_output(cached=32)])
        assert usage["cache_read_input_tokens"] == 96

    def test_zero_cached_tokens_omitted(self):
        """Zero prefix-cache hits is not a cache_read of zero worth recording."""
        usage = VLLMInstrumentor()._extract_usage([_v1_output(cached=0)])
        assert "cache_read_input_tokens" not in usage


def _vllm_output_with_reason(reason, prompt_ids=(1, 2), token_ids=(3,)):
    return SimpleNamespace(
        request_id="r",
        prompt_token_ids=list(prompt_ids),
        outputs=[SimpleNamespace(token_ids=list(token_ids), finish_reason=reason)],
        metrics=None,
        num_cached_tokens=0,
    )


class TestVLLMBatchFinishReasons:
    """A vLLM batch genuinely ends different ways; the array must say so.

    Found by live testing: a two-prompt batch where the engine reported
    ['stop', 'length'] emitted only ('stop',), hiding the truncated request --
    precisely the one an operator is looking for.
    """

    def test_mixed_batch_reports_every_reason(self):
        attrs = VLLMInstrumentor()._extract_response_attributes(
            [_vllm_output_with_reason("stop"), _vllm_output_with_reason("length")]
        )
        assert attrs["gen_ai.response.finish_reasons"] == ["stop", "length"]

    def test_uniform_batch_is_deduplicated(self):
        """512 prompts that all hit the cap report ["length"], not 512 copies."""
        attrs = VLLMInstrumentor()._extract_response_attributes(
            [_vllm_output_with_reason("length") for _ in range(8)]
        )
        assert attrs["gen_ai.response.finish_reasons"] == ["length"]

    def test_single_output(self):
        attrs = VLLMInstrumentor()._extract_response_attributes([_vllm_output_with_reason("stop")])
        assert attrs["gen_ai.response.finish_reasons"] == ["stop"]

    def test_absent_when_engine_reports_none(self):
        attrs = VLLMInstrumentor()._extract_response_attributes([_vllm_output_with_reason(None)])
        assert "gen_ai.response.finish_reasons" not in attrs


class TestVLLMAsyncStreaming:
    """AsyncLLM.generate is how streaming and server deployments reach vLLM.

    LLM.generate returns completed outputs, so it is never on that path. The
    generic async-generator tracing in base.py ends the span but records no
    result metrics, which live testing showed produced a span with no tokens,
    cost or finish reason -- hence the dedicated wrapper.
    """

    @staticmethod
    def _instrumentor():
        inst = VLLMInstrumentor()
        inst._instrumented = True
        inst.config = OTelConfig(service_name="test")
        return inst

    def test_stream_is_passed_through_unchanged(self):
        """Telemetry must never alter or truncate a caller's token stream."""
        import asyncio

        chunks = [_v1_output(), _v1_output(), _v1_output()]

        async def fake_generate(self, *a, **kw):
            for c in chunks:
                yield c

        inst = self._instrumentor()
        wrapped = inst._async_generate_wrapper(fake_generate)

        async def run():
            return [x async for x in wrapped(None, "prompt", None, "req-1")]

        assert asyncio.run(run()) == chunks

    def test_telemetry_failure_does_not_break_the_stream(self):
        """Losing a span is acceptable; losing the caller's tokens is not."""
        import asyncio

        async def fake_generate(self, *a, **kw):
            yield _v1_output()

        inst = self._instrumentor()
        inst._record_result_metrics = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        wrapped = inst._async_generate_wrapper(fake_generate)

        async def run():
            return [x async for x in wrapped(None, "prompt", None, "req-1")]

        assert len(asyncio.run(run())) == 1

    def test_generator_exception_propagates(self):
        import asyncio

        async def failing(self, *a, **kw):
            yield _v1_output()
            raise ValueError("engine died")

        inst = self._instrumentor()
        wrapped = inst._async_generate_wrapper(failing)

        async def run():
            return [x async for x in wrapped(None, "prompt", None, "req-1")]

        with pytest.raises(ValueError, match="engine died"):
            asyncio.run(run())

    def test_request_attributes_mark_the_call_as_streamed(self):
        inst = VLLMInstrumentor()
        attrs = inst._extract_async_generate_attributes(None, ("prompt", None, "req-9"), {})
        assert attrs["gen_ai.request.stream"] is True
        assert attrs["gen_ai.request.id"] == "req-9"
        assert attrs["gen_ai.request.input_count"] == 1

    def test_request_id_read_from_kwargs_too(self):
        inst = VLLMInstrumentor()
        attrs = inst._extract_async_generate_attributes(None, (), {"request_id": "kw-req"})
        assert attrs["gen_ai.request.id"] == "kw-req"


class TestLlamaCppNestedSpanDedup:
    """One span per user call, not one per internal delegation.

    Found live: `create_chat_completion` calls `create_completion` internally,
    so wrapping both emitted a `llamacpp.chat` AND a nested
    `llamacpp.completion` span carrying the same usage -- tokens and cost were
    counted twice for every chat call.
    """

    def test_inner_delegation_is_not_traced_again(self):
        from genai_otel.instrumentors import llamacpp_instrumentor as mod

        calls = []

        def original(self, *a, **kw):
            return {"origin": "original"}

        def traced(*a, **kw):
            calls.append("traced")
            return {"origin": "traced"}

        guarded = LlamaCppInstrumentor._dedup(traced, original)

        # Outermost call is traced.
        assert guarded(object())["origin"] == "traced"
        assert calls == ["traced"]

        # A call made while a llama.cpp span is already open is not.
        token = mod._LLAMACPP_SPAN_ACTIVE.set(True)
        try:
            assert guarded(object())["origin"] == "original"
        finally:
            mod._LLAMACPP_SPAN_ACTIVE.reset(token)
        assert calls == ["traced"]

    def test_guard_is_released_after_a_call(self):
        from genai_otel.instrumentors import llamacpp_instrumentor as mod

        guarded = LlamaCppInstrumentor._dedup(lambda *a, **k: {}, lambda *a, **k: {})
        guarded(object())
        assert mod._LLAMACPP_SPAN_ACTIVE.get() is False

    def test_guard_is_released_when_the_call_raises(self):
        from genai_otel.instrumentors import llamacpp_instrumentor as mod

        def boom(*a, **k):
            raise ValueError("nope")

        guarded = LlamaCppInstrumentor._dedup(boom, lambda *a, **k: {})
        with pytest.raises(ValueError):
            guarded(object())
        assert mod._LLAMACPP_SPAN_ACTIVE.get() is False

    def test_guard_is_held_until_a_stream_is_drained(self):
        """A delegated call mid-stream must not open a second span."""
        from genai_otel.instrumentors import llamacpp_instrumentor as mod

        def streaming(*a, **k):
            yield {"chunk": 1}
            yield {"chunk": 2}

        guarded = LlamaCppInstrumentor._dedup(streaming, lambda *a, **k: {})
        gen = guarded(object())
        next(gen)
        assert mod._LLAMACPP_SPAN_ACTIVE.get() is True, "guard released too early"
        list(gen)
        assert mod._LLAMACPP_SPAN_ACTIVE.get() is False


class TestLlamaCppStreamedFinishReason:
    """Streamed chunks carry no usage, but the last one carries the reason."""

    def test_reason_read_from_the_final_chunk(self):
        chunks = [
            {"choices": [{"finish_reason": None}]},
            {"choices": [{"finish_reason": None}]},
            {"choices": [{"finish_reason": "length"}]},
        ]
        assert LlamaCppInstrumentor()._extract_finish_reason(chunks) == "length"

    def test_no_reason_anywhere(self):
        chunks = [{"choices": [{"finish_reason": None}]}]
        assert LlamaCppInstrumentor()._extract_finish_reason(chunks) is None

    def test_non_streamed_response_still_works(self):
        assert LlamaCppInstrumentor()._extract_finish_reason(_llamacpp_result()) == "stop"
