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
from genai_otel.engine_latency import (
    llamacpp_latency_attributes,
    sglang_latency_attributes,
    vllm_latency_attributes,
)
from genai_otel.instrumentors.llamacpp_instrumentor import LlamaCppInstrumentor
from genai_otel.instrumentors.sglang_instrumentor import SGLangInstrumentor
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


class TestSGLangLatencyDerivation:
    def test_fields_read_independently(self):
        attrs = sglang_latency_attributes({"e2e_latency": 1.5, "ttft": 0.2})
        assert attrs["gen_ai.latency.e2e"] == 1.5
        assert attrs["gen_ai.latency.time_to_first_token"] == 0.2
        # A release that does not report prefill simply produces no attribute.
        assert "gen_ai.latency.time_in_model_prefill" not in attrs

    def test_non_mapping_is_safe(self):
        assert sglang_latency_attributes(None) == {}
        assert sglang_latency_attributes("not a dict") == {}


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
        (SGLangInstrumentor, "sglang", "_sglang_available"),
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
        (SGLangInstrumentor, "sglang", "Engine", "_sglang_available"),
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
# SGLang extraction
# ---------------------------------------------------------------------------


def _sglang_result(**meta):
    base = {
        "id": "sg-1",
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "finish_reason": {"type": "stop"},
        "e2e_latency": 1.25,
    }
    base.update(meta)
    return {"text": "hello", "meta_info": base}


class TestSGLangExtraction:
    def test_usage_and_cached_tokens(self):
        usage = SGLangInstrumentor()._extract_usage(_sglang_result(cached_tokens=5))
        assert usage["prompt_tokens"] == 12
        assert usage["completion_tokens"] == 8
        # SGLang's prefix-cache hits are the conventions' cache_read concept.
        assert usage["cache_read_input_tokens"] == 5

    def test_usage_summed_across_batch(self):
        usage = SGLangInstrumentor()._extract_usage([_sglang_result(), _sglang_result()])
        assert usage["total_tokens"] == 40

    def test_response_attributes(self):
        attrs = SGLangInstrumentor()._extract_response_attributes(_sglang_result())
        assert attrs["gen_ai.request.id"] == "sg-1"
        assert attrs["gen_ai.latency.e2e"] == 1.25

    def test_finish_reason_handles_both_shapes(self):
        inst = SGLangInstrumentor()
        assert inst._extract_finish_reason(_sglang_result()) == "stop"
        assert inst._extract_finish_reason(_sglang_result(finish_reason="length")) == "length"

    def test_no_meta_info_is_safe(self):
        inst = SGLangInstrumentor()
        assert inst._extract_usage({"text": "x"}) is None
        assert inst._extract_response_attributes({"text": "x"}) == {}


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
