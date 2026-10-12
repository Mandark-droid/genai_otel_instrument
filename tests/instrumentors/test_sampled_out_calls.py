"""A call whose span is sampled out still works, on every wrapper that builds its own span.

With ``GENAI_SAMPLING_RATE`` below 1 the tracer hands back a ``NonRecordingSpan``,
which has no ``name`` and no attributes. Wrappers that build their own span went
on to record metrics against it, and ``AttributeError`` came out of a call the
provider had already answered. The request must come back, be counted, and be
subject to block policies whatever the sampler decided.

``test_openai_async_sampling.py`` has the same checks for ``AsyncOpenAI``.
"""

import asyncio
import logging
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from genai_otel.config import OTelConfig
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors import base
from genai_otel.instrumentors.azure_openai_instrumentor import AzureOpenAIInstrumentor
from genai_otel.instrumentors.base import BaseInstrumentor
from genai_otel.instrumentors.groq_instrumentor import GroqInstrumentor
from genai_otel.instrumentors.litellm_latency_instrumentor import LiteLLMLatencyInstrumentor
from genai_otel.instrumentors.mistralai_instrumentor import MistralAIInstrumentor

CARD = "4111 1111 1111 1111"
MESSAGES = [{"role": "user", "content": "My card number is {0}.".format(CARD)}]


def _chat_result():
    message = SimpleNamespace(content="ok", role="assistant", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)
    return SimpleNamespace(id="x", model="m", choices=[choice], usage=usage)


class _Provider:
    """Stands in for the provider: answers, streams or fails, and keeps what it was sent."""

    def __init__(self):
        self.calls = []
        self.stream = iter(())
        self.error = None

    def answer(self, kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.stream if kwargs.get("stream") else _chat_result()


def _ready(instrumentor, tracer):
    instrumentor.tracer = tracer
    instrumentor.config = OTelConfig(service_name="t")
    instrumentor._instrumented = True
    instrumentor.request_counter = MagicMock()
    instrumentor.latency_histogram = MagicMock()
    return instrumentor


# Each driver wires one instrumentor around the stand-in provider and returns the
# instrumentor and a function that makes one call through it.


def _groq(tracer, provider):
    with patch.dict("sys.modules", {"groq": MagicMock()}):
        instrumentor = _ready(GroqInstrumentor(), tracer)
    completions = SimpleNamespace(create=lambda *args, **kwargs: provider.answer(kwargs))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    instrumentor._instrument_client(client)
    return instrumentor, lambda **kw: client.chat.completions.create(
        model="m", messages=MESSAGES, **kw
    )


def _azure_openai(tracer, provider):
    class OpenAIClient:
        def complete(self, *args, **kwargs):
            return provider.answer(kwargs)

    module = types.ModuleType("azure.ai.openai")
    module.OpenAIClient = OpenAIClient
    modules = {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": module}
    with patch.dict("sys.modules", modules):
        instrumentor = AzureOpenAIInstrumentor()
        instrumentor.tracer = tracer
        instrumentor.request_counter = MagicMock()
        instrumentor.latency_histogram = MagicMock()
        instrumentor.instrument(OTelConfig(service_name="t"))
    client = OpenAIClient()
    return instrumentor, lambda **kw: client.complete(model="m", messages=MESSAGES, **kw)


def _mistral(tracer, provider):
    with patch.dict("sys.modules", {"mistralai": MagicMock()}):
        instrumentor = _ready(MistralAIInstrumentor(), tracer)

    def call(**kw):
        kwargs = dict({"model": "m", "messages": MESSAGES}, **kw)
        return instrumentor._wrap_chat_complete(
            lambda *a, **k: provider.answer(k), None, (), kwargs
        )

    return instrumentor, call


def _litellm(tracer, provider):
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        instrumentor = _ready(LiteLLMLatencyInstrumentor(), tracer)
    wrapped = instrumentor._wrap_sync(
        lambda *args, **kwargs: provider.answer(kwargs), "litellm.completion"
    )
    return instrumentor, lambda **kw: wrapped(model="m", messages=MESSAGES, **kw)


def _litellm_async(tracer, provider):
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        instrumentor = _ready(LiteLLMLatencyInstrumentor(), tracer)

    async def acompletion(*args, **kwargs):
        return provider.answer(kwargs)

    wrapped = instrumentor._wrap_async(acompletion, "litellm.acompletion")
    return instrumentor, lambda **kw: asyncio.run(wrapped(model="m", messages=MESSAGES, **kw))


DRIVERS = {
    "groq": _groq,
    "azure_openai": _azure_openai,
    "mistral": _mistral,
    "litellm": _litellm,
    "litellm_async": _litellm_async,
}


@pytest.fixture
def no_detectors():
    cleared = {
        "_pii_detector": None,
        "_toxicity_detector": None,
        "_bias_detector": None,
        "_prompt_injection_detector": None,
        "_restricted_topics_detector": None,
        "_hallucination_detector": None,
        "_blocking_active": False,
        "_blocking_policies": frozenset(),
    }
    saved = {name: getattr(BaseInstrumentor, name) for name in cleared}
    for name, value in cleared.items():
        setattr(BaseInstrumentor, name, value)
    yield
    assert base.drain_evaluations(30)
    for name, value in saved.items():
        setattr(BaseInstrumentor, name, value)


@pytest.fixture(params=sorted(DRIVERS))
def sampled_out(request, no_detectors):
    """One wrapper around the stand-in provider, with every span dropped."""
    tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
    provider = _Provider()
    instrumentor, call = DRIVERS[request.param](tracer, provider)
    return SimpleNamespace(
        name=request.param, call=call, provider=provider, instrumentor=instrumentor
    )


def _found_a_card(text, **kwargs):
    return SimpleNamespace(
        has_pii=True,
        entity_counts={"CREDIT_CARD": 1},
        entities=["x"],
        score=0.9,
        redacted_text=None,
        blocked=False,
    )


def test_sampled_out_call_returns_the_answer(sampled_out):
    result = sampled_out.call()

    assert result.choices[0].message.content == "ok"
    assert len(sampled_out.provider.calls) == 1


def test_sampled_out_call_is_still_counted(sampled_out):
    """Client metrics do not depend on trace sampling."""
    sampled_out.call()

    if sampled_out.name.startswith("litellm"):
        # This wrapper keeps no request counter for any call, recorded or not:
        # the provider SDK beneath it counts the request. Its one request-level
        # measurement is the duration, and that is still taken.
        assert sampled_out.instrumentor.latency_histogram.record.call_count == 1
    else:
        assert sampled_out.instrumentor.request_counter.add.call_count == 1


def test_sampled_out_stream_is_handed_back_untouched(sampled_out):
    if sampled_out.name == "mistral":
        pytest.skip("chat.complete does not stream; chat.stream is tested below")
    assert sampled_out.call(stream=True) is sampled_out.provider.stream


def test_sampled_out_call_is_still_blocked(sampled_out):
    """Whether a request is blocked must not depend on trace sampling."""
    BaseInstrumentor._pii_detector = SimpleNamespace(detect=_found_a_card)
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True

    with pytest.raises(PolicyViolationError):
        sampled_out.call()

    assert sampled_out.provider.calls == []


def test_sampled_out_provider_error_reaches_the_caller_unchanged(sampled_out):
    sampled_out.provider.error = TimeoutError("provider timed out")

    with pytest.raises(TimeoutError):
        sampled_out.call()


@pytest.mark.parametrize("mode", ["background", "inline"])
def test_sampled_out_call_with_detectors_configured_runs_none_and_warns_of_nothing(
    sampled_out, caplog, mode
):
    """There is no span to write a finding to, so there is nothing to look for."""
    seen = []
    BaseInstrumentor._pii_detector = SimpleNamespace(
        detect=lambda text, **kwargs: seen.append(text) or _found_a_card(text)
    )
    sampled_out.instrumentor.config.evaluation_mode = mode

    with caplog.at_level(logging.WARNING, logger="genai_otel"):
        result = sampled_out.call()
    assert base.drain_evaluations(30)

    assert result.choices[0].message.content == "ok"
    assert seen == []
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_sampled_out_mistral_stream_can_be_read_to_its_end(no_detectors):
    """The stream wrapper records its timings when the stream runs out."""
    tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
    with patch.dict("sys.modules", {"mistralai": MagicMock()}):
        instrumentor = _ready(MistralAIInstrumentor(), tracer)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    chunks = [SimpleNamespace(data=SimpleNamespace(usage=None)) for _ in range(3)]
    chunks.append(SimpleNamespace(data=SimpleNamespace(usage=usage)))

    stream = instrumentor._wrap_chat_stream(
        lambda *a, **k: iter(chunks), None, (), {"model": "m", "messages": MESSAGES}
    )

    assert list(stream) == chunks


# --- the shared pieces those wrappers call ----------------------------------------


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}

    def _extract_response_attributes(self, result):
        return {"gen_ai.response.model": "m"}

    def _extract_finish_reason(self, result):
        return "stop"


@pytest.fixture
def shared(no_detectors):
    instrumentor = _Instrumentor()
    instrumentor.config = OTelConfig(service_name="t", metrics_profile="full")
    for metric in ("latency_histogram", "token_counter", "cost_counter", "request_counter"):
        setattr(instrumentor, metric, MagicMock())
    span = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test").start_span("dropped")
    assert span.is_recording() is False
    return instrumentor, span


def test_recording_metrics_against_a_dropped_span_does_not_raise(shared, caplog):
    instrumentor, span = shared

    with caplog.at_level(logging.WARNING, logger="genai_otel"):
        instrumentor._record_result_metrics(span, _chat_result(), 0.0, {"model": "m"})

    # The call was timed; nothing else is written anywhere.
    assert instrumentor.latency_histogram.record.call_count == 1
    instrumentor.token_counter.add.assert_not_called()
    assert caplog.records == []


def test_detectors_are_not_run_for_a_dropped_span(shared):
    instrumentor, span = shared
    BaseInstrumentor._pii_detector = detector = MagicMock()

    assert instrumentor._evaluate_or_defer(span, (), {"messages": MESSAGES}, None) is False
    instrumentor._run_evaluation_checks(span, (), {"messages": MESSAGES}, None)
    asyncio.run(instrumentor._run_evaluation_checks_async(span, (), {"messages": MESSAGES}, None))

    detector.detect.assert_not_called()


def test_a_dropped_span_handed_over_for_evaluation_is_simply_ended(shared, caplog, monkeypatch):
    instrumentor, span = shared
    BaseInstrumentor._pii_detector = detector = MagicMock()
    queue = base._EvaluationQueue()
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", queue)

    with caplog.at_level(logging.WARNING, logger="genai_otel"):
        instrumentor._end_span_after_evaluation(span, (), {"messages": MESSAGES}, None)

    assert queue.submitted == 0
    detector.detect.assert_not_called()
    assert caplog.records == []


def test_stream_timings_against_a_dropped_span_do_not_raise(shared):
    instrumentor, span = shared
    for metric in (
        "ttft_histogram",
        "time_to_first_token_histogram",
        "time_per_output_token_histogram",
        "tbt_histogram",
        "error_counter",
    ):
        setattr(instrumentor, metric, MagicMock())
    timing = base._StreamTiming(0.0)
    timing.tbt_sum, timing.tbt_count = 0.2, 2

    instrumentor._record_time_to_first_token(span, 0.1, "m")
    instrumentor._record_time_per_output_token(span, 0.1, 1.0, 5, "m")
    instrumentor._record_mean_time_between_tokens(span, timing, "m")
    instrumentor._observe_stream_chunk(span, base._StreamTiming(0.0), object(), "m")
    instrumentor._fail_stream(span, RuntimeError("dropped"))
