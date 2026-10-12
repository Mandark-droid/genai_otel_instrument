"""An AsyncOpenAI call whose span is sampled out still works.

With ``GENAI_SAMPLING_RATE`` below 1 the tracer hands back a ``NonRecordingSpan``,
which has no ``name`` and no attributes. The async wrapper went on to record
metrics against it and raised ``AttributeError`` out of
``await client.chat.completions.create(...)`` after the provider had answered.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from genai_otel.config import OTelConfig
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors.base import BaseInstrumentor
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

MESSAGES = [{"role": "user", "content": "My card number is 4111 1111 1111 1111."}]


def _chat_result():
    message = SimpleNamespace(content="ok", role="assistant", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)
    return SimpleNamespace(id="x", model="gpt-4o-mini", choices=[choice], usage=usage)


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
    for name, value in saved.items():
        setattr(BaseInstrumentor, name, value)


@pytest.fixture
def sampled_out():
    """The async wrapper around a fake provider call, with every span dropped."""
    tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
    calls = []
    stream = object()

    async def create(*args, **kwargs):
        calls.append(kwargs)
        return stream if kwargs.get("stream") else _chat_result()

    instrumentor = OpenAIInstrumentor()
    instrumentor.config = OTelConfig(service_name="t")
    instrumentor.request_counter = MagicMock()

    def run(**kwargs):
        with patch("opentelemetry.trace.get_tracer", return_value=tracer):
            wrapped = instrumentor._create_async_span_wrapper(
                span_name="openai.chat.completion",
                extract_attributes=instrumentor._extract_openai_attributes,
            )(create)
            return asyncio.run(wrapped(model="gpt-4o-mini", messages=MESSAGES, **kwargs))

    return SimpleNamespace(run=run, calls=calls, stream=stream, instrumentor=instrumentor)


def test_sampled_out_call_returns_the_answer(no_detectors, sampled_out):
    result = sampled_out.run()

    assert result.choices[0].message.content == "ok"
    assert len(sampled_out.calls) == 1


def test_sampled_out_call_is_still_counted(no_detectors, sampled_out):
    """Client metrics do not depend on trace sampling."""
    sampled_out.run()

    assert sampled_out.instrumentor.request_counter.add.call_count == 1


def test_sampled_out_stream_is_handed_back_untouched(no_detectors, sampled_out):
    assert sampled_out.run(stream=True) is sampled_out.stream


def test_sampled_out_call_is_still_blocked(no_detectors, sampled_out):
    """Whether a request is blocked must not depend on trace sampling."""
    found = SimpleNamespace(
        has_pii=True, entity_counts={"CREDIT_CARD": 1}, entities=["x"], score=0.9
    )
    BaseInstrumentor._pii_detector = SimpleNamespace(detect=lambda text: found)
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True

    with pytest.raises(PolicyViolationError):
        sampled_out.run()

    assert sampled_out.calls == []


def test_sampled_out_provider_error_reaches_the_caller_unchanged(no_detectors):
    tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")

    async def create(*args, **kwargs):
        raise TimeoutError("provider timed out")

    instrumentor = OpenAIInstrumentor()
    instrumentor.config = OTelConfig(service_name="t")
    with patch("opentelemetry.trace.get_tracer", return_value=tracer):
        wrapped = instrumentor._create_async_span_wrapper(
            span_name="openai.chat.completion",
            extract_attributes=instrumentor._extract_openai_attributes,
        )(create)
        with pytest.raises(TimeoutError):
            asyncio.run(wrapped(model="gpt-4o-mini", messages=MESSAGES))
