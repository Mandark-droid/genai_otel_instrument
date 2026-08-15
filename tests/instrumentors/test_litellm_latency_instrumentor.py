"""Tests for the opt-in litellm latency instrumentor (issue #22).

The risky part is not the measurement, it is the *suppression*: this wrapper
sits outside provider SDKs we already instrument, so it must stay silent when
an inner span already accounted for the request. Getting that wrong bills one
request twice.
"""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import (
    _note_inner_measurement,
    close_inner_measurement_scope,
    inner_measurement_scope,
)
from genai_otel.instrumentors.litellm_latency_instrumentor import LiteLLMLatencyInstrumentor

TTFT = "gen_ai.server.time_to_first_token"
TPOT = "gen_ai.server.time_per_output_token"
COST = "gen_ai.usage.cost.total"


class _Capture(SimpleSpanProcessor):
    def __init__(self, sink):  # pylint: disable=super-init-not-called
        self._sink = sink

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        self._sink.append(span)

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


@pytest.fixture
def inst_and_spans():
    spans = []
    provider = TracerProvider()
    provider.add_span_processor(_Capture(spans))
    inst = LiteLLMLatencyInstrumentor()
    inst.tracer = provider.get_tracer(__name__)
    inst.config = OTelConfig()
    return inst, spans


def _usage_chunk(prompt=4, completion=6):
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    chunk = MagicMock()
    chunk.usage = usage
    return chunk


class _FakeStream:
    """Stands in for litellm's CustomStreamWrapper."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


# --- scope semantics -------------------------------------------------------


def test_nested_scopes_share_one_holder():
    """litellm.acompletion re-enters litellm.completion; a nested scope must not
    shadow the outer one, or the provider's note never reaches the outer wrapper."""
    outer, outer_token = inner_measurement_scope()
    inner, inner_token = inner_measurement_scope()

    assert inner is outer, "nested scope shadowed the enclosing holder"
    assert inner_token is None, "nested scope must not own the reset token"

    _note_inner_measurement()
    assert outer, "note taken in the inner scope is invisible to the outer one"

    close_inner_measurement_scope(inner_token)  # no-op
    assert inner_measurement_scope()[0] is outer, "inner close tore down the outer scope"
    close_inner_measurement_scope(outer_token)


def test_note_outside_any_scope_is_harmless():
    """A direct provider call has no litellm wrapper around it."""
    _note_inner_measurement()  # must not raise


# --- suppression -----------------------------------------------------------


def test_streamed_call_measures_when_nothing_inner_measured(inst_and_spans):
    """The httpx-provider path: no inner span exists, so this one reports."""
    import asyncio

    inst, spans = inst_and_spans

    async def fake_acompletion(**kwargs):
        return _FakeStream([MagicMock(usage=None), _usage_chunk()])

    wrapped = inst._wrap_async(fake_acompletion, "litellm.acompletion")

    async def run():
        stream = await wrapped(model="gemini/gemini-2.5-flash", stream=True)
        assert not spans, "span closed before the stream was consumed"
        return [c async for c in stream]

    assert len(asyncio.run(run())) == 2
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert TTFT in attrs
    assert TPOT in attrs


def test_streamed_call_stays_silent_when_inner_span_measured(inst_and_spans):
    """The provider-SDK path: the inner span reported, so this one must not."""
    import asyncio

    inst, spans = inst_and_spans

    async def fake_acompletion(**kwargs):
        # Stand in for the inner OpenAI span taking over measurement.
        _note_inner_measurement()
        return _FakeStream([MagicMock(usage=None), _usage_chunk()])

    wrapped = inst._wrap_async(fake_acompletion, "litellm.acompletion")

    async def run():
        stream = await wrapped(model="gpt-4o-mini", stream=True)
        return [c async for c in stream]

    assert len(asyncio.run(run())) == 2
    assert len(spans) == 1, "the span must still exist as the parent"
    attrs = dict(spans[0].attributes or {})
    assert TTFT not in attrs, "double-counted TTFT"
    assert TPOT not in attrs
    assert COST not in attrs, "double-billed the request"


def test_non_streamed_call_stays_silent_when_inner_span_measured(inst_and_spans):
    """The path that double-billed before the fix: non-streamed via a provider SDK."""
    import asyncio

    inst, spans = inst_and_spans

    async def fake_acompletion(**kwargs):
        _note_inner_measurement()
        return _usage_chunk()

    wrapped = inst._wrap_async(fake_acompletion, "litellm.acompletion")
    asyncio.run(wrapped(model="gpt-4o-mini"))

    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert COST not in attrs, "double-billed the request"
    assert TTFT not in attrs


def test_non_streamed_call_carries_no_streaming_attributes(inst_and_spans):
    """Absent, never zero - even on the path this instrumentor owns."""
    import asyncio

    inst, spans = inst_and_spans

    async def fake_acompletion(**kwargs):
        return _usage_chunk()

    wrapped = inst._wrap_async(fake_acompletion, "litellm.acompletion")
    asyncio.run(wrapped(model="gemini/gemini-2.5-flash"))

    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert TTFT not in attrs
    assert TPOT not in attrs


def test_errors_propagate_and_close_the_span(inst_and_spans):
    import asyncio

    inst, spans = inst_and_spans

    async def fake_acompletion(**kwargs):
        raise RuntimeError("upstream exploded")

    wrapped = inst._wrap_async(fake_acompletion, "litellm.acompletion")
    with pytest.raises(RuntimeError, match="upstream exploded"):
        asyncio.run(wrapped(model="gpt-4o-mini"))

    assert len(spans) == 1, "span leaked on error"


def test_availability_check_does_not_import_litellm():
    """Constructing the instrumentor must not pay litellm's multi-second import."""
    import sys

    with patch.dict(sys.modules):
        sys.modules.pop("litellm", None)
        LiteLLMLatencyInstrumentor()
        assert "litellm" not in sys.modules, "constructor imported litellm"


def test_instrument_is_idempotent():
    """instrument() twice must not stack wrappers."""
    fake_litellm = MagicMock()
    fake_litellm._genai_otel_latency_instrumented = False
    original = fake_litellm.acompletion

    with patch.dict("sys.modules", {"litellm": fake_litellm}):
        inst = LiteLLMLatencyInstrumentor()
        inst._litellm_available = True
        inst.instrument(OTelConfig())
        after_first = fake_litellm.acompletion
        inst2 = LiteLLMLatencyInstrumentor()
        inst2._litellm_available = True
        inst2.instrument(OTelConfig())

    assert after_first is not original, "first instrument() did not wrap"
    assert fake_litellm.acompletion is after_first, "second instrument() stacked a wrapper"
