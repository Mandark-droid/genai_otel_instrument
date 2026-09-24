"""Async clients and streamed usage for the Anthropic instrumentor.

Two defects are covered here, both of which produced a span that looked fine
and carried no economics:

* ``AsyncAnthropic`` was never wrapped. Only ``anthropic.Anthropic.__init__``
  was, so an application built on the async client instrumented cleanly and
  emitted no spans at all. The same held for the Bedrock/Vertex clients.

* Streamed usage was read from the final chunk only. Anthropic sends input
  tokens on ``message_start`` and output tokens on ``message_delta``, then
  ends with a bare ``message_stop`` -- so the finalizer found nothing, set no
  token attributes, and never reached the cost calculation. That one hit the
  *sync* path too, which is why the streaming tests below fail against the
  code that shipped in 1.29.0.

``SimpleNamespace`` is used rather than ``MagicMock`` for the stream events on
purpose: a MagicMock auto-creates ``.usage``, so a bare ``message_stop`` would
appear to carry usage and the bug under test would be invisible.
"""

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor

MODEL = "claude-sonnet-4-6"
PROMPT_TOKENS = 15
COMPLETION_TOKENS = 25


class _Capture(SimpleSpanProcessor):
    """Collect finished spans without needing an exporter."""

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
def captured():
    """Give the instrumentor a real tracer whose finished spans we can read."""
    spans = []
    provider = TracerProvider()
    provider.add_span_processor(_Capture(spans))
    return spans, provider


def _raw_stream_events():
    """The event sequence a real ``messages.create(stream=True)`` produces.

    Input tokens arrive first, nested under ``message_start.message.usage``;
    output tokens arrive on ``message_delta.usage``; the stream ends on a
    ``message_stop`` that carries nothing at all.
    """
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=PROMPT_TOKENS, output_tokens=0)
            ),
        ),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="hello")),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(input_tokens=0, output_tokens=COMPLETION_TOKENS),
        ),
        SimpleNamespace(type="message_stop"),
    ]


def _buffered_response():
    """A non-streamed Message, with usage where the buffered API reports it."""
    return SimpleNamespace(
        id="msg_1",
        model=MODEL,
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=PROMPT_TOKENS, output_tokens=COMPLETION_TOKENS),
    )


def _make_fake_anthropic_module(sync_create=None, async_create=None, extra_clients=()):
    """Build a stand-in ``anthropic`` module exposing real client classes.

    Real classes rather than MagicMock attributes because ``instrument()``
    assigns to ``__init__``, which a MagicMock attribute cannot meaningfully
    accept.
    """
    module = types.ModuleType("anthropic")

    def _client_class(create):
        class _Client:
            def __init__(self, base_url=None, api_key=None):
                self.base_url = base_url
                self.api_key = api_key
                self.messages = SimpleNamespace(create=create)

        return _Client

    if sync_create is not None:
        module.Anthropic = _client_class(sync_create)
    if async_create is not None:
        module.AsyncAnthropic = _client_class(async_create)
    for name in extra_clients:
        setattr(module, name, _client_class(sync_create or (lambda **kw: None)))
    return module


def _prepare(instrumentor, provider):
    instrumentor.tracer = provider.get_tracer(__name__)
    return instrumentor


def _attrs(span):
    return dict(span.attributes or {})


# --------------------------------------------------------------------------
# Streaming usage folding -- these fail against 1.29.0
# --------------------------------------------------------------------------


def test_sync_streaming_folds_usage_split_across_events(captured):
    """Tokens and cost survive a stream that ends on a bare message_stop.

    Against the shipped code this fails on the very first assertion: the
    finalizer read ``message_stop``, found no ``.usage``, and left the span
    with no token attributes and no cost at all.
    """
    spans, provider = captured

    def create(**kwargs):
        yield from _raw_stream_events()

    fake = _make_fake_anthropic_module(sync_create=create)
    with patch.dict(sys.modules, {"anthropic": fake}):
        inst = _prepare(AnthropicInstrumentor(), provider)
        inst.instrument(OTelConfig(service_name="test"))

        client = fake.Anthropic(base_url=None, api_key="k")
        stream = client.messages.create(
            model=MODEL, max_tokens=64, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        assert spans == [], "span closed before the stream was consumed"
        assert len(list(stream)) == 4

    assert len(spans) == 1
    attrs = _attrs(spans[0])
    assert attrs["gen_ai.usage.prompt_tokens"] == PROMPT_TOKENS
    assert attrs["gen_ai.usage.completion_tokens"] == COMPLETION_TOKENS
    assert attrs["gen_ai.usage.total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS
    assert attrs["gen_ai.usage.cost.total"] > 0
    assert attrs["gen_ai.usage.cost.pricing_source"] == "table"


def test_async_streaming_folds_usage_split_across_events(captured):
    """The async client reports the same tokens and cost as the sync one."""
    spans, provider = captured

    async def create(**kwargs):
        async def _agen():
            for event in _raw_stream_events():
                yield event

        return _agen()

    fake = _make_fake_anthropic_module(async_create=create)
    with patch.dict(sys.modules, {"anthropic": fake}):
        inst = _prepare(AnthropicInstrumentor(), provider)
        inst.instrument(OTelConfig(service_name="test"))

        client = fake.AsyncAnthropic(base_url=None, api_key="k")

        async def run():
            stream = await client.messages.create(
                model=MODEL,
                max_tokens=64,
                stream=True,
                messages=[{"role": "user", "content": "hi"}],
            )
            assert spans == [], "span closed before the stream was consumed"
            return [event async for event in stream]

        assert len(asyncio.run(run())) == 4

    assert len(spans) == 1
    attrs = _attrs(spans[0])
    assert attrs["gen_ai.usage.prompt_tokens"] == PROMPT_TOKENS
    assert attrs["gen_ai.usage.completion_tokens"] == COMPLETION_TOKENS
    assert attrs["gen_ai.usage.cost.total"] > 0


def test_cumulative_message_deltas_are_not_summed(captured):
    """Several message_delta events restate a running total, they do not add up.

    Summing them would multiply both the reported output tokens and the bill.
    """
    spans, provider = captured

    def create(**kwargs):
        yield SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=0)),
        )
        yield SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=5))
        yield SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=9))
        yield SimpleNamespace(type="message_stop")

    fake = _make_fake_anthropic_module(sync_create=create)
    with patch.dict(sys.modules, {"anthropic": fake}):
        inst = _prepare(AnthropicInstrumentor(), provider)
        inst.instrument(OTelConfig(service_name="test"))
        client = fake.Anthropic(base_url=None, api_key="k")
        list(client.messages.create(model=MODEL, stream=True, messages=[]))

    attrs = _attrs(spans[0])
    assert attrs["gen_ai.usage.completion_tokens"] == 9
    assert attrs["gen_ai.usage.prompt_tokens"] == 10
    assert attrs["gen_ai.usage.total_tokens"] == 19


# --------------------------------------------------------------------------
# Async client wiring
# --------------------------------------------------------------------------


def test_async_client_emits_span_with_usage_and_cost(captured):
    """A buffered AsyncAnthropic call produces one priced span."""
    spans, provider = captured

    async def create(**kwargs):
        return _buffered_response()

    fake = _make_fake_anthropic_module(async_create=create)
    with patch.dict(sys.modules, {"anthropic": fake}):
        inst = _prepare(AnthropicInstrumentor(), provider)
        inst.instrument(OTelConfig(service_name="test"))

        client = fake.AsyncAnthropic(base_url=None, api_key="k")
        result = asyncio.run(
            client.messages.create(
                model=MODEL, max_tokens=64, messages=[{"role": "user", "content": "hi"}]
            )
        )

    assert result.id == "msg_1"
    assert len(spans) == 1
    attrs = _attrs(spans[0])
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == MODEL
    assert attrs["gen_ai.usage.prompt_tokens"] == PROMPT_TOKENS
    assert attrs["gen_ai.usage.completion_tokens"] == COMPLETION_TOKENS
    assert attrs["gen_ai.usage.cost.total"] > 0


def test_bedrock_and_vertex_clients_are_wrapped():
    """The cloud-hosted client classes get the same treatment as the direct one."""

    def create(**kwargs):
        return _buffered_response()

    fake = _make_fake_anthropic_module(
        sync_create=create,
        async_create=create,
        extra_clients=("AnthropicBedrock", "AsyncAnthropicBedrock", "AnthropicVertex"),
    )
    originals = {
        name: getattr(fake, name).__init__
        for name in ("Anthropic", "AsyncAnthropic", "AnthropicBedrock", "AnthropicVertex")
    }

    with patch.dict(sys.modules, {"anthropic": fake}):
        AnthropicInstrumentor().instrument(OTelConfig(service_name="test"))

    for name, original in originals.items():
        assert getattr(fake, name).__init__ is not original, f"{name}.__init__ was not wrapped"


def test_instrument_is_idempotent_and_still_covers_the_async_client():
    """Running instrument() twice wraps every client once, async included.

    The module-level "already instrumented" flag used to be set inside the
    sync branch, so a run that wrapped ``Anthropic`` marked the module done
    and the async client could never be picked up afterwards.
    """

    def create(**kwargs):
        return _buffered_response()

    fake = _make_fake_anthropic_module(sync_create=create, async_create=create)
    # Read through __dict__: a wrapt wrapper is a descriptor, so every
    # attribute access hands back a fresh BoundFunctionWrapper and an identity
    # check on the attribute would compare two bindings of the same function.
    original_async_init = fake.AsyncAnthropic.__dict__["__init__"]

    with patch.dict(sys.modules, {"anthropic": fake}):
        AnthropicInstrumentor().instrument(OTelConfig(service_name="test"))
        after_first = fake.AsyncAnthropic.__dict__["__init__"]
        assert after_first is not original_async_init, "async client was never wrapped"
        assert after_first.__wrapped__ is original_async_init

        AnthropicInstrumentor().instrument(OTelConfig(service_name="test"))
        after_second = fake.AsyncAnthropic.__dict__["__init__"]
        assert after_second is after_first, "__init__ was replaced on the second run"
        # The decisive check: one more layer here would mean every call runs
        # the instrumentation twice and reports the request twice.
        assert after_second.__wrapped__ is original_async_init, "wrappers were stacked"


def test_async_client_pointed_at_an_aggregator_is_skipped(captured):
    """The base-url claim still holds for async clients.

    A CometAPI client is traced by its own instrumentor; wrapping it here too
    would emit a duplicate span and double-count tokens and cost.

    The unclaimed client is exercised alongside it deliberately -- without that
    control the test would also pass if the async client were never wrapped at
    all, which is the very defect this file exists to catch.
    """
    spans, provider = captured

    async def create(**kwargs):
        return _buffered_response()

    # Both clients, as the real SDK has: CometAPI registers its base-url claim
    # only once it has something to wrap, and it wraps the sync client.
    fake = _make_fake_anthropic_module(sync_create=create, async_create=create)
    with patch.dict(sys.modules, {"anthropic": fake, "openai": None}):
        from genai_otel.instrumentors.cometapi_instrumentor import CometAPIInstrumentor

        config = OTelConfig(service_name="test")
        CometAPIInstrumentor().instrument(config)  # registers the base-url claim

        inst = _prepare(AnthropicInstrumentor(), provider)
        inst.instrument(config)

        claimed = fake.AsyncAnthropic(base_url="https://api.cometapi.com", api_key="k")
        asyncio.run(claimed.messages.create(model=MODEL, messages=[]))
        assert spans == [], "generic instrumentor traced a client claimed by CometAPI"

        direct = fake.AsyncAnthropic(base_url=None, api_key="k")
        asyncio.run(direct.messages.create(model=MODEL, messages=[]))

    assert len(spans) == 1, "the unclaimed async client should still be traced"


# --------------------------------------------------------------------------
# Usage extraction
# --------------------------------------------------------------------------


def test_extract_usage_reads_nested_message_usage():
    """message_start / message_stop carry usage one level down, on .message."""
    inst = AnthropicInstrumentor()
    event = SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=3)),
    )

    usage = inst._extract_usage(event)

    assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_extract_usage_returns_none_for_a_bare_message_stop():
    """A message_stop with no usage must report nothing, not zeros.

    Zeros would be indistinguishable from a real free call and would price the
    request at 0.00 instead of leaving it unpriced.
    """
    inst = AnthropicInstrumentor()

    assert inst._extract_usage(SimpleNamespace(type="message_stop")) is None


def test_extract_usage_tolerates_none_token_counts():
    """The streaming usage models declare several counts Optional."""
    inst = AnthropicInstrumentor()
    event = SimpleNamespace(usage=SimpleNamespace(input_tokens=None, output_tokens=12))

    usage = inst._extract_usage(event)

    assert usage == {"prompt_tokens": 0, "completion_tokens": 12, "total_tokens": 12}


def test_extract_usage_keeps_cache_token_counts():
    """Cache reads and writes are priced differently and must survive."""
    inst = AnthropicInstrumentor()
    event = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=2,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=50,
        )
    )

    usage = inst._extract_usage(event)

    assert usage["cache_read_input_tokens"] == 100
    assert usage["cache_creation_input_tokens"] == 50


@pytest.mark.parametrize(
    ("sdk_module", "instrumentor_module", "instrumentor_name"),
    [
        ("openai", "genai_otel.instrumentors.openai_instrumentor", "OpenAIInstrumentor"),
        ("groq", "genai_otel.instrumentors.groq_instrumentor", "GroqInstrumentor"),
    ],
)
def test_accumulator_is_inert_for_providers_that_do_not_override(
    sdk_module, instrumentor_module, instrumentor_name
):
    """The base hook must leave ``timing.usage`` alone for every other provider.

    A provider that does not override it keeps reading usage off the final
    chunk exactly as it always did, so adding Anthropic's accumulation moved
    no other provider's token counts or cost. OpenAI is named explicitly
    because it is the provider whose streamed cost accounting we most need to
    be able to state has not changed.
    """
    from genai_otel.instrumentors.base import _StreamTiming

    instrumentor_cls = getattr(importlib.import_module(instrumentor_module), instrumentor_name)

    with patch.dict(sys.modules, {sdk_module: MagicMock()}):
        timing = _StreamTiming(0.0)
        # A MagicMock chunk would happily yield usage to anything that looked
        # for it; the assertion is that the default hook does not look at all.
        instrumentor_cls()._accumulate_stream_usage(timing, MagicMock())

    assert timing.usage is None
