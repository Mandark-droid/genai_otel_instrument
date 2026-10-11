"""A traced coroutine is current where it runs, not where it was created.

Calling an ``async def`` only builds a coroutine; the work happens wherever it is
awaited. Under ``asyncio.run``, ``asyncio.gather`` or ``create_task`` that is a
different task with its own context. The span wrapper used to make its span
current when the coroutine was created and undo that when the coroutine
finished - two different contexts - so the undo failed, the caller was left with
an ended span as its current span, and everything it did afterwards joined that
trace. Two gathered calls nested the second under the first.
"""

import asyncio
import logging

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel import agent_budget
from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


@pytest.fixture(autouse=True)
def empty_context():
    """Start from a context with no current span, whatever ran before."""
    token = otel_context.attach(otel_context.Context())
    yield
    otel_context.detach(token)


@pytest.fixture
def harness():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    inst = _Instrumentor()
    inst.config = OTelConfig()
    inst._instrumented = True
    inst.tracer = tracer

    def spans():
        return {s.name: s for s in exporter.get_finished_spans()}

    return inst, tracer, spans


def _current_span_is_valid() -> bool:
    return trace.get_current_span().get_span_context().is_valid


def _parent_id(span):
    return span.parent.span_id if span.parent else None


def test_asyncio_run_leaves_no_current_span(harness):
    inst, _, spans = harness

    async def call():
        return "ok"

    traced = inst.create_span_wrapper("provider.call")(call)

    assert asyncio.run(traced()) == "ok"

    assert "provider.call" in spans()
    assert not _current_span_is_valid(), "an ended span was left as the caller's current span"


def test_asyncio_run_detaches_where_it_attached(harness, caplog):
    inst, _, _ = harness

    async def call():
        return "ok"

    traced = inst.create_span_wrapper("provider.call")(call)

    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        asyncio.run(traced())

    assert not [r for r in caplog.records if "detach" in r.getMessage().lower()]


def test_gathered_calls_are_siblings(harness):
    inst, tracer, spans = harness

    async def first():
        return 1

    async def second():
        return 2

    traced_first = inst.create_span_wrapper("first.call")(first)
    traced_second = inst.create_span_wrapper("second.call")(second)

    async def main():
        with tracer.start_as_current_span("request") as request:
            await asyncio.gather(traced_first(), traced_second())
            return request.get_span_context().span_id

    request_id = asyncio.run(main())

    got = spans()
    assert _parent_id(got["first.call"]) == request_id
    assert _parent_id(got["second.call"]) == request_id, "the second call nested under the first"


def test_work_inside_a_gathered_call_is_its_own_child(harness):
    inst, tracer, spans = harness

    def make(name):
        async def call():
            await asyncio.sleep(0)
            with tracer.start_as_current_span(name + ".inner"):
                await asyncio.sleep(0)

        return inst.create_span_wrapper(name + ".call")(call)

    first, second = make("first"), make("second")

    async def main():
        await asyncio.gather(first(), second())

    asyncio.run(main())

    got = spans()
    assert _parent_id(got["first.inner"]) == got["first.call"].context.span_id
    assert _parent_id(got["second.inner"]) == got["second.call"].context.span_id
    assert got["first.call"].context.trace_id != got["second.call"].context.trace_id


def test_task_started_from_a_traced_call_site_does_not_leak(harness):
    inst, tracer, spans = harness

    async def call():
        return "ok"

    traced = inst.create_span_wrapper("provider.call")(call)

    async def main():
        task = asyncio.create_task(traced())
        await task
        # What the caller does next must not join the finished call's trace.
        with tracer.start_as_current_span("afterwards"):
            pass

    asyncio.run(main())

    got = spans()
    assert got["afterwards"].parent is None
    assert got["afterwards"].context.trace_id != got["provider.call"].context.trace_id


def test_awaited_in_place_still_parents_nested_work(harness):
    """The ordinary case - ``await client.call()`` in one task - is unchanged."""
    inst, tracer, spans = harness

    async def call():
        with tracer.start_as_current_span("inner"):
            return "ok"

    traced = inst.create_span_wrapper("provider.call")(call)

    async def main():
        with tracer.start_as_current_span("request") as request:
            await traced()
            assert trace.get_current_span() is request

    asyncio.run(main())

    got = spans()
    assert _parent_id(got["inner"]) == got["provider.call"].context.span_id
    assert _parent_id(got["provider.call"]) == got["request"].context.span_id


def test_coroutine_error_is_recorded_and_the_context_restored(harness):
    inst, _, spans = harness

    async def call():
        raise RuntimeError("boom")

    traced = inst.create_span_wrapper("provider.call")(call)

    with pytest.raises(RuntimeError):
        asyncio.run(traced())

    assert spans()["provider.call"].status.status_code.name == "ERROR"
    assert not _current_span_is_valid()


def test_agent_coroutines_leave_no_budget_frame_open(harness):
    inst, _, spans = harness

    async def run_agent():
        # One inference inside the agent invocation.
        agent_budget.record_inference(40)
        return "ok"

    traced = inst.create_span_wrapper(
        "framework.agent.run",
        extract_attributes=lambda instance, args, kwargs: {"gen_ai.operation.name": "agent.run"},
    )(run_agent)

    async def main():
        for _ in range(3):
            await traced()
        return agent_budget._BUDGET_STACK.get()

    assert asyncio.run(main()) == ()
    assert agent_budget._BUDGET_STACK.get() == ()
    # The frame is closed in the task that opened it, so its consumption is reported.
    attrs = dict(spans()["framework.agent.run"].attributes)
    assert attrs.get("gen_ai.agent.token_budget.consumed") == 40


def test_sampled_out_coroutine_keeps_its_span_current_while_it_runs():
    """A sampled-out call still has to parent what it does, or the children start
    traces of their own and are sampled separately."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    inst = _Instrumentor()
    inst.config = OTelConfig()
    inst._instrumented = True
    inst.tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
    seen = {}

    async def call():
        seen["context"] = trace.get_current_span().get_span_context()
        return "ok"

    traced = inst.create_span_wrapper("provider.call")(call)

    assert asyncio.run(traced()) == "ok"
    assert seen["context"].is_valid, "the coroutine ran with no span to inherit from"
    assert not seen["context"].trace_flags.sampled
    assert not _current_span_is_valid()


def test_sampled_out_coroutine_is_counted_when_it_finishes():
    from unittest.mock import MagicMock

    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    inst = _Instrumentor()
    inst.config = OTelConfig()
    inst._instrumented = True
    inst.tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
    inst.request_counter = MagicMock()

    async def call():
        return "ok"

    traced = inst.create_span_wrapper("provider.call")(call)
    coroutine = traced()
    assert inst.request_counter.add.call_count == 0, "counted before it had run"

    asyncio.run(coroutine)
    assert inst.request_counter.add.call_count == 1


def test_context_set_before_the_await_is_visible_inside(harness):
    """What the caller puts in its context between building a coroutine and awaiting
    it belongs to the call."""
    inst, _, _ = harness
    key = otel_context.create_key("request-scoped")
    seen = {}

    async def call():
        seen["value"] = otel_context.get_value(key)

    traced = inst.create_span_wrapper("provider.call")(call)

    async def main():
        coroutine = traced()
        token = otel_context.attach(otel_context.set_value(key, "tenant-7"))
        try:
            await coroutine
        finally:
            otel_context.detach(token)

    asyncio.run(main())
    assert seen["value"] == "tenant-7"


def _agent_attributes(instance, args, kwargs):
    return {"gen_ai.operation.name": "agent.run"}


def test_agent_generators_leave_no_budget_frame_open(harness):
    """Generator results are iterated later, somewhere else. A frame opened at the
    call was never closed and swallowed the enclosing agent's consumption."""
    inst, _, _ = harness

    def stream_agent():
        yield "chunk"

    async def astream_agent():
        yield "chunk"

    traced = inst.create_span_wrapper("framework.agent.run_stream", _agent_attributes)(stream_agent)
    atraced = inst.create_span_wrapper("framework.agent.run_stream", _agent_attributes)(
        astream_agent
    )

    chunks = traced()
    assert agent_budget._BUDGET_STACK.get() == ()
    assert list(chunks) == ["chunk"]

    async def main():
        stream = atraced()
        assert agent_budget._BUDGET_STACK.get() == ()
        return [chunk async for chunk in stream]

    assert asyncio.run(main()) == ["chunk"]
    assert agent_budget._BUDGET_STACK.get() == ()
