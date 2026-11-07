"""A span wrapped around a generator is current only while the generator runs.

The wrapper used to attach the span when the generator was created and detach it in
the generator's ``finally``. A generator is resumed, and finalised, by whoever drives
it, so when the consumer broke out early, closed it from another task or left it to
the garbage collector, the detach ran in another context: OpenTelemetry logged
"Failed to detach context" and the span stayed current in the consumer's context, so
every later call (the next turn of a conversation, say) nested under a span that had
already ended. Google ADK's ``Runner.run_async`` is the case that found it.
"""

import asyncio
import gc
import logging

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from genai_otel.config import OTelConfig
from genai_otel.instrumentors._stream_context import drive_async_in_context, drive_in_context
from genai_otel.instrumentors.base import BaseInstrumentor


class _Instrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()
        self.items = []
        self.ends = []

    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None

    def _on_stream_item(self, span, item):
        self.items.append(item)

    def _on_stream_end(self, span, outcome):
        self.ends.append(outcome)


@pytest.fixture(autouse=True)
def empty_context():
    token = otel_context.attach(otel_context.Context())
    yield
    otel_context.detach(token)


class _DetachFailures(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        if "Failed to detach context" in record.getMessage():
            self.count += 1


@pytest.fixture
def detach_failures():
    handler = _DetachFailures()
    log = logging.getLogger("opentelemetry.context")
    log.addHandler(handler)
    yield handler
    log.removeHandler(handler)


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
    return inst, tracer, exporter


def _current_is_valid() -> bool:
    return trace.get_current_span().get_span_context().is_valid


def _by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


# --------------------------------------------------------------------------- async


def test_async_full_iteration_ends_ok_and_nests_inner_work(harness, detach_failures):
    inst, tracer, exporter = harness

    async def run():
        for i in range(3):
            with tracer.start_as_current_span("inner"):
                pass
            yield i

    traced = inst.create_span_wrapper("framework.run")(run)

    async def main():
        return [i async for i in traced()]

    assert asyncio.run(main()) == [0, 1, 2]
    (outer,) = _by_name(exporter, "framework.run")
    assert outer.status.status_code == StatusCode.OK
    assert "genai.stream.closed_early" not in outer.attributes
    inner = _by_name(exporter, "inner")
    assert len(inner) == 3
    assert all(s.parent.span_id == outer.context.span_id for s in inner)
    assert inst.items == [0, 1, 2]
    assert inst.ends == ["ok"]
    assert detach_failures.count == 0


def test_async_consumer_work_is_not_parented_to_the_stream_span(harness):
    """The consumer's context is untouched while it handles an item."""
    inst, tracer, exporter = harness

    async def run():
        yield 1
        yield 2

    traced = inst.create_span_wrapper("framework.run")(run)

    async def main():
        async for _ in traced():
            assert not _current_is_valid()
            with tracer.start_as_current_span("consumer"):
                pass

    asyncio.run(main())
    assert all(s.parent is None for s in _by_name(exporter, "consumer"))


def test_async_early_break_does_not_leak_into_later_turns(harness, detach_failures):
    """The Google ADK case: break on the final response, then run more turns."""
    inst, tracer, exporter = harness

    async def run():
        for i in range(5):
            with tracer.start_as_current_span("step"):
                pass
            yield i

    traced = inst.create_span_wrapper("framework.run")(run)

    async def turn():
        async for i in traced():
            if i == 1:
                break
        # Left to the garbage collector, as a caller that breaks out does.
        assert not _current_is_valid(), "the stream span was left current after a break"

    async def main():
        await turn()
        gc.collect()
        await asyncio.sleep(0)
        await turn()
        gc.collect()
        await asyncio.sleep(0)
        async for _ in traced():
            pass

    asyncio.run(main())
    gc.collect()

    runs = _by_name(exporter, "framework.run")
    assert len(runs) == 3
    assert len({s.context.trace_id for s in runs}) == 3, "turns merged into one trace"
    assert all(s.parent is None for s in runs)
    assert sum(1 for s in runs if s.attributes.get("genai.stream.closed_early")) == 2
    assert detach_failures.count == 0


def test_async_aclose_from_another_task(harness, detach_failures):
    inst, _, exporter = harness

    async def run():
        for i in range(10):
            yield i

    traced = inst.create_span_wrapper("framework.run")(run)

    async def main():
        agen = traced()
        assert await agen.__anext__() == 0
        await asyncio.get_running_loop().create_task(agen.aclose())
        assert not _current_is_valid()

    asyncio.run(main())
    (span,) = _by_name(exporter, "framework.run")
    assert span.attributes["genai.stream.closed_early"] is True
    assert span.status.status_code == StatusCode.UNSET
    assert inst.ends == ["closed"]
    assert detach_failures.count == 0


def test_async_generator_error_marks_the_span(harness):
    inst, _, exporter = harness

    async def run():
        yield 1
        raise ValueError("boom")

    traced = inst.create_span_wrapper("framework.run")(run)

    async def main():
        async for _ in traced():
            pass

    with pytest.raises(ValueError):
        asyncio.run(main())
    (span,) = _by_name(exporter, "framework.run")
    assert span.status.status_code == StatusCode.ERROR
    assert inst.ends == ["error"]


def test_async_asend_and_athrow_are_forwarded(harness):
    inst, _, _ = harness
    seen = []

    async def run():
        value = yield "first"
        seen.append(value)
        try:
            yield "second"
        except KeyError:
            seen.append("caught")
            yield "after-throw"

    traced = inst.create_span_wrapper("framework.run")(run)

    async def main():
        agen = traced()
        assert await agen.__anext__() == "first"
        assert await agen.asend("hello") == "second"
        assert await agen.athrow(KeyError("k")) == "after-throw"
        await agen.aclose()

    asyncio.run(main())
    assert seen == ["hello", "caught"]


def test_async_inner_generator_teardown_runs_under_the_span(harness):
    """Teardown inside the generator may trace; it must not start a trace of its own."""
    inst, tracer, exporter = harness

    async def run():
        try:
            yield 1
            yield 2
        finally:
            with tracer.start_as_current_span("teardown"):
                pass

    traced = inst.create_span_wrapper("framework.run")(run)

    async def main():
        agen = traced()
        await agen.__anext__()
        await agen.aclose()

    asyncio.run(main())
    (outer,) = _by_name(exporter, "framework.run")
    (teardown,) = _by_name(exporter, "teardown")
    assert teardown.parent.span_id == outer.context.span_id


# --------------------------------------------------------------------------- sync


def test_sync_early_break_does_not_leak(harness, detach_failures):
    inst, tracer, exporter = harness

    def run():
        for i in range(5):
            with tracer.start_as_current_span("step"):
                pass
            yield i

    traced = inst.create_span_wrapper("framework.stream")(run)

    for _ in range(2):
        gen = traced()
        for i in gen:
            if i == 1:
                break
        assert not _current_is_valid()
        del gen
        gc.collect()
    assert list(traced()) == [0, 1, 2, 3, 4]

    runs = _by_name(exporter, "framework.stream")
    assert len({s.context.trace_id for s in runs}) == 3
    assert [s.attributes.get("genai.stream.closed_early", False) for s in runs].count(True) == 2
    steps = _by_name(exporter, "step")
    run_ids = {s.context.span_id for s in runs}
    assert all(s.parent.span_id in run_ids for s in steps)
    assert detach_failures.count == 0


def test_sync_send_and_throw_are_forwarded(harness):
    inst, _, _ = harness
    seen = []

    def run():
        value = yield "first"
        seen.append(value)
        try:
            yield "second"
        except KeyError:
            seen.append("caught")
            yield "after-throw"

    traced = inst.create_span_wrapper("framework.stream")(run)
    gen = traced()
    assert next(gen) == "first"
    assert gen.send("hello") == "second"
    assert gen.throw(KeyError("k")) == "after-throw"
    gen.close()
    assert seen == ["hello", "caught"]


def test_sync_error_marks_the_span(harness):
    inst, _, exporter = harness

    def run():
        yield 1
        raise RuntimeError("boom")

    traced = inst.create_span_wrapper("framework.stream")(run)
    with pytest.raises(RuntimeError):
        list(traced())
    (span,) = _by_name(exporter, "framework.stream")
    assert span.status.status_code == StatusCode.ERROR


# --------------------------------------------------------------------------- drivers


def test_drivers_call_on_end_exactly_once():
    ends = []

    def gen():
        yield 1
        yield 2

    driven = drive_in_context(gen(), otel_context.get_current(), on_end=lambda o, e: ends.append(o))
    next(driven)
    driven.close()
    driven.close()
    assert ends == ["closed"]

    async def agen():
        yield 1

    async def main():
        async for _ in drive_async_in_context(
            agen(), otel_context.get_current(), on_end=lambda o, e: ends.append(o)
        ):
            pass

    asyncio.run(main())
    assert ends == ["closed", "ok"]


def test_a_failing_hook_never_breaks_the_stream(harness):
    inst, _, _ = harness

    def bad_item(span, item):
        raise RuntimeError("hook bug")

    inst._on_stream_item = bad_item

    def run():
        yield from range(3)

    assert list(inst.create_span_wrapper("framework.stream")(run)()) == [0, 1, 2]
