"""What the stream wrapper does for every chunk, and what it leaves to the end.

A metric sample per chunk was most of what streaming instrumentation cost the
caller. The time between chunks is now summed as the stream is read and recorded
once, as the stream's mean, when it ends. The ``full`` metrics profile keeps a
sample per chunk for those who want the distribution inside a stream.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors import base
from genai_otel.instrumentors.base import BaseInstrumentor, _StreamTiming
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

START = 100.0
# Chunks arrive 0.5 s after the request, then 0.1, 0.2 and 0.4 s apart.
ARRIVALS = [100.5, 100.6, 100.8, 101.2]
GAPS = [0.1, 0.2, 0.4]
END = 101.3


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


class _Clock:
    """Stands in for ``time.time``: one reading per chunk, then the time the stream ended."""

    def __init__(self, readings, afterwards):
        self._readings = list(readings)
        self._afterwards = afterwards

    def __call__(self):
        return self._readings.pop(0) if self._readings else self._afterwards


@pytest.fixture
def stream():
    """Read a stream of chunks through the wrapper; returns the finished span and
    the samples recorded into the time-between-tokens histogram."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def read(chunks=4, fail_after=None, emit_measurements=True, use_async=False, **config):
        instrumentor = _Instrumentor()
        instrumentor.config = OTelConfig(service_name="t", **config)
        instrumentor.tbt_histogram = MagicMock()
        span = provider.get_tracer("test").start_span("test.chat")
        arrivals = ARRIVALS[:chunks]

        def source():
            for number in range(chunks):
                if fail_after is not None and number == fail_after:
                    raise ConnectionError("connection reset")
                yield {"delta": number}

        async def async_source():
            for item in source():
                yield item

        with patch.object(base.time, "time", _Clock(arrivals, END)):
            if use_async:
                wrapped = instrumentor._wrap_async_streaming_response(
                    async_source(), span, START, "m", emit_measurements
                )

                async def consume():
                    return [chunk async for chunk in wrapped]

                asyncio.run(consume())
            else:
                wrapped = instrumentor._wrap_streaming_response(
                    source(), span, START, "m", emit_measurements
                )
                try:
                    for _chunk in wrapped:
                        pass
                except ConnectionError:
                    pass
        samples = [call.args[0] for call in instrumentor.tbt_histogram.record.call_args_list]
        attributes = [call.args[1] for call in instrumentor.tbt_histogram.record.call_args_list]
        return exporter.get_finished_spans()[-1], samples, attributes

    return read


def test_one_sample_per_stream_the_mean_time_between_chunks(stream):
    span, samples, attributes = stream()

    assert samples == [pytest.approx(sum(GAPS) / len(GAPS))]
    assert attributes == [{"model": "m", "operation": "test.chat"}]
    assert span.attributes["gen_ai.streaming.token_count"] == 4


def test_an_async_stream_is_measured_the_same_way(stream):
    _, samples, attributes = stream(use_async=True)

    assert samples == [pytest.approx(sum(GAPS) / len(GAPS))]
    assert attributes == [{"model": "m", "operation": "test.chat"}]


def test_the_full_profile_keeps_a_sample_per_chunk(stream):
    _, samples, attributes = stream(metrics_profile="full")

    assert samples == [pytest.approx(gap) for gap in GAPS]
    assert attributes == [{"model": "m", "operation": "test.chat"}] * 3


def test_a_stream_of_one_chunk_has_no_time_between_chunks(stream):
    _, samples, _ = stream(chunks=1)
    assert samples == []


def test_an_outer_span_that_leaves_measurements_to_the_inner_one_records_nothing(stream):
    _, samples, _ = stream(emit_measurements=False)
    assert samples == []


def test_a_stream_that_fails_part_way_still_reports_what_it_measured(stream):
    span, samples, _ = stream(fail_after=3)

    assert samples == [pytest.approx((0.1 + 0.2) / 2)]
    assert span.status.status_code.name == "ERROR"


def test_the_span_carries_the_same_attributes_under_either_profile(stream):
    standard, _, _ = stream()
    full, _, _ = stream(metrics_profile="full")

    assert dict(standard.attributes) == dict(full.attributes)
    assert standard.attributes["gen_ai.server.ttft"] == pytest.approx(0.5)
    assert standard.attributes["gen_ai.server.time_to_first_token"] == pytest.approx(0.5)
    assert standard.attributes["gen_ai.streaming.token_count"] == 4


def test_the_running_total_is_kept_on_the_stream_timing():
    instrumentor = _Instrumentor()
    instrumentor.config = OTelConfig(service_name="t")
    instrumentor.tbt_histogram = MagicMock()
    span = MagicMock()
    span.name = "test.chat"
    timing = _StreamTiming(START)

    with patch.object(base.time, "time", _Clock(ARRIVALS, END)):
        for number in range(4):
            instrumentor._observe_stream_chunk(span, timing, {"delta": number}, "m")

    assert timing.chunk_count == 4
    assert timing.tbt_count == 3
    assert timing.tbt_sum == pytest.approx(sum(GAPS))
    instrumentor.tbt_histogram.record.assert_not_called()


# --- OpenAI: which kind of stream this is, decided once -------------------------


class _ChatChunk:
    """A Chat Completions chunk: it has no ``type``. Counts how often it is asked."""

    asked = 0
    usage = None
    choices = ()

    def __getattr__(self, name):
        if name == "type":
            _ChatChunk.asked += 1
        raise AttributeError(name)


def test_a_chat_stream_is_asked_what_kind_it_is_once_not_for_every_chunk():
    """Asking a chunk for an attribute it does not have is slow on the SDK's
    models, and a chat stream never grows one part-way through."""
    _ChatChunk.asked = 0
    instrumentor = OpenAIInstrumentor()
    instrumentor.config = OTelConfig(service_name="t")
    instrumentor.tbt_histogram = MagicMock()
    span = MagicMock()
    span.name = "openai.chat.completion"
    timing = _StreamTiming(START)

    for _ in range(50):
        instrumentor._observe_stream_chunk(span, timing, _ChatChunk(), "gpt-4o")

    assert _ChatChunk.asked == 1
    assert timing.usage is None
    assert timing.chunk_count == 50


def test_a_responses_stream_still_reports_usage_from_its_last_event():
    instrumentor = OpenAIInstrumentor()
    timing = _StreamTiming(START)
    usage = SimpleNamespace(input_tokens=12, output_tokens=40, total_tokens=52)
    events = [
        SimpleNamespace(type="response.created", response=SimpleNamespace(usage=None)),
        SimpleNamespace(type="response.output_text.delta", delta="tok"),
        SimpleNamespace(type="response.output_text.delta", delta="tok"),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=usage)),
    ]

    for event in events:
        instrumentor._accumulate_stream_usage(timing, event)

    assert timing.usage["prompt_tokens"] == 12
    assert timing.usage["completion_tokens"] == 40


def test_each_stream_is_decided_on_its_own():
    """The same instrumentor reads chat streams and Responses streams."""
    instrumentor = OpenAIInstrumentor()
    chat, responses = _StreamTiming(START), _StreamTiming(START)
    usage = SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7)

    instrumentor._accumulate_stream_usage(chat, _ChatChunk())
    instrumentor._accumulate_stream_usage(
        responses, SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=usage))
    )
    instrumentor._accumulate_stream_usage(chat, _ChatChunk())

    assert chat.usage is None
    assert responses.usage["total_tokens"] == 7


# --- Anthropic messages.stream(): the manager ends the span ------------------------


class _Events:
    """Stands in for the SDK's stream manager: yields events, then fails if told to."""

    def __init__(self, count, fail_after=None):
        self._count, self._fail_after = count, fail_after

    def _events(self):
        for number in range(self._count):
            if self._fail_after is not None and number == self._fail_after:
                raise ConnectionError("connection reset")
            yield {"event": number}

    def __enter__(self):
        return self._events()

    def __exit__(self, *exc_info):
        return False

    async def __aenter__(self):
        return self._async_events()

    async def __aexit__(self, *exc_info):
        return False

    async def _async_events(self):
        for event in self._events():
            yield event


def _anthropic_manager(manager_class, events):
    provider = TracerProvider()
    instrumentor = _Instrumentor()
    instrumentor.config = OTelConfig(service_name="t")
    instrumentor.tbt_histogram = MagicMock()
    instrumentor._adopt_final_usage = lambda timing, stream: None
    span = provider.get_tracer("test").start_span("anthropic.messages.stream")
    return instrumentor, manager_class(events, instrumentor, span, "m")


def test_an_anthropic_stream_that_fails_part_way_still_reports_what_it_measured():
    """The manager, not the iterator, ends the span of ``messages.stream()``. It
    ended a failed stream without recording the time between the events that
    had arrived."""
    from genai_otel.instrumentors.anthropic_instrumentor import _MeasuredStreamManager

    instrumentor, manager = _anthropic_manager(_MeasuredStreamManager, _Events(10, fail_after=5))

    with pytest.raises(ConnectionError):
        with manager as events:
            for _ in events:
                pass

    assert instrumentor.tbt_histogram.record.call_count == 1


def test_an_async_anthropic_stream_that_fails_part_way_does_too():
    from genai_otel.instrumentors.anthropic_instrumentor import _MeasuredAsyncStreamManager

    instrumentor, manager = _anthropic_manager(
        _MeasuredAsyncStreamManager, _Events(10, fail_after=5)
    )

    async def read():
        async with manager as events:
            async for _ in events:
                pass

    with pytest.raises(ConnectionError):
        asyncio.run(read())

    assert instrumentor.tbt_histogram.record.call_count == 1


def test_an_anthropic_stream_read_to_its_end_reports_one_sample():
    from genai_otel.instrumentors.anthropic_instrumentor import _MeasuredStreamManager

    instrumentor, manager = _anthropic_manager(_MeasuredStreamManager, _Events(10))

    with manager as events:
        assert len(list(events)) == 10

    assert instrumentor.tbt_histogram.record.call_count == 1


def test_an_anthropic_stream_that_fails_before_its_second_event_has_nothing_to_report():
    from genai_otel.instrumentors.anthropic_instrumentor import _MeasuredStreamManager

    instrumentor, manager = _anthropic_manager(_MeasuredStreamManager, _Events(10, fail_after=1))

    with pytest.raises(ConnectionError):
        with manager as events:
            for _ in events:
                pass

    instrumentor.tbt_histogram.record.assert_not_called()
