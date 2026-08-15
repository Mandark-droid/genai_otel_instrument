"""Streaming latency (TTFT / TPOT) coverage for providers with bespoke wrappers.

Most instrumentors get streaming measurement from
``BaseInstrumentor.create_span_wrapper``. These three build their own spans, so
they route through ``_install_stream_measurement`` instead - and used to close
the span the moment the SDK handed back an iterator, which reported the
handshake as the whole call and lost TTFT, token usage and cost.

See issues #21 and #22.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.azure_openai_instrumentor import AzureOpenAIInstrumentor
from genai_otel.instrumentors.groq_instrumentor import GroqInstrumentor
from genai_otel.instrumentors.sarvam_instrumentor import SarvamAIInstrumentor

TTFT = "gen_ai.server.time_to_first_token"
TPOT = "gen_ai.server.time_per_output_token"


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
def captured(monkeypatch):
    """Give the instrumentor a real tracer whose finished spans we can read."""
    spans = []
    provider = TracerProvider()
    provider.add_span_processor(_Capture(spans))
    return spans, provider


def _usage_chunk(prompt=4, completion=6):
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    chunk = MagicMock()
    chunk.usage = usage
    return chunk


def _stream_of(chunks):
    def _gen():
        yield from chunks

    return _gen()


def _plain_chunk():
    chunk = MagicMock()
    chunk.usage = None
    return chunk


def _prepare(instrumentor, provider):
    instrumentor.tracer = provider.get_tracer(__name__)
    instrumentor.config = OTelConfig()
    return instrumentor


def test_groq_streaming_records_ttft_and_tpot(captured):
    spans, provider = captured
    with patch.dict("sys.modules", {"groq": MagicMock()}):
        inst = _prepare(GroqInstrumentor(), provider)
        client = MagicMock()
        client.chat.completions.create = lambda **kw: _stream_of([_plain_chunk(), _usage_chunk()])
        inst._instrument_client(client)

        stream = client.chat.completions.create(model="llama-3.3-70b", stream=True)
        assert spans == [], "span closed before the stream was consumed"
        assert len(list(stream)) == 2

    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert TTFT in attrs
    assert TPOT in attrs
    assert attrs["gen_ai.streaming.token_count"] == 2


def test_groq_non_streaming_omits_streaming_latency(captured):
    spans, provider = captured
    with patch.dict("sys.modules", {"groq": MagicMock()}):
        inst = _prepare(GroqInstrumentor(), provider)
        client = MagicMock()
        client.chat.completions.create = lambda **kw: _usage_chunk()
        inst._instrument_client(client)

        client.chat.completions.create(model="llama-3.3-70b")

    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert TTFT not in attrs
    assert TPOT not in attrs
    assert "gen_ai.server.ttft" not in attrs


def test_sarvam_streaming_records_ttft_and_tpot(captured):
    spans, provider = captured
    with patch.dict("sys.modules", {"sarvamai": MagicMock()}):
        inst = _prepare(SarvamAIInstrumentor(), provider)
        client = MagicMock()
        client.chat.completions = lambda **kw: _stream_of([_plain_chunk(), _usage_chunk()])
        inst._instrument_client(client)

        stream = client.chat.completions(model="sarvam-m", stream=True)
        assert spans == [], "span closed before the stream was consumed"
        assert len(list(stream)) == 2

    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert TTFT in attrs
    assert TPOT in attrs


def test_azure_openai_streaming_records_ttft_and_tpot(captured):
    spans, provider = captured

    class OpenAIClient:  # stand-in for azure.ai.openai.OpenAIClient
        def complete(self, **kwargs):
            return _stream_of([_plain_chunk(), _usage_chunk()])

    mock_azure_openai = MagicMock()
    mock_azure_openai.OpenAIClient = OpenAIClient

    with patch.dict(
        "sys.modules",
        {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": mock_azure_openai},
    ):
        inst = _prepare(AzureOpenAIInstrumentor(), provider)
        inst.instrument(OTelConfig())

        stream = OpenAIClient().complete(model="gpt-4o", stream=True)
        assert spans == [], "span closed before the stream was consumed"
        assert len(list(stream)) == 2

    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert TTFT in attrs
    assert TPOT in attrs


def test_streaming_helper_ignores_non_iterator_results():
    """A buffered response must never be mistaken for a stream."""
    with patch.dict("sys.modules", {"groq": MagicMock()}):
        inst = GroqInstrumentor()
    span = MagicMock()

    # Strings, dicts and lists iterate, but iterating them would invent a TTFT
    # out of characters or keys.
    for buffered in ("some text", {"choices": []}, [1, 2, 3], (1, 2)):
        handled, _ = inst._install_stream_measurement(span, buffered, 0.0, "m", {"stream": True})
        assert handled is False

    # And nothing is wrapped when the call was not streamed at all.
    handled, _ = inst._install_stream_measurement(span, _stream_of([]), 0.0, "m", {})
    assert handled is False


# --- Raw-response streams (issue #22) -------------------------------------
# A caller that wants the provider's response headers - litellm does, to read
# rate limits - reaches the OpenAI SDK through `with_raw_response.create`. That
# returns a LegacyAPIResponse/AsyncAPIResponse, so at await time there is
# nothing iterable to detect; the stream only exists once `.parse()` is called.


class _RawResponse:
    """Stands in for LegacyAPIResponse: sync .parse() yielding the stream."""

    def __init__(self, parsed):
        self._parsed = parsed
        self.headers = {"x-ratelimit-remaining-requests": "42"}

    def parse(self):
        return self._parsed


class _AsyncRawResponse(_RawResponse):
    """Stands in for AsyncAPIResponse, whose .parse() is a coroutine."""

    async def parse(self):  # type: ignore[override]
        return self._parsed


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


@contextmanager
def _openai_async_wrapper(provider, func):
    """Yield the wrapped call with the patched tracer still active.

    The wrapper resolves its tracer when it is *called*, not when it is built,
    so the patch has to stay in scope for the duration of the test.
    """
    from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

    with patch.dict("sys.modules", {"openai": MagicMock()}):
        inst = OpenAIInstrumentor()
    inst.config = OTelConfig()
    with patch("opentelemetry.trace.get_tracer", return_value=provider.get_tracer(__name__)):
        yield inst._create_async_span_wrapper(
            span_name="openai.chat.completion",
            extract_attributes=inst._extract_openai_attributes,
        )(func)


@pytest.mark.parametrize("raw_cls", [_RawResponse, _AsyncRawResponse])
def test_raw_response_stream_is_measured(captured, raw_cls):
    """The litellm path: TTFT/TPOT land once .parse() hands over the stream."""
    import asyncio

    spans, provider = captured
    stream = _FakeAsyncStream([_plain_chunk(), _usage_chunk()])

    async def create(**kwargs):
        return raw_cls(stream)

    with _openai_async_wrapper(provider, create) as wrapped:

        async def run():
            raw = await wrapped(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            assert spans == [], "span closed before the stream was parsed"
            # Callers read the headers, then parse - exactly what litellm does.
            assert raw.headers["x-ratelimit-remaining-requests"] == "42"
            parsed = raw.parse()
            if asyncio.iscoroutine(parsed):
                parsed = await parsed
            return [chunk async for chunk in parsed]

        chunks = asyncio.run(run())

    assert len(chunks) == 2
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert TTFT in attrs
    assert TPOT in attrs
    assert attrs["gen_ai.streaming.token_count"] == 2


def test_raw_response_that_is_not_a_stream_still_closes_its_span(captured):
    """A raw response whose parse() yields a plain object must not leak the span."""
    import asyncio

    spans, provider = captured

    class _BufferedCompletion:
        """A real buffered response is a plain model object - not iterable.

        Deliberately not a MagicMock: those auto-implement __aiter__ and would
        be mistaken for a stream here.
        """

        def __init__(self):
            self.usage = None
            self.choices = []

    async def create(**kwargs):
        return _RawResponse(_BufferedCompletion())  # buffered body, not a stream

    with _openai_async_wrapper(provider, create) as wrapped:

        async def run():
            raw = await wrapped(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True
            )
            return raw.parse()

        asyncio.run(run())

    assert len(spans) == 1, "span was leaked"
    attrs = dict(spans[0].attributes or {})
    assert TTFT not in attrs
    assert TPOT not in attrs


def test_sync_raw_response_stream_is_measured(captured):
    """Same deal on the sync path (litellm.completion -> with_raw_response)."""
    spans, provider = captured
    with patch.dict("sys.modules", {"groq": MagicMock()}):
        inst = _prepare(GroqInstrumentor(), provider)
        client = MagicMock()
        client.chat.completions.create = lambda **kw: _RawResponse(
            _stream_of([_plain_chunk(), _usage_chunk()])
        )
        inst._instrument_client(client)

        raw = client.chat.completions.create(model="llama-3.3-70b", stream=True)
        assert spans == [], "span closed before the stream was parsed"
        assert len(list(raw.parse())) == 2

    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert TTFT in attrs
    assert TPOT in attrs
