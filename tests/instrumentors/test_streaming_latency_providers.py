"""Streaming latency (TTFT / TPOT) coverage for providers with bespoke wrappers.

Most instrumentors get streaming measurement from
``BaseInstrumentor.create_span_wrapper``. These three build their own spans, so
they route through ``_wrap_stream_if_streaming`` instead - and used to close
the span the moment the SDK handed back an iterator, which reported the
handshake as the whole call and lost TTFT, token usage and cost.

See issue #21.
"""

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
        assert inst._wrap_stream_if_streaming(span, buffered, 0.0, "m", {"stream": True}) is None

    # And nothing is wrapped when the call was not streamed at all.
    assert inst._wrap_stream_if_streaming(span, _stream_of([]), 0.0, "m", {}) is None
