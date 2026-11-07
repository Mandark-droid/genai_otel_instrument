"""OpenAI speech endpoints traced: transcriptions, translations, text-to-speech (#46).

The real ``openai`` SDK talks to an ``httpx.MockTransport`` serving recorded response
shapes, so the SDK's own resource classes and its ``with_raw_response`` /
``with_streaming_response`` wrappers run; nothing is mocked at our boundary.
"""

import asyncio
import io
import json
import wave

import httpx
import pytest

openai = pytest.importorskip("openai")

from opentelemetry import context as otel_context  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from genai_otel.config import OTelConfig  # noqa: E402
from genai_otel.instrumentors.base import register_base_url_claim  # noqa: E402
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor  # noqa: E402


def _wav(seconds=2.0, rate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


WAV = _wav()
MP3 = b"ID3\x04\x00" + b"\x00" * 200
TOKENS_USAGE = {
    "type": "tokens",
    "input_tokens": 120,
    "output_tokens": 5,
    "total_tokens": 125,
    "input_token_details": {"audio_tokens": 110, "text_tokens": 10},
}


def _sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


class Harness:
    def __init__(self, routes, capture=False, media="off", tmp_path=None, **config):
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.inst = OpenAIInstrumentor()
        self.inst.tracer = provider.get_tracer("test")
        self.inst._instrumented = True
        extra = {}
        if media != "off":
            extra = {"media_store": "filesystem", "media_store_endpoint": str(tmp_path)}
        self.inst.config = OTelConfig(
            service_name="test",
            enable_content_capture=capture,
            content_max_length=0,
            media_capture_mode=media,
            **extra,
            **config,
        )
        self.routes = routes
        self.requests = []

    def _handler(self, request):
        self.requests.append(request)
        status, headers, body = self.routes(request)
        return httpx.Response(status, headers=headers, content=body)

    def client(self, is_async=False, **kwargs):
        kwargs.setdefault("api_key", "test")
        kwargs.setdefault("max_retries", 0)
        transport = httpx.MockTransport(self._handler)
        if is_async:
            client = openai.AsyncOpenAI(
                http_client=httpx.AsyncClient(transport=transport), **kwargs
            )
            self.inst._instrument_async_client(client)
        else:
            client = openai.OpenAI(http_client=httpx.Client(transport=transport), **kwargs)
            self.inst._instrument_client(client)
        return client

    def spans(self):
        return self.exporter.get_finished_spans()

    def span(self):
        (span,) = self.spans()
        return span

    def attrs(self):
        return dict(self.span().attributes)


def _json(body, status=200):
    return lambda request: (status, {"content-type": "application/json"}, json.dumps(body).encode())


def _run(is_async, fn):
    if is_async:
        return asyncio.run(fn())
    return fn()


@pytest.fixture(autouse=True)
def empty_context():
    token = otel_context.attach(otel_context.Context())
    yield
    otel_context.detach(token)


PARAMS = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])


# --------------------------------------------------------------------------- STT


@PARAMS
def test_token_billed_transcription(is_async):
    h = Harness(_json({"text": "hello world", "usage": TOKENS_USAGE}))
    client = h.client(is_async)

    def call():
        return client.audio.transcriptions.create(model="gpt-4o-transcribe", file=("a.wav", WAV))

    async def acall():
        return await client.audio.transcriptions.create(
            model="gpt-4o-transcribe", file=("a.wav", WAV)
        )

    result = _run(is_async, acall if is_async else call)
    assert result.text == "hello world"
    span = h.span()
    a = dict(span.attributes)
    assert span.name == "openai.audio.transcriptions"
    assert span.status.status_code == StatusCode.OK
    assert a["gen_ai.operation.name"] == "speech_to_text"
    assert a["gen_ai.provider.name"] == "openai" or a["gen_ai.system"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o-transcribe"
    assert a["gen_ai.response.transcript_length"] == len("hello world")
    assert a["gen_ai.usage.input_tokens"] == 120
    assert a["gen_ai.usage.output_tokens"] == 5
    assert a["gen_ai.usage.input_audio_tokens"] == 110
    assert a["gen_ai.usage.cost.total"] == pytest.approx(120 / 1000 * 0.0025 + 5 / 1000 * 0.01)
    # Content capture off: no transcript, no audio.
    assert not any(k.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for k in a)


def test_whisper_priced_by_reported_seconds():
    h = Harness(_json({"text": "hi", "usage": {"type": "duration", "seconds": 3.5}}))
    h.client().audio.transcriptions.create(model="whisper-1", file=("a.mp3", MP3))
    a = h.attrs()
    assert a["gen_ai.usage.audio_duration_seconds"] == 3.5
    assert a["gen_ai.usage.cost.total"] == pytest.approx(3.5 * 0.0001)


def _text(body):
    return lambda request: (200, {"content-type": "text/plain"}, body.encode())


def test_plain_text_format_is_priced_from_the_wav_header():
    h = Harness(_text("hi there\n"))
    result = h.client().audio.transcriptions.create(
        model="whisper-1", file=("a.wav", WAV), response_format="text"
    )
    assert isinstance(result, str)
    a = h.attrs()
    assert a["gen_ai.usage.audio_duration_seconds"] == pytest.approx(2.0)
    assert a["gen_ai.usage.cost.total"] == pytest.approx(2.0 * 0.0001)


def test_plain_text_format_on_mp3_has_no_cost_and_says_why():
    h = Harness(_text("hi there\n"))
    h.client().audio.transcriptions.create(
        model="whisper-1", file=("a.mp3", MP3), response_format="text"
    )
    a = h.attrs()
    assert "gen_ai.usage.cost.total" not in a
    assert a["gen_ai.usage.cost.unknown_reason"] == "no_usage_reported"


def test_translation_targets_english_and_reads_verbose_duration():
    h = Harness(_json({"text": "hello", "duration": 2.0, "language": "english", "segments": []}))
    h.client().audio.translations.create(
        model="whisper-1", file=("a.mp3", MP3), response_format="verbose_json"
    )
    a = h.attrs()
    assert h.span().name == "openai.audio.translations"
    assert a["gen_ai.request.target_language_code"] == "en"
    assert a["gen_ai.usage.cost.total"] == pytest.approx(2.0 * 0.0001)


@pytest.mark.parametrize("shape", ["bytes", "path", "file", "tuple"])
def test_upload_shapes_reach_the_api_whole(tmp_path, shape):
    h = Harness(_json({"text": "x", "usage": TOKENS_USAGE}), media="full", tmp_path=tmp_path)
    if shape == "bytes":
        file = WAV
    elif shape == "path":
        path = tmp_path / "a.wav"
        path.write_bytes(WAV)
        file = path
    elif shape == "file":
        file = io.BytesIO(WAV)
    else:
        file = ("a.wav", io.BytesIO(WAV))
    h.client().audio.transcriptions.create(model="gpt-4o-transcribe", file=file)
    body = h.requests[0].read()
    assert WAV[-64:] in body and WAV[:12] in body, "the upload was truncated"
    if shape == "file":
        assert file.tell() == len(WAV)  # the SDK read it to the end, from the start
    a = h.attrs()
    assert a["gen_ai.prompt.0.content.0.type"] == "audio"
    assert a["gen_ai.prompt.0.content.0.media_uri"].startswith("file:")


def test_capture_matrix_transcript_needs_content_capture(tmp_path):
    h = Harness(_json({"text": "secret words", "usage": TOKENS_USAGE}), capture=True)
    h.client().audio.transcriptions.create(model="gpt-4o-transcribe", file=("a.wav", WAV))
    a = h.attrs()
    assert a["gen_ai.completion.0.content.0.text"] == "secret words"
    assert "gen_ai.prompt.0.content.0.type" not in a  # media capture off: no audio


@PARAMS
def test_streamed_transcription_ends_when_read(is_async):
    events = [
        {"type": "transcript.text.delta", "delta": "hel"},
        {"type": "transcript.text.delta", "delta": "lo"},
        {"type": "transcript.text.done", "text": "hello", "usage": TOKENS_USAGE},
    ]
    h = Harness(lambda r: (200, {"content-type": "text/event-stream"}, _sse(events)))
    client = h.client(is_async)

    def call():
        stream = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", file=("a.wav", WAV), stream=True
        )
        assert not h.spans(), "the span ended before the stream was read"
        return [e.type for e in stream]

    async def acall():
        stream = await client.audio.transcriptions.create(
            model="gpt-4o-transcribe", file=("a.wav", WAV), stream=True
        )
        assert not h.spans()
        return [e.type async for e in stream]

    types = _run(is_async, acall if is_async else call)
    assert types[-1] == "transcript.text.done"
    a = h.attrs()
    assert a["gen_ai.response.transcript_length"] == 5
    assert a["gen_ai.usage.output_tokens"] == 5
    assert "gen_ai.server.time_to_first_token" in a
    assert a["gen_ai.request.streamed"] is True


def test_abandoned_transcription_stream_is_closed_early():
    events = [{"type": "transcript.text.delta", "delta": "a"}] * 3
    h = Harness(lambda r: (200, {"content-type": "text/event-stream"}, _sse(events)))
    stream = h.client().audio.transcriptions.create(
        model="gpt-4o-transcribe", file=("a.wav", WAV), stream=True
    )
    for _ in stream:
        break
    stream.close()
    span = h.span()
    assert span.attributes["genai.stream.closed_early"] is True
    assert span.status.status_code == StatusCode.UNSET


# --------------------------------------------------------------------------- TTS


def _audio(body=b"\xff\xfb" + b"\x00" * 500):
    return lambda request: (200, {"content-type": "audio/mpeg"}, body)


@PARAMS
def test_tts_priced_by_characters(is_async, tmp_path):
    h = Harness(_audio(), media="full", tmp_path=tmp_path)
    client = h.client(is_async)
    text = "Your refund has been issued."

    def call():
        return client.audio.speech.create(model="tts-1", voice="alloy", input=text)

    async def acall():
        return await client.audio.speech.create(model="tts-1", voice="alloy", input=text)

    _run(is_async, acall if is_async else call)
    a = h.attrs()
    assert h.span().name == "openai.audio.speech"
    assert a["gen_ai.operation.name"] == "text_to_speech"
    assert a["gen_ai.request.voice_id"] == "alloy"
    assert a["gen_ai.usage.characters"] == len(text)
    assert a["gen_ai.usage.cost.total"] == pytest.approx(len(text) / 1000 * 0.015)
    assert a["gen_ai.completion.0.content.0.type"] == "audio"
    assert a["gen_ai.completion.0.content.0.media_mime_type"] == "audio/mpeg"


def test_token_billed_tts_without_usage_has_no_cost():
    h = Harness(_audio())
    h.client().audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input="hi")
    a = h.attrs()
    assert "gen_ai.usage.cost.total" not in a
    assert a["gen_ai.usage.cost.unknown_reason"] == "no_usage_reported"


def test_tts_sse_reports_usage_and_is_priced():
    events = [
        {"type": "speech.audio.delta", "audio": "AAAA"},
        {"type": "speech.audio.done", "usage": {"input_tokens": 10, "output_tokens": 200}},
    ]
    h = Harness(lambda r: (200, {"content-type": "text/event-stream"}, _sse(events)))
    h.client().audio.speech.create(
        model="gpt-4o-mini-tts", voice="alloy", input="hi", stream_format="sse"
    )
    a = h.attrs()
    assert a["gen_ai.usage.input_tokens"] == 10
    assert a["gen_ai.usage.cost.total"] == pytest.approx(10 / 1000 * 0.0006 + 200 / 1000 * 0.012)


@PARAMS
def test_streaming_tts_span_lasts_until_the_audio_is_read(is_async):
    h = Harness(_audio(b"\xff\xfb" + b"\x01" * 4000))
    client = h.client(is_async)

    def call():
        with client.audio.speech.with_streaming_response.create(
            model="tts-1", voice="alloy", input="hello"
        ) as response:
            assert not h.spans(), "the span ended before the audio was read"
            return b"".join(response.iter_bytes(512))

    async def acall():
        async with client.audio.speech.with_streaming_response.create(
            model="tts-1", voice="alloy", input="hello"
        ) as response:
            assert not h.spans()
            out = b""
            async for chunk in response.iter_bytes(512):
                out += chunk
            return out

    audio = _run(is_async, acall if is_async else call)
    assert len(audio) == 4002
    a = h.attrs()
    assert "gen_ai.server.time_to_first_token" in a
    assert "genai.stream.closed_early" not in a
    assert a["gen_ai.usage.cost.total"] > 0


def test_streaming_tts_left_unread_is_closed_early_and_still_billed():
    h = Harness(_audio())
    with h.client().audio.speech.with_streaming_response.create(
        model="tts-1", voice="alloy", input="hello"
    ):
        pass
    a = h.attrs()
    assert a["genai.stream.closed_early"] is True
    assert a["gen_ai.usage.cost.total"] == pytest.approx(5 / 1000 * 0.015)
    assert "gen_ai.completion.0.content.0.type" not in a


def test_raw_response_ends_when_parsed():
    h = Harness(_json({"text": "hi", "usage": TOKENS_USAGE}))
    raw = h.client().audio.transcriptions.with_raw_response.create(
        model="gpt-4o-transcribe", file=("a.wav", WAV)
    )
    assert not h.spans()
    assert raw.parse().text == "hi"
    raw.parse()  # a second parse records nothing twice
    assert len(h.spans()) == 1
    assert h.attrs()["gen_ai.usage.output_tokens"] == 5


# --------------------------------------------------------------------------- wiring


def test_instrumenting_twice_still_gives_one_span():
    h = Harness(_audio())
    client = h.client()
    h.inst._instrument_client(client)
    client.audio.speech.create(model="tts-1", voice="alloy", input="hi")
    assert len(h.spans()) == 1


def test_an_aggregator_claimed_client_is_not_wrapped():
    register_base_url_claim("speech-aggregator.test", "test-aggregator")
    h = Harness(_audio())
    client = h.client(base_url="https://speech-aggregator.test/v1")
    client.audio.speech.create(model="tts-1", voice="alloy", input="hi")
    assert not h.spans()


def test_azure_deployment_name_is_traced_unpriced():
    h = Harness(_audio())
    transport = httpx.MockTransport(h._handler)
    client = openai.AzureOpenAI(
        api_key="test",
        api_version="2025-04-01-preview",
        azure_endpoint="https://example.openai.azure.com",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    h.inst._instrument_client(client)
    client.audio.speech.create(model="my-tts-deployment", voice="alloy", input="hi")
    a = h.attrs()
    assert a["gen_ai.request.model"] == "my-tts-deployment"
    assert a["gen_ai.usage.cost.unknown_reason"] == "unpriced_model"


def test_api_error_marks_the_span():
    h = Harness(_json({"error": {"message": "bad"}}, status=400))
    with pytest.raises(openai.BadRequestError):
        h.client().audio.speech.create(model="tts-1", voice="alloy", input="hi")
    assert h.span().status.status_code == StatusCode.ERROR
