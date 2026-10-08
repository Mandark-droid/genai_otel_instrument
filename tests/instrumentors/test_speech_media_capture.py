"""Speech SDKs capture their audio like every other media part.

The Sarvam and ElevenLabs instrumentors recorded characters and cost but never the audio:
only the chat providers went through the media offload, so a voice app had to upload its
own recordings to make a call reviewable. Speech-to-text now records the audio it was sent
(media capture) and the transcript (content capture); text-to-speech records the text
(content capture) and the audio it returned (media capture). Both are off by default.
"""

import base64
import io
import struct
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.elevenlabs_instrumentor import ElevenLabsInstrumentor
from genai_otel.instrumentors.sarvam_instrumentor import SarvamAIInstrumentor
from genai_otel.media.speech import audio_mime, read_audio


def _wav(seconds: float = 0.5, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buf.getvalue()


class _Span:
    def __init__(self):
        self.attrs = {}
        self.name = "test.span"

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_status(self, *_a, **_k):
        pass

    def record_exception(self, *_a, **_k):
        pass

    def get_span_context(self):
        return SimpleNamespace(trace_id=0xABC)

    def end(self):
        self.ended = True


def _config(tmp_path, media="full", content=True, **over):
    cfg = OTelConfig(service_name="test", enable_cost_tracking=False)
    cfg.media_capture_mode = media
    cfg.media_store = "filesystem"
    cfg.media_store_endpoint = str(tmp_path)
    cfg.enable_content_capture = content
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _ctx_tracer(span):
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    tracer.start_span.return_value = span
    return tracer


def _sarvam(cfg, span):
    inst = SarvamAIInstrumentor()
    inst.config = cfg
    inst.cost_calculator = MagicMock(calculate_cost=MagicMock(return_value=0))
    inst.latency_histogram = None
    inst.cost_counter = None
    inst.request_counter = None
    inst.tracer = _ctx_tracer(span)
    return inst


class TestSpeechHelpers:
    def test_mime_from_codec_or_extension(self):
        assert audio_mime(codec="mp3_44100_128") == "audio/mpeg"
        assert audio_mime(codec="ulaw_8000") == "audio/basic"
        assert audio_mime("call.flac") == "audio/flac"
        assert audio_mime() == "audio/wav"

    def test_read_is_bounded_and_leaves_a_file_where_it_was(self):
        f = io.BytesIO(b"x" * 100)
        data, _name, _size = read_audio(f, max_bytes=10)
        assert data == b"x" * 11  # cap + 1, enough to know it is over
        assert f.tell() == 0

    def test_a_tuple_upload_keeps_its_name(self):
        data, name, size = read_audio(("turn.wav", b"abc", "audio/wav"), max_bytes=10)
        assert (data, name, size) == (b"abc", "turn.wav", 3)

    @pytest.mark.parametrize("audio", [None, 42, "missing.wav", io.StringIO("text")])
    def test_unreadable_is_nothing_never_an_error(self, audio):
        assert read_audio(audio, max_bytes=10) == (None, None, None)


class TestSarvamSpeechToText:
    def _run(self, cfg, transcript="जी हाँ"):
        span = _Span()
        inst = _sarvam(cfg, span)
        client = MagicMock()
        seen = {}

        def transcribe(**kw):
            seen["read"] = kw["file"].read()  # the SDK consumes the file, as the real one does
            return SimpleNamespace(transcript=transcript)

        client.speech_to_text.transcribe = transcribe
        inst._instrument_client(client)
        audio = _wav()
        client.speech_to_text.transcribe(file=io.BytesIO(audio), model="saaras:v3")
        return span.attrs, seen, audio

    def test_uploads_the_audio_sent_and_records_the_transcript(self, tmp_path):
        attrs, seen, audio = self._run(_config(tmp_path))
        assert seen["read"] == audio, "the SDK must still read the whole file"
        assert attrs["gen_ai.prompt.0.role"] == "user"
        assert attrs["gen_ai.prompt.0.content.0.type"] == "audio"
        assert attrs["gen_ai.prompt.0.content.0.media_source"] == "inline_offloaded"
        assert attrs["gen_ai.prompt.0.content.0.media_mime_type"] == "audio/wav"
        assert attrs["gen_ai.prompt.0.content.0.media_byte_size"] == len(audio)
        assert attrs["gen_ai.completion.0.content.0.type"] == "text"
        assert attrs["gen_ai.completion.0.content.0.text"] == "जी हाँ"
        stored = list(tmp_path.rglob("*.*"))
        assert any(p.read_bytes() == audio for p in stored if p.is_file())

    def test_off_by_default_records_nothing(self, tmp_path):
        attrs, _seen, _audio = self._run(_config(tmp_path, media="off", content=False))
        assert not any(k.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for k in attrs)
        assert not any(p.is_file() for p in tmp_path.rglob("*"))

    def test_transcript_without_media_capture_records_text_only(self, tmp_path):
        attrs, _seen, _audio = self._run(_config(tmp_path, media="off", content=True))
        assert "gen_ai.prompt.0.content.0.type" not in attrs
        assert attrs["gen_ai.completion.0.content.0.text"] == "जी हाँ"

    def test_audio_without_content_capture_records_no_transcript(self, tmp_path):
        attrs, _seen, _audio = self._run(_config(tmp_path, content=False))
        assert attrs["gen_ai.prompt.0.content.0.type"] == "audio"
        assert "gen_ai.completion.0.content.0.text" not in attrs

    def test_over_the_cap_is_a_reference_with_its_real_size(self, tmp_path):
        attrs, _seen, audio = self._run(_config(tmp_path, media_max_bytes=100))
        assert attrs["gen_ai.prompt.0.content.0.media_source"] == "reference_only"
        assert attrs["gen_ai.media.stripped_reason"] == "size_exceeded"
        assert attrs["gen_ai.prompt.0.content.0.media_byte_size"] == len(audio)


class TestSarvamTextToSpeech:
    def test_records_the_text_and_uploads_the_returned_audio(self, tmp_path):
        span = _Span()
        inst = _sarvam(_config(tmp_path), span)
        audio = _wav()
        client = MagicMock()
        client.text_to_speech.convert = MagicMock(
            return_value=SimpleNamespace(audios=[base64.b64encode(audio).decode()])
        )
        inst._instrument_client(client)
        client.text_to_speech.convert(text="कुल बकाया 27846 rupees है", model="bulbul:v3")
        a = span.attrs
        assert a["gen_ai.prompt.0.content.0.text"] == "कुल बकाया 27846 rupees है"
        assert a["gen_ai.completion.0.role"] == "assistant"
        assert a["gen_ai.completion.0.content.0.type"] == "audio"
        assert a["gen_ai.completion.0.content.0.media_source"] == "inline_offloaded"
        assert a["gen_ai.completion.0.content.0.media_byte_size"] == len(audio)

    def test_returned_audio_too_large_to_decode_is_a_sized_reference(self, tmp_path):
        span = _Span()
        inst = _sarvam(_config(tmp_path, media_max_bytes=100), span)
        client = MagicMock()
        client.text_to_speech.convert = MagicMock(
            return_value={"audios": [base64.b64encode(_wav()).decode()]}
        )
        inst._instrument_client(client)
        client.text_to_speech.convert(text="hi", model="bulbul:v3")
        a = span.attrs
        assert a["gen_ai.completion.0.content.0.media_source"] == "reference_only"
        assert a["gen_ai.completion.0.content.0.media_byte_size"] > 100


def _eleven(cfg, span):
    inst = ElevenLabsInstrumentor()
    inst.config = cfg
    inst.cost_counter = None
    inst.request_counter = None
    inst.latency_histogram = None
    inst.ttft_histogram = None
    inst.tracer = _ctx_tracer(span)
    return inst


class TestElevenLabs:
    def test_tts_stream_reaches_the_caller_whole_and_its_audio_is_uploaded(self, tmp_path):
        span = _Span()
        inst = _eleven(_config(tmp_path), span)
        chunks = [b"ID3", b"aaaa", b"bbbb"]
        tts = SimpleNamespace(convert=lambda **kw: iter(chunks))
        inst._wrap_tts(tts, "convert", is_async=False)
        assert list(tts.convert(voice_id="v", text="hello there")) == chunks
        a = span.attrs
        assert a["gen_ai.prompt.0.content.0.text"] == "hello there"
        assert a["gen_ai.completion.0.content.0.media_mime_type"] == "audio/mpeg"
        assert a["gen_ai.completion.0.content.0.media_byte_size"] == 11
        assert a["gen_ai.completion.0.content.0.media_source"] == "inline_offloaded"

    def test_an_abandoned_stream_records_no_audio(self, tmp_path):
        span = _Span()
        inst = _eleven(_config(tmp_path), span)
        tts = SimpleNamespace(convert=lambda **kw: iter([b"aa", b"bb", b"cc"]))
        inst._wrap_tts(tts, "convert", is_async=False)
        stream = tts.convert(voice_id="v", text="hi")
        next(stream)
        stream.close()
        assert "gen_ai.completion.0.content.0.type" not in span.attrs

    def test_media_off_leaves_the_stream_unwrapped_by_the_buffer(self, tmp_path):
        span = _Span()
        inst = _eleven(_config(tmp_path, media="off", content=False), span)
        tts = SimpleNamespace(convert=lambda **kw: iter([b"aa"]))
        inst._wrap_tts(tts, "convert", is_async=False)
        assert list(tts.convert(voice_id="v", text="hi")) == [b"aa"]
        assert not any(k.startswith("gen_ai.completion.") for k in span.attrs)

    def test_stt_uploads_the_audio_and_records_the_transcript(self, tmp_path):
        span = _Span()
        inst = _eleven(_config(tmp_path), span)
        audio = _wav()
        stt = SimpleNamespace(
            convert=lambda **kw: SimpleNamespace(
                text="hello", audio_duration_secs=0.5, language_code="en"
            )
        )
        inst._wrap_stt(stt, is_async=False)
        stt.convert(model_id="scribe_v1", file=("turn.wav", audio))
        a = span.attrs
        assert a["gen_ai.prompt.0.content.0.media_mime_type"] == "audio/wav"
        assert a["gen_ai.prompt.0.content.0.media_byte_size"] == len(audio)
        assert a["gen_ai.completion.0.content.0.text"] == "hello"

    @pytest.mark.asyncio
    async def test_async_tts_stream_uploads_its_audio(self, tmp_path):
        span = _Span()
        inst = _eleven(_config(tmp_path), span)

        async def agen():
            for c in (b"aa", b"bbb"):
                yield c

        out = [
            c
            async for c in inst._wrap_async_audio_stream(agen(), span, 0.0, "m", mime="audio/mpeg")
        ]
        assert out == [b"aa", b"bbb"]
        assert span.attrs["gen_ai.completion.0.content.0.media_byte_size"] == 5


class TestSarvamStreamedSpeech:
    """sarvamai 0.1.x streams through `convert_stream`; only `stream` was wrapped, so every
    streamed call went untraced."""

    def _client(self, chunks):
        tts = SimpleNamespace(convert_stream=lambda **kw: iter(chunks))
        return SimpleNamespace(text_to_speech=tts)

    def test_convert_stream_is_traced_priced_and_reaches_the_caller_whole(self, tmp_path):
        span = _Span()
        inst = _sarvam(_config(tmp_path), span)
        client = self._client([b"ID3", b"aaaa"])
        inst._instrument_client(client)
        out = list(client.text_to_speech.convert_stream(text="नमस्ते", model="bulbul:v3"))
        assert out == [b"ID3", b"aaaa"]
        inst.tracer.start_span.assert_called_with("sarvam.text_to_speech.convert_stream")
        a = span.attrs
        assert a["gen_ai.operation.name"] == "text_to_speech"
        assert a["gen_ai.request.model"] == "bulbul:v3"
        assert a["gen_ai.usage.characters"] == len("नमस्ते")
        assert a["gen_ai.prompt.0.content.0.text"] == "नमस्ते"
        assert a["gen_ai.completion.0.content.0.media_mime_type"] == "audio/mpeg"
        assert a["gen_ai.completion.0.content.0.media_byte_size"] == 7
        assert getattr(span, "ended", False)

    def test_media_off_streams_untouched_and_records_no_audio(self, tmp_path):
        span = _Span()
        inst = _sarvam(_config(tmp_path, media="off", content=False), span)
        client = self._client([b"aa"])
        inst._instrument_client(client)
        assert list(client.text_to_speech.convert_stream(text="hi")) == [b"aa"]
        assert not any(k.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for k in span.attrs)

    @pytest.mark.asyncio
    async def test_async_convert_stream_records_its_audio(self, tmp_path):
        span = _Span()
        inst = _sarvam(_config(tmp_path), span)

        async def agen():
            for c in (b"aa", b"bbb"):
                yield c

        tts = SimpleNamespace(convert_stream=lambda **kw: agen())
        inst._instrument_client(SimpleNamespace(text_to_speech=tts))
        out = [c async for c in tts.convert_stream(text="hi", output_audio_codec="wav")]
        assert out == [b"aa", b"bbb"]
        assert span.attrs["gen_ai.completion.0.content.0.media_mime_type"] == "audio/wav"
        assert span.attrs["gen_ai.completion.0.content.0.media_byte_size"] == 5


class TestDurationIsReadBeforeTheSdkConsumesTheFile:
    """Found end to end: the real SDK reads (uploads) a file object, leaving it at its end.
    The duration was read afterwards, came back empty, and the call went unpriced."""

    def test_a_consumed_file_object_is_still_priced(self, tmp_path):
        span = _Span()
        inst = _sarvam(_config(tmp_path, media="off", content=False), span)
        inst.config.enable_cost_tracking = True
        from genai_otel.cost_calculator import CostCalculator

        inst.cost_calculator = CostCalculator()
        client = MagicMock()

        def transcribe(**kw):
            kw["file"].read()
            return SimpleNamespace(transcript="ok")

        client.speech_to_text.transcribe = transcribe
        inst._instrument_client(client)
        client.speech_to_text.transcribe(file=io.BytesIO(_wav(36.0)), model="saaras:v3")
        assert span.attrs["gen_ai.usage.audio_duration_seconds"] == pytest.approx(36.0)
        assert span.attrs.get("gen_ai.usage.cost.total", 0) > 0
