"""Audio inputs on DECISION spans (issue #47).

Decision models now take audio ("has the speaker finished?", "is the tone calm?").
Decision serialisation used to call any bytes an image, so an audio clip was
recorded as ``[image omitted]``, and no attribute said which modalities a request
carried. Audio is now detected and labelled, the modalities are recorded with
content capture off too, and with media capture on the audio goes through the
media pipeline.
"""

import base64
import io
import json
import struct
import wave

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import genai_otel
from genai_otel import decision as manual
from genai_otel.config import OTelConfig
from genai_otel.instrumentors import decision as d
from genai_otel.media.sniff import classify_data_url, sniff_bytes, wav_seconds

QUESTIONS = {"finished": {"type": "noul", "description": "Has the speaker finished?"}}
ANSWERS = {"finished": {"type": "noul", "probability": 0.9}}


def _wav(seconds=0.5, rate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def _mp3_frames(count=3):
    # MPEG-1 Layer III, 128 kbit/s, 44.1 kHz, no padding: 417-byte frames.
    header = bytes([0xFF, 0xFB, 0x90, 0x00])
    return (header + b"\x00" * 413) * count


SAMPLES = {
    "wav": (_wav(), ("audio", "audio/wav")),
    "mp3-id3": (b"ID3\x04\x00" + b"\x00" * 64, ("audio", "audio/mpeg")),
    "mp3-frames": (_mp3_frames(), ("audio", "audio/mpeg")),
    "ogg": (b"OggS" + b"\x00" * 60, ("audio", "audio/ogg")),
    "flac": (b"fLaC" + b"\x00" * 60, ("audio", "audio/flac")),
    "webm": (b"\x1a\x45\xdf\xa3" + b"\x00" * 60, ("audio", "audio/webm")),
    "png": (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, ("image", "image/png")),
    "jpeg": (b"\xff\xd8\xff\xe0" + b"\x00" * 20, ("image", "image/jpeg")),
    "gif": (b"GIF89a" + b"\x00" * 20, ("image", "image/gif")),
    "random": (bytes(range(7, 71)), ("binary", None)),
}


@pytest.mark.parametrize("name", list(SAMPLES))
def test_bytes_are_classified_by_header(name):
    data, expected = SAMPLES[name]
    assert sniff_bytes(data) == expected


def test_a_lone_sync_word_is_not_mp3():
    assert sniff_bytes(bytes([0xFF, 0xFB, 0x90, 0x00]) + bytes(range(1, 40))) == ("binary", None)


def test_wav_duration_from_the_header():
    assert wav_seconds(_wav(1.25)) == pytest.approx(1.25)
    assert wav_seconds(b"ID3....") is None


@pytest.mark.parametrize(
    "name,placeholder",
    [
        ("wav", d.AUDIO_OMITTED),
        ("mp3-frames", d.AUDIO_OMITTED),
        ("png", d.IMAGE_OMITTED),
        ("random", d.BINARY_OMITTED),
    ],
)
def test_bytes_regression_audio_is_no_longer_an_image(name, placeholder):
    data, _ = SAMPLES[name]
    assert d.to_jsonable(data) == placeholder
    assert json.loads(d.payload_json({"clip": data}))["clip"] == placeholder


def test_data_urls_and_parts_are_labelled_by_kind():
    wav_b64 = base64.b64encode(_wav()).decode()
    payload = {
        "state": f"data:audio/wav;base64,{wav_b64}",
        "image": "data:image/png;base64,AAAA",
        "parts": [{"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}],
    }
    loaded = json.loads(d.payload_json(payload))
    assert loaded["state"] == d.AUDIO_OMITTED
    assert loaded["image"] == d.IMAGE_OMITTED
    assert loaded["parts"][0] == {"type": "input_audio", "input_audio": d.AUDIO_OMITTED}
    assert wav_b64 not in d.payload_json(payload)
    assert classify_data_url("data:audio/ogg;base64,AAAA") == ("audio", "audio/ogg")
    # The old name still works.
    assert d.omit_images is d.omit_media


def test_asset_objects_by_mime_type():
    class Asset:
        def __init__(self, data, mime_type):
            self.data = data
            self.mime_type = mime_type

    assert d.to_jsonable(Asset(b"...", "audio/mpeg")) == d.AUDIO_OMITTED
    assert d.to_jsonable(Asset(b"...", "image/png")) == d.IMAGE_OMITTED
    assert d.to_jsonable(Asset(_wav(), None)) == d.AUDIO_OMITTED


@pytest.mark.parametrize(
    "value,modalities",
    [
        ("plain text", ["text"]),
        ({"text": "hi", "photo": "data:image/png;base64,AAAA"}, ["image", "text"]),
        ({"clip": _wav(), "note": "caller on hold"}, ["audio", "text"]),
        ([{"type": "input_audio", "input_audio": {"data": "AAAA"}}], ["audio"]),
        ({"a": _wav(), "i": SAMPLES["png"][0], "t": "x"}, ["audio", "image", "text"]),
    ],
)
def test_scan_reports_modalities(value, modalities):
    assert d.scan_inputs(value).modalities == modalities


# --------------------------------------------------------------------------- manual helper


@pytest.fixture
def spans(monkeypatch, tmp_path):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(manual, "_INSTRUMENTOR", None)
    monkeypatch.setattr(manual, "_CONFIG", None)

    def configure(**config):
        config.setdefault("media_store", "filesystem")
        config.setdefault("media_store_endpoint", str(tmp_path))
        manual.configure(OTelConfig(content_max_length=0, **config))
        manual._instrumentor().tracer = provider.get_tracer("test")
        return exporter

    return configure


def _attrs(exporter):
    (span,) = exporter.get_finished_spans()
    return dict(span.attributes)


def _part_keys(attrs):
    return {k: v for k, v in attrs.items() if k.startswith("gen_ai.prompt.0.content.")}


@pytest.mark.parametrize("shape", ["bytes", "path", "file"])
def test_manual_audio_with_capture_off_records_modality_and_duration_only(spans, tmp_path, shape):
    exporter = spans(enable_content_capture=False)
    clip = _wav(0.75)
    if shape == "bytes":
        audio = clip
    elif shape == "path":
        path = tmp_path / "turn.wav"
        path.write_bytes(clip)
        audio = str(path)
    else:
        audio = io.BytesIO(b"xx" + clip)
        audio.seek(2)
    with genai_otel.decision_span(model="eot", questions=QUESTIONS, state="ctx", audio=audio) as r:
        r.record(ANSWERS)
    attrs = _attrs(exporter)
    assert list(attrs["genai.decision.input.modalities"]) == ["audio", "text"]
    assert attrs["genai.decision.input.audio_seconds"] == pytest.approx(0.75)
    assert not _part_keys(attrs)
    assert "input.value" not in attrs
    if shape == "file":
        assert audio.tell() == 2, "the caller's file position moved"


def test_manual_audio_with_capture_on_says_audio_not_image(spans):
    exporter = spans(enable_content_capture=True)
    with genai_otel.decision_span(model="eot", questions=QUESTIONS, audio=_wav()) as r:
        r.record(ANSWERS)
    value = _attrs(exporter)["input.value"]
    assert d.AUDIO_OMITTED in value and d.IMAGE_OMITTED not in value


@pytest.mark.parametrize("mode", ["reference_only", "full"])
def test_media_capture_records_the_audio_as_a_media_part(spans, mode):
    exporter = spans(enable_content_capture=True, media_capture_mode=mode)
    with genai_otel.decision_span(model="eot", questions=QUESTIONS, audio=_wav()) as r:
        r.record(ANSWERS)
    attrs = _attrs(exporter)
    parts = _part_keys(attrs)
    assert parts["gen_ai.prompt.0.content.0.type"] == "audio"
    assert parts["gen_ai.prompt.0.content.0.media_mime_type"] == "audio/wav"
    if mode == "full":
        uri = parts["gen_ai.prompt.0.content.0.media_uri"]
        assert uri.startswith("file:")
        assert f"[audio: {uri}]" in attrs["input.value"]
    else:
        assert "gen_ai.prompt.0.content.0.media_uri" not in parts


def test_over_the_cap_is_recorded_without_bytes(spans):
    exporter = spans(media_capture_mode="full", media_max_bytes=100)
    with genai_otel.decision_span(model="eot", questions=QUESTIONS, audio=_wav(1.0)) as r:
        r.record(ANSWERS)
    parts = _part_keys(_attrs(exporter))
    assert parts["gen_ai.prompt.0.content.0.type"] == "audio"
    assert parts["gen_ai.prompt.0.content.0.media_byte_size"] == len(_wav(1.0))
    assert "gen_ai.prompt.0.content.0.media_uri" not in parts


def test_media_modality_allow_list_applies(spans):
    exporter = spans(media_capture_mode="full", media_allowed_modalities="image")
    with genai_otel.decision_span(model="eot", questions=QUESTIONS, audio=_wav()) as r:
        r.record(ANSWERS)
    parts = _part_keys(_attrs(exporter))
    assert "gen_ai.prompt.0.content.0.media_uri" not in parts


# --------------------------------------------------------------------------- SDK path


def test_openai_decisions_with_an_input_audio_part():
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = OpenAIInstrumentor()
    inst.tracer = provider.get_tracer("test")
    inst._instrumented = True
    inst.config = OTelConfig(service_name="test", enable_content_capture=True)

    body = {"answers": [{"type": "predicate", "probability": 0.8}], "model": "gpt-6-luna"}
    client = openai.OpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
        ),
    )
    inst._instrument_client(client)
    if not hasattr(client, "decisions"):
        pytest.skip("this openai release has no Decisions resource")
    wav_b64 = base64.b64encode(_wav()).decode()
    client.decisions.create(
        model="gpt-6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Has the caller finished?"},
                    {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
                ],
            }
        ],
        questions=[{"type": "predicate", "name": "finished", "instructions": "Finished?"}],
    )
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert list(attrs["genai.decision.input.modalities"]) == ["audio", "text"]
    assert wav_b64 not in attrs["input.value"]
    assert d.AUDIO_OMITTED in attrs["input.value"]
    assert d.IMAGE_OMITTED not in attrs["input.value"]
