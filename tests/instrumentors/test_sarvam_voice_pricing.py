"""Sarvam speech is priced from media, at the published rates.

Measured on a voice workload: transcription carried no cost at all (``saaras:v3`` was not in
the table, and STT was priced per transcript character although Sarvam bills per hour of
audio), and text-to-speech read about 1000x low because per-character rates sat in a
per-1,000 table.
"""

import io
import struct
import wave
from unittest.mock import MagicMock

import pytest

from genai_otel.config import OTelConfig
from genai_otel.cost_calculator import CostCalculator
from genai_otel.instrumentors.sarvam_instrumentor import SarvamAIInstrumentor, _wav_seconds


def _wav(seconds: float, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buf.getvalue()


def _instrumentor() -> SarvamAIInstrumentor:
    inst = SarvamAIInstrumentor()
    inst.config = OTelConfig(enable_cost_tracking=True)
    inst.cost_calculator = CostCalculator()
    inst.latency_histogram = MagicMock()
    inst.cost_counter = MagicMock()
    inst.request_counter = MagicMock()
    return inst


def _attrs(span: MagicMock) -> dict:
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}


class TestWavSeconds:
    def test_bytes_path_tuple_and_file_like(self, tmp_path):
        data = _wav(2.5)
        assert _wav_seconds(data) == pytest.approx(2.5)
        assert _wav_seconds(("call.wav", data, "audio/wav")) == pytest.approx(2.5)
        path = tmp_path / "a.wav"
        path.write_bytes(data)
        assert _wav_seconds(str(path)) == pytest.approx(2.5)

    def test_a_file_object_is_left_where_it_was(self):
        f = io.BytesIO(_wav(1.0))
        assert _wav_seconds(f) == pytest.approx(1.0)
        assert f.tell() == 0  # the SDK still has to read it

    @pytest.mark.parametrize("audio", [b"not a wav", None, 42, "missing.wav"])
    def test_anything_unreadable_is_none_not_zero(self, audio):
        assert _wav_seconds(audio) is None


class TestSpeechToTextIsPricedByTheSecond:
    def _transcribe(self, **kwargs):
        inst = _instrumentor()
        client = MagicMock()
        client.speech_to_text.transcribe = MagicMock(
            return_value=MagicMock(transcript="namaste ji")
        )
        span = MagicMock()
        inst.tracer = MagicMock()
        inst.tracer.start_as_current_span.return_value.__enter__.return_value = span
        inst._instrument_client(client)
        client.speech_to_text.transcribe(**kwargs)
        return _attrs(span)

    def test_saaras_v3_is_priced_from_the_audio_it_was_sent(self):
        attrs = self._transcribe(file=("t.wav", _wav(36.0)), model="saaras:v3")
        assert attrs["gen_ai.usage.audio_duration_seconds"] == pytest.approx(36.0)
        # INR 30/hour at INR 95/USD: 36 s = 0.01 h = INR 0.30 = $0.003158
        assert attrs["gen_ai.usage.cost.total"] == pytest.approx(0.30 / 95, rel=1e-3)

    def test_audio_it_cannot_read_has_no_cost_rather_than_a_guessed_one(self):
        attrs = self._transcribe(file="not-a-file.mp3", model="saaras:v3")
        assert "gen_ai.usage.cost.total" not in attrs
        assert "gen_ai.usage.audio_duration_seconds" not in attrs


class TestTextToSpeechIsPricedPerThousandCharacters:
    def test_bulbul_v3_at_the_published_rate(self):
        inst = _instrumentor()
        client = MagicMock()
        client.text_to_speech.convert = MagicMock(
            return_value=MagicMock(audio_duration_seconds=None, duration=None)
        )
        span = MagicMock()
        inst.tracer = MagicMock()
        inst.tracer.start_as_current_span.return_value.__enter__.return_value = span
        inst._instrument_client(client)
        client.text_to_speech.convert(
            text="x" * 500, model="bulbul:v3", target_language_code="hi-IN"
        )
        attrs = _attrs(span)
        assert attrs["gen_ai.request.model"] == "bulbul:v3"
        # INR 3.00 per 1,000 characters: 500 chars = INR 1.50 = $0.0158 -- it read ~$0.000018.
        assert attrs["gen_ai.usage.cost.total"] == pytest.approx(1.50 / 95, rel=1e-3)


def test_sarvam_105b_conversations_is_no_longer_priced_as_free():
    calc = CostCalculator()
    cost = calc.calculate_cost(
        "sarvam-105b-conversations", {"prompt_tokens": 1508, "completion_tokens": 25}, "chat"
    )
    expected = (1508 * 29.28 + 25 * 73.20) / 1_000_000 / 95
    assert cost == pytest.approx(expected, rel=1e-3)
    assert calc.pricing_source("sarvam-105b-conversations", "chat") == "table"


def test_per_character_text_services_are_read_in_their_unit():
    """mayura is INR 20 per 10K characters by its own note: 1,000 characters is INR 2."""
    calc = CostCalculator()
    entry = calc.pricing_data["speech_to_text"]["mayura:v1"]
    assert entry["promptPrice"] == pytest.approx(0.024)  # per 1K chars, was 0.000024
