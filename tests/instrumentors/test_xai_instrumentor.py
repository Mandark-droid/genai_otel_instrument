from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.xai_instrumentor import XAIInstrumentor


class RecordingSpan:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = dict(attributes)
        self.is_recording = lambda: True
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def end(self):
        self.ended = True

    def record_exception(self, exc):
        pass

    def set_status(self, status):
        pass


def test_xai_provider_ticks_are_recorded_as_exact_usd_cost():
    instrumentor = XAIInstrumentor()
    instrumentor.config = OTelConfig(enable_cost_tracking=True)
    instrumentor.cost_counter = MagicMock()
    instrumentor.latency_histogram = MagicMock()
    span = RecordingSpan(
        "xai.image.sample",
        {"gen_ai.request.model": "grok-imagine-image-2.0", "gen_ai.request.type": "image"},
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(cost_in_usd_ticks=400_000_000),
        cost_usd=0.04,
    )

    instrumentor._finish_span(span, response, 0, operation="image")

    assert span.attributes["gen_ai.usage.cost.total"] == 0.04
    assert span.attributes["gen_ai.usage.cost.pricing_source"] == "provider"
    instrumentor.cost_counter.add.assert_called_once_with(0.04, {"model": "grok-imagine-image-2.0"})
    assert span.ended


def test_xai_tts_character_usage_uses_published_audio_rate():
    instrumentor = XAIInstrumentor()
    instrumentor.config = OTelConfig(enable_cost_tracking=True)
    instrumentor.cost_counter = MagicMock()
    instrumentor.latency_histogram = MagicMock()
    span = RecordingSpan(
        "xai.voice.audio",
        {"gen_ai.request.model": "xai-tts", "gen_ai.request.type": "audio"},
    )
    response = SimpleNamespace(content=b"audio bytes")

    instrumentor._finish_span(
        span,
        response,
        0,
        operation="audio",
        usage_override={"characters": 100},
    )

    assert span.attributes["gen_ai.usage.cost.total"] == 0.0015
    assert span.attributes["gen_ai.usage.cost.pricing_source"] == "table"
    instrumentor.cost_counter.add.assert_called_once_with(0.0015, {"model": "xai-tts"})
    assert span.ended


def test_xai_stt_duration_uses_published_rest_rate():
    instrumentor = XAIInstrumentor()
    instrumentor.config = OTelConfig(enable_cost_tracking=True)
    instrumentor.cost_counter = MagicMock()
    instrumentor.latency_histogram = MagicMock()
    span = RecordingSpan(
        "xai.voice.speech_to_text",
        {"gen_ai.request.model": "xai-stt", "gen_ai.request.type": "speech_to_text"},
    )

    instrumentor._finish_span(
        span,
        SimpleNamespace(duration=3600),
        0,
        operation="speech_to_text",
        usage_override={"seconds": 3600},
    )

    assert span.attributes["gen_ai.usage.cost.total"] == pytest.approx(0.10)
    instrumentor.cost_counter.add.assert_called_once_with(pytest.approx(0.10), {"model": "xai-stt"})
    assert span.ended


def test_xai_tts_rest_wrapper_only_passes_character_count_to_telemetry():
    import requests

    session_type = requests.sessions.Session
    original_request = session_type.request
    original_flag = getattr(session_type, "_genai_otel_xai_voice_wrapped", None)
    if hasattr(session_type, "_genai_otel_xai_voice_wrapped"):
        delattr(session_type, "_genai_otel_xai_voice_wrapped")
    response = SimpleNamespace(status_code=200)
    raw_call = MagicMock(return_value=response)
    session_type.request = raw_call
    instrumentor = XAIInstrumentor()
    instrumentor._start_span = MagicMock(return_value=RecordingSpan("xai.voice.audio", {}))
    instrumentor._finish_span = MagicMock()

    try:
        instrumentor._instrument_voice_http()
        result = requests.Session().request(
            "POST",
            "https://api.x.ai/v1/tts",
            json={"text": "private text", "voice_id": "eve", "language": "en"},
        )
    finally:
        session_type.request = original_request
        if original_flag is not None:
            session_type._genai_otel_xai_voice_wrapped = original_flag
        elif hasattr(session_type, "_genai_otel_xai_voice_wrapped"):
            delattr(session_type, "_genai_otel_xai_voice_wrapped")

    assert result is response
    instrumentor._finish_span.assert_called_once()
    assert instrumentor._finish_span.call_args.kwargs["usage_override"] == {"characters": 12}
    raw_call.assert_called_once()
