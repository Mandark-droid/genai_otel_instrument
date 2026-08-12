import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.elevenlabs_instrumentor import (
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    ElevenLabsInstrumentor,
    _arg,
)


class _Span:
    """Minimal span double.

    A bare MagicMock records attribute writes but makes every assertion pass, so
    a real dict is used to assert on what was actually emitted.
    """

    def __init__(self):
        self.attrs = {}
        self.name = "test.span"

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_status(self, *_a, **_k):
        pass

    def record_exception(self, *_a, **_k):
        pass

    def end(self):
        self.ended = True


def _instrumentor(opt_in="gen_ai/dup", cost_tracking=True):
    inst = ElevenLabsInstrumentor()
    inst.config = OTelConfig(service_name="test", enable_cost_tracking=cost_tracking)
    inst.config.semconv_stability_opt_in = opt_in
    inst.cost_counter = None
    inst.request_counter = None
    inst.latency_histogram = None
    inst.ttft_histogram = None
    return inst


class TestAvailability(unittest.TestCase):
    def test_available(self):
        with patch.dict("sys.modules", {"elevenlabs": MagicMock()}):
            self.assertTrue(ElevenLabsInstrumentor()._elevenlabs_available)

    def test_not_available(self):
        with patch.dict("sys.modules", {"elevenlabs": None}):
            self.assertFalse(ElevenLabsInstrumentor()._elevenlabs_available)

    @patch("genai_otel.instrumentors.elevenlabs_instrumentor.logger")
    def test_instrument_skips_when_unavailable(self, mock_logger):
        with patch.dict("sys.modules", {"elevenlabs": None}):
            inst = ElevenLabsInstrumentor()
            inst.instrument(OTelConfig())
            mock_logger.debug.assert_any_call(
                "Skipping ElevenLabs instrumentation - library not available"
            )

    def test_instrument_respects_fail_on_error(self):
        inst = ElevenLabsInstrumentor()
        inst._elevenlabs_available = True
        cfg = OTelConfig()
        cfg.fail_on_error = True
        broken = MagicMock()
        type(broken).ElevenLabs = property(lambda _s: (_ for _ in ()).throw(RuntimeError("boom")))
        with patch.dict("sys.modules", {"elevenlabs": broken}):
            with self.assertRaises(RuntimeError):
                inst.instrument(cfg)

    def test_instrument_swallows_error_by_default(self):
        inst = ElevenLabsInstrumentor()
        inst._elevenlabs_available = True
        cfg = OTelConfig()
        cfg.fail_on_error = False
        broken = MagicMock()
        type(broken).ElevenLabs = property(lambda _s: (_ for _ in ()).throw(RuntimeError("boom")))
        with patch.dict("sys.modules", {"elevenlabs": broken}):
            inst.instrument(cfg)  # must not raise


class TestArgHelper(unittest.TestCase):
    def test_keyword_wins(self):
        self.assertEqual(_arg({"text": "kw"}, ("pos",), "text", 0), "kw")

    def test_positional_fallback(self):
        self.assertEqual(_arg({}, ("pos",), "text", 0), "pos")

    def test_default_when_absent(self):
        self.assertEqual(_arg({}, (), "text", 3, "dflt"), "dflt")


class TestProviderSpelling(unittest.TestCase):
    def test_dup_emits_both_spellings(self):
        inst = _instrumentor(opt_in="gen_ai/dup")
        span = _Span()
        inst._set_provider(span)
        self.assertEqual(span.attrs["gen_ai.provider.name"], "elevenlabs")
        self.assertEqual(span.attrs["gen_ai.system"], "elevenlabs")

    def test_current_only_omits_superseded(self):
        inst = _instrumentor(opt_in="gen_ai")
        span = _Span()
        inst._set_provider(span)
        self.assertEqual(span.attrs["gen_ai.provider.name"], "elevenlabs")
        self.assertNotIn("gen_ai.system", span.attrs)


class TestCostResolution(unittest.TestCase):
    """The pricing table keys TTS bare and Scribe provider-qualified."""

    def test_tts_multilingual_per_1k_characters(self):
        inst = _instrumentor()
        span = _Span()
        inst._record_media_cost(span, "eleven_multilingual_v2", {"characters": 1000})
        self.assertAlmostEqual(span.attrs["gen_ai.usage.cost.total"], 0.10, places=6)

    def test_tts_turbo_is_cheaper_than_multilingual(self):
        inst = _instrumentor()
        span = _Span()
        inst._record_media_cost(span, "eleven_turbo_v2", {"characters": 1000})
        self.assertAlmostEqual(span.attrs["gen_ai.usage.cost.total"], 0.05, places=6)

    def test_scribe_resolves_from_bare_model_id(self):
        """Regression: the SDK passes 'scribe_v1' but the table keys it
        'elevenlabs/scribe_v1'. Without the qualified lookup this silently
        priced every transcription at zero."""
        inst = _instrumentor()
        span = _Span()
        inst._record_media_cost(span, DEFAULT_STT_MODEL, {"seconds": 3600})
        self.assertAlmostEqual(span.attrs["gen_ai.usage.cost.total"], 0.21996, places=5)

    def test_scribe_priced_per_second_not_per_minute(self):
        """60 seconds must cost 1/60th of an hour, not a full hour."""
        inst = _instrumentor()
        hour, minute = _Span(), _Span()
        inst._record_media_cost(hour, DEFAULT_STT_MODEL, {"seconds": 3600})
        inst._record_media_cost(minute, DEFAULT_STT_MODEL, {"seconds": 60})
        self.assertAlmostEqual(
            minute.attrs["gen_ai.usage.cost.total"] * 60,
            hour.attrs["gen_ai.usage.cost.total"],
            places=6,
        )

    def test_unpriced_model_records_no_cost(self):
        """A zero cost must not be written - it would read as 'this was free'."""
        inst = _instrumentor()
        span = _Span()
        inst._record_media_cost(span, "totally_unknown_model_xyz", {"characters": 1000})
        self.assertNotIn("gen_ai.usage.cost.total", span.attrs)

    def test_cost_tracking_disabled(self):
        inst = _instrumentor(cost_tracking=False)
        span = _Span()
        inst._record_media_cost(span, "eleven_multilingual_v2", {"characters": 1000})
        self.assertNotIn("gen_ai.usage.cost.total", span.attrs)


class TestTTSSpan(unittest.TestCase):
    def test_span_setup_from_keywords(self):
        inst = _instrumentor()
        span = _Span()
        model = inst._tts_span_setup(
            span, (), {"voice_id": "v123", "text": "hello", "model_id": "eleven_turbo_v2"}
        )
        self.assertEqual(model, "eleven_turbo_v2")
        self.assertEqual(span.attrs["gen_ai.operation.name"], "text_to_speech")
        self.assertEqual(span.attrs["gen_ai.request.model"], "eleven_turbo_v2")
        self.assertEqual(span.attrs["gen_ai.request.voice_id"], "v123")
        self.assertEqual(span.attrs["gen_ai.usage.characters"], 5)

    def test_span_setup_positional_and_default_model(self):
        inst = _instrumentor()
        span = _Span()
        model = inst._tts_span_setup(span, ("voice-1", "abcdefgh"), {})
        self.assertEqual(model, DEFAULT_TTS_MODEL)
        self.assertEqual(span.attrs["gen_ai.usage.characters"], 8)
        self.assertEqual(span.attrs["gen_ai.request.voice_id"], "voice-1")

    def test_cost_scales_with_character_count(self):
        inst = _instrumentor()
        span = _Span()
        inst._tts_span_setup(span, (), {"text": "x" * 500, "model_id": "eleven_multilingual_v2"})
        self.assertAlmostEqual(span.attrs["gen_ai.usage.cost.total"], 0.05, places=6)

    def test_convert_returns_iterator_without_consuming_it(self):
        """The SDK returns Iterator[bytes]; draining it in the wrapper would
        hand the caller an empty stream."""
        inst = _instrumentor()
        inst.tracer = MagicMock()
        inst.tracer.start_span.return_value = _Span()
        chunks = [b"aa", b"bb", b"cc"]
        tts = SimpleNamespace(convert=lambda **kw: iter(chunks))
        inst._wrap_tts(tts, "convert", is_async=False)

        result = tts.convert(voice_id="v", text="hello")
        self.assertEqual(list(result), chunks)

    def test_streaming_sets_ttft(self):
        inst = _instrumentor()
        span = _Span()
        inst.tracer = MagicMock()
        inst.tracer.start_span.return_value = span
        tts = SimpleNamespace(stream=lambda **kw: iter([b"a", b"b"]))
        inst._wrap_tts(tts, "stream", is_async=False)

        list(tts.stream(voice_id="v", text="hi"))
        self.assertIn("gen_ai.server.ttft", span.attrs)


class TestSTTSpan(unittest.TestCase):
    def _response(self, seconds=120.0, text="hello world", language="en"):
        return SimpleNamespace(audio_duration_secs=seconds, text=text, language_code=language)

    def test_records_duration_transcript_and_cost(self):
        inst = _instrumentor()
        span = _Span()
        inst._stt_record_result(span, DEFAULT_STT_MODEL, self._response(), start_time=0.0)
        self.assertEqual(span.attrs["gen_ai.usage.audio_duration_seconds"], 120.0)
        self.assertEqual(span.attrs["gen_ai.response.transcript_length"], 11)
        self.assertEqual(span.attrs["gen_ai.response.language_code"], "en")
        self.assertAlmostEqual(span.attrs["gen_ai.usage.cost.total"], 6.11e-05 * 120, places=8)

    def test_missing_duration_records_no_cost(self):
        inst = _instrumentor()
        span = _Span()
        inst._stt_record_result(span, DEFAULT_STT_MODEL, SimpleNamespace(text="hi"), start_time=0.0)
        self.assertNotIn("gen_ai.usage.cost.total", span.attrs)
        self.assertNotIn("gen_ai.usage.audio_duration_seconds", span.attrs)

    def test_span_setup_defaults_and_language(self):
        inst = _instrumentor()
        span = _Span()
        model = inst._stt_span_setup(span, (), {"language_code": "hi"})
        self.assertEqual(model, DEFAULT_STT_MODEL)
        self.assertEqual(span.attrs["gen_ai.operation.name"], "speech_to_text")
        self.assertEqual(span.attrs["gen_ai.request.language_code"], "hi")

    def test_convert_wrapper_returns_result(self):
        inst = _instrumentor()
        inst.tracer = MagicMock()
        response = self._response()
        stt = SimpleNamespace(convert=lambda **kw: response)
        inst._wrap_stt(stt, is_async=False)
        self.assertIs(stt.convert(model_id="scribe_v1", file=b""), response)


class TestClientInstrumentation(unittest.TestCase):
    def test_wraps_available_methods_only(self):
        inst = _instrumentor()
        inst.tracer = MagicMock()
        tts = SimpleNamespace(convert=lambda **kw: iter([]))  # no .stream
        stt = SimpleNamespace(convert=lambda **kw: SimpleNamespace())
        client = SimpleNamespace(text_to_speech=tts, speech_to_text=stt)
        original_convert = tts.convert

        inst._instrument_client(client)

        self.assertIsNot(tts.convert, original_convert)
        self.assertFalse(hasattr(tts, "stream"))

    def test_client_without_audio_surfaces_is_safe(self):
        inst = _instrumentor()
        inst.tracer = MagicMock()
        inst._instrument_client(SimpleNamespace())  # must not raise


if __name__ == "__main__":
    unittest.main()
