import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.sarvam_instrumentor import SarvamAIInstrumentor


class TestSarvamAIInstrumentor(unittest.TestCase):
    """Tests for SarvamAIInstrumentor"""

    @patch("genai_otel.instrumentors.sarvam_instrumentor.logger")
    def test_init_with_sarvam_available(self, mock_logger):
        """Test that __init__ detects sarvamai availability."""
        with patch.dict("sys.modules", {"sarvamai": MagicMock()}):
            instrumentor = SarvamAIInstrumentor()

            self.assertTrue(instrumentor._sarvam_available)
            mock_logger.debug.assert_called_with(
                "Sarvam AI library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.sarvam_instrumentor.logger")
    def test_init_with_sarvam_not_available(self, mock_logger):
        """Test that __init__ handles missing sarvamai gracefully."""
        with patch.dict("sys.modules", {"sarvamai": None}):
            instrumentor = SarvamAIInstrumentor()

            self.assertFalse(instrumentor._sarvam_available)
            mock_logger.debug.assert_called_with(
                "Sarvam AI library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.sarvam_instrumentor.logger")
    def test_instrument_with_sarvam_not_available(self, mock_logger):
        """Test that instrument skips when sarvamai is not available."""
        with patch.dict("sys.modules", {"sarvamai": None}):
            instrumentor = SarvamAIInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Sarvam AI instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.sarvam_instrumentor.logger")
    def test_instrument_with_sarvam_available(self, mock_logger):
        """Test that instrument wraps SarvamAI client when available."""

        class MockSarvamAIClass:
            def __init__(self, *args, **kwargs):
                self.chat = MagicMock()
                self.chat.completions = MagicMock()
                self.text = MagicMock()
                self.text.translate = MagicMock()
                self.text.transliterate = MagicMock()
                self.text.identify_language = MagicMock()
                self.speech_to_text = MagicMock()
                self.speech_to_text.transcribe = MagicMock()
                self.speech_to_text.translate = MagicMock()
                self.text_to_speech = MagicMock()
                self.text_to_speech.convert = MagicMock()

        mock_sarvamai = MagicMock()
        mock_sarvamai.SarvamAI = MockSarvamAIClass

        with patch.dict("sys.modules", {"sarvamai": mock_sarvamai}):
            instrumentor = SarvamAIInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            # Verify instrumentor state is correctly set
            self.assertTrue(instrumentor._instrumented)
            self.assertEqual(instrumentor.config, config)
            mock_logger.info.assert_called_with("Sarvam AI instrumentation enabled")

    @patch("genai_otel.instrumentors.sarvam_instrumentor.logger")
    def test_instrument_with_exception_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        mock_sarvamai = MagicMock()
        type(mock_sarvamai.SarvamAI).__init__ = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"sarvamai": mock_sarvamai}):
            instrumentor = SarvamAIInstrumentor()
            config = OTelConfig(fail_on_error=False)

            instrumentor.instrument(config)

            mock_logger.error.assert_called()

    def test_instrument_with_exception_fail_on_error_true(self):
        """Test that exceptions are raised when fail_on_error is True."""
        mock_sarvamai = MagicMock()
        type(mock_sarvamai.SarvamAI).__init__ = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"sarvamai": mock_sarvamai}):
            instrumentor = SarvamAIInstrumentor()
            config = OTelConfig(fail_on_error=True)

            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Access failed")

    def test_instrument_client_chat_completions(self):
        """Test that _instrument_client wraps chat completions."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_completions = MagicMock(return_value="result")
        mock_client.chat.completions = original_completions

        instrumentor.tracer = MagicMock()
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        self.assertNotEqual(mock_client.chat.completions, original_completions)

    def test_instrument_client_text_translate(self):
        """Test that _instrument_client wraps text.translate."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_translate = MagicMock(return_value="result")
        mock_client.text.translate = original_translate

        instrumentor.tracer = MagicMock()
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        self.assertNotEqual(mock_client.text.translate, original_translate)

    def test_instrument_client_speech_to_text(self):
        """Test that _instrument_client wraps speech_to_text.transcribe."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_transcribe = MagicMock(return_value="result")
        mock_client.speech_to_text.transcribe = original_transcribe

        instrumentor.tracer = MagicMock()
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        self.assertNotEqual(mock_client.speech_to_text.transcribe, original_transcribe)

    def test_instrument_client_text_to_speech(self):
        """Test that _instrument_client wraps text_to_speech.convert."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_convert = MagicMock(return_value="result")
        mock_client.text_to_speech.convert = original_convert

        instrumentor.tracer = MagicMock()
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        self.assertNotEqual(mock_client.text_to_speech.convert, original_convert)

    def test_wrapped_chat_completions_execution(self):
        """Test that wrapped chat completions method executes correctly."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_completions = MagicMock(
            return_value=MagicMock(
                usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                choices=[MagicMock(message=MagicMock(content="Hello from Sarvam"))],
            )
        )
        mock_client.chat.completions = original_completions

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_result_metrics = MagicMock()

        instrumentor._instrument_client(mock_client)

        result = mock_client.chat.completions(
            model="sarvam-m", messages=[{"role": "user", "content": "Namaste"}]
        )

        instrumentor.tracer.start_as_current_span.assert_called_once_with("sarvam.chat.completions")

        mock_span.set_attribute.assert_any_call("gen_ai.system", "sarvam")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "sarvam-m")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "chat")

        instrumentor.request_counter.add.assert_called_once_with(
            1, {"model": "sarvam-m", "provider": "sarvam"}
        )

        original_completions.assert_called_once_with(
            model="sarvam-m", messages=[{"role": "user", "content": "Namaste"}]
        )

        instrumentor._record_result_metrics.assert_called_once()

    def test_wrapped_translate_execution(self):
        """Test that wrapped translate method executes correctly."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.translated_text = "Translated text in Hindi"
        original_translate = MagicMock(return_value=mock_result)
        mock_client.text.translate = original_translate

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        result = mock_client.text.translate(
            input="Hello", source_language_code="en-IN", target_language_code="hi-IN"
        )

        instrumentor.tracer.start_as_current_span.assert_called_with("sarvam.text.translate")

        mock_span.set_attribute.assert_any_call("gen_ai.system", "sarvam")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "mayura:v1")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "translate")
        mock_span.set_attribute.assert_any_call("sarvam.source_language", "en-IN")
        mock_span.set_attribute.assert_any_call("sarvam.target_language", "hi-IN")

        instrumentor.request_counter.add.assert_called_once_with(
            1, {"model": "mayura:v1", "operation": "translate", "provider": "sarvam"}
        )

    def test_wrapped_speech_to_text_execution(self):
        """Test that wrapped speech_to_text.transcribe executes correctly."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.transcript = "Transcribed text in Hindi"
        original_transcribe = MagicMock(return_value=mock_result)
        mock_client.speech_to_text.transcribe = original_transcribe

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        result = mock_client.speech_to_text.transcribe(
            file="audio.wav", model="saarika:v2.5", language_code="hi-IN"
        )

        instrumentor.tracer.start_as_current_span.assert_called_with(
            "sarvam.speech_to_text.transcribe"
        )

        mock_span.set_attribute.assert_any_call("gen_ai.system", "sarvam")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "saarika:v2.5")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "speech_to_text")
        mock_span.set_attribute.assert_any_call("sarvam.language_code", "hi-IN")

    def test_wrapped_text_to_speech_execution(self):
        """Test that wrapped text_to_speech.convert executes correctly."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_convert = MagicMock(return_value=MagicMock())
        mock_client.text_to_speech.convert = original_convert

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        result = mock_client.text_to_speech.convert(
            text="Namaste", target_language_code="hi-IN", speaker="shubh"
        )

        instrumentor.tracer.start_as_current_span.assert_called_with(
            "sarvam.text_to_speech.convert"
        )

        mock_span.set_attribute.assert_any_call("gen_ai.system", "sarvam")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "text_to_speech")
        mock_span.set_attribute.assert_any_call("sarvam.target_language", "hi-IN")
        mock_span.set_attribute.assert_any_call("sarvam.speaker", "shubh")
        mock_span.set_attribute.assert_any_call("sarvam.input_text_length", 7)

    def test_wrapped_chat_with_default_model(self):
        """Test that wrapped chat completions uses default model when not specified."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_completions = MagicMock(return_value=MagicMock())
        mock_client.chat.completions = original_completions

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_result_metrics = MagicMock()

        instrumentor._instrument_client(mock_client)

        result = mock_client.chat.completions(messages=[{"role": "user", "content": "hi"}])

        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "sarvam-m")

    def test_extract_usage_with_usage_field(self):
        """Test that _extract_usage extracts from usage field."""
        instrumentor = SarvamAIInstrumentor()

        result = MagicMock()
        result.usage.prompt_tokens = 15
        result.usage.completion_tokens = 25
        result.usage.total_tokens = 40

        usage = instrumentor._extract_usage(result)

        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 15)
        self.assertEqual(usage["completion_tokens"], 25)
        self.assertEqual(usage["total_tokens"], 40)

    def test_extract_usage_without_usage_field(self):
        """Test that _extract_usage returns None when no usage field."""
        instrumentor = SarvamAIInstrumentor()

        result = MagicMock(spec=[])

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)

    def test_extract_response_attributes_with_content(self):
        """Test that response content is captured for evaluation support."""
        instrumentor = SarvamAIInstrumentor()

        mock_message = MagicMock()
        mock_message.content = "AI aur machine learning ka matlab hai..."

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        attrs = instrumentor._extract_response_attributes(mock_response)

        self.assertIn("gen_ai.response", attrs)
        self.assertEqual(attrs["gen_ai.response"], "AI aur machine learning ka matlab hai...")

    def test_extract_response_attributes_without_content(self):
        """Test graceful handling when response has no content."""
        instrumentor = SarvamAIInstrumentor()

        mock_response = MagicMock()
        mock_response.choices = []

        attrs = instrumentor._extract_response_attributes(mock_response)
        self.assertNotIn("gen_ai.response", attrs)

        mock_response_none = MagicMock()
        mock_response_none.choices = None

        attrs = instrumentor._extract_response_attributes(mock_response_none)
        self.assertNotIn("gen_ai.response", attrs)

    def test_evaluation_support_request_capture(self):
        """Test that request content is captured for evaluation support."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_completions = MagicMock(return_value=MagicMock())
        mock_client.chat.completions = original_completions

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_result_metrics = MagicMock()
        instrumentor._extract_response_attributes = MagicMock(return_value={})

        instrumentor._instrument_client(mock_client)

        mock_client.chat.completions(
            model="sarvam-m",
            messages=[{"role": "user", "content": "What is artificial intelligence?"}],
        )

        set_attribute_calls = [call[0] for call in mock_span.set_attribute.call_args_list]
        self.assertTrue(
            any(
                "gen_ai.request.first_message" in call and "artificial intelligence" in str(call)
                for call in set_attribute_calls
            )
        )

    def test_wrapped_transliterate_execution(self):
        """Test that wrapped transliterate method executes correctly."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        original_transliterate = MagicMock(return_value=MagicMock())
        mock_client.text.transliterate = original_transliterate

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        result = mock_client.text.transliterate(
            source_language_code="hi-IN", target_language_code="en-IN"
        )

        instrumentor.tracer.start_as_current_span.assert_called_with("sarvam.text.transliterate")

        mock_span.set_attribute.assert_any_call("gen_ai.system", "sarvam")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "transliterate")

    def test_wrapped_identify_language_execution(self):
        """Test that wrapped identify_language method executes correctly."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.language_code = "hi-IN"
        original_identify = MagicMock(return_value=mock_result)
        mock_client.text.identify_language = original_identify

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()

        instrumentor._instrument_client(mock_client)

        result = mock_client.text.identify_language(input="Namaste")

        instrumentor.tracer.start_as_current_span.assert_called_with(
            "sarvam.text.identify_language"
        )

        mock_span.set_attribute.assert_any_call("gen_ai.system", "sarvam")
        mock_span.set_attribute.assert_any_call("sarvam.detected_language", "hi-IN")

    def test_safe_kwarg_returns_value(self):
        """Test that _safe_kwarg returns the actual value when present."""
        from genai_otel.instrumentors.sarvam_instrumentor import _safe_kwarg

        self.assertEqual(_safe_kwarg({"key": "value"}, "key"), "value")
        self.assertEqual(_safe_kwarg({"key": 42}, "key"), 42)

    def test_safe_kwarg_returns_default_for_missing(self):
        """Test that _safe_kwarg returns default when key is missing."""
        from genai_otel.instrumentors.sarvam_instrumentor import _safe_kwarg

        self.assertIsNone(_safe_kwarg({}, "key"))
        self.assertEqual(_safe_kwarg({}, "key", "default"), "default")

    def test_safe_kwarg_returns_default_for_none(self):
        """Test that _safe_kwarg returns default when value is None."""
        from genai_otel.instrumentors.sarvam_instrumentor import _safe_kwarg

        self.assertEqual(_safe_kwarg({"key": None}, "key", "default"), "default")

    def test_safe_kwarg_handles_omit_sentinel(self):
        """Test that _safe_kwarg detects OMIT/NotGiven sentinels."""
        from genai_otel.instrumentors.sarvam_instrumentor import _safe_kwarg

        class NotGiven:
            pass

        class OMIT:
            pass

        self.assertEqual(_safe_kwarg({"k": NotGiven()}, "k", "default"), "default")
        self.assertEqual(_safe_kwarg({"k": OMIT()}, "k", "default"), "default")

    def test_normalize_sarvam_tts_model_bulbul(self):
        """Test TTS model normalization for bulbul models."""
        self.assertEqual(SarvamAIInstrumentor._normalize_sarvam_tts_model("bulbul:v2"), "bulbul-v2")
        self.assertEqual(SarvamAIInstrumentor._normalize_sarvam_tts_model("bulbul:v3"), "bulbul-v3")

    def test_normalize_sarvam_tts_model_non_bulbul(self):
        """Test TTS model normalization leaves non-bulbul models unchanged."""
        self.assertEqual(
            SarvamAIInstrumentor._normalize_sarvam_tts_model("saarika:v2.5"), "saarika:v2.5"
        )
        self.assertEqual(SarvamAIInstrumentor._normalize_sarvam_tts_model("mayura:v1"), "mayura:v1")

    def test_record_sarvam_cost_with_pricing(self):
        """Test that _record_sarvam_cost records cost when pricing is available."""
        instrumentor = SarvamAIInstrumentor()
        instrumentor.config = OTelConfig(enable_cost_tracking=True)
        instrumentor.cost_calculator = MagicMock()
        instrumentor.cost_calculator.pricing_data = {
            "speech_to_text": {
                "mayura:v1": {"promptPrice": 0.000024, "completionPrice": 0},
            }
        }
        instrumentor.latency_histogram = MagicMock()
        instrumentor.cost_counter = MagicMock()

        mock_span = MagicMock()
        mock_span.name = "sarvam.text.translate"

        import time

        start_time = time.time() - 0.5  # Simulate 0.5s elapsed

        instrumentor._record_sarvam_cost(mock_span, "mayura:v1", 1000, start_time)

        # Should record character count
        mock_span.set_attribute.assert_any_call("gen_ai.usage.characters", 1000)
        # Should record cost: (1000/1000) * 0.000024 = 0.000024
        mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.total", 0.000024)
        # Should record latency
        instrumentor.latency_histogram.record.assert_called_once()

    def test_record_sarvam_cost_no_pricing(self):
        """Test that _record_sarvam_cost handles missing pricing gracefully."""
        instrumentor = SarvamAIInstrumentor()
        instrumentor.config = OTelConfig(enable_cost_tracking=True)
        instrumentor.cost_calculator = MagicMock()
        instrumentor.cost_calculator.pricing_data = {"speech_to_text": {}}
        instrumentor.latency_histogram = MagicMock()
        instrumentor.cost_counter = MagicMock()

        mock_span = MagicMock()
        mock_span.name = "test"

        import time

        instrumentor._record_sarvam_cost(mock_span, "unknown-model", 100, time.time())

        # Should still record character count
        mock_span.set_attribute.assert_any_call("gen_ai.usage.characters", 100)
        # Should not set cost attribute (no pricing found)
        cost_calls = [
            c
            for c in mock_span.set_attribute.call_args_list
            if c[0][0] == "gen_ai.usage.cost.total"
        ]
        self.assertEqual(len(cost_calls), 0)

    def test_translate_with_explicit_model(self):
        """Test translate with explicitly specified model."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.translated_text = "Translated text"
        original_translate = MagicMock(return_value=mock_result)
        mock_client.text.translate = original_translate

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_sarvam_cost = MagicMock()

        instrumentor._instrument_client(mock_client)

        mock_client.text.translate(
            input="Hello",
            source_language_code="en-IN",
            target_language_code="hi-IN",
            model="sarvam-translate:v1",
        )

        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "sarvam-translate:v1")
        instrumentor.request_counter.add.assert_called_once_with(
            1, {"model": "sarvam-translate:v1", "operation": "translate", "provider": "sarvam"}
        )

    def test_translate_captures_metadata(self):
        """Test translate captures Sarvam-specific metadata params."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.translated_text = "Translated"
        mock_client.text.translate = MagicMock(return_value=mock_result)

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_sarvam_cost = MagicMock()

        instrumentor._instrument_client(mock_client)

        mock_client.text.translate(
            input="Hello",
            source_language_code="en-IN",
            target_language_code="hi-IN",
            mode="classic-colloquial",
            speaker_gender="male",
            numerals_format="international",
        )

        mock_span.set_attribute.assert_any_call("sarvam.translate.mode", "classic-colloquial")
        mock_span.set_attribute.assert_any_call("sarvam.translate.speaker_gender", "male")
        mock_span.set_attribute.assert_any_call("sarvam.translate.numerals_format", "international")

    def test_tts_captures_metadata(self):
        """Test TTS captures Sarvam-specific metadata params."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = MagicMock(return_value=MagicMock())

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_sarvam_cost = MagicMock()

        instrumentor._instrument_client(mock_client)

        mock_client.text_to_speech.convert(
            text="Namaste",
            target_language_code="hi-IN",
            speaker="shubh",
            model="bulbul:v3",
            pace=1.2,
            pitch=0.5,
            loudness=1.5,
        )

        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "bulbul-v3")
        mock_span.set_attribute.assert_any_call("sarvam.tts.pace", 1.2)
        mock_span.set_attribute.assert_any_call("sarvam.tts.pitch", 0.5)
        mock_span.set_attribute.assert_any_call("sarvam.tts.loudness", 1.5)

    def test_tts_default_model_is_bulbul_v2(self):
        """Test TTS uses bulbul-v2 as default model when not specified."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = MagicMock(return_value=MagicMock())

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_sarvam_cost = MagicMock()

        instrumentor._instrument_client(mock_client)

        mock_client.text_to_speech.convert(
            text="Hello", target_language_code="en-IN", speaker="shubh"
        )

        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "bulbul-v2")

    def test_transliterate_has_model_attribute(self):
        """Test transliterate sets model attribute to sarvam-transliterate."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_client.text.transliterate = MagicMock(return_value=MagicMock())

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_sarvam_cost = MagicMock()

        instrumentor._instrument_client(mock_client)

        mock_client.text.transliterate(
            input="Namaste",
            source_language_code="en-IN",
            target_language_code="hi-IN",
        )

        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "sarvam-transliterate")

    def test_identify_language_has_model_attribute(self):
        """Test identify_language sets model attribute to sarvam-detect-language."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.language_code = "hi-IN"
        mock_client.text.identify_language = MagicMock(return_value=mock_result)

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_sarvam_cost = MagicMock()

        instrumentor._instrument_client(mock_client)

        mock_client.text.identify_language(input="Namaste")

        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "sarvam-detect-language")

    def test_chat_completions_records_start_time(self):
        """Test that chat completions passes real start_time to _record_result_metrics."""
        instrumentor = SarvamAIInstrumentor()

        mock_client = MagicMock()
        mock_client.chat.completions = MagicMock(return_value=MagicMock())

        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_result_metrics = MagicMock()

        instrumentor._instrument_client(mock_client)

        import time

        before = time.time()
        mock_client.chat.completions(messages=[{"role": "user", "content": "hi"}])
        after = time.time()

        call_args = instrumentor._record_result_metrics.call_args
        start_time_arg = call_args[0][2]  # Third positional arg is start_time

        # start_time should be a real timestamp, not 0
        self.assertGreater(start_time_arg, 0)
        self.assertGreaterEqual(start_time_arg, before)
        self.assertLessEqual(start_time_arg, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
