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

        instrumentor.tracer.start_as_current_span.assert_called_once_with(
            "sarvam.chat.completions"
        )

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
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "translate")
        mock_span.set_attribute.assert_any_call("sarvam.source_language", "en-IN")
        mock_span.set_attribute.assert_any_call("sarvam.target_language", "hi-IN")

        instrumentor.request_counter.add.assert_called_once_with(
            1, {"operation": "translate", "provider": "sarvam"}
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

        result = mock_client.chat.completions(
            messages=[{"role": "user", "content": "hi"}]
        )

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

        instrumentor.tracer.start_as_current_span.assert_called_with(
            "sarvam.text.transliterate"
        )

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
