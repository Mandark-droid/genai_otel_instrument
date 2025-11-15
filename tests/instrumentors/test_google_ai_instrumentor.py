import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.google_ai_instrumentor import GoogleAIInstrumentor


class TestGoogleAIInstrumentor(unittest.TestCase):
    """Tests for GoogleAIInstrumentor"""

    def test_init_with_new_sdk_available(self):
        """Test that __init__ detects new google-genai SDK availability."""
        # Mock the new SDK (from google import genai)
        mock_google = MagicMock()
        mock_google.genai = MagicMock()

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_google.genai}):
            instrumentor = GoogleAIInstrumentor()
            self.assertTrue(instrumentor._google_available)
            self.assertTrue(instrumentor._using_new_sdk)

    def test_init_with_legacy_sdk_available(self):
        """Test that __init__ detects legacy google.generativeai availability."""
        # Only patch google.generativeai, not the google namespace package
        with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
            instrumentor = GoogleAIInstrumentor()
            self.assertTrue(instrumentor._google_available)
            self.assertFalse(instrumentor._using_new_sdk)

    def test_init_with_google_not_available(self):
        """Test that __init__ handles missing google SDKs gracefully."""
        # Remove both SDKs from sys.modules
        with patch.dict("sys.modules", {"google.generativeai": None, "google": None}):
            instrumentor = GoogleAIInstrumentor()
            self.assertFalse(instrumentor._google_available)

    def test_instrument_with_google_not_available(self):
        """Test that instrument skips when google SDKs are not available."""
        with patch.dict("sys.modules", {"google": None, "google.generativeai": None}):
            instrumentor = GoogleAIInstrumentor()
            config = OTelConfig()

            # Should not raise
            instrumentor.instrument(config)

    def test_instrument_with_legacy_sdk(self):
        """Test that instrument wraps GenerativeModel when legacy SDK available."""

        # Create mock GenerativeModel
        class MockGenerativeModel:
            def generate_content(self, *args, **kwargs):
                return "result"

        # Create mock google.generativeai module
        mock_genai = MagicMock()
        mock_genai.GenerativeModel = MockGenerativeModel

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            instrumentor = GoogleAIInstrumentor()
            config = OTelConfig()

            # Mock the create_span_wrapper to return a wrapper function
            def mock_wrapper(original_func):
                def wrapper(*args, **kwargs):
                    return original_func(*args, **kwargs)

                return wrapper

            instrumentor.create_span_wrapper = MagicMock(return_value=mock_wrapper)

            # Call instrument
            instrumentor.instrument(config)

            # Verify create_span_wrapper was called
            instrumentor.create_span_wrapper.assert_called_once()
            call_kwargs = instrumentor.create_span_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "google.generativeai.generate_content")
            self.assertEqual(
                call_kwargs["extract_attributes"], instrumentor._extract_google_ai_attributes
            )

            # Verify generate_content was replaced
            self.assertIsNotNone(mock_genai.GenerativeModel.generate_content)
            self.assertTrue(instrumentor._instrumented)

    def test_instrument_with_new_sdk(self):
        """Test that instrument wraps Client when new SDK available."""

        # Create mock Client
        class MockClient:
            def __init__(self, *args, **kwargs):
                self.models = MagicMock()
                self.models.generate_content = MagicMock(return_value="result")

        # Create mock google.genai module (new SDK)
        mock_google = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client = MockClient
        mock_google.genai = mock_genai

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.genai": mock_genai,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = GoogleAIInstrumentor()
            config = OTelConfig()

            # Call instrument
            instrumentor.instrument(config)

            # Verify FunctionWrapper was called to wrap Client.__init__
            self.assertTrue(mock_wrapt.FunctionWrapper.called)
            self.assertTrue(instrumentor._instrumented)

    def test_instrument_with_missing_generate_content(self):
        """Test that instrument handles missing generate_content method."""
        # Create mock GenerativeModel without generate_content attribute
        mock_genai = MagicMock()
        mock_genai.GenerativeModel = MagicMock(spec=[])  # Empty spec means no attributes

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            instrumentor = GoogleAIInstrumentor()
            config = OTelConfig()

            # Call instrument
            instrumentor.instrument(config)

            # _instrumented flag is set to True even if generate_content is missing
            self.assertTrue(instrumentor._instrumented)

    def test_instrument_with_exception_fail_on_error_false(self):
        """Test that exceptions are logged when fail_on_error is False."""
        # Create mock that raises
        mock_genai = MagicMock()
        type(mock_genai).GenerativeModel = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            instrumentor = GoogleAIInstrumentor()
            config = OTelConfig(fail_on_error=False)

            # Should not raise
            instrumentor.instrument(config)

    def test_instrument_with_exception_fail_on_error_true(self):
        """Test that exceptions are raised when fail_on_error is True."""

        # Create mock GenerativeModel
        class MockGenerativeModel:
            def generate_content(self, *args, **kwargs):
                return "result"

        # Create mock google.generativeai module
        mock_genai = MagicMock()
        mock_genai.GenerativeModel = MockGenerativeModel

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            instrumentor = GoogleAIInstrumentor()
            config = OTelConfig(fail_on_error=True)

            # Mock create_span_wrapper to raise exception
            instrumentor.create_span_wrapper = MagicMock(
                side_effect=RuntimeError("Wrapper creation failed")
            )

            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Wrapper creation failed")

    def test_extract_google_ai_attributes_legacy_sdk(self):
        """Test that _extract_google_ai_attributes extracts correct attributes for legacy SDK."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock instance with model_name
        mock_instance = MagicMock()
        mock_instance.model_name = "gemini-pro"

        # Mock generation config
        mock_config = MagicMock()
        mock_config.temperature = 0.7
        mock_config.top_p = 0.9
        mock_config.max_output_tokens = 1024

        kwargs = {"generation_config": mock_config, "safety_settings": [MagicMock(), MagicMock()]}

        attrs = instrumentor._extract_google_ai_attributes(mock_instance, None, kwargs)

        self.assertEqual(attrs["gen_ai.system"], "google")
        self.assertEqual(attrs["gen_ai.request.model"], "gemini-pro")
        self.assertEqual(attrs["gen_ai.operation.name"], "chat")
        self.assertEqual(attrs["gen_ai.request.temperature"], 0.7)
        self.assertEqual(attrs["gen_ai.request.top_p"], 0.9)
        self.assertEqual(attrs["gen_ai.request.max_tokens"], 1024)
        self.assertEqual(attrs["gen_ai.request.safety_settings_count"], 2)

    def test_extract_google_ai_attributes_new_sdk(self):
        """Test that _extract_google_ai_attributes_new_sdk extracts correct attributes."""
        instrumentor = GoogleAIInstrumentor()

        kwargs = {
            "model": "gemini-2.0-flash",
            "config": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            },
        }

        attrs = instrumentor._extract_google_ai_attributes_new_sdk(None, None, kwargs)

        self.assertEqual(attrs["gen_ai.system"], "google")
        self.assertEqual(attrs["gen_ai.operation.name"], "chat")
        self.assertEqual(attrs["gen_ai.request.model"], "gemini-2.0-flash")
        self.assertEqual(attrs["gen_ai.request.temperature"], 0.8)
        self.assertEqual(attrs["gen_ai.request.top_p"], 0.95)
        self.assertEqual(attrs["gen_ai.request.max_tokens"], 2048)

    def test_extract_google_ai_attributes_with_unknown_model(self):
        """Test that _extract_google_ai_attributes uses 'unknown' as default."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock instance without model_name
        mock_instance = MagicMock(spec=[])
        if hasattr(mock_instance, "model_name"):
            delattr(mock_instance, "model_name")

        attrs = instrumentor._extract_google_ai_attributes(mock_instance, None, {})

        self.assertEqual(attrs["gen_ai.system"], "google")
        self.assertEqual(attrs["gen_ai.request.model"], "unknown")

    def test_extract_usage_with_usage_metadata(self):
        """Test that _extract_usage extracts from usage_metadata field."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result with usage_metadata
        result = MagicMock()
        result.usage_metadata.prompt_token_count = 15
        result.usage_metadata.candidates_token_count = 25
        result.usage_metadata.total_token_count = 40

        usage = instrumentor._extract_usage(result)

        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 15)
        self.assertEqual(usage["completion_tokens"], 25)
        self.assertEqual(usage["total_tokens"], 40)

    def test_extract_usage_with_alternative_format(self):
        """Test that _extract_usage handles alternative attribute names."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result with alternative usage format
        result = MagicMock()
        del result.usage_metadata  # Remove usage_metadata
        result.usage = MagicMock()
        result.usage.prompt_tokens = 20
        result.usage.completion_tokens = 30
        result.usage.total_tokens = 50

        usage = instrumentor._extract_usage(result)

        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 20)
        self.assertEqual(usage["completion_tokens"], 30)
        self.assertEqual(usage["total_tokens"], 50)

    def test_extract_usage_without_usage_metadata(self):
        """Test that _extract_usage returns None when no usage fields."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result without usage fields
        result = MagicMock(spec=[])
        if hasattr(result, "usage_metadata"):
            delattr(result, "usage_metadata")
        if hasattr(result, "usage"):
            delattr(result, "usage")

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)

    def test_extract_usage_with_none_usage_metadata(self):
        """Test that _extract_usage returns None when usage_metadata is None."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result with None usage_metadata and no usage attribute
        result = MagicMock(spec=["usage_metadata"])
        result.usage_metadata = None

        usage = instrumentor._extract_usage(result)

        # Should return None when no usage data available
        self.assertIsNone(usage)

    def test_extract_usage_with_missing_token_counts(self):
        """Test that _extract_usage handles missing token count attributes."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result with usage_metadata but missing token attributes
        result = MagicMock()
        result.usage_metadata = MagicMock(spec=[])
        if hasattr(result.usage_metadata, "prompt_token_count"):
            delattr(result.usage_metadata, "prompt_token_count")
        if hasattr(result.usage_metadata, "candidates_token_count"):
            delattr(result.usage_metadata, "candidates_token_count")
        if hasattr(result.usage_metadata, "total_token_count"):
            delattr(result.usage_metadata, "total_token_count")

        usage = instrumentor._extract_usage(result)

        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 0)
        self.assertEqual(usage["completion_tokens"], 0)
        self.assertEqual(usage["total_tokens"], 0)

    def test_extract_response_attributes(self):
        """Test that _extract_response_attributes extracts correct attributes."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result
        result = MagicMock()
        result.model = "gemini-1.5-pro"

        # Create mock candidates with finish reasons
        candidate1 = MagicMock()
        candidate1.finish_reason = "STOP"
        candidate1.safety_ratings = [
            MagicMock(category="HARM_CATEGORY_HATE_SPEECH", probability="LOW"),
            MagicMock(category="HARM_CATEGORY_DANGEROUS_CONTENT", probability="NEGLIGIBLE"),
        ]
        result.candidates = [candidate1]

        attrs = instrumentor._extract_response_attributes(result)

        self.assertEqual(attrs["gen_ai.response.model"], "gemini-1.5-pro")
        self.assertIn("STOP", attrs["gen_ai.response.finish_reasons"])
        self.assertEqual(attrs["gen_ai.safety.HARM_CATEGORY_HATE_SPEECH"], "LOW")
        self.assertEqual(attrs["gen_ai.safety.HARM_CATEGORY_DANGEROUS_CONTENT"], "NEGLIGIBLE")

    def test_extract_finish_reason(self):
        """Test that _extract_finish_reason extracts finish reason."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result with candidates
        result = MagicMock()
        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        result.candidates = [candidate]

        finish_reason = instrumentor._extract_finish_reason(result)

        self.assertEqual(finish_reason, "STOP")

    def test_extract_finish_reason_no_candidates(self):
        """Test that _extract_finish_reason returns None when no candidates."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result without candidates
        result = MagicMock()
        result.candidates = []

        finish_reason = instrumentor._extract_finish_reason(result)

        self.assertIsNone(finish_reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
