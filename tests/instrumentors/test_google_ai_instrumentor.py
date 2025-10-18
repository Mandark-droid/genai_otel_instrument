import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.google_ai_instrumentor import GoogleAIInstrumentor


class TestGoogleAIInstrumentor(unittest.TestCase):
    """Tests for GoogleAIInstrumentor"""

    def test_init_with_google_available(self):
        """Test that __init__ detects google.generativeai availability."""
        # Only patch google.generativeai, not the google namespace package
        with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
            instrumentor = GoogleAIInstrumentor()
            self.assertTrue(instrumentor._google_available)

    def test_init_with_google_not_available(self):
        """Test that __init__ handles missing google.generativeai gracefully."""
        # Remove google.generativeai from sys.modules if it exists
        with patch.dict("sys.modules", {"google.generativeai": None}):
            instrumentor = GoogleAIInstrumentor()
            self.assertFalse(instrumentor._google_available)

    def test_instrument_with_google_not_available(self):
        """Test that instrument skips when google.generativeai is not available."""
        with patch.dict("sys.modules", {"google": None, "google.generativeai": None}):
            instrumentor = GoogleAIInstrumentor()
            config = OTelConfig()

            # Should not raise
            instrumentor.instrument(config)

    def test_instrument_with_google_available(self):
        """Test that instrument wraps GenerativeModel when available."""

        # Create mock GenerativeModel
        class MockGenerativeModel:
            def generate_content(self, *args, **kwargs):
                return "result"

        # Create mock google.generativeai module
        mock_genai = MagicMock()
        mock_genai.GenerativeModel = MockGenerativeModel

        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
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

    def test_instrument_with_missing_generate_content(self):
        """Test that instrument handles missing generate_content method."""
        # Create mock GenerativeModel without generate_content attribute
        mock_genai = MagicMock()
        mock_genai.GenerativeModel = MagicMock(spec=[])  # Empty spec means no attributes

        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
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

        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
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

        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            instrumentor = GoogleAIInstrumentor()
            config = OTelConfig(fail_on_error=True)

            # Mock create_span_wrapper to raise exception
            instrumentor.create_span_wrapper = MagicMock(
                side_effect=RuntimeError("Wrapper creation failed")
            )

            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Wrapper creation failed")

    def test_extract_google_ai_attributes(self):
        """Test that _extract_google_ai_attributes extracts correct attributes."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock instance with model_name
        mock_instance = MagicMock()
        mock_instance.model_name = "gemini-pro"

        attrs = instrumentor._extract_google_ai_attributes(mock_instance, None, None)

        self.assertEqual(attrs["gen_ai.system"], "google")
        self.assertEqual(attrs["gen_ai.request.model"], "gemini-pro")

    def test_extract_google_ai_attributes_with_unknown_model(self):
        """Test that _extract_google_ai_attributes uses 'unknown' as default."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock instance without model_name
        mock_instance = MagicMock(spec=[])
        if hasattr(mock_instance, "model_name"):
            delattr(mock_instance, "model_name")

        attrs = instrumentor._extract_google_ai_attributes(mock_instance, None, None)

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

    def test_extract_usage_without_usage_metadata(self):
        """Test that _extract_usage returns None when no usage_metadata field."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result without usage_metadata
        result = MagicMock(spec=[])
        if hasattr(result, "usage_metadata"):
            delattr(result, "usage_metadata")

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)

    def test_extract_usage_with_none_usage_metadata(self):
        """Test that _extract_usage returns None when usage_metadata is None."""
        instrumentor = GoogleAIInstrumentor()

        # Create mock result with None usage_metadata
        result = MagicMock()
        result.usage_metadata = None

        usage = instrumentor._extract_usage(result)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
