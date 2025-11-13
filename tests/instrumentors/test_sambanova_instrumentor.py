import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.sambanova_instrumentor import SambaNovaInstrumentor


class TestSambaNovaInstrumentor(unittest.TestCase):
    """Tests for SambaNovaInstrumentor"""

    @patch("genai_otel.instrumentors.sambanova_instrumentor.logger")
    def test_init_with_sambanova_available(self, mock_logger):
        """Test that __init__ detects sambanova availability."""
        with patch.dict("sys.modules", {"sambanova": MagicMock()}):
            instrumentor = SambaNovaInstrumentor()

            self.assertTrue(instrumentor._sambanova_available)
            mock_logger.debug.assert_called_with(
                "SambaNova library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.sambanova_instrumentor.logger")
    def test_init_with_sambanova_not_available(self, mock_logger):
        """Test that __init__ handles missing sambanova gracefully."""
        with patch.dict("sys.modules", {"sambanova": None}):
            instrumentor = SambaNovaInstrumentor()

            self.assertFalse(instrumentor._sambanova_available)
            mock_logger.debug.assert_called_with(
                "SambaNova library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.sambanova_instrumentor.logger")
    def test_instrument_with_sambanova_not_available(self, mock_logger):
        """Test that instrument skips when sambanova is not available."""
        with patch.dict("sys.modules", {"sambanova": None}):
            instrumentor = SambaNovaInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping SambaNova instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.sambanova_instrumentor.logger")
    def test_instrument_with_sambanova_available(self, mock_logger):
        """Test that instrument wraps sambanova client when available."""

        # Create a real class to mock SambaNova
        class MockSambaNovaClass:
            def __init__(self, *args, **kwargs):
                self.chat = MagicMock()
                self.chat.completions = MagicMock()
                self.chat.completions.create = MagicMock()

        # Create mock sambanova module
        mock_sambanova = MagicMock()
        mock_sambanova.SambaNova = MockSambaNovaClass

        with patch.dict("sys.modules", {"sambanova": mock_sambanova}):
            instrumentor = SambaNovaInstrumentor()
            config = OTelConfig()

            # Store original __init__
            original_init = MockSambaNovaClass.__init__

            # Call instrument
            instrumentor.instrument(config)

            # Verify that SambaNova.__init__ was replaced
            self.assertNotEqual(mock_sambanova.SambaNova.__init__, original_init)
            self.assertTrue(instrumentor._instrumented)
            self.assertEqual(instrumentor.config, config)
            mock_logger.info.assert_called_with("SambaNova instrumentation enabled")

    @patch("genai_otel.instrumentors.sambanova_instrumentor.logger")
    def test_instrument_with_exception_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        # Create a mock that raises when __init__ is accessed
        mock_sambanova = MagicMock()
        # Make SambaNova raise an exception when trying to access __init__
        type(mock_sambanova.SambaNova).__init__ = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"sambanova": mock_sambanova}):
            instrumentor = SambaNovaInstrumentor()
            config = OTelConfig(fail_on_error=False)

            # Should not raise
            instrumentor.instrument(config)

            # Verify error was logged
            mock_logger.error.assert_called()

    def test_instrument_with_exception_fail_on_error_true(self):
        """Test that exceptions are raised when fail_on_error is True."""
        # Create a mock that raises when __init__ is accessed
        mock_sambanova = MagicMock()
        # Make SambaNova raise an exception when trying to access __init__
        type(mock_sambanova.SambaNova).__init__ = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"sambanova": mock_sambanova}):
            instrumentor = SambaNovaInstrumentor()
            config = OTelConfig(fail_on_error=True)

            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Access failed")

    def test_extract_usage_with_usage_field(self):
        """Test that _extract_usage extracts from usage field."""
        instrumentor = SambaNovaInstrumentor()

        # Create mock result with usage
        result = MagicMock()
        result.usage.prompt_tokens = 100
        result.usage.completion_tokens = 200
        result.usage.total_tokens = 300

        usage = instrumentor._extract_usage(result)

        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 200)
        self.assertEqual(usage["total_tokens"], 300)

    def test_extract_usage_without_usage_field(self):
        """Test that _extract_usage returns None when no usage field."""
        instrumentor = SambaNovaInstrumentor()

        # Create mock result without usage
        result = MagicMock(spec=[])
        if hasattr(result, "usage"):
            delattr(result, "usage")

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)

    def test_extract_sambanova_attributes(self):
        """Test extraction of SambaNova-specific attributes."""
        instrumentor = SambaNovaInstrumentor()

        kwargs = {
            "model": "Llama-4-Maverick-17B-128E-Instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 500,
        }

        attrs = instrumentor._extract_sambanova_attributes(None, None, kwargs)

        self.assertEqual(attrs["gen_ai.system"], "sambanova")
        self.assertEqual(attrs["gen_ai.request.model"], "Llama-4-Maverick-17B-128E-Instruct")
        self.assertEqual(attrs["gen_ai.operation.name"], "chat")
        self.assertEqual(attrs["gen_ai.request.message_count"], 1)
        self.assertEqual(attrs["gen_ai.request.temperature"], 0.7)
        self.assertEqual(attrs["gen_ai.request.top_p"], 0.9)
        self.assertEqual(attrs["gen_ai.request.max_tokens"], 500)

    def test_extract_response_attributes(self):
        """Test extraction of response attributes."""
        instrumentor = SambaNovaInstrumentor()

        # Create mock result
        result = MagicMock()
        result.id = "resp-123"
        result.model = "Llama-4-Maverick-17B-128E-Instruct"
        result.choices = [MagicMock(finish_reason="stop")]

        attrs = instrumentor._extract_response_attributes(result)

        self.assertEqual(attrs["gen_ai.response.id"], "resp-123")
        self.assertEqual(attrs["gen_ai.response.model"], "Llama-4-Maverick-17B-128E-Instruct")
        self.assertEqual(attrs["gen_ai.response.finish_reasons"], ["stop"])

    def test_extract_finish_reason(self):
        """Test extraction of finish reason."""
        instrumentor = SambaNovaInstrumentor()

        result = MagicMock()
        result.choices = [MagicMock(finish_reason="stop")]

        finish_reason = instrumentor._extract_finish_reason(result)

        self.assertEqual(finish_reason, "stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
