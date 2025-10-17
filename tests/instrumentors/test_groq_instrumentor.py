import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.groq_instrumentor import GroqInstrumentor


class TestGroqInstrumentor(unittest.TestCase):
    """Tests for GroqInstrumentor"""

    @patch("genai_otel.instrumentors.groq_instrumentor.logger")
    def test_init_with_groq_available(self, mock_logger):
        """Test that __init__ detects groq availability."""
        with patch.dict("sys.modules", {"groq": MagicMock()}):
            instrumentor = GroqInstrumentor()

            self.assertTrue(instrumentor._groq_available)
            mock_logger.debug.assert_called_with(
                "Groq library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.groq_instrumentor.logger")
    def test_init_with_groq_not_available(self, mock_logger):
        """Test that __init__ handles missing groq gracefully."""
        with patch.dict("sys.modules", {"groq": None}):
            instrumentor = GroqInstrumentor()

            self.assertFalse(instrumentor._groq_available)
            mock_logger.debug.assert_called_with(
                "Groq library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.groq_instrumentor.logger")
    def test_instrument_with_groq_not_available(self, mock_logger):
        """Test that instrument skips when groq is not available."""
        with patch.dict("sys.modules", {"groq": None}):
            instrumentor = GroqInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Groq instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.groq_instrumentor.logger")
    def test_instrument_with_groq_available(self, mock_logger):
        """Test that instrument wraps groq client when available."""

        # Create a real class to mock Groq
        class MockGroqClass:
            def __init__(self, *args, **kwargs):
                self.chat = MagicMock()
                self.chat.completions = MagicMock()
                self.chat.completions.create = MagicMock()

        # Create mock groq module
        mock_groq = MagicMock()
        mock_groq.Groq = MockGroqClass

        with patch.dict("sys.modules", {"groq": mock_groq}):
            instrumentor = GroqInstrumentor()
            config = OTelConfig()

            # Store original __init__
            original_init = MockGroqClass.__init__

            # Call instrument
            instrumentor.instrument(config)

            # Verify that Groq.__init__ was replaced
            self.assertNotEqual(mock_groq.Groq.__init__, original_init)
            self.assertTrue(instrumentor._instrumented)
            self.assertEqual(instrumentor.config, config)
            mock_logger.info.assert_called_with("Groq instrumentation enabled")

    @patch("genai_otel.instrumentors.groq_instrumentor.logger")
    def test_instrument_with_exception_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        # Create a mock that raises when __init__ is accessed
        mock_groq = MagicMock()
        # Make Groq raise an exception when trying to access __init__
        type(mock_groq.Groq).__init__ = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"groq": mock_groq}):
            instrumentor = GroqInstrumentor()
            config = OTelConfig(fail_on_error=False)

            # Should not raise
            instrumentor.instrument(config)

            # Verify error was logged
            mock_logger.error.assert_called()

    def test_instrument_with_exception_fail_on_error_true(self):
        """Test that exceptions are raised when fail_on_error is True."""
        # Create a mock that raises when __init__ is accessed
        mock_groq = MagicMock()
        # Make Groq raise an exception when trying to access __init__
        type(mock_groq.Groq).__init__ = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"groq": mock_groq}):
            instrumentor = GroqInstrumentor()
            config = OTelConfig(fail_on_error=True)

            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Access failed")

    def test_instrument_client(self):
        """Test that _instrument_client wraps chat completions create."""
        instrumentor = GroqInstrumentor()

        # Create mock client
        mock_client = MagicMock()
        original_create = MagicMock(return_value="result")
        mock_client.chat.completions.create = original_create

        # Mock create_span_wrapper
        instrumentor.tracer = MagicMock()
        instrumentor.request_counter = MagicMock()

        # Call _instrument_client
        instrumentor._instrument_client(mock_client)

        # Verify create was replaced
        self.assertNotEqual(mock_client.chat.completions.create, original_create)

    def test_wrapped_create_execution(self):
        """Test that wrapped create method executes correctly."""
        instrumentor = GroqInstrumentor()

        # Create mock client
        mock_client = MagicMock()
        original_create = MagicMock(
            return_value=MagicMock(
                usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            )
        )
        mock_client.chat.completions.create = original_create

        # Mock tracer and metrics
        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_result_metrics = MagicMock()

        # Instrument the client
        instrumentor._instrument_client(mock_client)

        # Call the wrapped method
        result = mock_client.chat.completions.create(model="llama-3.1-70b")

        # Verify tracer was called
        instrumentor.tracer.start_as_current_span.assert_called_once_with("groq.chat.completions")

        # Verify span attributes were set
        mock_span.set_attribute.assert_any_call("gen_ai.system", "groq")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "llama-3.1-70b")

        # Verify metrics were recorded
        instrumentor.request_counter.add.assert_called_once_with(
            1, {"model": "llama-3.1-70b", "provider": "groq"}
        )

        # Verify original create was called
        original_create.assert_called_once_with(model="llama-3.1-70b")

        # Verify result metrics were recorded
        instrumentor._record_result_metrics.assert_called_once()

    def test_wrapped_create_with_unknown_model(self):
        """Test that wrapped create handles missing model parameter."""
        instrumentor = GroqInstrumentor()

        # Create mock client
        mock_client = MagicMock()
        original_create = MagicMock(return_value=MagicMock())
        mock_client.chat.completions.create = original_create

        # Mock tracer and metrics
        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()
        instrumentor._record_result_metrics = MagicMock()

        # Instrument the client
        instrumentor._instrument_client(mock_client)

        # Call the wrapped method without model parameter
        result = mock_client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

        # Verify span attributes were set with "unknown" model
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "unknown")

    def test_extract_usage_with_usage_field(self):
        """Test that _extract_usage extracts from usage field."""
        instrumentor = GroqInstrumentor()

        # Create mock result with usage
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
        instrumentor = GroqInstrumentor()

        # Create mock result without usage
        result = MagicMock(spec=[])
        if hasattr(result, "usage"):
            delattr(result, "usage")

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
