import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor


class TestAnthropicInstrumentor(unittest.TestCase):
    """Tests for AnthropicInstrumentor"""

    @patch("genai_otel.instrumentors.anthropic_instrumentor.logger")
    def test_init_with_anthropic_available(self, mock_logger):
        """Test that __init__ detects anthropic availability."""
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            instrumentor = AnthropicInstrumentor()

            self.assertTrue(instrumentor._anthropic_available)
            mock_logger.debug.assert_called_with(
                "Anthropic library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.anthropic_instrumentor.logger")
    def test_init_with_anthropic_not_available(self, mock_logger):
        """Test that __init__ handles missing anthropic gracefully."""
        with patch.dict("sys.modules", {"anthropic": None}):
            instrumentor = AnthropicInstrumentor()

            self.assertFalse(instrumentor._anthropic_available)
            mock_logger.debug.assert_called_with(
                "Anthropic library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.anthropic_instrumentor.logger")
    def test_instrument_with_anthropic_not_available(self, mock_logger):
        """Test that instrument skips when anthropic is not available."""
        with patch.dict("sys.modules", {"anthropic": None}):
            instrumentor = AnthropicInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Anthropic instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.anthropic_instrumentor.logger")
    def test_instrument_with_anthropic_available(self, mock_logger):
        """Test that instrument wraps anthropic client when available."""

        # Create a real class to mock Anthropic
        class MockAnthropicClass:
            def __init__(self, *args, **kwargs):
                self.messages = MagicMock()
                self.messages.create = MagicMock()

        # Create mock anthropic module
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MockAnthropicClass
        mock_wrapt = MagicMock()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic, "wrapt": mock_wrapt}):
            instrumentor = AnthropicInstrumentor()
            config = OTelConfig()

            # Store original __init__
            original_init = MockAnthropicClass.__init__

            # Call instrument
            instrumentor.instrument(config)

            # Verify that instrumentation was set up
            self.assertTrue(instrumentor._instrumented)
            self.assertEqual(instrumentor.config, config)
            mock_logger.info.assert_called_with("Anthropic instrumentation enabled")

    @patch("genai_otel.instrumentors.anthropic_instrumentor.logger")
    def test_instrument_with_exception_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        # Create a mock that raises when accessed
        mock_anthropic = MagicMock()
        # Make accessing Anthropic raise an exception
        type(mock_anthropic).Anthropic = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            instrumentor = AnthropicInstrumentor()
            config = OTelConfig(fail_on_error=False)

            # Should not raise
            instrumentor.instrument(config)

            # Verify error was logged
            mock_logger.error.assert_called()

    def test_instrument_with_exception_fail_on_error_true(self):
        """Test that exceptions are raised when fail_on_error is True."""
        # Create a mock that raises when accessed
        mock_anthropic = MagicMock()
        # Make accessing Anthropic raise an exception
        type(mock_anthropic).Anthropic = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            instrumentor = AnthropicInstrumentor()
            config = OTelConfig(fail_on_error=True)

            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Access failed")

    def test_instrument_client_with_messages(self):
        """Test that _instrument_client wraps messages.create."""
        instrumentor = AnthropicInstrumentor()

        # Create mock client
        mock_client = MagicMock()
        original_create = MagicMock(return_value="result")
        mock_client.messages.create = original_create

        # Mock create_span_wrapper
        mock_wrapper = MagicMock()
        # create_span_wrapper returns a decorator, so we need to return a callable
        # that when called with original_create returns mock_wrapper
        mock_decorator = MagicMock(return_value=mock_wrapper)
        instrumentor.create_span_wrapper = MagicMock(return_value=mock_decorator)

        # Call _instrument_client
        instrumentor._instrument_client(mock_client)

        # Verify create_span_wrapper was called
        instrumentor.create_span_wrapper.assert_called_once_with(
            span_name="anthropic.messages.create",
            extract_attributes=instrumentor._extract_anthropic_attributes,
        )

        # Verify the decorator was called with original_create
        mock_decorator.assert_called_once_with(original_create)

        # Verify messages.create was replaced with mock_wrapper
        self.assertEqual(mock_client.messages.create, mock_wrapper)

    def test_instrument_client_without_messages(self):
        """Test that _instrument_client handles clients without messages."""
        instrumentor = AnthropicInstrumentor()

        # Create mock client without messages
        mock_client = MagicMock(spec=[])
        if hasattr(mock_client, "messages"):
            delattr(mock_client, "messages")

        # Mock create_span_wrapper
        instrumentor.create_span_wrapper = MagicMock()

        # Call _instrument_client - should not raise
        instrumentor._instrument_client(mock_client)

        # Verify create_span_wrapper was NOT called
        instrumentor.create_span_wrapper.assert_not_called()

    def test_instrument_client_without_create_method(self):
        """Test that _instrument_client handles messages without create method."""
        instrumentor = AnthropicInstrumentor()

        # Create mock client with messages but no create
        mock_client = MagicMock()
        mock_client.messages = MagicMock(spec=[])
        if hasattr(mock_client.messages, "create"):
            delattr(mock_client.messages, "create")

        # Mock create_span_wrapper
        instrumentor.create_span_wrapper = MagicMock()

        # Call _instrument_client - should not raise
        instrumentor._instrument_client(mock_client)

        # Verify create_span_wrapper was NOT called
        instrumentor.create_span_wrapper.assert_not_called()

    def test_extract_anthropic_attributes(self):
        """Test that _extract_anthropic_attributes extracts correct attributes."""
        instrumentor = AnthropicInstrumentor()

        kwargs = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        }

        attrs = instrumentor._extract_anthropic_attributes(None, None, kwargs)

        self.assertEqual(attrs["gen_ai.system"], "anthropic")
        self.assertEqual(attrs["gen_ai.request.model"], "claude-3-opus-20240229")
        self.assertEqual(attrs["gen_ai.request.message_count"], 3)

    def test_extract_anthropic_attributes_with_unknown_model(self):
        """Test that _extract_anthropic_attributes uses 'unknown' as default."""
        instrumentor = AnthropicInstrumentor()

        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

        attrs = instrumentor._extract_anthropic_attributes(None, None, kwargs)

        self.assertEqual(attrs["gen_ai.system"], "anthropic")
        self.assertEqual(attrs["gen_ai.request.model"], "unknown")
        self.assertEqual(attrs["gen_ai.request.message_count"], 1)

    def test_extract_anthropic_attributes_with_no_messages(self):
        """Test that _extract_anthropic_attributes handles missing messages."""
        instrumentor = AnthropicInstrumentor()

        kwargs = {"model": "claude-3-sonnet-20240229"}

        attrs = instrumentor._extract_anthropic_attributes(None, None, kwargs)

        self.assertEqual(attrs["gen_ai.system"], "anthropic")
        self.assertEqual(attrs["gen_ai.request.model"], "claude-3-sonnet-20240229")
        self.assertEqual(attrs["gen_ai.request.message_count"], 0)

    def test_extract_usage_with_usage_field(self):
        """Test that _extract_usage extracts from usage field."""
        instrumentor = AnthropicInstrumentor()

        # Create mock result with usage
        result = MagicMock()
        result.usage.input_tokens = 15
        result.usage.output_tokens = 25

        usage = instrumentor._extract_usage(result)

        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 15)
        self.assertEqual(usage["completion_tokens"], 25)
        self.assertEqual(usage["total_tokens"], 40)

    def test_extract_usage_without_usage_field(self):
        """Test that _extract_usage returns None when no usage field."""
        instrumentor = AnthropicInstrumentor()

        # Create mock result without usage
        result = MagicMock(spec=[])
        if hasattr(result, "usage"):
            delattr(result, "usage")

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)

    def test_extract_usage_with_none_usage(self):
        """Test that _extract_usage returns None when usage is None."""
        instrumentor = AnthropicInstrumentor()

        # Create mock result with None usage
        result = MagicMock()
        result.usage = None

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)

    def test_extract_usage_with_missing_tokens(self):
        """Test that _extract_usage handles missing token attributes."""
        instrumentor = AnthropicInstrumentor()

        # Create mock result with usage but missing token attributes
        result = MagicMock()
        result.usage = MagicMock(spec=[])
        if hasattr(result.usage, "input_tokens"):
            delattr(result.usage, "input_tokens")
        if hasattr(result.usage, "output_tokens"):
            delattr(result.usage, "output_tokens")

        usage = instrumentor._extract_usage(result)

        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 0)
        self.assertEqual(usage["completion_tokens"], 0)
        self.assertEqual(usage["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
