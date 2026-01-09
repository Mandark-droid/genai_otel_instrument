import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.openrouter_instrumentor import OpenRouterInstrumentor


class TestOpenRouterInstrumentor(unittest.TestCase):
    """Tests for OpenRouterInstrumentor"""

    @patch("genai_otel.instrumentors.openrouter_instrumentor.logger")
    def test_init_with_openai_available(self, mock_logger):
        """Test that __init__ detects OpenAI library availability (used by OpenRouter)."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            self.assertTrue(instrumentor._openrouter_available)
            mock_logger.debug.assert_called_with(
                "OpenAI library detected, OpenRouter instrumentation available"
            )

    @patch("genai_otel.instrumentors.openrouter_instrumentor.logger")
    def test_init_with_openai_not_available(self, mock_logger):
        """Test that __init__ handles missing OpenAI library gracefully."""
        with patch.dict("sys.modules", {"openai": None}):
            instrumentor = OpenRouterInstrumentor()

            self.assertFalse(instrumentor._openrouter_available)
            mock_logger.debug.assert_called_with(
                "OpenAI library not installed, OpenRouter instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.openrouter_instrumentor.logger")
    def test_instrument_when_openai_not_available(self, mock_logger):
        """Test that instrument skips when OpenAI library is not available."""
        with patch.dict("sys.modules", {"openai": None}):
            instrumentor = OpenRouterInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping OpenRouter instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.openrouter_instrumentor.logger")
    def test_instrument_with_openai_available(self, mock_logger):
        """Test that instrument wraps OpenAI client when available."""

        # Create a real class (not a MagicMock) so we can set __init__
        class MockOpenAI:
            def __init__(self):
                pass

        # Create mock OpenAI module
        mock_openai = MagicMock()
        mock_openai.OpenAI = MockOpenAI

        # Create a mock wrapt module
        mock_wrapt = MagicMock()

        with patch.dict("sys.modules", {"openai": mock_openai, "wrapt": mock_wrapt}):
            instrumentor = OpenRouterInstrumentor()
            config = MagicMock()

            # Mock _instrument_client and _is_openrouter_client to avoid complex setup
            mock_instrument_client = MagicMock()
            instrumentor._instrument_client = mock_instrument_client

            # Act
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("OpenRouter instrumentation enabled")
            # Verify FunctionWrapper was called to wrap __init__
            mock_wrapt.FunctionWrapper.assert_called_once()

    @patch("genai_otel.instrumentors.openrouter_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        # Create mock OpenAI module
        mock_openai = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "OpenAI":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"openai": mock_openai, "wrapt": MagicMock()}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = OpenRouterInstrumentor()
                config = MagicMock()
                config.fail_on_error = False

                # Should not raise exception
                instrumentor.instrument(config)

                mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.openrouter_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that instrument raises exceptions when fail_on_error is True."""
        # Create mock OpenAI module
        mock_openai = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "OpenAI":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"openai": mock_openai, "wrapt": MagicMock()}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = OpenRouterInstrumentor()
                config = MagicMock()
                config.fail_on_error = True

                # Should raise exception
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_is_openrouter_client_with_openrouter_base_url(self):
        """Test that _is_openrouter_client detects OpenRouter clients correctly."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock client with OpenRouter base_url
            mock_client = MagicMock()
            mock_client.base_url = "https://openrouter.ai/api/v1"

            result = instrumentor._is_openrouter_client(mock_client)

            self.assertTrue(result)

    def test_is_openrouter_client_with_non_openrouter_base_url(self):
        """Test that _is_openrouter_client returns False for non-OpenRouter clients."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock client with non-OpenRouter base_url
            mock_client = MagicMock()
            mock_client.base_url = "https://api.openai.com/v1"

            result = instrumentor._is_openrouter_client(mock_client)

            self.assertFalse(result)

    def test_is_openrouter_client_without_base_url(self):
        """Test that _is_openrouter_client handles missing base_url."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock client without base_url
            mock_client = MagicMock()
            del mock_client.base_url

            result = instrumentor._is_openrouter_client(mock_client)

            self.assertFalse(result)

    def test_instrument_client(self):
        """Test that _instrument_client wraps the chat.completions.create method."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock client with chat.completions.create
            mock_client = MagicMock()
            original_create = MagicMock()
            mock_client.chat.completions.create = original_create

            # Create mock wrapper
            mock_wrapper = MagicMock()
            # create_span_wrapper returns a decorator, so we need to return a callable
            # that when called with original_create returns mock_wrapper
            mock_decorator = MagicMock(return_value=mock_wrapper)
            instrumentor.create_span_wrapper = MagicMock(return_value=mock_decorator)

            # Act
            instrumentor._instrument_client(mock_client)

            # Assert that create_span_wrapper was called with correct arguments
            instrumentor.create_span_wrapper.assert_called_once_with(
                span_name="openrouter.chat.completion",
                extract_attributes=instrumentor._extract_openrouter_attributes,
            )

            # Assert that the decorator was called with original_create
            mock_decorator.assert_called_once_with(original_create)

            # Assert that the create method was replaced with mock_wrapper
            self.assertEqual(mock_client.chat.completions.create, mock_wrapper)

    def test_extract_openrouter_attributes_with_messages(self):
        """Test that _extract_openrouter_attributes extracts attributes correctly."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            kwargs = {
                "model": "anthropic/claude-3-opus",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ],
            }

            attrs = instrumentor._extract_openrouter_attributes(None, [], kwargs)

            self.assertEqual(attrs["gen_ai.system"], "openrouter")
            self.assertEqual(attrs["gen_ai.request.model"], "anthropic/claude-3-opus")
            self.assertEqual(attrs["gen_ai.request.message_count"], 2)
            self.assertIn("gen_ai.request.first_message", attrs)
            # Check that first_message is truncated to 200 chars
            self.assertLessEqual(len(attrs["gen_ai.request.first_message"]), 200)

    def test_extract_openrouter_attributes_without_messages(self):
        """Test that _extract_openrouter_attributes handles missing messages."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            kwargs = {"model": "openai/gpt-4"}

            attrs = instrumentor._extract_openrouter_attributes(None, [], kwargs)

            self.assertEqual(attrs["gen_ai.system"], "openrouter")
            self.assertEqual(attrs["gen_ai.request.model"], "openai/gpt-4")
            self.assertEqual(attrs["gen_ai.request.message_count"], 0)
            self.assertNotIn("gen_ai.request.first_message", attrs)

    def test_extract_openrouter_attributes_with_openrouter_params(self):
        """Test that OpenRouter-specific parameters are captured."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            kwargs = {
                "model": "anthropic/claude-3-opus",
                "messages": [{"role": "user", "content": "Hello"}],
                "provider": {"order": ["Anthropic"]},
                "route": "fallback",
            }

            attrs = instrumentor._extract_openrouter_attributes(None, [], kwargs)

            self.assertIn("openrouter.provider", attrs)
            self.assertIn("openrouter.route", attrs)

    def test_extract_usage_with_valid_response(self):
        """Test that _extract_usage extracts token usage correctly."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock response object
            mock_result = MagicMock()
            mock_result.usage.prompt_tokens = 100
            mock_result.usage.completion_tokens = 50
            mock_result.usage.total_tokens = 150

            usage = instrumentor._extract_usage(mock_result)

            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 100)
            self.assertEqual(usage["completion_tokens"], 50)
            self.assertEqual(usage["total_tokens"], 150)

    def test_extract_usage_without_usage(self):
        """Test that _extract_usage handles missing usage information."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock response without usage
            mock_result = MagicMock()
            del mock_result.usage

            usage = instrumentor._extract_usage(mock_result)

            self.assertIsNone(usage)

    def test_extract_response_attributes_with_finish_reasons(self):
        """Test that _extract_response_attributes extracts finish reasons."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock response
            mock_result = MagicMock()
            mock_result.id = "response-123"
            mock_result.model = "anthropic/claude-3-opus"

            mock_choice1 = MagicMock()
            mock_choice1.finish_reason = "stop"
            mock_choice2 = MagicMock()
            mock_choice2.finish_reason = "length"

            mock_result.choices = [mock_choice1, mock_choice2]

            attrs = instrumentor._extract_response_attributes(mock_result)

            self.assertEqual(attrs["gen_ai.response.id"], "response-123")
            self.assertEqual(attrs["gen_ai.response.model"], "anthropic/claude-3-opus")
            self.assertEqual(attrs["gen_ai.response.finish_reasons"], ["stop", "length"])

    def test_extract_finish_reason(self):
        """Test that _extract_finish_reason extracts the first finish reason."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock response
            mock_result = MagicMock()
            mock_choice = MagicMock()
            mock_choice.finish_reason = "stop"
            mock_result.choices = [mock_choice]

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            self.assertEqual(finish_reason, "stop")

    def test_extract_finish_reason_without_choices(self):
        """Test that _extract_finish_reason handles missing choices."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenRouterInstrumentor()

            # Create mock response without choices
            mock_result = MagicMock()
            mock_result.choices = []

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            self.assertIsNone(finish_reason)


if __name__ == "__main__":
    unittest.main()
