import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor


class TestOpenAIInstrumentor(unittest.TestCase):
    """Tests for OpenAIInstrumentor"""

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
    def test_init_with_openai_available(self, mock_logger):
        """Test that __init__ detects OpenAI availability."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            self.assertTrue(instrumentor._openai_available)
            mock_logger.debug.assert_called_with(
                "OpenAI library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
    def test_init_with_openai_not_available(self, mock_logger):
        """Test that __init__ handles missing OpenAI gracefully."""
        with patch.dict("sys.modules", {"openai": None}):
            instrumentor = OpenAIInstrumentor()

            self.assertFalse(instrumentor._openai_available)
            mock_logger.debug.assert_called_with(
                "OpenAI library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
    def test_instrument_when_openai_not_available(self, mock_logger):
        """Test that instrument skips when OpenAI is not available."""
        with patch.dict("sys.modules", {"openai": None}):
            instrumentor = OpenAIInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping OpenAI instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
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
            instrumentor = OpenAIInstrumentor()
            config = MagicMock()

            # Mock _instrument_client to avoid complex setup
            mock_instrument_client = MagicMock()
            instrumentor._instrument_client = mock_instrument_client

            # Act
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("OpenAI instrumentation enabled")
            # Verify FunctionWrapper was called to wrap __init__
            mock_wrapt.FunctionWrapper.assert_called_once()

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
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
                instrumentor = OpenAIInstrumentor()
                config = MagicMock()
                config.fail_on_error = False

                # Should not raise exception
                instrumentor.instrument(config)

                mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
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
                instrumentor = OpenAIInstrumentor()
                config = MagicMock()
                config.fail_on_error = True

                # Should raise exception
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_instrument_client(self):
        """Test that _instrument_client wraps the chat.completions.create method."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock client with chat.completions.create
            mock_client = MagicMock()
            original_create = MagicMock()
            mock_client.chat.completions.create = original_create

            # Create mock wrapper
            mock_wrapper = MagicMock()
            instrumentor.create_span_wrapper = MagicMock(return_value=mock_wrapper)

            # Act
            instrumentor._instrument_client(mock_client)

            # Assert that create_span_wrapper was called with correct arguments
            instrumentor.create_span_wrapper.assert_called_once_with(
                span_name="openai.chat.completion",
                extract_attributes=instrumentor._extract_openai_attributes,
            )

            # Assert that the create method was replaced
            self.assertEqual(mock_client.chat.completions.create, mock_wrapper)

    def test_extract_openai_attributes_with_messages(self):
        """Test that _extract_openai_attributes extracts attributes correctly."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            kwargs = {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ],
            }

            attrs = instrumentor._extract_openai_attributes(None, [], kwargs)

            self.assertEqual(attrs["gen_ai.system"], "openai")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-4")
            self.assertEqual(attrs["gen_ai.request.message_count"], 2)
            self.assertIn("gen_ai.request.first_message", attrs)
            # Check that first_message is truncated to 200 chars
            self.assertLessEqual(len(attrs["gen_ai.request.first_message"]), 200)

    def test_extract_openai_attributes_without_messages(self):
        """Test that _extract_openai_attributes handles missing messages."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            kwargs = {"model": "gpt-4"}

            attrs = instrumentor._extract_openai_attributes(None, [], kwargs)

            self.assertEqual(attrs["gen_ai.system"], "openai")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-4")
            self.assertEqual(attrs["gen_ai.request.message_count"], 0)
            self.assertNotIn("gen_ai.request.first_message", attrs)

    def test_extract_openai_attributes_with_long_message(self):
        """Test that first message is truncated to 200 chars."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            long_content = "x" * 300
            kwargs = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": long_content}],
            }

            attrs = instrumentor._extract_openai_attributes(None, [], kwargs)

            self.assertIn("gen_ai.request.first_message", attrs)
            self.assertLessEqual(len(attrs["gen_ai.request.first_message"]), 200)

    def test_extract_usage_with_usage_object(self):
        """Test that _extract_usage extracts token counts from response."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock result with usage
            result = MagicMock()
            result.usage = MagicMock()
            result.usage.prompt_tokens = 10
            result.usage.completion_tokens = 20
            result.usage.total_tokens = 30

            usage = instrumentor._extract_usage(result)

            self.assertEqual(usage["prompt_tokens"], 10)
            self.assertEqual(usage["completion_tokens"], 20)
            self.assertEqual(usage["total_tokens"], 30)

    def test_extract_usage_without_usage_object(self):
        """Test that _extract_usage returns None when usage is missing."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock result without usage
            result = MagicMock()
            result.usage = None

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    def test_extract_usage_without_usage_attribute(self):
        """Test that _extract_usage returns None when result has no usage attribute."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock result without usage attribute
            result = MagicMock(spec=[])  # spec=[] means no attributes

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
    def test_wrapped_init_calls_instrument_client(self, mock_logger):
        """Test that the wrapped __init__ calls _instrument_client on the instance."""

        # Create a real class (not a MagicMock) so we can set __init__
        class MockOpenAI:
            def __init__(self):
                pass

        # Create mock OpenAI module
        mock_openai = MagicMock()
        mock_openai.OpenAI = MockOpenAI

        # Create a mock wrapt module that actually executes wrapped functions
        import wrapt as real_wrapt

        with patch.dict("sys.modules", {"openai": mock_openai, "wrapt": real_wrapt}):
            instrumentor = OpenAIInstrumentor()
            config = MagicMock()

            # Mock _instrument_client
            mock_instrument_client = MagicMock()
            instrumentor._instrument_client = mock_instrument_client

            # Act - instrument the class
            instrumentor.instrument(config)

            # Now create an instance - this should call the wrapped __init__
            instance = mock_openai.OpenAI()

            # Verify _instrument_client was called with the instance
            mock_instrument_client.assert_called_once_with(instance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
