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
            # create_span_wrapper returns a decorator, so we need to return a callable
            # that when called with original_create returns mock_wrapper
            mock_decorator = MagicMock(return_value=mock_wrapper)
            instrumentor.create_span_wrapper = MagicMock(return_value=mock_decorator)

            # Act
            instrumentor._instrument_client(mock_client)

            # Assert that create_span_wrapper was called with correct arguments
            instrumentor.create_span_wrapper.assert_called_once_with(
                span_name="openai.chat.completion",
                extract_attributes=instrumentor._extract_openai_attributes,
            )

            # Assert that the decorator was called with original_create
            mock_decorator.assert_called_once_with(original_create)

            # Assert that the create method was replaced with mock_wrapper
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

    def test_extract_openai_attributes_with_request_parameters(self):
        """Test that _extract_openai_attributes extracts request parameters."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            kwargs = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
            }

            attrs = instrumentor._extract_openai_attributes(None, [], kwargs)

            self.assertEqual(attrs["gen_ai.system"], "openai")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-4")
            self.assertEqual(attrs["gen_ai.operation.name"], "chat")
            self.assertEqual(attrs["gen_ai.request.temperature"], 0.7)
            self.assertEqual(attrs["gen_ai.request.top_p"], 0.9)
            self.assertEqual(attrs["gen_ai.request.max_tokens"], 100)
            self.assertEqual(attrs["gen_ai.request.frequency_penalty"], 0.5)
            self.assertEqual(attrs["gen_ai.request.presence_penalty"], 0.3)

    def test_extract_response_attributes_complete(self):
        """Test that _extract_response_attributes extracts all response attributes."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock result with all response attributes
            result = MagicMock()
            result.id = "chatcmpl-123"
            result.model = "gpt-4-0613"
            result.choices = [
                MagicMock(finish_reason="stop"),
                MagicMock(finish_reason="length"),
            ]

            attrs = instrumentor._extract_response_attributes(result)

            self.assertEqual(attrs["gen_ai.response.id"], "chatcmpl-123")
            self.assertEqual(attrs["gen_ai.response.model"], "gpt-4-0613")
            self.assertEqual(attrs["gen_ai.response.finish_reasons"], ["stop", "length"])

    def test_extract_response_attributes_partial(self):
        """Test that _extract_response_attributes handles partial response data."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock result with only some attributes
            result = MagicMock()
            result.id = "chatcmpl-456"
            result.model = None  # Model might be None in some cases
            result.choices = []

            attrs = instrumentor._extract_response_attributes(result)

            self.assertEqual(attrs["gen_ai.response.id"], "chatcmpl-456")
            # Should not include finish_reasons if choices is empty
            self.assertNotIn("gen_ai.response.finish_reasons", attrs)

    def test_extract_response_attributes_missing(self):
        """Test that _extract_response_attributes handles missing attributes."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock result without response attributes
            result = MagicMock(spec=[])

            attrs = instrumentor._extract_response_attributes(result)

            # Should return empty dict when no attributes available
            self.assertEqual(attrs, {})

    def test_add_content_events(self):
        """Test that _add_content_events adds prompt and completion events."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock span
            mock_span = MagicMock()

            # Create mock result with completion content
            result = MagicMock()
            choice = MagicMock()
            choice.message.content = "This is the completion"
            result.choices = [choice]

            # Create request kwargs with messages
            request_kwargs = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }

            # Call method
            instrumentor._add_content_events(mock_span, result, request_kwargs)

            # Verify prompt events were added
            assert mock_span.add_event.call_count == 3  # 2 prompts + 1 completion
            calls = mock_span.add_event.call_args_list

            # Check prompt events
            self.assertEqual(calls[0][0][0], "gen_ai.prompt.0")
            self.assertEqual(calls[0][1]["attributes"]["gen_ai.prompt.role"], "user")
            self.assertEqual(calls[0][1]["attributes"]["gen_ai.prompt.content"], "Hello")

            self.assertEqual(calls[1][0][0], "gen_ai.prompt.1")
            self.assertEqual(calls[1][1]["attributes"]["gen_ai.prompt.role"], "assistant")
            self.assertEqual(calls[1][1]["attributes"]["gen_ai.prompt.content"], "Hi there")

            # Check completion event
            self.assertEqual(calls[2][0][0], "gen_ai.completion.0")
            self.assertEqual(calls[2][1]["attributes"]["gen_ai.completion.role"], "assistant")
            self.assertEqual(
                calls[2][1]["attributes"]["gen_ai.completion.content"], "This is the completion"
            )

    def test_add_content_events_empty_messages(self):
        """Test that _add_content_events handles empty messages."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            mock_span = MagicMock()
            result = MagicMock()
            result.choices = []
            request_kwargs = {"messages": []}

            # Should not raise any errors
            instrumentor._add_content_events(mock_span, result, request_kwargs)

            # No events should be added
            mock_span.add_event.assert_not_called()

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

    def test_extract_openai_attributes_with_tools(self):
        """Test that _extract_openai_attributes extracts tool definitions."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    },
                }
            ]

            kwargs = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": tools,
            }

            attrs = instrumentor._extract_openai_attributes(None, [], kwargs)

            self.assertIn("llm.tools", attrs)
            # Verify it's JSON-serialized
            import json

            parsed_tools = json.loads(attrs["llm.tools"])
            self.assertEqual(len(parsed_tools), 1)
            self.assertEqual(parsed_tools[0]["function"]["name"], "get_weather")

    def test_extract_response_attributes_with_tool_calls(self):
        """Test that _extract_response_attributes extracts tool calls from response."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create mock result with tool calls
            result = MagicMock()
            result.id = "chatcmpl-123"
            result.model = "gpt-4-0613"

            # Create mock tool call
            tool_call = MagicMock()
            tool_call.id = "call_abc123"
            tool_call.function.name = "get_weather"
            tool_call.function.arguments = '{"location": "San Francisco"}'

            # Create mock choice with tool calls
            choice = MagicMock()
            choice.finish_reason = "tool_calls"
            choice.message.tool_calls = [tool_call]
            result.choices = [choice]

            attrs = instrumentor._extract_response_attributes(result)

            self.assertEqual(attrs["gen_ai.response.id"], "chatcmpl-123")
            self.assertEqual(attrs["gen_ai.response.model"], "gpt-4-0613")
            self.assertEqual(attrs["gen_ai.response.finish_reasons"], ["tool_calls"])

            # Check tool call attributes
            prefix = "llm.output_messages.0.message.tool_calls.0"
            self.assertEqual(attrs[f"{prefix}.tool_call.id"], "call_abc123")
            self.assertEqual(attrs[f"{prefix}.tool_call.function.name"], "get_weather")
            self.assertEqual(
                attrs[f"{prefix}.tool_call.function.arguments"], '{"location": "San Francisco"}'
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
