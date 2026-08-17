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

        class MockAsyncOpenAI:
            def __init__(self):
                pass

        # Create mock OpenAI module
        mock_openai = MagicMock()
        mock_openai.OpenAI = MockOpenAI
        mock_openai.AsyncOpenAI = MockAsyncOpenAI

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
            mock_logger.info.assert_any_call("OpenAI instrumentation enabled")
            mock_logger.info.assert_any_call("AsyncOpenAI instrumentation enabled")
            # Verify FunctionWrapper was called for both sync and async
            self.assertEqual(mock_wrapt.FunctionWrapper.call_count, 2)

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

            # Chat and embeddings are both wrapped, so assert on the chat call
            # specifically rather than on it being the only one.
            instrumentor.create_span_wrapper.assert_any_call(
                span_name="openai.chat.completion",
                extract_attributes=instrumentor._extract_openai_attributes,
            )

            # Assert that the decorator was called with original_create
            mock_decorator.assert_any_call(original_create)

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
        """Test that first message content is truncated to content_max_length."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            long_content = "x" * 300
            kwargs = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": long_content}],
            }

            attrs = instrumentor._extract_openai_attributes(None, [], kwargs)

            self.assertIn("gen_ai.request.first_message", attrs)
            # Content is truncated to 200 (default), but the full attribute
            # includes the dict wrapper: {'role': 'user', 'content': '...'}
            import ast

            parsed = ast.literal_eval(attrs["gen_ai.request.first_message"])
            self.assertLessEqual(len(parsed["content"]), 200)

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

        class MockAsyncOpenAI:
            def __init__(self):
                pass

        # Create mock OpenAI module
        mock_openai = MagicMock()
        mock_openai.OpenAI = MockOpenAI
        mock_openai.AsyncOpenAI = MockAsyncOpenAI

        # Create a mock wrapt module that actually executes wrapped functions
        import wrapt as real_wrapt

        with patch.dict("sys.modules", {"openai": mock_openai, "wrapt": real_wrapt}):
            instrumentor = OpenAIInstrumentor()
            config = MagicMock()

            # Mock _instrument_client
            mock_instrument_client = MagicMock()
            instrumentor._instrument_client = mock_instrument_client
            mock_instrument_async_client = MagicMock()
            instrumentor._instrument_async_client = mock_instrument_async_client

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

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
    def test_instrument_with_async_openai_available(self, mock_logger):
        """Test that instrument wraps AsyncOpenAI client when available."""

        class MockAsyncOpenAI:
            def __init__(self):
                pass

        mock_openai = MagicMock()
        mock_openai.OpenAI = type("OpenAI", (), {"__init__": lambda self: None})
        mock_openai.AsyncOpenAI = MockAsyncOpenAI

        import wrapt as real_wrapt

        with patch.dict("sys.modules", {"openai": mock_openai, "wrapt": real_wrapt}):
            instrumentor = OpenAIInstrumentor()
            config = MagicMock()

            instrumentor._instrument_client = MagicMock()
            instrumentor._instrument_async_client = MagicMock()

            instrumentor.instrument(config)

            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_any_call("AsyncOpenAI instrumentation enabled")

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
    def test_wrapped_async_init_calls_instrument_async_client(self, mock_logger):
        """Test that the wrapped AsyncOpenAI __init__ calls _instrument_async_client."""

        class MockAsyncOpenAI:
            def __init__(self):
                pass

        mock_openai = MagicMock()
        mock_openai.OpenAI = type("OpenAI", (), {"__init__": lambda self: None})
        mock_openai.AsyncOpenAI = MockAsyncOpenAI

        import wrapt as real_wrapt

        with patch.dict("sys.modules", {"openai": mock_openai, "wrapt": real_wrapt}):
            instrumentor = OpenAIInstrumentor()
            config = MagicMock()

            mock_instrument_client = MagicMock()
            mock_instrument_async_client = MagicMock()
            instrumentor._instrument_client = mock_instrument_client
            instrumentor._instrument_async_client = mock_instrument_async_client

            instrumentor.instrument(config)

            instance = mock_openai.AsyncOpenAI()
            mock_instrument_async_client.assert_called_once_with(instance)

    def test_instrument_async_client(self):
        """Test that _instrument_async_client wraps async chat.completions.create."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            mock_client = MagicMock()
            original_create = MagicMock()
            mock_client.chat.completions.create = original_create

            mock_async_wrapper = MagicMock()
            mock_decorator = MagicMock(return_value=mock_async_wrapper)
            instrumentor._create_async_span_wrapper = MagicMock(return_value=mock_decorator)

            instrumentor._instrument_async_client(mock_client)

            # Chat and embeddings are both wrapped, so assert on the chat call
            # specifically rather than on it being the only one.
            instrumentor._create_async_span_wrapper.assert_any_call(
                span_name="openai.chat.completion",
                extract_attributes=instrumentor._extract_openai_attributes,
            )
            mock_decorator.assert_any_call(original_create)
            self.assertEqual(mock_client.chat.completions.create, mock_async_wrapper)

    def test_create_async_span_wrapper(self):
        """Test that _create_async_span_wrapper creates a working async wrapper."""
        import asyncio

        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            # Create a mock async function
            mock_result = MagicMock()
            mock_result.usage = None
            mock_result.choices = []

            async def mock_create(*args, **kwargs):
                return mock_result

            wrapper_decorator = instrumentor._create_async_span_wrapper(
                span_name="openai.chat.completion",
                extract_attributes=instrumentor._extract_openai_attributes,
            )
            wrapped = wrapper_decorator(mock_create)

            # Run the async wrapper
            result = asyncio.run(
                wrapped(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
            )

            self.assertEqual(result, mock_result)

    def test_async_streaming_records_ttft_and_tpot(self):
        """An async stream keeps its span open and reports TTFT/TPOT (issue #21).

        Awaiting an async streaming call only yields the stream object - the
        model has generated nothing yet - so a span closed at that point times
        the handshake and reports no streaming latency at all.
        """
        import asyncio

        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        captured = []

        class _Capture(SimpleSpanProcessor):
            def __init__(self):  # pylint: disable=super-init-not-called
                pass

            def on_start(self, span, parent_context=None):
                pass

            def on_end(self, span):
                captured.append(span)

            def shutdown(self):
                pass

            def force_flush(self, timeout_millis=30000):
                return True

        provider = TracerProvider()
        provider.add_span_processor(_Capture())

        class FakeAsyncStream:
            def __aiter__(self):
                return self._chunks()

            async def _chunks(self):
                usage = MagicMock()
                usage.prompt_tokens = 4
                usage.completion_tokens = 6
                usage.total_tokens = 10
                first = MagicMock()
                first.usage = None
                last = MagicMock()
                last.usage = usage
                yield first
                yield last

        async def mock_create(*args, **kwargs):
            return FakeAsyncStream()

        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()
            with patch(
                "opentelemetry.trace.get_tracer",
                return_value=provider.get_tracer(__name__),
            ):
                wrapped = instrumentor._create_async_span_wrapper(
                    span_name="openai.chat.completion",
                    extract_attributes=instrumentor._extract_openai_attributes,
                )(mock_create)

                async def run():
                    stream = await wrapped(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "hi"}],
                        stream=True,
                    )
                    self.assertEqual(captured, [], "span closed before the stream was consumed")
                    return [chunk async for chunk in stream]

                chunks = asyncio.run(run())

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(captured), 1)
        attrs = dict(captured[0].attributes or {})
        self.assertIn("gen_ai.server.time_to_first_token", attrs)
        self.assertIn("gen_ai.server.time_per_output_token", attrs)
        self.assertEqual(attrs.get("gen_ai.streaming.token_count"), 2)

    def test_async_non_streaming_omits_streaming_latency(self):
        """A non-streamed async call must carry neither TTFT nor TPOT."""
        import asyncio

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        captured = []

        class _Capture(SimpleSpanProcessor):
            def __init__(self):  # pylint: disable=super-init-not-called
                pass

            def on_start(self, span, parent_context=None):
                pass

            def on_end(self, span):
                captured.append(span)

            def shutdown(self):
                pass

            def force_flush(self, timeout_millis=30000):
                return True

        provider = TracerProvider()
        provider.add_span_processor(_Capture())

        mock_result = MagicMock()
        mock_result.usage = None
        mock_result.choices = []

        async def mock_create(*args, **kwargs):
            return mock_result

        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()
            with patch(
                "opentelemetry.trace.get_tracer",
                return_value=provider.get_tracer(__name__),
            ):
                wrapped = instrumentor._create_async_span_wrapper(
                    span_name="openai.chat.completion",
                    extract_attributes=instrumentor._extract_openai_attributes,
                )(mock_create)
                asyncio.run(
                    wrapped(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
                )

        self.assertEqual(len(captured), 1)
        attrs = dict(captured[0].attributes or {})
        self.assertNotIn("gen_ai.server.time_to_first_token", attrs)
        self.assertNotIn("gen_ai.server.time_per_output_token", attrs)
        self.assertNotIn("gen_ai.server.ttft", attrs)


class TestOpenAIEmbeddingsInstrumentation(unittest.TestCase):
    """Tests for embeddings instrumentation.

    Embeddings are the retrieval half of a RAG trace. Without a span here, a
    retrieval-augmented call shows only its generation leg, so the lookup that
    chose the context is invisible and its tokens and cost go unbilled.
    """

    @staticmethod
    def _embedding_response(dimensions=3, count=1, model="text-embedding-3-small"):
        """Build a response shaped like openai.resources.embeddings returns."""
        response = MagicMock()
        response.model = model
        response.data = []
        for index in range(count):
            item = MagicMock()
            item.index = index
            item.embedding = [0.1] * dimensions
            response.data.append(item)
        usage = MagicMock(spec=["prompt_tokens", "total_tokens"])
        usage.prompt_tokens = 8
        usage.total_tokens = 8
        response.usage = usage
        # An embeddings response has no choices; guard against a chat-shaped read.
        del response.choices
        del response.id
        return response

    def test_instrument_client_wraps_embeddings_create(self):
        """_instrument_client must wrap embeddings.create, not only chat."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            mock_client = MagicMock()
            original_embeddings_create = MagicMock()
            mock_client.embeddings.create = original_embeddings_create

            wrappers = {}

            def fake_create_span_wrapper(span_name, extract_attributes=None):
                def decorator(func):
                    wrappers[span_name] = (func, extract_attributes)
                    return MagicMock(name=f"wrapped:{span_name}")

                return decorator

            instrumentor.create_span_wrapper = fake_create_span_wrapper

            instrumentor._instrument_client(mock_client)

            self.assertIn("openai.embeddings", wrappers)
            wrapped_func, extractor = wrappers["openai.embeddings"]
            self.assertIs(wrapped_func, original_embeddings_create)
            self.assertEqual(extractor, instrumentor._extract_embedding_attributes)
            self.assertIsNot(mock_client.embeddings.create, original_embeddings_create)

    def test_instrument_async_client_wraps_embeddings_create(self):
        """The async client needs the same coverage as the sync one."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            mock_client = MagicMock()
            original_embeddings_create = MagicMock()
            mock_client.embeddings.create = original_embeddings_create

            wrappers = {}

            def fake_async_wrapper(span_name, extract_attributes=None):
                def decorator(func):
                    wrappers[span_name] = (func, extract_attributes)
                    return MagicMock(name=f"wrapped:{span_name}")

                return decorator

            instrumentor._create_async_span_wrapper = fake_async_wrapper

            instrumentor._instrument_async_client(mock_client)

            self.assertIn("openai.embeddings", wrappers)
            _, extractor = wrappers["openai.embeddings"]
            self.assertEqual(extractor, instrumentor._extract_embedding_attributes)

    def test_instrument_client_without_embeddings_attribute(self):
        """A client lacking .embeddings must not raise."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            class BareClient:
                base_url = None

            instrumentor._instrument_client(BareClient())  # must not raise

    def test_extract_embedding_attributes_single_string(self):
        """A single string input is one item, and the call type drives pricing."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            attrs = instrumentor._extract_embedding_attributes(
                None, (), {"model": "text-embedding-3-small", "input": "hello world"}
            )

            self.assertEqual(attrs["gen_ai.system"], "openai")
            self.assertEqual(attrs["gen_ai.request.model"], "text-embedding-3-small")
            self.assertEqual(attrs["gen_ai.operation.name"], "embeddings")
            # "embedding" (singular) is what CostCalculator.calculate_cost
            # dispatches on; "embeddings" would silently price as chat.
            self.assertEqual(attrs["gen_ai.request.type"], "embedding")
            self.assertEqual(attrs["gen_ai.request.input_count"], 1)

    def test_extract_embedding_attributes_batch(self):
        """RAG indexing embeds in batches; the batch size must be visible."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            attrs = instrumentor._extract_embedding_attributes(
                None,
                (),
                {"model": "text-embedding-3-large", "input": ["chunk a", "chunk b", "chunk c"]},
            )

            self.assertEqual(attrs["gen_ai.request.input_count"], 3)

    def test_extract_embedding_attributes_optional_parameters(self):
        """dimensions/encoding_format are request parameters worth recording."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            attrs = instrumentor._extract_embedding_attributes(
                None,
                (),
                {
                    "model": "text-embedding-3-small",
                    "input": "x",
                    "dimensions": 256,
                    "encoding_format": "float",
                },
            )

            self.assertEqual(attrs["gen_ai.request.dimensions"], 256)
            self.assertEqual(attrs["gen_ai.request.encoding_format"], "float")

    def test_extract_embedding_attributes_token_input(self):
        """The API also accepts pre-tokenised input; count it as one item."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            attrs = instrumentor._extract_embedding_attributes(
                None, (), {"model": "text-embedding-3-small", "input": [1234, 5678]}
            )

            self.assertEqual(attrs["gen_ai.request.input_count"], 1)

    def test_extract_usage_from_embedding_response(self):
        """Embeddings report prompt/total tokens and no completion tokens."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            usage = instrumentor._extract_usage(self._embedding_response())

            self.assertEqual(usage["prompt_tokens"], 8)
            self.assertEqual(usage["completion_tokens"], 0)
            self.assertEqual(usage["total_tokens"], 8)

    def test_extract_response_attributes_for_embedding(self):
        """Vector count and dimension are the response facts worth keeping."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            attrs = instrumentor._extract_response_attributes(
                self._embedding_response(dimensions=1536, count=2)
            )

            self.assertEqual(attrs["gen_ai.response.model"], "text-embedding-3-small")
            self.assertEqual(attrs["gen_ai.response.embedding_count"], 2)
            self.assertEqual(attrs["gen_ai.response.vector_size"], 1536)

    def test_extract_response_attributes_chat_still_works(self):
        """The embedding branch must not disturb chat response extraction."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            result = MagicMock()
            result.id = "chatcmpl-123"
            result.model = "gpt-4o-mini"
            choice = MagicMock()
            choice.finish_reason = "stop"
            choice.message = MagicMock(spec=["content"])
            choice.message.content = "hi"
            result.choices = [choice]

            attrs = instrumentor._extract_response_attributes(result)

            self.assertEqual(attrs["gen_ai.response.id"], "chatcmpl-123")
            self.assertEqual(attrs["gen_ai.response.finish_reasons"], ["stop"])
            self.assertNotIn("gen_ai.response.embedding_count", attrs)

    def test_embedding_input_text_captured_only_when_enabled(self):
        """Embedded text is user content: it must honour the capture switch."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()
            instrumentor.config = OTelConfig(
                service_name="t", enable_content_capture=True, content_max_length=0
            )

            span = MagicMock()
            captured = {}
            span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)

            instrumentor._add_content_events(
                span,
                self._embedding_response(),
                {"model": "text-embedding-3-small", "input": "retrieve me"},
            )

            self.assertEqual(captured.get("embedding.model_name"), "text-embedding-3-small")
            self.assertIn("retrieve me", captured.get("embedding.text", ""))

    def test_embedding_input_text_truncated_to_max_length(self):
        """content_max_length applies to embedded text as it does to prompts."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()
            instrumentor.config = OTelConfig(
                service_name="t", enable_content_capture=True, content_max_length=10
            )

            span = MagicMock()
            captured = {}
            span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)

            instrumentor._add_content_events(
                span,
                self._embedding_response(),
                {"model": "text-embedding-3-small", "input": "x" * 500},
            )

            self.assertEqual(len(captured.get("embedding.text", "")), 10)

    def test_embedding_vectors_not_captured_by_default(self):
        """Vectors are large and rarely wanted; they stay off unless asked for."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()
            instrumentor.config = OTelConfig(service_name="t", enable_content_capture=True)

            span = MagicMock()
            captured = {}
            span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)

            instrumentor._add_content_events(
                span,
                self._embedding_response(dimensions=1536),
                {"model": "text-embedding-3-small", "input": "hello"},
            )

            self.assertNotIn("embedding.vector", captured)

    def test_embedding_cost_uses_embedding_pricing(self):
        """The recorded call type must resolve against the embeddings table."""
        from genai_otel.cost_calculator import CostCalculator

        calculator = CostCalculator()
        attrs = OpenAIInstrumentor.__new__(OpenAIInstrumentor)._extract_embedding_attributes(
            None, (), {"model": "text-embedding-3-small", "input": "hi"}
        )

        cost = calculator.calculate_cost(
            "text-embedding-3-small",
            {"prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000},
            attrs["gen_ai.request.type"],
        )

        self.assertGreater(cost, 0.0)

    def test_embedding_span_not_added_for_aggregator_clients(self):
        """Aggregator clients are owned by their own instrumentor, as for chat."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            instrumentor = OpenAIInstrumentor()

            mock_client = MagicMock()
            original = MagicMock()
            mock_client.embeddings.create = original

            with patch(
                "genai_otel.instrumentors.openai_instrumentor.find_base_url_claim",
                return_value="openrouter",
            ):
                instrumentor._instrument_client(mock_client)

            self.assertIs(mock_client.embeddings.create, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
