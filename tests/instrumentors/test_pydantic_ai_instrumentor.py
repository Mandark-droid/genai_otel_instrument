import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.pydantic_ai_instrumentor import PydanticAIInstrumentor


class TestPydanticAIInstrumentor(unittest.TestCase):
    """Tests for PydanticAIInstrumentor"""

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_init_with_pydantic_ai_available(self, mock_logger):
        """Test that __init__ detects Pydantic AI availability."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            self.assertTrue(instrumentor._pydantic_ai_available)
            mock_logger.debug.assert_called_with(
                "Pydantic AI library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_init_with_pydantic_ai_not_available(self, mock_logger):
        """Test that __init__ handles missing Pydantic AI gracefully."""
        with patch.dict("sys.modules", {"pydantic_ai": None}):
            instrumentor = PydanticAIInstrumentor()

            self.assertFalse(instrumentor._pydantic_ai_available)
            mock_logger.debug.assert_called_with(
                "Pydantic AI library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_instrument_when_pydantic_ai_not_available(self, mock_logger):
        """Test that instrument skips when Pydantic AI is not available."""
        with patch.dict("sys.modules", {"pydantic_ai": None}):
            instrumentor = PydanticAIInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Pydantic AI instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_instrument_with_agent_run(self, mock_logger):
        """Test that instrument wraps Agent.run method."""

        # Create mock Agent class
        class MockAgent:
            def run(self, user_prompt, **kwargs):
                return {"data": "response"}

        # Create mock pydantic_ai module
        mock_pydantic_ai = MagicMock()
        mock_pydantic_ai.Agent = MockAgent

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "pydantic_ai": mock_pydantic_ai,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = PydanticAIInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("Pydantic AI instrumentation enabled")
            # Verify FunctionWrapper was called
            self.assertTrue(mock_wrapt.FunctionWrapper.called)

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_instrument_with_agent_run_sync(self, mock_logger):
        """Test that instrument wraps Agent.run_sync method."""

        # Create mock Agent class
        class MockAgent:
            def run_sync(self, user_prompt, **kwargs):
                return {"data": "response"}

        # Create mock pydantic_ai module
        mock_pydantic_ai = MagicMock()
        mock_pydantic_ai.Agent = MockAgent

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "pydantic_ai": mock_pydantic_ai,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = PydanticAIInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("Pydantic AI instrumentation enabled")
            # Verify FunctionWrapper was called
            self.assertTrue(mock_wrapt.FunctionWrapper.called)

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_instrument_with_agent_run_stream(self, mock_logger):
        """Test that instrument wraps Agent.run_stream method."""

        # Create mock Agent class
        class MockAgent:
            def run_stream(self, user_prompt, **kwargs):
                return iter([{"data": "chunk1"}, {"data": "chunk2"}])

        # Create mock pydantic_ai module
        mock_pydantic_ai = MagicMock()
        mock_pydantic_ai.Agent = MockAgent

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "pydantic_ai": mock_pydantic_ai,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = PydanticAIInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("Pydantic AI instrumentation enabled")
            # Verify FunctionWrapper was called
            self.assertTrue(mock_wrapt.FunctionWrapper.called)

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        # Create mock that raises
        mock_pydantic_ai = MagicMock()
        type(mock_pydantic_ai).Agent = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"pydantic_ai": mock_pydantic_ai, "wrapt": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            # Should not raise
            instrumentor.instrument(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that exceptions are raised when fail_on_error is True."""
        # Create mock that raises
        mock_pydantic_ai = MagicMock()
        type(mock_pydantic_ai).Agent = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"pydantic_ai": mock_pydantic_ai, "wrapt": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            # Should raise
            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Access failed")

    def test_extract_agent_attributes_basic(self):
        """Test extraction of basic agent attributes."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.name = "test_agent"

            # Create mock model
            mock_model = MagicMock()
            mock_model.name = "gpt-4"
            mock_model.__class__.__name__ = "OpenAIModel"
            mock_agent.model = mock_model

            # User prompt
            user_prompt = "What is the capital of France?"

            args = (user_prompt,)
            kwargs = {}

            attrs = instrumentor._extract_agent_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "pydantic_ai")
            self.assertEqual(attrs["gen_ai.operation.name"], "agent.run")
            self.assertEqual(attrs["pydantic_ai.agent.name"], "test_agent")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-4")
            self.assertEqual(attrs["pydantic_ai.model.name"], "gpt-4")
            self.assertEqual(attrs["pydantic_ai.model.provider"], "OpenAIModel")
            self.assertEqual(attrs["pydantic_ai.user_prompt"], user_prompt)

    def test_extract_agent_attributes_with_tools(self):
        """Test extraction of agent attributes with tools."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.name = "agent_with_tools"
            mock_agent.model = MagicMock()

            # Mock tools
            mock_agent._function_tools = {
                "search_web": MagicMock(),
                "calculate": MagicMock(),
                "get_weather": MagicMock(),
            }

            args = ("Test prompt",)
            kwargs = {}

            attrs = instrumentor._extract_agent_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["pydantic_ai.tools.count"], 3)
            self.assertIn("search_web", attrs["pydantic_ai.tools"])
            self.assertIn("calculate", attrs["pydantic_ai.tools"])
            self.assertIn("get_weather", attrs["pydantic_ai.tools"])

    def test_extract_agent_attributes_with_system_prompts(self):
        """Test extraction of agent attributes with system prompts."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.name = "agent_with_prompts"
            mock_agent.model = MagicMock()

            # Mock system prompts
            mock_agent._system_prompts = [
                "You are a helpful assistant.",
                "Be concise and accurate.",
            ]

            args = ("Test prompt",)
            kwargs = {}

            attrs = instrumentor._extract_agent_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertIn("pydantic_ai.system_prompts", attrs)
            self.assertEqual(len(attrs["pydantic_ai.system_prompts"]), 2)
            self.assertEqual(attrs["pydantic_ai.system_prompts"][0], "You are a helpful assistant.")

    def test_extract_agent_attributes_with_result_type(self):
        """Test extraction of agent attributes with result type."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.name = "typed_agent"
            mock_agent.model = MagicMock()
            mock_agent._result_type = "str"

            args = ("Test prompt",)
            kwargs = {}

            attrs = instrumentor._extract_agent_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["pydantic_ai.result_type"], "str")

    def test_extract_agent_attributes_with_model_settings(self):
        """Test extraction of agent attributes with model settings."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.model = MagicMock()

            args = ("Test prompt",)
            kwargs = {
                "model_settings": {
                    "temperature": 0.8,
                    "max_tokens": 2000,
                    "top_p": 0.95,
                }
            }

            attrs = instrumentor._extract_agent_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.request.temperature"], 0.8)
            self.assertEqual(attrs["gen_ai.request.max_tokens"], 2000)
            self.assertEqual(attrs["gen_ai.request.top_p"], 0.95)

    def test_extract_agent_attributes_with_message_history(self):
        """Test extraction of agent attributes with message history."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.model = MagicMock()

            # Mock message history
            message_history = [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Follow-up"},
            ]

            args = ("Current prompt",)
            kwargs = {"message_history": message_history}

            attrs = instrumentor._extract_agent_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["pydantic_ai.message_history.count"], 3)

    def test_extract_agent_attributes_with_kwargs_prompt(self):
        """Test extraction of agent attributes with prompt in kwargs."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.model = MagicMock()

            args = ()
            kwargs = {"user_prompt": "Test prompt from kwargs"}

            attrs = instrumentor._extract_agent_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["pydantic_ai.user_prompt"], "Test prompt from kwargs")

    def test_extract_usage_with_usage_attribute(self):
        """Test that _extract_usage extracts from usage attribute."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock result with usage
            result = MagicMock()
            result.usage.request_tokens = 120
            result.usage.response_tokens = 80
            result.usage.total_tokens = 200

            usage = instrumentor._extract_usage(result)

            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 120)
            self.assertEqual(usage["completion_tokens"], 80)
            self.assertEqual(usage["total_tokens"], 200)

    def test_extract_usage_without_usage_attribute(self):
        """Test that _extract_usage returns None when no usage."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock result without usage
            result = MagicMock(spec=[])
            if hasattr(result, "usage"):
                delattr(result, "usage")
            if hasattr(result, "_usage"):
                delattr(result, "_usage")

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    def test_extract_response_attributes_with_data(self):
        """Test extraction of response attributes with data."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock result
            result = MagicMock()
            result.data = "This is the agent response"
            result.model = "gpt-4"

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["pydantic_ai.result.data"], "This is the agent response")
            self.assertEqual(attrs["gen_ai.response.model"], "gpt-4")

    def test_extract_response_attributes_with_pydantic_model(self):
        """Test extraction of response attributes with Pydantic model."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock Pydantic model
            mock_data = MagicMock()
            mock_data.model_dump.return_value = {"name": "John", "age": 30}

            # Create mock result
            result = MagicMock()
            result.data = mock_data

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertIn("pydantic_ai.result.data", attrs)
            self.assertIn("John", attrs["pydantic_ai.result.data"])

    def test_extract_response_attributes_with_messages(self):
        """Test extraction of response attributes with messages."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock messages
            mock_msg1 = MagicMock()
            mock_msg1.content = "First message"
            mock_msg1.role = "user"

            mock_msg2 = MagicMock()
            mock_msg2.content = "Second message"
            mock_msg2.role = "assistant"

            # Create mock result
            result = MagicMock()
            result.messages = [mock_msg1, mock_msg2]

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["pydantic_ai.result.messages.count"], 2)
            self.assertEqual(attrs["pydantic_ai.result.last_message"], "Second message")
            self.assertEqual(attrs["pydantic_ai.result.last_role"], "assistant")

    def test_extract_response_attributes_with_cost(self):
        """Test extraction of response attributes with cost."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock result with cost
            result = MagicMock()
            result.cost = 0.045

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["pydantic_ai.result.cost"], 0.045)

    def test_extract_response_attributes_with_timestamp(self):
        """Test extraction of response attributes with timestamp."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock result with timestamp
            result = MagicMock()
            result.timestamp = "2024-12-20T10:30:00Z"

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["pydantic_ai.result.timestamp"], "2024-12-20T10:30:00Z")

    def test_extract_finish_reason_with_finish_reason(self):
        """Test extraction of finish reason with finish_reason attribute."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock result
            result = MagicMock()
            result.finish_reason = "stop"

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertEqual(finish_reason, "stop")

    def test_extract_finish_reason_with_data(self):
        """Test extraction of finish reason with data (implies completion)."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock result with data but no finish_reason
            result = MagicMock(spec=["data"])
            result.data = "Some response"

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_finish_reason_from_messages(self):
        """Test extraction of finish reason from last message."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock message with finish_reason
            mock_msg = MagicMock()
            mock_msg.finish_reason = "length"

            # Create mock result
            result = MagicMock()
            result.messages = [mock_msg]
            del result.finish_reason  # Remove direct finish_reason

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertEqual(finish_reason, "length")

    def test_extract_finish_reason_none(self):
        """Test extraction of finish reason returns None when not available."""
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()

            # Create mock result without finish_reason or data
            result = MagicMock(spec=[])
            if hasattr(result, "finish_reason"):
                delattr(result, "finish_reason")
            if hasattr(result, "messages"):
                delattr(result, "messages")
            if hasattr(result, "data"):
                delattr(result, "data")

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertIsNone(finish_reason)


if __name__ == "__main__":
    unittest.main()
