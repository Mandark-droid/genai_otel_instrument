import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.openai_agents_instrumentor import OpenAIAgentsInstrumentor


class TestOpenAIAgentsInstrumentor(unittest.TestCase):
    """Tests for OpenAIAgentsInstrumentor"""

    @patch("genai_otel.instrumentors.openai_agents_instrumentor.logger")
    def test_init_with_agents_available(self, mock_logger):
        """Test that __init__ detects OpenAI Agents SDK availability."""
        with patch.dict("sys.modules", {"agents": MagicMock()}):
            instrumentor = OpenAIAgentsInstrumentor()

            self.assertTrue(instrumentor._agents_available)
            mock_logger.debug.assert_called_with(
                "OpenAI Agents SDK detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.openai_agents_instrumentor.logger")
    def test_init_with_agents_not_available(self, mock_logger):
        """Test that __init__ handles missing OpenAI Agents SDK gracefully."""
        with patch.dict("sys.modules", {"agents": None}):
            instrumentor = OpenAIAgentsInstrumentor()

            self.assertFalse(instrumentor._agents_available)
            mock_logger.debug.assert_called_with(
                "OpenAI Agents SDK not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.openai_agents_instrumentor.logger")
    def test_instrument_when_agents_not_available(self, mock_logger):
        """Test that instrument skips when OpenAI Agents SDK is not available."""
        with patch.dict("sys.modules", {"agents": None}):
            instrumentor = OpenAIAgentsInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping OpenAI Agents instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.openai_agents_instrumentor.logger")
    def test_instrument_with_agents_available(self, mock_logger):
        """Test that instrument wraps Runner methods when available."""

        # Create a real Runner class
        class MockRunner:
            @staticmethod
            def run(agent, input_data, session=None):
                return "run_result"

            @staticmethod
            def run_sync(agent, input_data, session=None):
                return "run_sync_result"

        # Create mock agents module
        mock_agents = MagicMock()
        mock_agents.Runner = MockRunner

        # Create a mock wrapt module
        mock_wrapt = MagicMock()

        with patch.dict("sys.modules", {"agents": mock_agents, "wrapt": mock_wrapt}):
            instrumentor = OpenAIAgentsInstrumentor()
            config = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("OpenAI Agents SDK instrumentation enabled")
            # Verify FunctionWrapper was called to wrap both methods
            self.assertEqual(mock_wrapt.FunctionWrapper.call_count, 2)

    @patch("genai_otel.instrumentors.openai_agents_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        # Create mock agents module
        mock_agents = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "Runner":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"agents": mock_agents, "wrapt": MagicMock()}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = OpenAIAgentsInstrumentor()
                config = MagicMock()
                config.fail_on_error = False

                # Should not raise exception
                instrumentor.instrument(config)

                mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.openai_agents_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that instrument raises exceptions when fail_on_error is True."""
        # Create mock agents module
        mock_agents = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "Runner":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"agents": mock_agents, "wrapt": MagicMock()}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = OpenAIAgentsInstrumentor()
                config = MagicMock()
                config.fail_on_error = True

                # Should raise exception
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_extract_runner_attributes_with_agent(self):
        """Test extraction of runner attributes with agent information."""
        with patch.dict("sys.modules", {"agents": MagicMock()}):
            instrumentor = OpenAIAgentsInstrumentor()

            # Create mock agent
            mock_agent = MagicMock()
            mock_agent.name = "TestAgent"
            mock_agent.model = "gpt-4"
            mock_agent.instructions = "You are a helpful assistant"
            mock_agent.tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
            mock_agent.handoffs = [MagicMock(name="agent2")]
            mock_agent.guardrails = [MagicMock()]

            # Create mock args (agent, input_data)
            args = (mock_agent, "test input", None)
            kwargs = {}

            attrs = instrumentor._extract_runner_attributes(None, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "openai_agents")
            self.assertEqual(attrs["gen_ai.operation.name"], "agent.run")
            self.assertEqual(attrs["openai.agent.name"], "TestAgent")
            self.assertEqual(attrs["openai.agent.model"], "gpt-4")
            self.assertEqual(attrs["gen_ai.request.model"], "TestAgent")
            self.assertIn("openai.agent.instructions", attrs)
            self.assertEqual(attrs["openai.agent.tool_count"], 2)
            self.assertEqual(attrs["openai.agent.handoff_count"], 1)
            self.assertTrue(attrs["openai.agent.guardrails_enabled"])
            self.assertEqual(attrs["openai.agent.guardrail_count"], 1)

    def test_extract_runner_attributes_with_session(self):
        """Test extraction of runner attributes with session information."""
        with patch.dict("sys.modules", {"agents": MagicMock()}):
            instrumentor = OpenAIAgentsInstrumentor()

            # Create mock agent and session
            mock_agent = MagicMock()
            mock_agent.name = "TestAgent"

            mock_session = MagicMock()
            mock_session.session_id = "sess_123"

            # Create mock args (agent, input_data, session)
            args = (mock_agent, "test input", mock_session)
            kwargs = {}

            attrs = instrumentor._extract_runner_attributes(None, args, kwargs)

            # Assert
            self.assertEqual(attrs["openai.session.id"], "sess_123")
            self.assertEqual(attrs["session.id"], "sess_123")
            self.assertIn("openai.session.type", attrs)

    def test_extract_response_attributes(self):
        """Test extraction of response attributes from agent run result."""
        with patch.dict("sys.modules", {"agents": MagicMock()}):
            instrumentor = OpenAIAgentsInstrumentor()

            # Create mock result
            mock_result = MagicMock()
            mock_result.final_output = "This is the agent's response"
            mock_result.handoff = MagicMock()
            mock_result.handoff.target_agent = "agent2"

            attrs = instrumentor._extract_response_attributes(mock_result)

            # Assert
            self.assertIn("openai.agent.output", attrs)
            self.assertTrue(attrs["openai.handoff.occurred"])
            self.assertEqual(attrs["openai.handoff.to_agent"], "agent2")

    def test_extract_usage_when_available(self):
        """Test extraction of usage information when available."""
        with patch.dict("sys.modules", {"agents": MagicMock()}):
            instrumentor = OpenAIAgentsInstrumentor()

            # Create mock result with usage
            mock_result = MagicMock()
            mock_result.usage = MagicMock()
            mock_result.usage.prompt_tokens = 100
            mock_result.usage.completion_tokens = 50
            mock_result.usage.total_tokens = 150

            usage = instrumentor._extract_usage(mock_result)

            # Assert
            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 100)
            self.assertEqual(usage["completion_tokens"], 50)
            self.assertEqual(usage["total_tokens"], 150)

    def test_extract_usage_when_not_available(self):
        """Test extraction of usage information when not available."""
        with patch.dict("sys.modules", {"agents": MagicMock()}):
            instrumentor = OpenAIAgentsInstrumentor()

            # Create mock result without usage
            mock_result = MagicMock()
            del mock_result.usage

            usage = instrumentor._extract_usage(mock_result)

            # Assert
            self.assertIsNone(usage)

    def test_extract_finish_reason(self):
        """Test extraction of finish reason from result."""
        with patch.dict("sys.modules", {"agents": MagicMock()}):
            instrumentor = OpenAIAgentsInstrumentor()

            # Create mock result with finish_reason
            mock_result = MagicMock()
            mock_result.finish_reason = "stop"

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
