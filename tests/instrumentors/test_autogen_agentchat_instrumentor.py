import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.autogen_agentchat_instrumentor import AutoGenAgentChatInstrumentor


class TestAutoGenAgentChatInstrumentor(unittest.TestCase):
    """Tests for AutoGenAgentChatInstrumentor"""

    @patch("genai_otel.instrumentors.autogen_agentchat_instrumentor.logger")
    def test_init_with_agentchat_available(self, mock_logger):
        """Test that __init__ detects autogen-agentchat availability."""
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            instrumentor = AutoGenAgentChatInstrumentor()

            self.assertTrue(instrumentor._agentchat_available)
            mock_logger.debug.assert_called_with(
                "autogen-agentchat library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.autogen_agentchat_instrumentor.logger")
    def test_init_with_agentchat_not_available(self, mock_logger):
        """Test that __init__ handles missing autogen-agentchat gracefully."""
        with patch.dict("sys.modules", {"autogen_agentchat": None}):
            instrumentor = AutoGenAgentChatInstrumentor()

            self.assertFalse(instrumentor._agentchat_available)
            mock_logger.debug.assert_called_with(
                "autogen-agentchat library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.autogen_agentchat_instrumentor.logger")
    def test_instrument_when_agentchat_not_available(self, mock_logger):
        """Test that instrument skips when autogen-agentchat is not available."""
        with patch.dict("sys.modules", {"autogen_agentchat": None}):
            instrumentor = AutoGenAgentChatInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping AutoGen AgentChat instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.autogen_agentchat_instrumentor.logger")
    def test_instrument_with_agentchat_available(self, mock_logger):
        """Test that instrument wraps ChatAgent and BaseGroupChat methods."""

        class MockChatAgent:
            def run(self, task=None):
                pass

            def run_stream(self, task=None):
                pass

            def on_messages(self, messages=None):
                pass

        class MockBaseGroupChat:
            def run(self, task=None):
                pass

            def run_stream(self, task=None):
                pass

        mock_base = MagicMock()
        mock_base.ChatAgent = MockChatAgent

        mock_teams = MagicMock()
        mock_teams.BaseGroupChat = MockBaseGroupChat

        # Save original methods
        original_agent_run = MockChatAgent.run
        original_team_run = MockBaseGroupChat.run

        with patch.dict(
            "sys.modules",
            {
                "autogen_agentchat": MagicMock(),
                "autogen_agentchat.base": mock_base,
                "autogen_agentchat.teams": mock_teams,
            },
        ):
            instrumentor = AutoGenAgentChatInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("AutoGen AgentChat instrumentation enabled")
            # Verify methods were wrapped
            self.assertIsNot(MockChatAgent.run, original_agent_run)
            self.assertIsNot(MockBaseGroupChat.run, original_team_run)

    def test_extract_agent_run_attributes_basic(self):
        """Test extraction of basic agent run attributes."""
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            instrumentor = AutoGenAgentChatInstrumentor()

            mock_agent = MagicMock()
            mock_agent.name = "assistant"
            mock_agent.__class__.__name__ = "AssistantAgent"
            mock_agent.model_client.model = "gpt-4o"

            attrs = instrumentor._extract_agent_run_attributes(
                mock_agent, ("Write a poem about AI",), {}
            )

            self.assertEqual(attrs["gen_ai.system"], "autogen_agentchat")
            self.assertEqual(attrs["gen_ai.operation.name"], "agent.run")
            self.assertEqual(attrs["autogen_agentchat.agent.name"], "assistant")
            self.assertEqual(attrs["autogen_agentchat.task"], "Write a poem about AI")

    def test_extract_team_run_attributes(self):
        """Test extraction of team run attributes."""
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            instrumentor = AutoGenAgentChatInstrumentor()

            # Create mock team
            mock_team = MagicMock()
            mock_team.__class__.__name__ = "RoundRobinGroupChat"
            mock_team.name = "research_team"

            # Create mock participants
            mock_p1 = MagicMock()
            mock_p1.name = "researcher"
            mock_p2 = MagicMock()
            mock_p2.name = "writer"
            mock_team._participants = [mock_p1, mock_p2]

            mock_team._termination_condition.__class__.__name__ = "MaxMessageTermination"

            attrs = instrumentor._extract_team_run_attributes(
                mock_team, ("Research AI trends",), {}
            )

            self.assertEqual(attrs["gen_ai.system"], "autogen_agentchat")
            self.assertEqual(attrs["gen_ai.operation.name"], "team.run")
            self.assertEqual(attrs["autogen_agentchat.team.type"], "RoundRobinGroupChat")
            self.assertEqual(attrs["autogen_agentchat.team.name"], "research_team")
            self.assertEqual(attrs["autogen_agentchat.team.participant_count"], 2)
            self.assertIn("researcher", attrs["autogen_agentchat.team.participants"])
            self.assertIn("writer", attrs["autogen_agentchat.team.participants"])
            self.assertEqual(attrs["autogen_agentchat.task"], "Research AI trends")

    def test_extract_on_messages_attributes(self):
        """Test extraction of on_messages attributes."""
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            instrumentor = AutoGenAgentChatInstrumentor()

            mock_agent = MagicMock()
            mock_agent.name = "assistant"
            mock_agent.__class__.__name__ = "AssistantAgent"

            messages = [MagicMock(), MagicMock(), MagicMock()]

            attrs = instrumentor._extract_on_messages_attributes(mock_agent, (messages,), {})

            self.assertEqual(attrs["gen_ai.system"], "autogen_agentchat")
            self.assertEqual(attrs["gen_ai.operation.name"], "agent.on_messages")
            self.assertEqual(attrs["autogen_agentchat.message_count"], 3)

    def test_extract_usage_returns_none(self):
        """Test that _extract_usage returns None (relies on provider instrumentors)."""
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            instrumentor = AutoGenAgentChatInstrumentor()
            self.assertIsNone(instrumentor._extract_usage(MagicMock()))

    def test_extract_finish_reason_with_stop_reason(self):
        """Test extraction of finish reason from TaskResult with stop_reason."""
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            instrumentor = AutoGenAgentChatInstrumentor()

            mock_result = MagicMock()
            mock_result.stop_reason = "MaxMessageTermination"

            self.assertEqual(
                instrumentor._extract_finish_reason(mock_result), "MaxMessageTermination"
            )

    def test_extract_finish_reason_completed(self):
        """Test extraction of finish reason when result exists but no stop_reason."""
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            instrumentor = AutoGenAgentChatInstrumentor()

            mock_result = MagicMock(spec=[])  # No stop_reason attribute

            self.assertEqual(instrumentor._extract_finish_reason(mock_result), "completed")

    def test_extract_response_attributes(self):
        """Test extraction of response attributes from TaskResult."""
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            instrumentor = AutoGenAgentChatInstrumentor()

            mock_result = MagicMock()
            mock_result.messages = [MagicMock(), MagicMock()]
            mock_result.stop_reason = "MaxMessageTermination"

            attrs = instrumentor._extract_response_attributes(mock_result)

            self.assertEqual(attrs["autogen_agentchat.result.message_count"], 2)
            self.assertEqual(attrs["autogen_agentchat.result.stop_reason"], "MaxMessageTermination")

    @patch("genai_otel.instrumentors.autogen_agentchat_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        with patch.dict(
            "sys.modules",
            {
                "autogen_agentchat": MagicMock(),
                "autogen_agentchat.base": MagicMock(),
                "autogen_agentchat.teams": MagicMock(),
            },
        ):
            instrumentor = AutoGenAgentChatInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            # Make create_span_wrapper raise to trigger the except block
            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=RuntimeError("Test error")
            ):
                instrumentor.instrument(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.autogen_agentchat_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that instrument raises exceptions when fail_on_error is True."""
        with patch.dict(
            "sys.modules",
            {
                "autogen_agentchat": MagicMock(),
                "autogen_agentchat.base": MagicMock(),
                "autogen_agentchat.teams": MagicMock(),
            },
        ):
            instrumentor = AutoGenAgentChatInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=RuntimeError("Test error")
            ):
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)


if __name__ == "__main__":
    unittest.main()
