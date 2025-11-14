import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.autogen_instrumentor import AutoGenInstrumentor


class TestAutoGenInstrumentor(unittest.TestCase):
    """Tests for AutoGenInstrumentor"""

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_init_with_autogen_available(self, mock_logger):
        """Test that __init__ detects AutoGen availability."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            self.assertTrue(instrumentor._autogen_available)
            mock_logger.debug.assert_called_with(
                "AutoGen library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_init_with_pyautogen_available(self, mock_logger):
        """Test that __init__ detects pyautogen (legacy package name)."""
        # Simulate autogen not available but pyautogen is
        with patch.dict("sys.modules", {"autogen": None, "pyautogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            self.assertTrue(instrumentor._autogen_available)
            mock_logger.debug.assert_called_with(
                "AutoGen library (pyautogen) detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_init_with_autogen_not_available(self, mock_logger):
        """Test that __init__ handles missing AutoGen gracefully."""
        with patch.dict("sys.modules", {"autogen": None, "pyautogen": None}):
            instrumentor = AutoGenInstrumentor()

            self.assertFalse(instrumentor._autogen_available)
            mock_logger.debug.assert_called_with(
                "AutoGen library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_instrument_when_autogen_not_available(self, mock_logger):
        """Test that instrument skips when AutoGen is not available."""
        with patch.dict("sys.modules", {"autogen": None, "pyautogen": None}):
            instrumentor = AutoGenInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping AutoGen instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_instrument_with_conversable_agent(self, mock_logger):
        """Test that instrument wraps ConversableAgent.initiate_chat."""

        # Create mock ConversableAgent class
        class MockConversableAgent:
            def initiate_chat(self, recipient, message=None, **kwargs):
                return {"chat_history": [{"role": "assistant", "content": "response"}]}

        # Create mock autogen module
        mock_autogen = MagicMock()
        mock_autogen.ConversableAgent = MockConversableAgent

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "autogen": mock_autogen,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = AutoGenInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("AutoGen instrumentation enabled")
            # Verify FunctionWrapper was called
            self.assertTrue(mock_wrapt.FunctionWrapper.called)

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_instrument_with_group_chat(self, mock_logger):
        """Test that instrument wraps GroupChat.select_speaker."""

        # Create mock GroupChat class
        class MockGroupChat:
            def __init__(self):
                self.agents = []
                self.speaker_selection_method = "round_robin"

            def select_speaker(self, last_speaker, selector):
                return self.agents[0] if self.agents else None

        # Create mock autogen module
        mock_autogen = MagicMock()
        mock_autogen.GroupChat = MockGroupChat

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "autogen": mock_autogen,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = AutoGenInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("AutoGen instrumentation enabled")
            # Verify FunctionWrapper was called
            self.assertTrue(mock_wrapt.FunctionWrapper.called)

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_instrument_with_group_chat_manager(self, mock_logger):
        """Test that instrument wraps GroupChatManager.run."""

        # Create mock GroupChatManager class
        class MockGroupChatManager:
            def __init__(self):
                self.name = "test_manager"
                self.groupchat = MagicMock()

            def run(self):
                return {"result": "completed"}

        # Create mock autogen module
        mock_autogen = MagicMock()
        mock_autogen.GroupChatManager = MockGroupChatManager

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "autogen": mock_autogen,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = AutoGenInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("AutoGen instrumentation enabled")
            # Verify FunctionWrapper was called
            self.assertTrue(mock_wrapt.FunctionWrapper.called)

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        # Create mock that raises
        mock_autogen = MagicMock()
        type(mock_autogen).ConversableAgent = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"autogen": mock_autogen, "wrapt": MagicMock()}):
            instrumentor = AutoGenInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            # Should not raise
            instrumentor.instrument(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that exceptions are raised when fail_on_error is True."""
        # Create mock that raises
        mock_autogen = MagicMock()
        type(mock_autogen).ConversableAgent = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"autogen": mock_autogen, "wrapt": MagicMock()}):
            instrumentor = AutoGenInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            # Should raise
            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Access failed")

    def test_extract_chat_attributes_basic(self):
        """Test extraction of basic chat attributes."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.name = "sender_agent"

            # Create mock recipient
            mock_recipient = MagicMock()
            mock_recipient.name = "recipient_agent"

            # Message
            message = "Hello, let's discuss the project."

            args = (mock_recipient, message)
            kwargs = {}

            attrs = instrumentor._extract_chat_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "autogen")
            self.assertEqual(attrs["gen_ai.operation.name"], "conversation.initiate")
            self.assertEqual(attrs["autogen.agent.name"], "sender_agent")
            self.assertEqual(attrs["autogen.conversation.sender"], "sender_agent")
            self.assertEqual(attrs["autogen.conversation.recipient"], "recipient_agent")
            self.assertEqual(attrs["autogen.message"], message)
            self.assertEqual(attrs["autogen.message.type"], "string")

    def test_extract_chat_attributes_with_dict_message(self):
        """Test extraction of chat attributes with dict message."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.name = "agent1"

            # Create mock recipient
            mock_recipient = MagicMock()
            mock_recipient.name = "agent2"

            # Dict message
            message = {"content": "This is a structured message", "metadata": "test"}

            args = (mock_recipient, message)
            kwargs = {"max_turns": 5, "silent": True, "clear_history": False}

            attrs = instrumentor._extract_chat_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["autogen.message"], "This is a structured message")
            self.assertEqual(attrs["autogen.message.type"], "dict")
            self.assertEqual(attrs["autogen.conversation.max_turns"], 5)
            self.assertTrue(attrs["autogen.conversation.silent"])
            self.assertFalse(attrs["autogen.conversation.clear_history"])

    def test_extract_chat_attributes_with_kwargs(self):
        """Test extraction of chat attributes with kwargs."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock agent instance
            mock_agent = MagicMock()
            mock_agent.name = "test_agent"

            # Create mock recipient
            mock_recipient = MagicMock()
            mock_recipient.name = "target_agent"

            args = ()
            kwargs = {
                "recipient": mock_recipient,
                "message": "Testing kwargs",
                "max_turns": 10,
            }

            attrs = instrumentor._extract_chat_attributes(mock_agent, args, kwargs)

            # Assert
            self.assertEqual(attrs["autogen.conversation.recipient"], "target_agent")
            self.assertEqual(attrs["autogen.message"], "Testing kwargs")
            self.assertEqual(attrs["autogen.conversation.max_turns"], 10)

    def test_extract_group_chat_attributes(self):
        """Test extraction of group chat attributes."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock GroupChat instance
            mock_groupchat = MagicMock()
            mock_agent1 = MagicMock()
            mock_agent1.name = "agent1"
            mock_agent2 = MagicMock()
            mock_agent2.name = "agent2"
            mock_agent3 = MagicMock()
            mock_agent3.name = "agent3"

            mock_groupchat.agents = [mock_agent1, mock_agent2, mock_agent3]
            mock_groupchat.speaker_selection_method = "auto"
            mock_groupchat.max_round = 20

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_group_chat_attributes(mock_groupchat, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "autogen")
            self.assertEqual(attrs["gen_ai.operation.name"], "group_chat.select_speaker")
            self.assertEqual(attrs["autogen.group_chat.agent_count"], 3)
            self.assertIn("agent1", attrs["autogen.group_chat.agents"])
            self.assertIn("agent2", attrs["autogen.group_chat.agents"])
            self.assertIn("agent3", attrs["autogen.group_chat.agents"])
            self.assertEqual(attrs["autogen.group_chat.selection_mode"], "auto")
            self.assertEqual(attrs["autogen.group_chat.max_round"], 20)

    def test_extract_group_chat_manager_attributes(self):
        """Test extraction of group chat manager attributes."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock GroupChatManager instance
            mock_manager = MagicMock()
            mock_manager.name = "group_manager"

            # Create mock groupchat
            mock_groupchat = MagicMock()
            mock_agent1 = MagicMock()
            mock_agent1.name = "agent1"
            mock_agent2 = MagicMock()
            mock_agent2.name = "agent2"

            mock_groupchat.agents = [mock_agent1, mock_agent2]
            mock_groupchat.speaker_selection_method = "round_robin"
            mock_manager.groupchat = mock_groupchat

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_group_chat_manager_attributes(mock_manager, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "autogen")
            self.assertEqual(attrs["gen_ai.operation.name"], "group_chat.run")
            self.assertEqual(attrs["autogen.manager.name"], "group_manager")
            self.assertEqual(attrs["autogen.group_chat.agent_count"], 2)
            self.assertIn("agent1", attrs["autogen.group_chat.agents"])
            self.assertEqual(attrs["autogen.group_chat.selection_mode"], "round_robin")

    def test_extract_usage_with_usage_attribute(self):
        """Test that _extract_usage extracts from usage attribute."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock result with usage
            result = MagicMock()
            result.usage.prompt_tokens = 100
            result.usage.completion_tokens = 50
            result.usage.total_tokens = 150

            usage = instrumentor._extract_usage(result)

            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 100)
            self.assertEqual(usage["completion_tokens"], 50)
            self.assertEqual(usage["total_tokens"], 150)

    def test_extract_usage_without_usage_attribute(self):
        """Test that _extract_usage returns None when no usage."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock result without usage
            result = MagicMock(spec=[])
            if hasattr(result, "usage"):
                delattr(result, "usage")

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    def test_extract_response_attributes_with_chat_history(self):
        """Test extraction of response attributes with chat history."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock result
            result = MagicMock()
            result.chat_history = [
                {"role": "user", "content": "First message", "name": "user_agent"},
                {"role": "assistant", "content": "Second message", "name": "assistant_agent"},
                {"role": "assistant", "content": "Final response", "name": "assistant_agent"},
            ]
            result.summary = "Conversation completed successfully"

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["autogen.conversation.messages"], 3)
            self.assertEqual(attrs["autogen.conversation.last_message"], "Final response")
            self.assertEqual(attrs["autogen.conversation.last_role"], "assistant")
            self.assertEqual(attrs["autogen.conversation.last_speaker"], "assistant_agent")
            self.assertEqual(
                attrs["autogen.conversation.summary"], "Conversation completed successfully"
            )

    def test_extract_response_attributes_with_cost(self):
        """Test extraction of response attributes with cost."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock result with cost
            result = MagicMock()
            result.chat_history = []
            result.cost = {"total": 0.05, "gpt-4": 0.03, "gpt-3.5": 0.02}

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["autogen.cost.total"], 0.05)
            self.assertEqual(attrs["autogen.cost.gpt-4"], 0.03)
            self.assertEqual(attrs["autogen.cost.gpt-3.5"], 0.02)

    def test_extract_finish_reason_with_chat_history(self):
        """Test extraction of finish reason with chat history."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock result
            result = MagicMock()
            result.chat_history = [{"content": "message"}]

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_finish_reason_without_chat_history(self):
        """Test extraction of finish reason without chat history."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock result without chat_history
            result = MagicMock(spec=[])
            if hasattr(result, "chat_history"):
                delattr(result, "chat_history")

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertIsNone(finish_reason)


if __name__ == "__main__":
    unittest.main()
