import types
import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.autogen_instrumentor import AutoGenInstrumentor
from genai_otel.instrumentors.base import BaseInstrumentor

USAGE = {"prompt_tokens": 27, "completion_tokens": 2, "total_tokens": 29}


def _fake_autogen_module():
    """A minimal ``autogen`` module shaped like pyautogen 0.2 and AG2 0.14.

    What matters is how a group chat runs: ``GroupChatManager`` has no ``run()`` of
    its own in pyautogen 0.2 (in AG2 it inherits a generic, threaded one); it
    registers ``GroupChatManager.run_chat`` as a reply function when it is built,
    and ``initiate_chat`` reaches it through ``generate_reply``.
    """

    class ChatResult:
        def __init__(self, chat_history, cost=None, summary=""):
            self.chat_history = chat_history
            self.cost = cost
            self.summary = summary

    class ConversableAgent:
        def __init__(self, name, llm_config=None, reply="ok", llm_call=None):
            self.name = name
            self.llm_config = llm_config
            self._reply = reply
            self._llm_call = llm_call
            self._reply_funcs = []

        def register_reply(self, reply_func, config=None):
            self._reply_funcs.append((reply_func, config))

        def generate_reply(self, messages, sender):
            for reply_func, config in self._reply_funcs:
                final, reply = reply_func(self, messages=messages, sender=sender, config=config)
                if final:
                    return reply
            if self._llm_call is not None:
                self._llm_call()
            return self._reply

        def initiate_chat(self, recipient, message=None, **kwargs):
            reply = recipient.generate_reply([{"content": message}], self)
            cost = None
            if recipient._llm_call is not None:
                usage = {"total_cost": 5.25e-06, "gpt-4o-mini": dict(USAGE, cost=5.25e-06)}
                cost = {
                    "usage_including_cached_inference": usage,
                    "usage_excluding_cached_inference": usage,
                }
            history = [
                {"content": message, "role": "assistant", "name": self.name},
                {"content": reply, "role": "user", "name": recipient.name},
            ]
            return ChatResult(chat_history=history, cost=cost, summary=reply)

    class GroupChat:
        def __init__(self, agents, speaker_selection_method="round_robin", max_round=3):
            self.agents = agents
            self.speaker_selection_method = speaker_selection_method
            self.max_round = max_round

        def select_speaker(self, last_speaker, selector):
            return self.agents[(self.agents.index(last_speaker) + 1) % len(self.agents)]

    class GroupChatManager(ConversableAgent):
        def __init__(self, groupchat, name="chat_manager"):
            super().__init__(name)
            self.groupchat = groupchat
            self.register_reply(GroupChatManager.run_chat, config=groupchat)

        def run_chat(self, messages=None, sender=None, config=None):
            speaker = sender
            for _ in range(config.max_round - 1):
                speaker = config.select_speaker(speaker, self)
            return True, "last word from " + speaker.name

    module = types.ModuleType("autogen")
    module.ChatResult = ChatResult
    module.ConversableAgent = ConversableAgent
    module.GroupChat = GroupChat
    module.GroupChatManager = GroupChatManager
    return module


class _FakeLLMInstrumentor(BaseInstrumentor):
    """Stands in for a provider instrumentor recording usage on its own span."""

    def instrument(self, config):
        self.config = config
        self._instrumented = True

    def _extract_usage(self, result):
        return dict(USAGE)


class TestAgainstRealPackageLayout(unittest.TestCase):
    """Behaviour against modules shaped like the packages users install today."""

    def _instrument(self, module):
        patcher = patch.dict("sys.modules", {"autogen": module})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        instrumentor = AutoGenInstrumentor()
        instrumentor.tracer = self.provider.get_tracer("test")
        instrumentor.instrument(OTelConfig())
        return instrumentor

    def _spans(self, name):
        return [s for s in self.exporter.get_finished_spans() if s.name == name]

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_empty_pyautogen_proxy_is_not_treated_as_autogen(self, mock_logger):
        """pyautogen 0.10+ is an empty proxy package for autogen-agentchat: it imports,
        but there is no ConversableAgent in it and nothing to wrap."""
        proxy = types.ModuleType("pyautogen")
        with patch.dict("sys.modules", {"autogen": None, "pyautogen": proxy}):
            instrumentor = AutoGenInstrumentor()
            instrumentor.instrument(OTelConfig())

        self.assertFalse(instrumentor._autogen_available)
        self.assertFalse(instrumentor._instrumented)
        mock_logger.info.assert_not_called()

    def test_group_chat_run_span_covers_the_group_chat(self):
        """The group chat runs in GroupChatManager.run_chat, not in a run() method."""
        module = _fake_autogen_module()
        self._instrument(module)
        user = module.ConversableAgent("user")
        writer = module.ConversableAgent("writer")
        manager = module.GroupChatManager(module.GroupChat([user, writer]))

        result = user.initiate_chat(manager, message="slogan please")

        self.assertEqual(result.summary, "last word from user")
        (chat_span,) = self._spans("autogen.initiate_chat")
        (run_span,) = self._spans("autogen.group_chat.run")
        selections = self._spans("autogen.group_chat.select_speaker")
        self.assertEqual(run_span.parent.span_id, chat_span.context.span_id)
        self.assertEqual(run_span.attributes["autogen.manager.name"], "chat_manager")
        self.assertEqual(run_span.attributes["autogen.group_chat.agent_count"], 2)
        self.assertEqual(len(selections), 2)
        for selection in selections:
            self.assertEqual(selection.parent.span_id, run_span.context.span_id)

    def test_threaded_run_is_not_reported_as_the_group_chat(self):
        """AG2's GroupChatManager.run() only starts a background thread and returns."""
        module = _fake_autogen_module()
        module.GroupChatManager.run = lambda self, **kwargs: "RunResponse"
        self._instrument(module)
        manager = module.GroupChatManager(module.GroupChat([module.ConversableAgent("user")]))

        self.assertEqual(manager.run(message="slogan please"), "RunResponse")

        self.assertEqual(self._spans("autogen.group_chat.run"), [])

    def test_model_read_from_llm_config_object(self):
        """AG2 turns llm_config into an LLMConfig object: not a dict, with
        attribute-style entries in ``config_list``."""

        class LLMConfig:
            def __init__(self, *entries):
                self.config_list = list(entries)

            def get(self, key, default=None):
                return getattr(self, key, default)

        module = _fake_autogen_module()
        self._instrument(module)
        entry = types.SimpleNamespace(model="gpt-4o-mini", api_type="openai")
        assistant = module.ConversableAgent("assistant", llm_config=LLMConfig(entry))
        user = module.ConversableAgent("user", llm_config=False)

        user.initiate_chat(assistant, message="hello")

        (chat_span,) = self._spans("autogen.initiate_chat")
        self.assertEqual(chat_span.attributes["gen_ai.request.model"], "gpt-4o-mini")

    def _chat_with_llm_call(self, llm_call):
        module = _fake_autogen_module()
        self._instrument(module)
        assistant = module.ConversableAgent(
            "assistant", llm_config={"model": "gpt-4o-mini"}, llm_call=llm_call
        )
        module.ConversableAgent("user").initiate_chat(assistant, message="hello")
        (chat_span,) = self._spans("autogen.initiate_chat")
        return chat_span

    def test_cost_left_to_the_llm_span_that_recorded_it(self):
        """ChatResult.cost repeats what the LLM spans beneath already carry; writing
        it on the conversation span as well doubles the cost of the trace."""
        llm = _FakeLLMInstrumentor()
        llm.instrument(OTelConfig())
        llm.cost_calculator = MagicMock()
        llm.cost_calculator.calculate_cost.return_value = 5.25e-06
        llm.cost_calculator.calculate_granular_cost.return_value = {"total": 5.25e-06}
        traced_call = llm.create_span_wrapper(
            span_name="openai.chat.completion",
            extract_attributes=lambda instance, args, kwargs: {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
            },
        )(lambda: "completion")

        def llm_call():
            # Bound late: the tracer under test exists only once _instrument() ran.
            llm.tracer = self.provider.get_tracer("test")
            return traced_call()

        chat_span = self._chat_with_llm_call(llm_call)

        (llm_span,) = self._spans("openai.chat.completion")
        self.assertEqual(llm_span.parent.span_id, chat_span.context.span_id)
        self.assertEqual(llm_span.attributes["gen_ai.usage.input_tokens"], 27)
        self.assertEqual(chat_span.attributes["gen_ai.usage.recorded_by"], "descendant")
        self.assertNotIn("gen_ai.usage.cost.total", chat_span.attributes)
        self.assertNotIn("gen_ai.usage.input_tokens", chat_span.attributes)

    def test_cost_kept_when_no_llm_span_recorded_it(self):
        """With an LLM client nobody instruments, the conversation span is the only
        place the usage can go."""
        chat_span = self._chat_with_llm_call(lambda: "completion")

        self.assertEqual(chat_span.attributes["gen_ai.usage.input_tokens"], 27)
        self.assertGreater(chat_span.attributes["gen_ai.usage.cost.total"], 0)
        self.assertNotIn("gen_ai.usage.recorded_by", chat_span.attributes)


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

        original_initiate_chat = MockConversableAgent.initiate_chat

        # Create mock autogen module
        mock_autogen = MagicMock()
        mock_autogen.ConversableAgent = MockConversableAgent

        with patch.dict(
            "sys.modules",
            {
                "autogen": mock_autogen,
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
            # Verify initiate_chat was replaced (no longer the original)
            self.assertIsNot(MockConversableAgent.initiate_chat, original_initiate_chat)

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

        original_select_speaker = MockGroupChat.select_speaker

        # Create mock autogen module
        mock_autogen = MagicMock()
        mock_autogen.GroupChat = MockGroupChat

        with patch.dict(
            "sys.modules",
            {
                "autogen": mock_autogen,
            },
        ):
            instrumentor = AutoGenInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("AutoGen instrumentation enabled")
            # Verify select_speaker was replaced
            self.assertIsNot(MockGroupChat.select_speaker, original_select_speaker)

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

        original_run = MockGroupChatManager.run

        # Create mock autogen module
        mock_autogen = MagicMock()
        mock_autogen.GroupChatManager = MockGroupChatManager

        with patch.dict(
            "sys.modules",
            {
                "autogen": mock_autogen,
            },
        ):
            instrumentor = AutoGenInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("AutoGen instrumentation enabled")
            # Verify run was replaced
            self.assertIsNot(MockGroupChatManager.run, original_run)

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
        """Test that _extract_usage extracts from ChatResult.cost CostDict."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock result with AutoGen CostDict format
            result = MagicMock()
            result.cost = {
                "usage_including_cached_inference": {
                    "gpt-4": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "cost": 0.005,
                    }
                },
                "usage_excluding_cached_inference": {},
            }

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
        """Test extraction of response attributes with AutoGen CostDict."""
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            instrumentor = AutoGenInstrumentor()

            # Create mock result with AutoGen CostDict format
            result = MagicMock()
            result.chat_history = []
            result.cost = {
                "usage_including_cached_inference": {
                    "gpt-4": {
                        "prompt_tokens": 200,
                        "completion_tokens": 100,
                        "cost": 0.03,
                    },
                    "gpt-3.5-turbo": {
                        "prompt_tokens": 150,
                        "completion_tokens": 80,
                        "cost": 0.02,
                    },
                },
                "usage_excluding_cached_inference": {},
            }

            attrs = instrumentor._extract_response_attributes(result)

            # Assert total cost is aggregated
            self.assertAlmostEqual(attrs["gen_ai.usage.cost.total"], 0.05)

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

    @patch("genai_otel.instrumentors.autogen_instrumentor.logger")
    def test_instrument_idempotent_guard(self, mock_logger):
        """Repeated instrument() must return early via the idempotency guard."""
        instrumentor = AutoGenInstrumentor()
        instrumentor._autogen_available = True
        instrumentor._instrumented = True
        instrumentor.instrument(OTelConfig())
        mock_logger.debug.assert_any_call(
            "AutoGen already instrumented, skipping repeat instrument()"
        )

    def test_cap_content_respects_config(self):
        """_cap_content honours content_max_length (default, small cap, unlimited)."""
        instrumentor = AutoGenInstrumentor()
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 200)
        cfg = OTelConfig()
        cfg.content_max_length = 3
        instrumentor.config = cfg
        self.assertEqual(instrumentor._cap_content("abcdef"), "abc")
        cfg.content_max_length = 0
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 300)


if __name__ == "__main__":
    unittest.main()
