import asyncio
import types
import unittest
from unittest.mock import MagicMock, patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.autogen_agentchat_instrumentor import AutoGenAgentChatInstrumentor

# Stands in for the tracer autogen-core uses for its own invoke_agent spans.
_NATIVE_TRACER = TracerProvider().get_tracer("autogen-core")


def _fake_agentchat_modules():
    """Fake ``autogen_agentchat`` modules laid out like the real package.

    The layout is what matters, and it has been the same from 0.4.0 to 0.7.5:
    ``base.ChatAgent`` only declares the API, ``agents.BaseChatAgent`` implements
    ``run()`` / ``run_stream()``, each concrete agent implements ``on_messages()``,
    and ``BaseGroupChat.run()`` is built on its own ``run_stream()``.
    """

    class TaskResult:
        def __init__(self, messages, stop_reason=None):
            self.messages = messages
            self.stop_reason = stop_reason

    class Response:
        def __init__(self, chat_message):
            self.chat_message = chat_message

    class ChatAgent:
        async def run(self, *, task=None): ...

        def run_stream(self, *, task=None): ...

        async def on_messages(self, messages, cancellation_token): ...

        on_messages.__isabstractmethod__ = True

    class BaseChatAgent(ChatAgent):
        def __init__(self, name):
            self._name = name

        @property
        def name(self):
            return self._name

        async def on_messages(self, messages, cancellation_token): ...

        on_messages.__isabstractmethod__ = True

        async def run(self, *, task=None):
            response = await self.on_messages([task], None)
            return TaskResult(messages=[task, response.chat_message])

        async def run_stream(self, *, task=None):
            self.stream_closed = False
            try:
                # autogen-core opens its own span around the whole stream.
                with _NATIVE_TRACER.start_as_current_span("invoke_agent " + self._name):
                    yield task
                    response = await self.on_messages([task], None)
                    yield TaskResult(messages=[task, response.chat_message])
            finally:
                self.stream_closed = True

    class AssistantAgent(BaseChatAgent):
        def __init__(self, name, model_client=None):
            super().__init__(name)
            self._model_client = model_client

        async def on_messages(self, messages, cancellation_token):
            await asyncio.sleep(0)
            # What an LLM call made by the agent would see as its parent.
            self.span_during_call = trace.get_current_span().get_span_context().span_id
            return Response(chat_message="reply from " + self._name)

    class BaseGroupChat:
        def __init__(self, participants):
            self._participants = participants
            self._termination_condition = None

        async def run(self, *, task=None):
            result = None
            async for message in self.run_stream(task=task):
                if isinstance(message, TaskResult):
                    result = message
            return result

        async def run_stream(self, *, task=None):
            yield task
            await asyncio.sleep(0)
            yield TaskResult(messages=[task], stop_reason="Maximum number of messages reached")

    base = types.ModuleType("autogen_agentchat.base")
    base.ChatAgent = ChatAgent
    base.TaskResult = TaskResult
    base.Response = Response
    agents = types.ModuleType("autogen_agentchat.agents")
    agents.BaseChatAgent = BaseChatAgent
    agents.AssistantAgent = AssistantAgent
    teams = types.ModuleType("autogen_agentchat.teams")
    teams.BaseGroupChat = BaseGroupChat
    package = types.ModuleType("autogen_agentchat")
    package.base, package.agents, package.teams = base, agents, teams
    modules = {
        "autogen_agentchat": package,
        "autogen_agentchat.base": base,
        "autogen_agentchat.agents": agents,
        "autogen_agentchat.teams": teams,
    }
    return modules, types.SimpleNamespace(
        AssistantAgent=AssistantAgent, BaseGroupChat=BaseGroupChat, TaskResult=TaskResult
    )


class TestAgainstRealPackageLayout(unittest.TestCase):
    """The instrumentor must wrap the classes real agents and teams actually run on."""

    def setUp(self):
        modules, self.lib = _fake_agentchat_modules()
        patcher = patch.dict("sys.modules", modules)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.instrumentor = AutoGenAgentChatInstrumentor()
        self.instrumentor.tracer = provider.get_tracer("test")
        self.instrumentor.instrument(OTelConfig())

    def _spans(self, name):
        return [s for s in self.exporter.get_finished_spans() if s.name == name]

    def test_agent_run_is_traced(self):
        """agent.run() runs BaseChatAgent.run(), not the ChatAgent protocol stub."""
        agent = self.lib.AssistantAgent("assistant")

        result = asyncio.run(agent.run(task="hello"))

        self.assertEqual(result.messages, ["hello", "reply from assistant"])
        (run_span,) = self._spans("autogen_agentchat.agent.run")
        self.assertEqual(run_span.attributes["autogen_agentchat.agent.name"], "assistant")
        self.assertEqual(run_span.attributes["autogen_agentchat.task"], "hello")
        self.assertEqual(run_span.attributes["autogen_agentchat.result.message_count"], 2)
        self.assertEqual(run_span.status.status_code, trace.StatusCode.OK)

    def test_agent_on_messages_is_traced_under_run(self):
        """on_messages() is implemented per agent class; it nests under run()."""
        agent = self.lib.AssistantAgent("assistant")

        asyncio.run(agent.run(task="hello"))

        (run_span,) = self._spans("autogen_agentchat.agent.run")
        (messages_span,) = self._spans("autogen_agentchat.agent.on_messages")
        self.assertEqual(messages_span.parent.span_id, run_span.context.span_id)
        self.assertEqual(messages_span.attributes["autogen_agentchat.message_count"], 1)
        self.assertEqual(agent.span_during_call, messages_span.context.span_id)

    def test_agent_on_messages_called_directly_is_traced(self):
        agent = self.lib.AssistantAgent("assistant")

        response = asyncio.run(agent.on_messages(["hello"], None))

        self.assertEqual(response.chat_message, "reply from assistant")
        (messages_span,) = self._spans("autogen_agentchat.agent.on_messages")
        self.assertIsNone(messages_span.parent)

    def test_agent_run_stream_is_traced_until_exhausted(self):
        agent = self.lib.AssistantAgent("assistant")

        async def consume():
            items = []
            async for item in agent.run_stream(task="hello"):
                # The span must still be open while the stream is being consumed.
                self.assertEqual(self._spans("autogen_agentchat.agent.run_stream"), [])
                items.append(item)
            return items

        items = asyncio.run(consume())

        self.assertEqual(len(items), 2)
        self.assertIsInstance(items[-1], self.lib.TaskResult)
        (stream_span,) = self._spans("autogen_agentchat.agent.run_stream")
        self.assertEqual(stream_span.attributes["gen_ai.agent.name"], "assistant")
        self.assertEqual(stream_span.status.status_code, trace.StatusCode.OK)

    def test_stream_closed_early_ends_the_span_and_restores_the_context(self):
        team = self.lib.BaseGroupChat([self.lib.AssistantAgent("writer")])

        async def stop_after_first_item():
            stream = team.run_stream(task="hello")
            async for _ in stream:
                break
            await stream.aclose()
            return trace.get_current_span().get_span_context().is_valid

        span_left_current = asyncio.run(stop_after_first_item())

        self.assertEqual(len(self._spans("autogen_agentchat.team.run_stream")), 1)
        self.assertFalse(span_left_current, "the stream's span was left as the current span")

    def test_closing_a_stream_early_closes_the_agent_generator_in_the_same_call(self):
        """Without this the agent's own generator was left to the garbage collector:
        its cleanup ran later, in another task, and the span AutoGen itself opened
        around the stream stayed current for the caller."""
        agent = self.lib.AssistantAgent("assistant")

        async def stop_after_first_item():
            stream = agent.run_stream(task="hello")
            async for _ in stream:
                break
            await stream.aclose()
            return agent.stream_closed, trace.get_current_span().get_span_context().is_valid

        closed, span_left_current = asyncio.run(stop_after_first_item())

        self.assertTrue(closed, "the agent's generator was still open after aclose()")
        self.assertFalse(span_left_current, "a span was left as the current span")
        self.assertEqual(len(self._spans("autogen_agentchat.agent.run_stream")), 1)

    def test_agent_runs_leave_no_budget_frame_open(self):
        """Every AgentChat entry point is async, and the base wrapper only closes an
        agent budget frame on its synchronous path."""
        from genai_otel import agent_budget

        agent = self.lib.AssistantAgent("assistant")

        async def run_repeatedly():
            for _ in range(3):
                await agent.run(task="hello")
                async for _ in agent.run_stream(task="hello"):
                    pass
            return agent_budget._BUDGET_STACK.get()

        self.assertEqual(asyncio.run(run_repeatedly()), ())

    def test_team_run_emits_one_span(self):
        """BaseGroupChat.run() drives run_stream() itself; that is one call, one span."""
        team = self.lib.BaseGroupChat([self.lib.AssistantAgent("writer")])

        result = asyncio.run(team.run(task="hello"))

        self.assertEqual(result.stop_reason, "Maximum number of messages reached")
        (run_span,) = self._spans("autogen_agentchat.team.run")
        self.assertEqual(self._spans("autogen_agentchat.team.run_stream"), [])
        self.assertEqual(run_span.attributes["autogen_agentchat.result.message_count"], 1)
        self.assertEqual(
            run_span.attributes["autogen_agentchat.result.stop_reason"],
            "Maximum number of messages reached",
        )

    def test_team_run_stream_called_directly_is_still_traced(self):
        team = self.lib.BaseGroupChat([self.lib.AssistantAgent("writer")])

        async def consume():
            await team.run(task="first")  # must not leave run_stream() untraced afterwards
            return [item async for item in team.run_stream(task="second")]

        items = asyncio.run(consume())

        self.assertEqual(len(items), 2)
        (stream_span,) = self._spans("autogen_agentchat.team.run_stream")
        self.assertEqual(stream_span.attributes["autogen_agentchat.task"], "second")
        self.assertEqual(len(self._spans("autogen_agentchat.team.run")), 1)

    def test_concurrent_team_runs_are_separate_traces(self):
        """asyncio.gather() calls every run() before any of them starts executing."""
        first = self.lib.BaseGroupChat([self.lib.AssistantAgent("a")])
        second = self.lib.BaseGroupChat([self.lib.AssistantAgent("b")])

        async def run_both():
            await asyncio.gather(first.run(task="one"), second.run(task="two"))
            return trace.get_current_span().get_span_context().is_valid

        span_left_current = asyncio.run(run_both())

        spans = self._spans("autogen_agentchat.team.run")
        self.assertEqual(len(spans), 2)
        self.assertEqual([s.parent for s in spans], [None, None])
        self.assertEqual(len({s.context.trace_id for s in spans}), 2)
        self.assertFalse(span_left_current, "a finished span was left as the current span")

    def test_concurrent_agent_runs_keep_their_own_children(self):
        first = self.lib.AssistantAgent("a")
        second = self.lib.AssistantAgent("b")

        async def run_both():
            await asyncio.gather(first.run(task="one"), second.run(task="two"))
            return trace.get_current_span().get_span_context().is_valid

        span_left_current = asyncio.run(run_both())

        run_spans = {
            s.attributes["gen_ai.agent.name"]: s for s in self._spans("autogen_agentchat.agent.run")
        }
        self.assertEqual(sorted(run_spans), ["a", "b"])
        self.assertIsNone(run_spans["a"].parent)
        self.assertIsNone(run_spans["b"].parent)
        for messages_span in self._spans("autogen_agentchat.agent.on_messages"):
            owner = run_spans[messages_span.attributes["gen_ai.agent.name"]]
            self.assertEqual(messages_span.parent.span_id, owner.context.span_id)
        self.assertFalse(span_left_current, "a finished span was left as the current span")

    def test_wrapped_methods_keep_their_async_nature(self):
        """Frameworks and users introspect these; the wrappers must not hide it."""
        import inspect

        agent = self.lib.AssistantAgent("assistant")
        team = self.lib.BaseGroupChat([agent])

        self.assertTrue(inspect.iscoroutinefunction(agent.run))
        self.assertTrue(inspect.isasyncgenfunction(agent.run_stream))
        self.assertTrue(inspect.iscoroutinefunction(agent.on_messages))
        self.assertTrue(inspect.iscoroutinefunction(team.run))
        self.assertTrue(inspect.isasyncgenfunction(team.run_stream))

    def test_agent_run_error_ends_the_span_with_error_status(self):
        agent = self.lib.AssistantAgent("assistant")

        async def boom(messages, cancellation_token):
            raise RuntimeError("model backend exploded")

        agent.on_messages = boom

        with self.assertRaises(RuntimeError):
            asyncio.run(agent.run(task="hello"))

        (run_span,) = self._spans("autogen_agentchat.agent.run")
        self.assertEqual(run_span.status.status_code, trace.StatusCode.ERROR)

    def test_model_read_from_private_model_client(self):
        """AssistantAgent keeps its client in ``_model_client``; clients keep the model
        name in ``_create_args`` (autogen-ext OpenAI, Anthropic, Ollama, Azure)."""
        client = types.SimpleNamespace(_create_args={"model": "gpt-4o-mini", "max_tokens": 40})
        agent = self.lib.AssistantAgent("assistant", model_client=client)

        asyncio.run(agent.run(task="hello"))

        (run_span,) = self._spans("autogen_agentchat.agent.run")
        self.assertEqual(run_span.attributes["gen_ai.request.model"], "gpt-4o-mini")


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

    @patch("genai_otel.instrumentors.autogen_agentchat_instrumentor.logger")
    def test_instrument_idempotent_guard(self, mock_logger):
        """Repeated instrument() must return early via the idempotency guard."""
        instrumentor = AutoGenAgentChatInstrumentor()
        instrumentor._agentchat_available = True
        instrumentor._instrumented = True
        instrumentor.instrument(OTelConfig())
        mock_logger.debug.assert_any_call(
            "AutoGen AgentChat already instrumented, skipping repeat instrument()"
        )

    def test_cap_content_respects_config(self):
        """_cap_content honours content_max_length (default, small cap, unlimited)."""
        instrumentor = AutoGenAgentChatInstrumentor()
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 200)
        cfg = OTelConfig()
        cfg.content_max_length = 3
        instrumentor.config = cfg
        self.assertEqual(instrumentor._cap_content("abcdef"), "abc")
        cfg.content_max_length = 0
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 300)


if __name__ == "__main__":
    unittest.main()
