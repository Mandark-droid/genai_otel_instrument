import asyncio
import builtins
import contextvars
import copy
import logging
import os
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import StatusCode

from genai_otel.config import OTelConfig
from genai_otel.instrumentors import crewai_instrumentor as crewai_module
from genai_otel.instrumentors.crewai_instrumentor import CrewAIInstrumentor


class TestCrewAIInstrumentor(unittest.TestCase):
    """Tests for CrewAIInstrumentor"""

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_init_with_crewai_available(self, mock_logger):
        """Test that __init__ detects CrewAI availability."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            self.assertTrue(instrumentor._crewai_available)
            mock_logger.debug.assert_called_with(
                "CrewAI library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_init_with_crewai_not_available(self, mock_logger):
        """Test that __init__ handles missing CrewAI gracefully."""
        with patch.dict("sys.modules", {"crewai": None}):
            instrumentor = CrewAIInstrumentor()

            self.assertFalse(instrumentor._crewai_available)
            mock_logger.debug.assert_called_with(
                "CrewAI library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_instrument_when_crewai_not_available(self, mock_logger):
        """Test that instrument skips when CrewAI is not available."""
        with patch.dict("sys.modules", {"crewai": None}):
            instrumentor = CrewAIInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping CrewAI instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_instrument_with_crewai_available(self, mock_logger):
        """Test that instrument wraps Crew.kickoff, Task, and Agent methods when available."""

        # Create a real Crew class with all kickoff variants
        class MockCrew:
            def kickoff(self, inputs=None):
                return "crew_result"

            def kickoff_async(self, inputs=None):
                return "crew_result"

            async def akickoff(self, inputs=None):
                return "crew_result"

            def kickoff_for_each(self, inputs=None):
                return "crew_result"

            def kickoff_for_each_async(self, inputs=None):
                return "crew_result"

            async def akickoff_for_each(self, inputs=None):
                return "crew_result"

        # Create mock Task and Agent classes
        class MockTask:
            def execute_sync(self):
                return "task_result"

            def execute_async(self):
                return "task_result"

        class MockAgent:
            def execute_task(self):
                return "agent_result"

        # Create mock crewai module
        mock_crewai = MagicMock()
        mock_crewai.Crew = MockCrew
        mock_crewai.Task = MockTask
        mock_crewai.Agent = MockAgent

        # Save original methods to verify they get wrapped
        original_kickoff = MockCrew.kickoff
        original_execute_sync = MockTask.execute_sync
        original_execute_task = MockAgent.execute_task

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
            instrumentor = CrewAIInstrumentor()
            config = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with(
                "CrewAI instrumentation enabled with automatic context propagation"
            )
            # Verify methods were wrapped (no longer the original functions)
            self.assertIsNot(MockCrew.kickoff, original_kickoff)
            self.assertIsNot(MockTask.execute_sync, original_execute_sync)
            self.assertIsNot(MockAgent.execute_task, original_execute_task)

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        # Create mock crewai module
        mock_crewai = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "Crew":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = CrewAIInstrumentor()
                config = MagicMock()
                config.fail_on_error = False

                # Should not raise exception
                instrumentor.instrument(config)

                mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that instrument raises exceptions when fail_on_error is True."""
        # Create mock crewai module
        mock_crewai = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "Crew":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = CrewAIInstrumentor()
                config = MagicMock()
                config.fail_on_error = True

                # Should raise exception
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_extract_crew_attributes_basic(self):
        """Test extraction of basic crew attributes."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock crew
            mock_crew = MagicMock()
            mock_crew.id = "crew_123"
            mock_crew.name = "Research Crew"
            mock_crew.process = "sequential"
            mock_crew.verbose = True

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "crewai")
            self.assertEqual(attrs["gen_ai.operation.name"], "crew.execution")
            self.assertEqual(attrs["crewai.crew.id"], "crew_123")
            self.assertEqual(attrs["crewai.crew.name"], "Research Crew")
            self.assertEqual(attrs["crewai.process.type"], "sequential")
            self.assertTrue(attrs["crewai.verbose"])
            # Session ID should be set from crew.id
            self.assertEqual(attrs["session.id"], "crew_123")
            self.assertEqual(attrs["crewai.session.id"], "crew_123")

    def test_extract_crew_attributes_with_agents(self):
        """Test extraction of crew attributes with agents."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock agents
            mock_agent1 = MagicMock()
            mock_agent1.role = "Researcher"
            mock_agent1.goal = "Research AI trends"

            # Create mock tools with proper name attributes
            search_tool = MagicMock()
            search_tool.name = "search_tool"
            scrape_tool = MagicMock()
            scrape_tool.name = "scrape_tool"
            mock_agent1.tools = [search_tool, scrape_tool]

            mock_agent2 = MagicMock()
            mock_agent2.role = "Writer"
            mock_agent2.goal = "Write articles"
            mock_agent2.tools = []

            # Create mock crew
            mock_crew = MagicMock()
            mock_crew.agents = [mock_agent1, mock_agent2]

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["crewai.agent_count"], 2)
            self.assertIn("Researcher", attrs["crewai.agent.roles"])
            self.assertIn("Writer", attrs["crewai.agent.roles"])
            self.assertIn("Research AI trends", attrs["crewai.agent.goals"])
            self.assertEqual(attrs["crewai.tool_count"], 2)
            self.assertIn("search_tool", attrs["crewai.tools"])
            self.assertIn("scrape_tool", attrs["crewai.tools"])

    def test_extract_crew_attributes_with_tasks(self):
        """Test extraction of crew attributes with tasks."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock tasks
            mock_task1 = MagicMock()
            mock_task1.description = "Research latest AI developments"

            mock_task2 = MagicMock()
            mock_task2.description = "Write a summary report"

            # Create mock crew
            mock_crew = MagicMock()
            mock_crew.tasks = [mock_task1, mock_task2]

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["crewai.task_count"], 2)
            self.assertIn("Research latest AI developments", attrs["crewai.task.descriptions"])
            self.assertIn("Write a summary report", attrs["crewai.task.descriptions"])

    def test_extract_crew_attributes_with_inputs(self):
        """Test extraction of crew attributes with inputs."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock crew
            mock_crew = MagicMock()

            # Test with dict inputs
            inputs = {"topic": "AI Agents", "format": "markdown"}
            args = (inputs,)
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["crewai.inputs.keys"], ["topic", "format"])
            self.assertEqual(attrs["crewai.inputs.topic"], "AI Agents")
            self.assertEqual(attrs["crewai.inputs.format"], "markdown")

    def test_extract_crew_attributes_with_manager(self):
        """Test extraction of crew attributes with hierarchical manager."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock manager agent
            mock_manager = MagicMock()
            mock_manager.role = "Project Manager"

            # Create mock crew
            mock_crew = MagicMock()
            mock_crew.manager_agent = mock_manager

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["crewai.manager.role"], "Project Manager")

    def test_extract_response_attributes_string_result(self):
        """Test extraction of response attributes from string result."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Test with string result
            result = "This is the crew output"

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["crewai.output"], "This is the crew output")
            self.assertEqual(attrs["crewai.output_length"], 23)

    def test_extract_response_attributes_crew_output(self):
        """Test extraction of response attributes from CrewOutput object."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock CrewOutput
            mock_result = MagicMock()
            mock_result.raw = "This is the raw output"

            # Create mock task outputs
            mock_task_output1 = MagicMock()
            mock_task_output1.raw = "Task 1 result"

            mock_task_output2 = MagicMock()
            mock_task_output2.raw = "Task 2 result"

            mock_result.tasks_output = [mock_task_output1, mock_task_output2]

            attrs = instrumentor._extract_response_attributes(mock_result)

            # Assert
            self.assertIn("crewai.output", attrs)
            self.assertEqual(attrs["crewai.tasks_completed"], 2)
            self.assertIn("Task 1 result", attrs["crewai.task_outputs"])
            self.assertIn("Task 2 result", attrs["crewai.task_outputs"])

    def test_extract_usage_when_available(self):
        """Test extraction of usage information when available."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock result with usage (mimics CrewAI UsageMetrics pydantic model)
            mock_result = MagicMock()
            mock_result.token_usage = MagicMock()
            mock_result.token_usage.prompt_tokens = 100
            mock_result.token_usage.completion_tokens = 50
            mock_result.token_usage.total_tokens = 150
            mock_result.token_usage.cached_prompt_tokens = 0

            usage = instrumentor._extract_usage(mock_result)

            # Assert
            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 100)
            self.assertEqual(usage["completion_tokens"], 50)
            self.assertEqual(usage["total_tokens"], 150)

    def test_extract_usage_when_not_available(self):
        """Test extraction of usage information when not available."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock result without usage
            mock_result = MagicMock()
            del mock_result.token_usage

            usage = instrumentor._extract_usage(mock_result)

            # Assert
            self.assertIsNone(usage)

    def test_extract_finish_reason(self):
        """Test extraction of finish reason from result."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock result
            mock_result = "Some output"

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_crew_attributes_session_id_from_crew_id(self):
        """Test that session.id is set from crew instance ID."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew.id = "crew-uuid-123"

            attrs = instrumentor._extract_crew_attributes(mock_crew, (), {})

            self.assertEqual(attrs["session.id"], "crew-uuid-123")
            self.assertEqual(attrs["crewai.session.id"], "crew-uuid-123")
            self.assertEqual(mock_crew._genai_otel_session_id, "crew-uuid-123")

    def test_extract_crew_attributes_session_id_from_inputs(self):
        """Test that session_id from inputs takes priority over crew ID."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew.id = "crew-uuid-123"

            inputs = {"topic": "AI", "session_id": "app-session-456"}
            attrs = instrumentor._extract_crew_attributes(mock_crew, (inputs,), {})

            # App-provided session_id takes priority over crew ID
            self.assertEqual(attrs["session.id"], "app-session-456")
            self.assertEqual(attrs["crewai.session.id"], "app-session-456")

    def test_extract_crew_attributes_session_id_from_inputs_kwarg(self):
        """Test that session_id from inputs kwarg works."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew.id = "crew-uuid-123"

            inputs = {"session.id": "dot-session-789"}
            attrs = instrumentor._extract_crew_attributes(mock_crew, (), {"inputs": inputs})

            self.assertEqual(attrs["session.id"], "dot-session-789")

    def test_extract_crew_attributes_session_id_from_extractor(self):
        """Test that session_id_extractor takes priority over crew ID."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()
            instrumentor.config = MagicMock()
            instrumentor.config.session_id_extractor = lambda inst, a, k: "extractor-session"

            mock_crew = MagicMock()
            mock_crew.id = "crew-uuid-123"

            attrs = instrumentor._extract_crew_attributes(mock_crew, (), {})

            # Extractor takes priority over crew ID
            self.assertEqual(attrs["session.id"], "extractor-session")

    def test_extract_crew_attributes_session_id_autogenerated(self):
        """Test that a UUID is auto-generated when no other source exists."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock(spec=[])  # No attributes at all

            attrs = instrumentor._extract_crew_attributes(mock_crew, (), {})

            # Should have auto-generated UUID
            self.assertIn("session.id", attrs)
            self.assertIn("crewai.session.id", attrs)
            # Verify it's a valid UUID format
            import uuid

            uuid.UUID(attrs["session.id"])  # Will raise if not valid

    def test_extract_task_attributes_session_propagation_via_crew(self):
        """Test that session.id is propagated from crew to task spans via task.crew."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew._genai_otel_session_id = "session-from-crew"

            mock_task = MagicMock()
            mock_task.description = "Test task"
            mock_task.crew = mock_crew

            attrs = instrumentor._extract_task_attributes(mock_task, (), {})

            self.assertEqual(attrs["session.id"], "session-from-crew")

    def test_extract_task_attributes_session_propagation_via_agent(self):
        """Test that session.id is propagated via task.agent.crew when task.crew is absent."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew._genai_otel_session_id = "session-via-agent"

            mock_agent = MagicMock()
            mock_agent.crew = mock_crew
            mock_agent.role = "Researcher"

            # Task without direct crew attr but with agent.crew
            mock_task = MagicMock(spec=["description", "expected_output", "id", "agent"])
            mock_task.description = "Test task"
            mock_task.expected_output = "Expected"
            mock_task.id = "task-1"
            mock_task.agent = mock_agent

            attrs = instrumentor._extract_task_attributes(mock_task, (), {})

            self.assertEqual(attrs["session.id"], "session-via-agent")

    def test_extract_agent_attributes_session_propagation(self):
        """Test that session.id is propagated from crew to agent spans."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock crew with session ID
            mock_crew = MagicMock()
            mock_crew._genai_otel_session_id = "session-from-crew"

            # Create mock agent with reference to crew
            mock_agent = MagicMock()
            mock_agent.role = "Researcher"
            mock_agent.crew = mock_crew

            attrs = instrumentor._extract_agent_attributes(mock_agent, (), {})

            self.assertEqual(attrs["session.id"], "session-from-crew")

    def test_extract_task_attributes_no_session_without_crew(self):
        """Test that task spans without crew reference don't set session.id."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_task = MagicMock(spec=["description", "expected_output", "id"])
            mock_task.description = "Test task"
            mock_task.expected_output = "Expected"
            mock_task.id = "task-1"

            attrs = instrumentor._extract_task_attributes(mock_task, (), {})

            self.assertNotIn("session.id", attrs)

    def test_thread_pool_executor_patch_is_idempotent(self):
        """_patch_thread_pool_executor must not stack wrappers on repeat calls."""
        from concurrent.futures import ThreadPoolExecutor

        saved_submit = ThreadPoolExecutor.submit
        try:
            with patch.dict("sys.modules", {"crewai": MagicMock()}):
                instrumentor = CrewAIInstrumentor()
                instrumentor._patch_thread_pool_executor()
                first = ThreadPoolExecutor.submit
                self.assertTrue(getattr(first, "_genai_otel_wrapped", False))

                # Second call sees the guard marker and is a no-op.
                instrumentor._patch_thread_pool_executor()
                self.assertIs(ThreadPoolExecutor.submit, first)
        finally:
            ThreadPoolExecutor.submit = saved_submit

    def test_patched_submit_still_executes_task(self):
        """The patched submit must still run submitted callables (unrelated work)."""
        from concurrent.futures import ThreadPoolExecutor

        saved_submit = ThreadPoolExecutor.submit
        try:
            with patch.dict("sys.modules", {"crewai": MagicMock()}):
                instrumentor = CrewAIInstrumentor()
                instrumentor._patch_thread_pool_executor()

                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(lambda x: x + 1, 41)
                    self.assertEqual(fut.result(timeout=5), 42)
        finally:
            ThreadPoolExecutor.submit = saved_submit

    def test_cap_content_respects_config(self):
        """_cap_content honours content_max_length (default, small cap, unlimited)."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 200)

        cfg = OTelConfig()
        cfg.content_max_length = 4
        instrumentor.config = cfg
        self.assertEqual(instrumentor._cap_content("abcdefgh"), "abcd")

        cfg.content_max_length = 0
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 300)


class _Usage:
    """Stand-in for CrewAI's UsageMetrics."""

    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.cached_prompt_tokens = 0


class _FakeLLM:
    """Counts usage the way CrewAI 1.x does: on the LLM object, for its whole lifetime."""

    model = "gpt-4o-mini"

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def call(self):
        self.prompt_tokens += 10
        self.completion_tokens += 5

    def get_token_usage_summary(self):
        return _Usage(self.prompt_tokens, self.completion_tokens)


class _CountingLLM:
    """Keeps its counters in a dict, as CrewAI does, so a shallow copy shares them."""

    model = "gpt-4o-mini"

    def __init__(self):
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def call(self):
        self._token_usage["prompt_tokens"] += 10
        self._token_usage["completion_tokens"] += 5

    @property
    def prompt_tokens(self):
        return self._token_usage["prompt_tokens"]

    @property
    def completion_tokens(self):
        return self._token_usage["completion_tokens"]

    def get_token_usage_summary(self):
        return _Usage(self.prompt_tokens, self.completion_tokens)


class _ErrorRecords(logging.Handler):
    """Collects ERROR records; assertNoLogs is not available on Python 3.9."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _CrewResult:
    """Stand-in for CrewOutput: token_usage is whatever the crew summed up."""

    def __init__(self, raw, token_usage):
        self.raw = raw
        self.token_usage = token_usage


def _fake_crewai():
    """A crewai module whose classes call each other the way CrewAI 1.x does.

    The classes are created per call, so wrapping them in one test never leaks into
    another.
    """

    class Agent:
        goal = "Find things"
        backstory = "Thorough."
        crew = None

        def __init__(self, role="Researcher", llm=None):
            self.role = role
            self.llm = llm

        def _work(self):
            if self.llm is not None:
                self.llm.call()
            return "answer from " + self.role

        def execute_task(self, task=None, context=None, tools=None):
            return self._work()

        async def aexecute_task(self, task=None, context=None, tools=None):
            return self._work()

    class Task:
        description = "Do the thing"
        expected_output = "The thing"
        id = "task-1"
        fail = False

        def __init__(self, agent):
            self.agent = agent

        def execute_sync(self, agent=None, context=None, tools=None):
            return self.agent.execute_task(task=self)

        async def aexecute_sync(self, agent=None, context=None, tools=None):
            return await self.agent.aexecute_task(task=self)

        def execute_async(self, agent=None, context=None, tools=None):
            # Same shape as CrewAI: hand back a Future, do the work on a thread
            # that runs in a copy of the caller's context.
            future = Future()
            ctx = contextvars.copy_context()

            def work():
                time.sleep(0.05)
                if self.fail:
                    future.set_exception(RuntimeError("task blew up"))
                else:
                    future.set_result(self.agent.execute_task(task=self))

            threading.Thread(target=ctx.run, args=(work,), daemon=True).start()
            return future

    class Crew:
        name = "fake crew"
        process = "sequential"
        verbose = False

        def __init__(self, tasks, crew_id="crew-1"):
            self.id = crew_id
            self.tasks = tasks
            self.agents = [task.agent for task in tasks]

        def _result(self, outputs):
            # CrewAI adds every agent's LLM lifetime counters together.
            prompt = completion = 0
            for agent in self.agents:
                if agent.llm is not None:
                    prompt += agent.llm.prompt_tokens
                    completion += agent.llm.completion_tokens
            return _CrewResult(outputs[-1], _Usage(prompt, completion))

        def kickoff(self, inputs=None):
            outputs = []
            for task in self.tasks:
                if getattr(task, "async_execution", False):
                    outputs.append(task.execute_async().result(timeout=5))
                else:
                    outputs.append(task.execute_sync())
            return self._result(outputs)

        async def kickoff_async(self, inputs=None):
            return await asyncio.to_thread(self.kickoff, inputs)

        async def akickoff(self, inputs=None):
            return self._result([await task.aexecute_sync() for task in self.tasks])

        def kickoff_for_each(self, inputs):
            results = []
            for index, item in enumerate(inputs):
                copy = type(self)(self.tasks, crew_id="copy-{0}".format(index))
                results.append(copy.kickoff(inputs=item))
            return results

    module = MagicMock()
    module.Crew = Crew
    module.Task = Task
    module.Agent = Agent
    return module


class TestCrewAICallPaths(unittest.TestCase):
    """The paths a real CrewAI 1.x run takes, against classes shaped like CrewAI's."""

    def setUp(self):
        # Start from an empty context, whatever earlier tests left current.
        self._context_token = otel_context.attach(otel_context.Context())
        self._saved_submit = ThreadPoolExecutor.submit
        self.crewai = _fake_crewai()
        with patch.dict("sys.modules", {"crewai": self.crewai}):
            self.instrumentor = CrewAIInstrumentor()
            self.exporter = InMemorySpanExporter()
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(self.exporter))
            self.instrumentor.tracer = provider.get_tracer("test")
            self.instrumentor.instrument(OTelConfig())

    def tearDown(self):
        ThreadPoolExecutor.submit = self._saved_submit
        otel_context.detach(self._context_token)

    def _crew(self, agents=None, **task_attrs):
        agents = agents or [self.crewai.Agent()]
        tasks = [self.crewai.Task(agent) for agent in agents]
        for task in tasks:
            for key, value in task_attrs.items():
                setattr(task, key, value)
        return self.crewai.Crew(tasks)

    def _spans(self, name=None):
        spans = self.exporter.get_finished_spans()
        return [s for s in spans if name is None or s.name == name]

    def _wait_for(self, name):
        """A Future's callbacks may still be running when result() has returned."""
        deadline = time.time() + 2
        while not self._spans(name) and time.time() < deadline:
            time.sleep(0.01)
        return self._spans(name)

    def _parent_name(self, span):
        by_id = {s.context.span_id: s for s in self._spans()}
        return by_id[span.parent.span_id].name if span.parent else None

    def test_native_async_kickoff_traces_tasks_and_agents(self):
        """akickoff() runs Task.aexecute_sync / Agent.aexecute_task, not the sync pair."""
        asyncio.run(self._crew().akickoff())

        (crew_span,) = self._spans("crewai.crew.execution")
        (task_span,) = self._spans("crewai.task.execution")
        (agent_span,) = self._spans("crewai.agent.execution")
        self.assertEqual(self._parent_name(task_span), "crewai.crew.execution")
        self.assertEqual(self._parent_name(agent_span), "crewai.task.execution")
        self.assertIsNone(crew_span.parent)
        self.assertEqual(task_span.attributes["crewai.output"], "answer from Researcher")

    def test_async_kickoff_does_not_leak_its_span_into_the_caller(self):
        """asyncio.run(crew.akickoff()) must leave the caller's context as it found it."""
        before = trace.get_current_span()
        errors = _ErrorRecords()
        context_logger = logging.getLogger("opentelemetry.context")
        context_logger.addHandler(errors)
        try:
            asyncio.run(self._crew().akickoff())
            asyncio.run(self._crew().kickoff_async())
        finally:
            context_logger.removeHandler(errors)

        self.assertEqual(errors.messages, [])
        self.assertIs(trace.get_current_span(), before)
        first, second = self._spans("crewai.crew.execution")
        self.assertIsNone(first.parent)
        self.assertIsNone(second.parent)
        self.assertNotEqual(first.context.trace_id, second.context.trace_id)

    def test_task_started_with_create_task_does_not_adopt_its_sibling(self):
        """A coroutine handed to create_task opens its span in the task that runs it."""
        crewai = self.crewai

        async def run():
            first = crewai.Task(crewai.Agent(role="One"))
            second = crewai.Task(crewai.Agent(role="Two"))
            pending = asyncio.create_task(first.aexecute_sync())
            await second.aexecute_sync()
            await pending

        asyncio.run(run())

        one, two = sorted(
            self._spans("crewai.task.execution"), key=lambda s: s.attributes["gen_ai.agent.name"]
        )
        self.assertIsNone(one.parent)
        self.assertIsNone(two.parent)

    def test_kickoff_async_produces_one_crew_span(self):
        """kickoff_async() runs kickoff() on a thread; that is one crew run, not two."""
        asyncio.run(self._crew().kickoff_async())

        (crew_span,) = self._spans("crewai.crew.execution")
        (task_span,) = self._spans("crewai.task.execution")
        self.assertEqual(self._parent_name(task_span), "crewai.crew.execution")
        self.assertEqual(crew_span.attributes["crewai.output"], "answer from Researcher")

    def test_kickoff_for_each_still_traces_every_copy(self):
        """The batch span and one crew span per input copy are all kept."""
        self._crew().kickoff_for_each([{"n": 1}, {"n": 2}])

        crew_spans = self._spans("crewai.crew.execution")
        self.assertEqual(len(crew_spans), 3)
        batch = [s for s in crew_spans if s.parent is None]
        self.assertEqual(len(batch), 1)
        self.assertEqual(len(self._spans("crewai.task.execution")), 2)

    def test_repeated_kickoff_is_traced_each_time(self):
        """The guard against nested crew spans must not outlive the call."""
        crew = self._crew()
        crew.kickoff()
        crew.kickoff()
        self.assertEqual(len(self._spans("crewai.crew.execution")), 2)

    def test_async_execution_task_span_covers_the_work(self):
        """Task.execute_async returns a Future at once; the span ends when it resolves."""
        self._crew(async_execution=True).kickoff()

        (task_span,) = self._wait_for("crewai.task.execution")
        (agent_span,) = self._spans("crewai.agent.execution")
        self.assertEqual(self._parent_name(agent_span), "crewai.task.execution")
        self.assertGreaterEqual(task_span.end_time, agent_span.end_time)
        self.assertGreaterEqual(task_span.end_time - task_span.start_time, 30_000_000)
        self.assertEqual(task_span.attributes["crewai.output"], "answer from Researcher")
        self.assertEqual(task_span.status.status_code, StatusCode.OK)

    def test_async_execution_task_span_records_failure(self):
        """A task that fails on its thread must not leave an OK span behind."""
        with self.assertRaises(RuntimeError):
            self._crew(async_execution=True, fail=True).kickoff()

        (task_span,) = self._wait_for("crewai.task.execution")
        self.assertEqual(task_span.status.status_code, StatusCode.ERROR)
        self.assertIn("task blew up", task_span.status.description)

    def test_async_execution_that_fails_to_start_is_an_error_span(self):
        """execute_async raising before it returns a Future still ends the span."""

        def explode(task, agent=None, context=None, tools=None):
            raise RuntimeError("no thread for you")

        traced = self.instrumentor._traced_future_method(
            explode, "crewai.task.execution", self.instrumentor._extract_task_attributes
        )

        with self.assertRaises(RuntimeError):
            traced(self.crewai.Task(self.crewai.Agent()))

        (task_span,) = self._spans("crewai.task.execution")
        self.assertEqual(task_span.status.status_code, StatusCode.ERROR)
        self.assertFalse(trace.get_current_span().get_span_context().is_valid)

    def test_async_execution_result_that_is_not_a_future_ends_the_span(self):
        traced = self.instrumentor._traced_future_method(
            lambda task: "done already",
            "crewai.task.execution",
            self.instrumentor._extract_task_attributes,
        )

        self.assertEqual(traced(self.crewai.Task(self.crewai.Agent())), "done already")

        (task_span,) = self._spans("crewai.task.execution")
        self.assertEqual(task_span.attributes["crewai.output"], "done already")

    def test_async_execution_task_span_carries_session_and_user(self):
        """The configured extractors apply to this span as they do to the others."""
        self.instrumentor.config.session_id_extractor = lambda instance, args, kwargs: "sess-9"
        self.instrumentor.config.user_id_extractor = lambda instance, args, kwargs: "user-9"

        self._crew(async_execution=True).kickoff()

        (task_span,) = self._wait_for("crewai.task.execution")
        self.assertEqual(task_span.attributes["session.id"], "sess-9")
        self.assertEqual(task_span.attributes["gen_ai.conversation.id"], "sess-9")
        self.assertEqual(task_span.attributes["user.id"], "user-9")

    def test_async_execution_passes_through_when_not_instrumented(self):
        self.instrumentor._instrumented = False

        future = self.crewai.Task(self.crewai.Agent()).execute_async()

        self.assertEqual(future.result(timeout=5), "answer from Researcher")
        self.assertEqual(self._spans(), [])

    def test_async_execution_survives_a_sampled_out_span(self):
        """A non-recording span has no name or attributes; the task must still run."""
        self.instrumentor.tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")

        future = self.crewai.Task(self.crewai.Agent()).execute_async()

        self.assertEqual(future.result(timeout=5), "answer from Researcher")
        self.assertEqual(self._spans(), [])

    def test_async_entry_point_on_a_crew_already_being_traced_adds_no_span(self):
        """The nested-span guard covers coroutine entry points too."""
        crew = self._crew()

        async def run():
            token = crewai_module._ACTIVE_CREWS.set(frozenset({id(crew)}))
            try:
                return await crew.akickoff()
            finally:
                crewai_module._ACTIVE_CREWS.reset(token)

        result = asyncio.run(run())

        self.assertEqual(result.raw, "answer from Researcher")
        self.assertEqual(self._spans("crewai.crew.execution"), [])
        self.assertEqual(len(self._spans("crewai.task.execution")), 1)

    def test_crew_usage_is_this_kickoff_not_the_llm_lifetime(self):
        """CrewOutput.token_usage keeps growing across kickoffs; the span must not."""
        crew = self._crew(agents=[self.crewai.Agent(llm=_FakeLLM())])
        crew.kickoff()
        second = crew.kickoff()

        self.assertEqual(second.token_usage.prompt_tokens, 20)
        first_span, second_span = self._spans("crewai.crew.execution")
        self.assertEqual(first_span.attributes["gen_ai.usage.input_tokens"], 10)
        self.assertEqual(second_span.attributes["gen_ai.usage.input_tokens"], 10)
        self.assertEqual(second_span.attributes["gen_ai.usage.output_tokens"], 5)
        self.assertEqual(second_span.attributes["gen_ai.usage.total_tokens"], 15)

    def test_crew_usage_counts_a_shared_llm_once(self):
        """Two agents on one LLM object: CrewAI adds its counters twice."""
        llm = _FakeLLM()
        agents = [self.crewai.Agent(role="One", llm=llm), self.crewai.Agent(role="Two", llm=llm)]
        result = self._crew(agents=agents).kickoff()

        self.assertEqual(result.token_usage.prompt_tokens, 40)
        (crew_span,) = self._spans("crewai.crew.execution")
        self.assertEqual(crew_span.attributes["gen_ai.usage.input_tokens"], 20)
        self.assertEqual(crew_span.attributes["gen_ai.usage.output_tokens"], 10)

    def test_crew_usage_counts_llm_copies_sharing_one_counter_once(self):
        """Crew.copy() shallow-copies each agent's LLM, and the copies share one counter.

        kickoff_for_each() runs such copies, so counting per LLM object reported the
        same tokens once for every copy.
        """
        llm = _CountingLLM()
        agents = [
            self.crewai.Agent(role="One", llm=llm),
            self.crewai.Agent(role="Two", llm=copy.copy(llm)),
        ]
        self._crew(agents=agents).kickoff()

        (crew_span,) = self._spans("crewai.crew.execution")
        self.assertEqual(crew_span.attributes["gen_ai.usage.input_tokens"], 20)
        self.assertEqual(crew_span.attributes["gen_ai.usage.output_tokens"], 10)

    def test_result_whose_token_usage_cannot_be_read_is_still_returned(self):
        """Measuring usage must never cost the caller a crew that already succeeded."""

        class _Unreadable:
            raw = "answer"

            @property
            def token_usage(self):
                raise RuntimeError("usage unavailable")

        crew = self._crew(agents=[self.crewai.Agent(llm=_FakeLLM())])
        crew._result = lambda outputs: _Unreadable()

        self.assertEqual(crew.kickoff().raw, "answer")

    def test_crew_usage_counts_the_manager_llm_once(self):
        """Hierarchical process: the manager agent only exists once the run is under way."""
        manager_llm = _FakeLLM()
        # Usage from before this kickoff, which must not be counted into it.
        manager_llm.call()
        crew = self._crew(agents=[self.crewai.Agent(llm=_FakeLLM())])
        crew.manager_llm = manager_llm
        summed_by_crewai = crew._result

        def result_after_manager_ran(outputs):
            manager_llm.call()
            crew.manager_agent = self.crewai.Agent(role="Manager", llm=manager_llm)
            return summed_by_crewai(outputs)

        crew._result = result_after_manager_ran
        crew.kickoff()

        (crew_span,) = self._spans("crewai.crew.execution")
        self.assertEqual(crew_span.attributes["gen_ai.usage.input_tokens"], 20)
        self.assertEqual(crew_span.attributes["gen_ai.usage.output_tokens"], 10)

    def test_crew_usage_falls_back_to_token_usage_without_llm_counters(self):
        """Older CrewAI has no counters on the LLM; token_usage is all there is."""
        crew = self._crew()
        crew._result = lambda outputs: _CrewResult(outputs[-1], _Usage(7, 3))
        crew.kickoff()

        (crew_span,) = self._spans("crewai.crew.execution")
        self.assertEqual(crew_span.attributes["gen_ai.usage.input_tokens"], 7)
        self.assertEqual(crew_span.attributes["gen_ai.usage.output_tokens"], 3)


class TestCrewAIResponseAttributes(unittest.TestCase):
    """Result objects as CrewAI 1.x returns them."""

    def setUp(self):
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            self.instrumentor = CrewAIInstrumentor()

    def test_pydantic_output_is_recorded(self):
        """CrewOutput.json raises unless the task asked for JSON; pydantic still counts."""

        class Output:
            raw = '{"text": "hi"}'
            pydantic = "Greeting(text='hi')"

            @property
            def json(self):
                raise ValueError("No JSON output found in the final task.")

        attrs = self.instrumentor._extract_response_attributes(Output())

        self.assertEqual(attrs["crewai.output"], '{"text": "hi"}')
        self.assertEqual(attrs["crewai.output.pydantic"], "Greeting(text='hi')")
        self.assertNotIn("crewai.output.json", attrs)

    def test_pydantic_output_follows_the_content_switch(self):
        """A structured output is the model's answer, so it is content."""

        class Output:
            raw = '{"text": "hi"}'
            pydantic = "Greeting(text='hi')"
            json = None

        self.instrumentor.config = OTelConfig(enable_content_capture=False)
        self.assertNotIn(
            "crewai.output.pydantic", self.instrumentor._extract_response_attributes(Output())
        )

        self.instrumentor.config = OTelConfig(enable_content_capture=True)
        self.assertEqual(
            self.instrumentor._extract_response_attributes(Output())["crewai.output.pydantic"],
            "Greeting(text='hi')",
        )

    def test_absent_pydantic_output_is_not_reported_as_none(self):
        class Output:
            raw = "plain"
            pydantic = None
            json = '{"a": 1}'

        attrs = self.instrumentor._extract_response_attributes(Output())

        self.assertEqual(attrs["crewai.output.json"], '{"a": 1}')
        self.assertNotIn("crewai.output.pydantic", attrs)

    def test_structured_agent_result_is_reported_as_its_content(self):
        """An agent with a response model returns the model, whose .json is a method."""

        class Greeting:
            def model_dump_json(self):
                return '{"text": "hi"}'

            def json(self):
                return '{"text": "hi"}'

        attrs = self.instrumentor._extract_response_attributes(Greeting())

        self.assertEqual(attrs["crewai.output"], '{"text": "hi"}')
        self.assertEqual(attrs["crewai.output_length"], 14)
        self.assertNotIn("crewai.output.json", attrs)


class TestCrewAITelemetrySwitch(unittest.TestCase):
    """CrewAI's own telemetry is switched off with the variable CrewAI actually reads."""

    def setUp(self):
        self._saved_submit = ThreadPoolExecutor.submit
        self._env = patch.dict(os.environ)
        self._env.start()
        os.environ.pop("CREWAI_DISABLE_TELEMETRY", None)

    def tearDown(self):
        self._env.stop()
        ThreadPoolExecutor.submit = self._saved_submit

    def test_instrument_sets_the_switch_crewai_reads(self):
        with patch.dict("sys.modules", {"crewai": _fake_crewai()}):
            CrewAIInstrumentor().instrument(OTelConfig())

        self.assertEqual(os.environ.get("CREWAI_DISABLE_TELEMETRY"), "true")

    def test_switch_is_set_before_crewai_is_imported(self):
        """CrewAI builds its telemetry exporter while it is being imported."""
        seen = []
        real_import = builtins.__import__

        def spy(name, *args, **kwargs):
            if name == "crewai":
                seen.append(os.environ.get("CREWAI_DISABLE_TELEMETRY"))
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            with patch("builtins.__import__", side_effect=spy):
                CrewAIInstrumentor()

        self.assertEqual(seen[0], "true")

    def test_switch_set_by_the_application_is_left_alone(self):
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "false"

        with patch.dict("sys.modules", {"crewai": _fake_crewai()}):
            CrewAIInstrumentor().instrument(OTelConfig())

        self.assertEqual(os.environ["CREWAI_DISABLE_TELEMETRY"], "false")

    def test_switch_is_not_left_behind_when_crewai_is_missing(self):
        with patch.dict("sys.modules", {"crewai": None}):
            CrewAIInstrumentor()

        self.assertNotIn("CREWAI_DISABLE_TELEMETRY", os.environ)

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_warns_when_crewai_telemetry_owns_the_global_provider(self, mock_logger):
        """CrewAI before 1.15.15 installs its telemetry provider globally on import."""
        crewai = _fake_crewai()
        crewai.telemetry.Telemetry._instance.provider = trace.get_tracer_provider()

        with patch.dict("sys.modules", {"crewai": crewai}):
            CrewAIInstrumentor().instrument(OTelConfig())

        (warning,) = mock_logger.warning.call_args_list
        self.assertIn("telemetry", warning.args[0])

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_no_warning_when_crewai_keeps_its_provider_private(self, mock_logger):
        with patch.dict("sys.modules", {"crewai": _fake_crewai()}):
            CrewAIInstrumentor().instrument(OTelConfig())

        mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Agent graph: the crew is the root node, agents hang beneath it, tasks are steps.
# ---------------------------------------------------------------------------


def _graph_instrumentor(**config):
    from genai_otel.config import OTelConfig
    from genai_otel.instrumentors.crewai_instrumentor import CrewAIInstrumentor

    instrumentor = CrewAIInstrumentor()
    instrumentor.config = OTelConfig(**config)
    return instrumentor


def test_crew_span_is_the_graph_root():
    import types

    instrumentor = _graph_instrumentor()
    named = types.SimpleNamespace(id="c1", name="research_crew", agents=[], tasks=[])
    attrs = instrumentor._extract_crew_attributes(named, (), {})
    assert attrs["graph.node.id"] == "research_crew"
    assert attrs["openinference.span.kind"] == "CHAIN"

    unnamed = types.SimpleNamespace(id="c2", agents=[], tasks=[])
    assert instrumentor._extract_crew_attributes(unnamed, (), {})["graph.node.id"] == "crew"


def test_task_span_is_a_step_not_a_node():
    import types

    instrumentor = _graph_instrumentor()
    agent = types.SimpleNamespace(role="Researcher")
    task = types.SimpleNamespace(description="d", expected_output="e", agent=agent)
    attrs = instrumentor._extract_task_attributes(task, (), {})
    assert attrs["gen_ai.agent.name"] == "Researcher"
    assert attrs["openinference.span.kind"] == "CHAIN"
    assert "graph.node.id" not in attrs


def test_graph_keys_follow_the_agent_graph_flag():
    import types

    instrumentor = _graph_instrumentor(enable_agent_graph_attributes=False)
    crew = types.SimpleNamespace(id="c1", name="research_crew", agents=[], tasks=[])
    task = types.SimpleNamespace(
        description="d", expected_output="e", agent=types.SimpleNamespace(role="R")
    )
    assert "graph.node.id" not in instrumentor._extract_crew_attributes(crew, (), {})
    assert "openinference.span.kind" not in instrumentor._extract_task_attributes(task, (), {})


def test_crew_task_agent_form_one_graph_through_the_wrappers():
    """Real span wrappers and the agent graph processor: one node per agent, under the crew."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from genai_otel.agent_graph import AgentGraphSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AgentGraphSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = _graph_instrumentor()
    instrumentor.tracer = provider.get_tracer("test")
    instrumentor._instrumented = True

    crew_wrap = instrumentor.create_span_wrapper(
        "crewai.crew.execution", extract_attributes=instrumentor._extract_crew_attributes
    )
    task_wrap = instrumentor.create_span_wrapper(
        "crewai.task.execution", extract_attributes=instrumentor._extract_task_attributes
    )
    agent_wrap = instrumentor.create_span_wrapper(
        "crewai.agent.execution", extract_attributes=instrumentor._extract_agent_attributes
    )

    class Agent:
        def __init__(self, role):
            self.role, self.goal, self.backstory = role, "g", "b"

        @agent_wrap
        def execute_task(self):
            return "done"

    class Task:
        def __init__(self, agent):
            self.agent, self.description, self.expected_output = agent, "d", "e"

        @task_wrap
        def execute_sync(self):
            return self.agent.execute_task()

    class Crew:
        def __init__(self, tasks):
            self.id, self.name, self.tasks, self.agents = "c1", "research_crew", tasks, []

        @crew_wrap
        def kickoff(self):
            return [task.execute_sync() for task in self.tasks]

    Crew([Task(Agent("Researcher")), Task(Agent("Writer"))]).kickoff()

    spans = exporter.get_finished_spans()
    nodes = {
        (
            span.name if span.name != "crewai.agent.execution" else span.attributes["graph.node.id"]
        ): dict(span.attributes)
        for span in spans
        if "graph.node.id" in span.attributes
    }
    assert set(nodes) == {"crewai.crew.execution", "Researcher", "Writer"}
    assert "graph.node.parent_id" not in nodes["crewai.crew.execution"]
    assert nodes["Researcher"]["graph.node.parent_id"] == "research_crew"
    assert nodes["Writer"]["graph.node.parent_id"] == "research_crew"
    for span in spans:
        if span.name == "crewai.task.execution":
            assert "graph.node.id" not in span.attributes
