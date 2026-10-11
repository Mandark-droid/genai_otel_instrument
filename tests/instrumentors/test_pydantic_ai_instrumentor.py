import asyncio
import inspect
import logging
import types
import unittest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from genai_otel import agent_budget
from genai_otel.config import OTelConfig
from genai_otel.instrumentors.pydantic_ai_instrumentor import PydanticAIInstrumentor


def _fake_pydantic_ai():
    """A fake ``pydantic_ai`` that behaves like the real package where it matters.

    Written against Pydantic AI 2.55. What the instrumentor depends on:

    * ``run()`` is a coroutine function, ``run_stream()`` and ``iter()`` are built with
      ``@asynccontextmanager``, and ``run()`` / ``run_stream()`` both drive ``iter()``.
    * ``run_sync()`` hands ``run()`` to a new task, so it runs in a copy of the context.
    * An unnamed agent is named after the variable that holds it, found by scanning the
      locals of the frame that called the entry point.
    * Results expose ``output``, ``usage`` (a property with ``input_tokens`` /
      ``output_tokens``), ``response`` and ``all_messages()``; the agent keeps its tools in
      ``_function_toolset`` and its output type in ``output_type``.
    """

    class RunUsage:
        def __init__(self, input_tokens=0, output_tokens=0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

        @property
        def total_tokens(self):
            return self.input_tokens + self.output_tokens

    class ModelResponse:
        model_name = "gpt-4o-mini-2024-07-18"
        finish_reason = "stop"

    class AgentRunResult:
        def __init__(self, output, usage):
            self.output = output
            self._state_usage = usage

        @property
        def usage(self):
            return self._state_usage

        @property
        def response(self):
            return ModelResponse()

        def all_messages(self):
            return ["request", "response"]

    class AgentRun:
        """What ``iter()`` yields; ``drive()`` stands in for iterating the graph."""

        def __init__(self, agent, user_prompt):
            self._agent = agent
            self._user_prompt = user_prompt
            self.result = None
            self.usage = RunUsage()

        async def drive(self):
            await asyncio.sleep(0)
            self._agent.model_request()
            self.usage = RunUsage(input_tokens=12, output_tokens=5)
            self.result = AgentRunResult("reply to " + str(self._user_prompt), self.usage)
            return self.result

    class StreamedRunResult:
        def __init__(self, agent_run):
            self._agent_run = agent_run
            self.is_complete = False

        @property
        def usage(self):
            return self._agent_run.usage

        @property
        def response(self):
            return ModelResponse()

        @property
        def result(self):
            if not self.is_complete:
                raise RuntimeError("The run is still streaming")
            return self._agent_run.result

        async def get_output(self):
            self.is_complete = True
            return self._agent_run.result.output

    class OpenAIResponsesModel:
        model_name = "gpt-4o-mini"

    class Agent:
        def __init__(self, name=None, output_type=str):
            self.name = name
            self.model = OpenAIResponsesModel()
            self._system_prompts = ("You answer questions about cities.",)
            self._function_toolset = types.SimpleNamespace(tools={"get_country": object()})
            self._output_type = output_type
            # Replaced by tests to observe what a model request made by the agent sees.
            self.model_request = lambda: None

        @property
        def output_type(self):
            return self._output_type

        def _infer_name(self, function_frame):
            assert self.name is None, "Name already set"
            parent_frame = function_frame.f_back if function_frame is not None else None
            if parent_frame is not None:
                for name, item in parent_frame.f_locals.items():
                    if item is self:
                        self.name = name
                        return
                if parent_frame.f_locals != parent_frame.f_globals:
                    for name, item in parent_frame.f_globals.items():
                        if item is self:
                            self.name = name
                            return

        async def run(self, user_prompt=None, *, infer_name=True, model_settings=None):
            if infer_name and self.name is None:
                self._infer_name(inspect.currentframe())
            async with self.iter(user_prompt, infer_name=False) as agent_run:
                await agent_run.drive()
            return agent_run.result

        def run_sync(self, user_prompt=None, *, infer_name=True, model_settings=None):
            if infer_name and self.name is None:
                self._infer_name(inspect.currentframe())
            loop = asyncio.new_event_loop()
            try:
                task = asyncio.ensure_future(
                    self.run(user_prompt, infer_name=False, model_settings=model_settings),
                    loop=loop,
                )
                return loop.run_until_complete(task)
            finally:
                loop.close()

        @asynccontextmanager
        async def run_stream(self, user_prompt=None, *, infer_name=True, model_settings=None):
            if infer_name and self.name is None:
                # f_back because `asynccontextmanager` adds one frame
                self._infer_name(inspect.currentframe().f_back)
            async with self.iter(user_prompt, infer_name=False) as agent_run:
                await agent_run.drive()
                yield StreamedRunResult(agent_run)

        @asynccontextmanager
        async def iter(self, user_prompt=None, *, infer_name=True, model_settings=None):
            if infer_name and self.name is None:
                self._infer_name(inspect.currentframe())
            yield AgentRun(self, user_prompt)

    package = types.ModuleType("pydantic_ai")
    package.Agent = Agent
    return package, types.SimpleNamespace(Agent=Agent, RunUsage=RunUsage)


class _LogRecords(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class TestAgainstRealPackageLayout(unittest.TestCase):
    """Behaviour that only shows up when the wrappers run around a real-shaped agent."""

    def setUp(self):
        package, self.lib = _fake_pydantic_ai()
        patcher = patch.dict("sys.modules", {"pydantic_ai": package})
        patcher.start()
        self.addCleanup(patcher.stop)

        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = provider.get_tracer("test")
        self.instrumentor = PydanticAIInstrumentor()
        self.instrumentor.tracer = self.tracer
        # Content capture on, so the tests can see that the output is read from the
        # current API. It is recorded only on request.
        self.instrumentor.instrument(OTelConfig(enable_content_capture=True))

        # OpenTelemetry reports a context detached in the wrong Context by logging.
        self.context_log = _LogRecords()
        context_logger = logging.getLogger("opentelemetry.context")
        context_logger.addHandler(self.context_log)
        self.addCleanup(context_logger.removeHandler, self.context_log)

    def _spans(self, name):
        return [s for s in self.exporter.get_finished_spans() if s.name == name]

    def _agent(self, **kwargs):
        """An agent whose model requests open a child span, as an LLM call would."""
        agent = self.lib.Agent(**kwargs)

        def model_request():
            with self.tracer.start_as_current_span("fake.llm") as llm_span:
                llm_span.set_attribute("agent", str(agent.name))

        agent.model_request = model_request
        return agent

    # -- agent name inference ------------------------------------------------

    def test_run_sync_keeps_the_inferred_agent_name(self):
        """Pydantic AI scans its caller's locals; the wrapper must not be that caller."""
        city_agent = self._agent()

        city_agent.run_sync("hello")

        self.assertEqual(city_agent.name, "city_agent")
        (sync_span,) = self._spans("pydantic_ai.agent.run_sync")
        self.assertEqual(sync_span.attributes["gen_ai.agent.name"], "city_agent")

    def test_awaited_run_keeps_the_inferred_agent_name(self):
        async def main():
            travel_agent = self._agent()
            await travel_agent.run("hello")
            return travel_agent.name

        self.assertEqual(asyncio.run(main()), "travel_agent")
        (run_span,) = self._spans("pydantic_ai.agent.run")
        self.assertEqual(run_span.attributes["gen_ai.agent.name"], "travel_agent")

    def test_run_stream_keeps_the_inferred_agent_name(self):
        async def main():
            stream_agent = self._agent()
            async with stream_agent.run_stream("hello") as stream:
                await stream.get_output()
            return stream_agent.name

        self.assertEqual(asyncio.run(main()), "stream_agent")

    def test_agent_that_cannot_be_named_stays_unnamed(self):
        """Held in an attribute there is no variable to name it after: None, not a
        name taken from this library's own frames."""
        holder = types.SimpleNamespace(agent=self._agent())

        holder.agent.run_sync("hello")

        self.assertIsNone(holder.agent.name)

    def test_explicit_name_and_infer_name_false_are_respected(self):
        named = self._agent(name="explicit")
        opted_out = self._agent()

        named.run_sync("hello")
        opted_out.run_sync("hello", infer_name=False)

        self.assertEqual(named.name, "explicit")
        self.assertIsNone(opted_out.name)

    # -- span lifetime and parenting ------------------------------------------

    def test_run_sync_nests_run_and_the_model_request(self):
        agent = self._agent(name="city_agent")

        result = agent.run_sync("hello")

        self.assertEqual(result.output, "reply to hello")
        (sync_span,) = self._spans("pydantic_ai.agent.run_sync")
        (run_span,) = self._spans("pydantic_ai.agent.run")
        (llm_span,) = self._spans("fake.llm")
        self.assertIsNone(sync_span.parent)
        self.assertEqual(run_span.parent.span_id, sync_span.context.span_id)
        self.assertEqual(llm_span.parent.span_id, run_span.context.span_id)
        self.assertEqual(self._spans("pydantic_ai.agent.iter"), [])

    def test_run_sync_detaches_its_context_where_it_attached_it(self):
        """run_sync() runs run() in a new task: a span attached when run() is called
        and detached when it finishes crosses two Contexts."""
        agent = self._agent(name="city_agent")

        agent.run_sync("hello")

        self.assertEqual(self.context_log.messages, [])
        self.assertFalse(trace.get_current_span().get_span_context().is_valid)

    def test_awaited_run_emits_one_agent_span(self):
        """run() drives iter() itself; that is one run, one span."""
        agent = self._agent(name="city_agent")

        asyncio.run(agent.run("hello"))

        (run_span,) = self._spans("pydantic_ai.agent.run")
        (llm_span,) = self._spans("fake.llm")
        self.assertEqual(self._spans("pydantic_ai.agent.iter"), [])
        self.assertEqual(llm_span.parent.span_id, run_span.context.span_id)
        self.assertEqual(run_span.status.status_code, trace.StatusCode.OK)

    def test_concurrent_runs_keep_their_own_children(self):
        """asyncio.gather() calls every run() before any of them starts executing."""
        first = self._agent(name="first")
        second = self._agent(name="second")

        async def run_both():
            await asyncio.gather(first.run("one"), second.run("two"))
            return trace.get_current_span().get_span_context().is_valid

        span_left_current = asyncio.run(run_both())

        run_spans = {
            s.attributes["gen_ai.agent.name"]: s for s in self._spans("pydantic_ai.agent.run")
        }
        self.assertEqual(sorted(run_spans), ["first", "second"])
        self.assertIsNone(run_spans["first"].parent)
        self.assertIsNone(run_spans["second"].parent)
        for llm_span in self._spans("fake.llm"):
            owner = run_spans[llm_span.attributes["agent"]]
            self.assertEqual(llm_span.parent.span_id, owner.context.span_id)
        self.assertFalse(span_left_current, "a finished span was left as the current span")
        self.assertEqual(self.context_log.messages, [])

    def test_run_stream_span_stays_open_for_the_whole_block(self):
        """run_stream() returns a context manager and does nothing until it is entered."""
        agent = self._agent(name="city_agent")

        async def consume():
            async with agent.run_stream("hello") as stream:
                # The model request has happened and the caller is still streaming.
                self.assertEqual(len(self._spans("fake.llm")), 1)
                self.assertEqual(self._spans("pydantic_ai.agent.run_stream"), [])
                inside = trace.get_current_span().get_span_context().span_id
                output = await stream.get_output()
            return inside, output, trace.get_current_span().get_span_context().is_valid

        inside, output, span_left_current = asyncio.run(consume())

        self.assertEqual(output, "reply to hello")
        (stream_span,) = self._spans("pydantic_ai.agent.run_stream")
        (llm_span,) = self._spans("fake.llm")
        self.assertEqual(inside, stream_span.context.span_id)
        self.assertEqual(llm_span.parent.span_id, stream_span.context.span_id)
        self.assertEqual(stream_span.status.status_code, trace.StatusCode.OK)
        self.assertEqual(stream_span.attributes["pydantic_ai.result.data"], "reply to hello")
        self.assertEqual(stream_span.attributes["gen_ai.usage.input_tokens"], 12)
        self.assertEqual(self._spans("pydantic_ai.agent.iter"), [])
        self.assertFalse(span_left_current, "the stream's span was left as the current span")

    def test_run_stream_error_ends_the_span_with_error_status(self):
        agent = self._agent(name="city_agent")

        async def consume():
            async with agent.run_stream("hello"):
                raise RuntimeError("consumer gave up")

        with self.assertRaises(RuntimeError):
            asyncio.run(consume())

        (stream_span,) = self._spans("pydantic_ai.agent.run_stream")
        self.assertEqual(stream_span.status.status_code, trace.StatusCode.ERROR)

    def test_iter_called_directly_is_traced(self):
        """Applications drive iter() themselves; the model requests need a parent."""
        agent = self._agent(name="city_agent")

        async def drive():
            async with agent.iter("hello") as agent_run:
                await agent_run.drive()
            return agent_run.result.output

        self.assertEqual(asyncio.run(drive()), "reply to hello")

        (iter_span,) = self._spans("pydantic_ai.agent.iter")
        (llm_span,) = self._spans("fake.llm")
        self.assertEqual(llm_span.parent.span_id, iter_span.context.span_id)
        self.assertEqual(iter_span.attributes["gen_ai.agent.name"], "city_agent")
        self.assertEqual(iter_span.attributes["pydantic_ai.result.data"], "reply to hello")
        self.assertEqual(iter_span.status.status_code, trace.StatusCode.OK)

    def test_iter_is_still_traced_after_a_run(self):
        agent = self._agent(name="city_agent")

        async def drive():
            await agent.run("first")  # must not leave iter() untraced afterwards
            async with agent.iter("second") as agent_run:
                await agent_run.drive()

        asyncio.run(drive())

        (iter_span,) = self._spans("pydantic_ai.agent.iter")
        self.assertEqual(iter_span.attributes["pydantic_ai.user_prompt"], "second")
        self.assertEqual(len(self._spans("pydantic_ai.agent.run")), 1)

    def test_run_error_ends_the_span_with_error_status(self):
        agent = self._agent(name="city_agent")

        def boom():
            raise RuntimeError("model backend exploded")

        agent.model_request = boom

        with self.assertRaises(RuntimeError):
            asyncio.run(agent.run("hello"))

        (run_span,) = self._spans("pydantic_ai.agent.run")
        self.assertEqual(run_span.status.status_code, trace.StatusCode.ERROR)
        self.assertEqual(run_span.events[0].name, "exception")

    def test_sampled_out_runs_execute_exactly_once(self):
        """A dropped span is a NonRecordingSpan; the run must neither fail nor repeat."""
        self.instrumentor.tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
        agent = self.lib.Agent(name="city_agent")
        requests = []
        agent.model_request = lambda: requests.append(1)

        async def drive():
            result = await agent.run("hello")
            async with agent.run_stream("hello") as stream:
                await stream.get_output()
            async with agent.iter("hello") as agent_run:
                await agent_run.drive()
            return result.output, trace.get_current_span().get_span_context().is_valid

        output, span_left_current = asyncio.run(drive())
        agent.run_sync("hello")

        self.assertEqual(output, "reply to hello")
        self.assertEqual(len(requests), 4)
        self.assertFalse(span_left_current)
        self.assertEqual(agent_budget._BUDGET_STACK.get(), ())
        self.assertEqual(self.context_log.messages, [])

    def test_wrapped_methods_keep_their_nature(self):
        """Frameworks and users introspect these; the wrappers must not hide it."""
        agent = self._agent()

        self.assertTrue(inspect.iscoroutinefunction(agent.run))
        self.assertFalse(inspect.iscoroutinefunction(agent.run_sync))
        coroutine = agent.run("hello")
        self.assertTrue(inspect.iscoroutine(coroutine))
        coroutine.close()
        self.assertTrue(hasattr(agent.run_stream("hello"), "__aenter__"))
        self.assertTrue(hasattr(agent.iter("hello"), "__aenter__"))
        # Nothing ran, so nothing may have been opened or left attached.
        self.assertEqual(self.exporter.get_finished_spans(), ())
        self.assertFalse(trace.get_current_span().get_span_context().is_valid)

    # -- what the spans carry --------------------------------------------------

    def test_run_span_carries_current_api_attributes(self):
        """Tools, output type, output, tokens and response model under their 2.x names."""
        agent = self._agent(name="city_agent", output_type=dict)

        asyncio.run(agent.run("hello", model_settings={"temperature": 0.0, "max_tokens": 80}))

        (run_span,) = self._spans("pydantic_ai.agent.run")
        attrs = run_span.attributes
        self.assertEqual(attrs["gen_ai.operation.name"], "agent.run")
        self.assertEqual(attrs["gen_ai.request.model"], "gpt-4o-mini")
        self.assertEqual(attrs["pydantic_ai.tools.count"], 1)
        self.assertIn("get_country", attrs["pydantic_ai.tools"])
        self.assertEqual(attrs["pydantic_ai.result_type"], "<class 'dict'>")
        self.assertEqual(attrs["pydantic_ai.result.data"], "reply to hello")
        self.assertEqual(attrs["pydantic_ai.result.messages.count"], 2)
        self.assertEqual(attrs["gen_ai.response.model"], "gpt-4o-mini-2024-07-18")
        self.assertEqual(attrs["gen_ai.response.finish_reasons"], ("stop",))
        self.assertEqual(attrs["gen_ai.usage.input_tokens"], 12)
        self.assertEqual(attrs["gen_ai.usage.output_tokens"], 5)
        self.assertEqual(attrs["gen_ai.usage.total_tokens"], 17)
        self.assertEqual(attrs["gen_ai.request.max_tokens"], 80)

    def test_usage_is_recorded_once_for_run_sync(self):
        """run_sync() and the run() inside it see the same usage object."""
        agent = self._agent(name="city_agent")

        agent.run_sync("hello")

        (sync_span,) = self._spans("pydantic_ai.agent.run_sync")
        (run_span,) = self._spans("pydantic_ai.agent.run")
        self.assertEqual(run_span.attributes["gen_ai.usage.input_tokens"], 12)
        self.assertNotIn("gen_ai.usage.input_tokens", sync_span.attributes)
        self.assertEqual(sync_span.attributes["gen_ai.usage.recorded_by"], "descendant")

    def test_budget_consumption_is_reported_and_no_frame_is_left_open(self):
        """The base wrapper closes an agent budget frame only on its synchronous path."""
        agent = self._agent(name="city_agent")

        async def run_repeatedly():
            for _ in range(3):
                await agent.run("hello")
                async with agent.run_stream("hello") as stream:
                    await stream.get_output()
                async with agent.iter("hello") as agent_run:
                    await agent_run.drive()
            return agent_budget._BUDGET_STACK.get()

        self.assertEqual(asyncio.run(run_repeatedly()), ())
        agent.run_sync("hello")
        self.assertEqual(agent_budget._BUDGET_STACK.get(), ())

        for name in (
            "pydantic_ai.agent.run_stream",
            "pydantic_ai.agent.iter",
            "pydantic_ai.agent.run_sync",
        ):
            span = self._spans(name)[0]
            self.assertEqual(span.attributes["gen_ai.agent.token_budget.consumed"], 17, name)
        # Awaited directly, run() is the invocation; inside run_sync() it is not.
        run_spans = self._spans("pydantic_ai.agent.run")
        awaited, inside_run_sync = run_spans[0], run_spans[-1]
        self.assertEqual(awaited.attributes["gen_ai.agent.token_budget.consumed"], 17)
        self.assertNotIn("gen_ai.agent.token_budget.consumed", inside_run_sync.attributes)

    def test_extract_usage_reads_current_and_method_style_usage(self):
        """input_tokens / output_tokens; usage() was a method in older releases."""
        usage = self.lib.RunUsage(input_tokens=215, output_tokens=34)
        as_property = types.SimpleNamespace(usage=usage)
        as_method = types.SimpleNamespace(usage=lambda: usage)

        expected = {"prompt_tokens": 215, "completion_tokens": 34, "total_tokens": 249}
        self.assertEqual(self.instrumentor._extract_usage(as_property), expected)
        self.assertEqual(self.instrumentor._extract_usage(as_method), expected)

    def test_extract_finish_reason_falls_back_to_completed(self):
        """A response without a finish reason (test and function models) still completed."""
        response = types.SimpleNamespace(finish_reason=None, model_name="test")
        result = types.SimpleNamespace(output="done", response=response)

        self.assertEqual(self.instrumentor._extract_finish_reason(result), "completed")


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

            # Create mock result (older shape: `data`, and no `output`)
            result = MagicMock(spec=["data", "model"])
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

            # Create mock result (older shape: `data`, and no `output`)
            result = MagicMock(spec=["data"])
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

    @patch("genai_otel.instrumentors.pydantic_ai_instrumentor.logger")
    def test_instrument_idempotent_guard(self, mock_logger):
        """Repeated instrument() must return early via the idempotency guard."""
        instrumentor = PydanticAIInstrumentor()
        instrumentor._pydantic_ai_available = True
        instrumentor._instrumented = True
        instrumentor.instrument(OTelConfig())
        mock_logger.debug.assert_any_call(
            "Pydantic AI already instrumented, skipping repeat instrument()"
        )

    def test_cap_content_respects_config(self):
        """_cap_content honours content_max_length (default, small cap, unlimited)."""
        instrumentor = PydanticAIInstrumentor()
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 200)
        cfg = OTelConfig()
        cfg.content_max_length = 3
        instrumentor.config = cfg
        self.assertEqual(instrumentor._cap_content("abcdef"), "abc")
        cfg.content_max_length = 0
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 300)


class TestOlderPydanticAIShapes(unittest.TestCase):
    """Result and stream shapes from before Pydantic AI 2.0."""

    def _response_attributes(self, result):
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            return PydanticAIInstrumentor()._extract_response_attributes(result)

    def test_method_style_timestamp_is_called_not_stringified(self):
        """timestamp was a method before 2.0. str() of a bound method embeds the repr
        of the object it is bound to, which for a result is the whole conversation."""

        class _OldResult:
            output = "done"

            def __repr__(self):
                return "OldResult(messages=['THE WHOLE CONVERSATION'])"

            def timestamp(self):
                return "2026-01-01 00:00:00+00:00"

        attrs = self._response_attributes(_OldResult())

        self.assertEqual(attrs["pydantic_ai.result.timestamp"], "2026-01-01 00:00:00+00:00")

    def test_timestamp_that_cannot_be_read_is_left_out(self):
        class _NoResponseYet:
            output = "done"

            def __repr__(self):
                return "Result(messages=['THE WHOLE CONVERSATION'])"

            def timestamp(self):
                raise RuntimeError("no response yet")

        attrs = self._response_attributes(_NoResponseYet())

        self.assertNotIn("pydantic_ai.result.timestamp", attrs)

    def test_all_zero_usage_is_no_usage(self):
        """An untouched usage object is not a record of usage. Reported as one, it
        told every enclosing span that its tokens had been counted beneath it."""
        usage = types.SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)

        self.assertIsNone(PydanticAIInstrumentor._usage_counts(usage))

    def test_run_stream_sync_that_never_leaves_its_block_is_not_traced(self):
        """Before 2.0, run_stream_sync() entered run_stream() and never left the
        block, so a span opened for it could never end."""
        package, lib = _fake_pydantic_ai()
        handed_back = []

        def run_stream_sync(agent, user_prompt=None):
            handed_back.append(agent.run_stream(user_prompt))
            return "stream"

        lib.Agent.run_stream_sync = run_stream_sync
        with patch.dict("sys.modules", {"pydantic_ai": package}):
            instrumentor = PydanticAIInstrumentor()
            instrumentor.tracer = TracerProvider().get_tracer("test")
            instrumentor.instrument(OTelConfig())
            agent = lib.Agent(name="streamer")

            self.assertEqual(agent.run_stream_sync("hi"), "stream")
            # Pydantic AI got its own context manager back, not one holding a span.
            self.assertNotEqual(type(handed_back[0]).__name__, "_TracedAgentContext")
            # An ordinary run_stream() is still traced.
            self.assertEqual(type(agent.run_stream("hi")).__name__, "_TracedAgentContext")


class TestOutputFollowsTheContentSwitch(unittest.TestCase):
    """The model's output is content. It is recorded only when content capture is on."""

    def _response_attributes(self, enable_content_capture):
        with patch.dict("sys.modules", {"pydantic_ai": MagicMock()}):
            instrumentor = PydanticAIInstrumentor()
            instrumentor.config = OTelConfig(enable_content_capture=enable_content_capture)
            result = MagicMock(spec=["output"])
            result.output = "the model's answer"
            return instrumentor._extract_response_attributes(result)

    def test_output_is_not_recorded_with_content_capture_off(self):
        self.assertNotIn("pydantic_ai.result.data", self._response_attributes(False))

    def test_output_is_recorded_with_content_capture_on(self):
        attrs = self._response_attributes(True)

        self.assertEqual(attrs["pydantic_ai.result.data"], "the model's answer")


if __name__ == "__main__":
    unittest.main()
