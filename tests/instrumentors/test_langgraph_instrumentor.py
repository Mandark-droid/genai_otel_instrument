import asyncio
import gc
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from genai_otel import agent_budget
from genai_otel.config import OTelConfig
from genai_otel.instrumentors.langgraph_instrumentor import LangGraphInstrumentor


class TestLangGraphInstrumentor(unittest.TestCase):
    """Tests for LangGraphInstrumentor"""

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_init_with_langgraph_available(self, mock_logger):
        """Test that __init__ detects LangGraph availability."""
        with patch.dict("sys.modules", {"langgraph": MagicMock(), "langgraph.pregel": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            self.assertTrue(instrumentor._langgraph_available)
            mock_logger.debug.assert_called_with(
                "LangGraph library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_init_with_langgraph_not_available(self, mock_logger):
        """Test that __init__ handles missing LangGraph gracefully."""
        with patch.dict("sys.modules", {"langgraph": None}):
            instrumentor = LangGraphInstrumentor()

            self.assertFalse(instrumentor._langgraph_available)
            mock_logger.debug.assert_called_with(
                "LangGraph library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_when_langgraph_not_available(self, mock_logger):
        """Test that instrument skips when LangGraph is not available."""
        with patch.dict("sys.modules", {"langgraph": None}):
            instrumentor = LangGraphInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping LangGraph instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_with_langgraph_available(self, mock_logger):
        """Test that instrument wraps the run methods of the Pregel class when available."""

        class MockPregel:
            def invoke(self, input, config=None):
                return {"result": "test"}

            def stream(self, input, config=None):
                yield {"result": "test"}

            async def ainvoke(self, input, config=None):
                return {"result": "test"}

            async def astream(self, input, config=None):
                yield {"result": "test"}

        mock_pregel_module = MagicMock()
        mock_pregel_module.Pregel = MockPregel

        # Create a mock wrapt module
        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "langgraph": MagicMock(),
                "langgraph.pregel": mock_pregel_module,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = LangGraphInstrumentor()
            config = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("LangGraph instrumentation enabled")
            # Verify each run method was replaced by a FunctionWrapper, on the class
            self.assertEqual(mock_wrapt.FunctionWrapper.call_count, 4)
            for name in ("invoke", "stream", "ainvoke", "astream"):
                self.assertIs(getattr(MockPregel, name), mock_wrapt.FunctionWrapper.return_value)

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        # Make the Pregel access raise an exception
        mock_pregel_module = MagicMock()
        type(mock_pregel_module).Pregel = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
        )

        with patch.dict(
            "sys.modules",
            {
                "langgraph": MagicMock(),
                "langgraph.pregel": mock_pregel_module,
                "wrapt": MagicMock(),
            },
        ):
            instrumentor = LangGraphInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            # Should not raise exception
            instrumentor.instrument(config)

            mock_logger.error.assert_called_once()
            self.assertFalse(instrumentor._instrumented)

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that instrument raises exceptions when fail_on_error is True."""
        # Make the Pregel access raise an exception
        mock_pregel_module = MagicMock()
        type(mock_pregel_module).Pregel = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
        )

        with patch.dict(
            "sys.modules",
            {
                "langgraph": MagicMock(),
                "langgraph.pregel": mock_pregel_module,
                "wrapt": MagicMock(),
            },
        ):
            instrumentor = LangGraphInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            # Should raise exception
            with self.assertRaises(RuntimeError):
                instrumentor.instrument(config)

    def test_extract_graph_attributes_basic(self):
        """Test extraction of basic graph attributes."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock state graph
            mock_state_graph = MagicMock()
            mock_state_graph.nodes = {"node1": MagicMock(), "node2": MagicMock()}
            mock_state_graph.edges = [("node1", "node2")]
            mock_state_graph.channels = {"messages": MagicMock(), "state": MagicMock()}

            # Input state
            input_state = {"messages": ["Hello"], "query": "test"}
            args = (input_state,)
            kwargs = {}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "langgraph")
            self.assertEqual(attrs["gen_ai.operation.name"], "graph.execution")
            self.assertEqual(attrs["langgraph.node_count"], 2)
            self.assertIn("node1", attrs["langgraph.nodes"])
            self.assertIn("node2", attrs["langgraph.nodes"])
            self.assertEqual(attrs["langgraph.edge_count"], 1)
            self.assertEqual(attrs["langgraph.channel_count"], 2)
            self.assertIn("messages", attrs["langgraph.channels"])

    def test_extract_graph_attributes_with_input_state(self):
        """Test extraction of graph attributes with input state."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock state graph
            mock_state_graph = MagicMock()

            # Input state with various keys
            input_state = {
                "messages": ["message1", "message2"],
                "query": "What is LangGraph?",
                "custom_key": "custom_value",
            }
            args = (input_state,)
            kwargs = {}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            # Assert
            self.assertIn("messages", attrs["langgraph.input.keys"])
            self.assertIn("query", attrs["langgraph.input.keys"])
            self.assertIn("custom_key", attrs["langgraph.input.keys"])
            self.assertIn("langgraph.input.messages", attrs)
            self.assertIn("langgraph.input.query", attrs)

    def test_extract_graph_attributes_with_config(self):
        """Test extraction of graph attributes with config."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock state graph
            mock_state_graph = MagicMock()

            # Input with config
            input_state = {"messages": []}
            config = {
                "configurable": {"thread_id": "thread_123", "checkpoint_id": "checkpoint_456"},
                "recursion_limit": 25,
            }
            args = (input_state,)
            kwargs = {"config": config}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            # Assert
            self.assertEqual(attrs["langgraph.thread_id"], "thread_123")
            self.assertEqual(attrs["langgraph.checkpoint_id"], "checkpoint_456")
            self.assertEqual(attrs["langgraph.recursion_limit"], 25)

    def test_extract_response_attributes_dict_result(self):
        """Test extraction of response attributes from dict result."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Test with dict result
            result = {
                "messages": ["msg1", "msg2", "msg3"],
                "answer": "This is the answer",
                "output": "Final output",
            }

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertIn("messages", attrs["langgraph.output.keys"])
            self.assertIn("answer", attrs["langgraph.output.keys"])
            self.assertIn("output", attrs["langgraph.output.keys"])
            self.assertEqual(attrs["langgraph.message_count"], 3)
            self.assertIn("langgraph.output.messages", attrs)
            self.assertIn("langgraph.output.answer", attrs)

    def test_extract_response_attributes_with_metadata(self):
        """Test extraction of response attributes with metadata."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock result with metadata using a class that acts like a dict
            class ResultWithMetadata(dict):
                pass

            result = ResultWithMetadata({"result": "test"})
            result.__metadata__ = {"step": 5}

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["langgraph.steps"], 5)

    def test_extract_usage_returns_none(self):
        """Test that extract_usage returns None (LangGraph doesn't provide usage)."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock result
            mock_result = MagicMock()

            usage = instrumentor._extract_usage(mock_result)

            # Assert
            self.assertIsNone(usage)

    def test_extract_finish_reason(self):
        """Test extraction of finish reason from result."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock result
            mock_result = {"output": "test"}

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_finish_reason_none_result(self):
        """Test extraction of finish reason with None result."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # None result
            finish_reason = instrumentor._extract_finish_reason(None)

            # Assert
            self.assertIsNone(finish_reason)

    def test_extract_graph_attributes_session_id_from_input(self):
        """Test session.id from input state session_id key."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()
            instrumentor.config = None

            mock_state_graph = MagicMock()
            input_state = {"messages": [], "session_id": "input-session-001"}
            args = (input_state,)
            kwargs = {}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            self.assertEqual(attrs["session.id"], "input-session-001")
            self.assertEqual(attrs["langgraph.session.id"], "input-session-001")

    def test_extract_graph_attributes_session_id_from_thread_id(self):
        """Test session.id falls back to config thread_id."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()
            instrumentor.config = None

            mock_state_graph = MagicMock()
            input_state = {"messages": []}
            config = {"configurable": {"thread_id": "thread-abc"}}
            args = (input_state,)
            kwargs = {"config": config}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            self.assertEqual(attrs["session.id"], "thread-abc")
            self.assertEqual(attrs["langgraph.session.id"], "thread-abc")

    def test_extract_graph_attributes_session_id_input_over_thread(self):
        """Test that input session_id takes priority over thread_id."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()
            instrumentor.config = None

            mock_state_graph = MagicMock()
            input_state = {"messages": [], "session_id": "explicit-session"}
            config = {"configurable": {"thread_id": "thread-xyz"}}
            args = (input_state,)
            kwargs = {"config": config}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            self.assertEqual(attrs["session.id"], "explicit-session")

    def test_extract_graph_attributes_session_id_from_extractor(self):
        """Test session.id from OTelConfig extractor."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()
            instrumentor.config = MagicMock()
            instrumentor.config.session_id_extractor = lambda i, a, k: "extractor-session"

            mock_state_graph = MagicMock()
            args = ({"messages": []},)
            kwargs = {}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            self.assertEqual(attrs["session.id"], "extractor-session")

    def test_extract_graph_attributes_session_id_autogenerated(self):
        """Test that UUID is auto-generated when no session source exists."""
        import uuid

        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()
            instrumentor.config = None

            mock_state_graph = MagicMock()
            args = ({"messages": []},)
            kwargs = {}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            self.assertIn("session.id", attrs)
            self.assertIn("langgraph.session.id", attrs)
            # Verify it's a valid UUID
            uuid.UUID(attrs["session.id"])

    def test_extract_graph_attributes_basic_has_session_id(self):
        """Test that basic graph attributes include session.id."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()
            instrumentor.config = None

            mock_state_graph = MagicMock()
            mock_state_graph.nodes = {"node1": MagicMock()}
            mock_state_graph.edges = [("node1", "node2")]

            input_state = {"messages": ["Hello"]}
            args = (input_state,)
            kwargs = {}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            self.assertIn("session.id", attrs)
            self.assertIn("langgraph.session.id", attrs)

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_idempotent_guard(self, mock_logger):
        """Repeated instrument() must return early via the idempotency guard."""
        instrumentor = LangGraphInstrumentor()
        instrumentor._langgraph_available = True
        instrumentor._instrumented = True
        instrumentor.instrument(OTelConfig())
        mock_logger.debug.assert_any_call(
            "LangGraph already instrumented, skipping repeat instrument()"
        )

    def test_cap_content_respects_config(self):
        """_cap_content honours content_max_length (default, small cap, unlimited)."""
        instrumentor = LangGraphInstrumentor()
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 200)
        cfg = OTelConfig()
        cfg.content_max_length = 3
        instrumentor.config = cfg
        self.assertEqual(instrumentor._cap_content("abcdef"), "abc")
        cfg.content_max_length = 0
        self.assertEqual(instrumentor._cap_content("x" * 300), "x" * 300)


# ---------------------------------------------------------------------------
# Behaviour against a stand-in shaped like the real framework.
#
# The tests above mock LangGraph away, so they could not see any of what follows. Each
# test below reproduces something a run against real LangGraph (0.1.19, 0.2.76, 0.6.11
# and 1.2.14) showed.
# ---------------------------------------------------------------------------


def _fake_langgraph_modules():
    """A stand-in LangGraph with the real one's shape.

    As in LangGraph: ``StateGraph.compile()`` returns a ``Pregel``; ``invoke`` is a loop
    over ``self.stream`` and ``ainvoke`` over ``self.astream``; ``stream`` does nothing
    until it is iterated; and ``copy()``, which ``with_config()`` uses, feeds every
    instance attribute back into ``__init__``.
    """

    class Pregel:
        on_node = None  # test hook, called while a node "runs"

        def __init__(self, *, builder=None, config=None):
            self.builder = builder
            self.config = config

        def _run_node(self, name, input):
            hook = type(self).on_node
            if hook is not None:
                hook(name)
            if input.get("fail_at") == name:
                raise ValueError("node exploded")
            return {name: {"question": input.get("question")}}

        def stream(self, input, config=None, **kwargs):
            for name in ("first", "second"):
                yield self._run_node(name, input)

        def invoke(self, input, config=None, **kwargs):
            chunks = list(self.stream(input, config, **kwargs))
            return {"question": input.get("question"), "answer": "done", "steps": len(chunks)}

        async def astream(self, input, config=None, **kwargs):
            for name in ("first", "second"):
                await asyncio.sleep(0)
                yield self._run_node(name, input)

        async def ainvoke(self, input, config=None, **kwargs):
            chunks = [chunk async for chunk in self.astream(input, config, **kwargs)]
            return {"question": input.get("question"), "answer": "done", "steps": len(chunks)}

        def copy(self, update=None):
            attrs = dict(self.__dict__)
            attrs.update(update or {})
            return self.__class__(**attrs)

        def with_config(self, config=None):
            return self.copy({"config": config})

    class StateGraph:
        def __init__(self):
            self.nodes = {"first": object(), "second": object()}
            self.edges = {("first", "second")}
            self.channels = {"question": object(), "answer": object()}

        def compile(self, **kwargs):
            return Pregel(builder=self, **kwargs)

    package = types.ModuleType("langgraph")
    package.__path__ = []  # a namespace package, as the real one is
    graph_module = types.ModuleType("langgraph.graph")
    graph_module.StateGraph = StateGraph
    pregel_module = types.ModuleType("langgraph.pregel")
    pregel_module.Pregel = Pregel
    return {
        "langgraph": package,
        "langgraph.graph": graph_module,
        "langgraph.pregel": pregel_module,
    }


@pytest.fixture
def lg():
    """The stand-in LangGraph, instrumented, with every finished span captured."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    modules = _fake_langgraph_modules()
    # Start from an empty context, whatever an earlier test left current.
    clean = otel_context.attach(otel_context.Context())
    try:
        with patch.dict("sys.modules", modules):
            instrumentor = LangGraphInstrumentor()
            instrumentor.tracer = tracer
            instrumentor.instrument(OTelConfig())
            yield types.SimpleNamespace(
                instrumentor=instrumentor,
                tracer=tracer,
                Pregel=modules["langgraph.pregel"].Pregel,
                compile=lambda **kwargs: modules["langgraph.graph"].StateGraph().compile(**kwargs),
                spans=exporter.get_finished_spans,
                names=lambda: [span.name for span in exporter.get_finished_spans()],
            )
    finally:
        otel_context.detach(clean)


def _a_span_is_current() -> bool:
    return trace.get_current_span().get_span_context().is_valid


def _run(graph, how, payload=None):
    """Run ``graph`` to completion through one of its four entry points."""
    payload = payload or {"question": "q"}
    if how == "invoke":
        return graph.invoke(payload)
    if how == "stream":
        return list(graph.stream(payload))
    if how == "ainvoke":
        return asyncio.run(graph.ainvoke(payload))

    async def consume():
        return [chunk async for chunk in graph.astream(payload)]

    return asyncio.run(consume())


RUN_METHODS = ["invoke", "stream", "ainvoke", "astream"]


def test_namespace_only_install_is_not_available():
    """`import langgraph` is not proof of LangGraph.

    ``langgraph`` is a namespace package that langgraph-checkpoint and langgraph-prebuilt
    install into too, so it imports with the graph runtime absent. That is "not
    installed": instrument() must skip, not fail on its own import.
    """
    namespace = types.ModuleType("langgraph")
    namespace.__path__ = []
    with patch.dict("sys.modules", {"langgraph": namespace}):
        for name in ("langgraph.pregel", "langgraph.graph"):
            sys.modules.pop(name, None)

        instrumentor = LangGraphInstrumentor()
        assert instrumentor._langgraph_available is False

        config = OTelConfig()
        config.fail_on_error = True
        with patch("genai_otel.instrumentors.langgraph_instrumentor.logger") as mock_logger:
            instrumentor.instrument(config)  # must not raise
        mock_logger.error.assert_not_called()
        assert instrumentor._instrumented is False


def test_invoke_is_one_graph_span(lg):
    """invoke() loops over self.stream(), which is wrapped too. That inner call must not
    add a second graph span: it did, with a session id of its own."""
    result = lg.compile().invoke({"question": "q"})

    assert result["answer"] == "done"
    assert lg.names() == ["langgraph.graph.invoke"]
    (span,) = lg.spans()
    assert span.status.status_code is StatusCode.OK
    assert span.attributes["langgraph.node_count"] == 2
    assert span.attributes["langgraph.output.answer"] == "done"


def test_ainvoke_is_one_graph_span(lg):
    result = asyncio.run(lg.compile().ainvoke({"question": "q"}))

    assert result["answer"] == "done"
    assert lg.names() == ["langgraph.graph.ainvoke"]
    (span,) = lg.spans()
    assert span.status.status_code is StatusCode.OK
    assert span.attributes["langgraph.output.answer"] == "done"


@pytest.mark.parametrize("how", ["stream", "astream"])
def test_stream_called_directly_is_traced_until_exhausted(lg, how):
    chunks = _run(lg.compile(), how)

    assert [list(chunk) for chunk in chunks] == [["first"], ["second"]]
    assert lg.names() == ["langgraph.graph." + how]
    (span,) = lg.spans()
    assert span.status.status_code is StatusCode.OK
    assert span.attributes["langgraph.input.question"] == "q"


def test_with_config_copy_is_traced(lg):
    """with_config() builds a new graph from the old one's instance attributes, and
    langchain.agents.create_agent returns such a copy. Wrappers kept on the instance were
    dropped by the copy on current LangGraph (no spans at all) and were passed to
    ``__init__`` on 0.2.x, where with_config() then raised TypeError."""
    graph = lg.compile()

    copy = graph.with_config({"tags": ["copied"]})
    assert copy.invoke({"question": "q"})["answer"] == "done"
    assert lg.names() == ["langgraph.graph.invoke"]

    assert copy.with_config({"tags": ["again"]}).invoke({"question": "q"})["answer"] == "done"
    assert lg.names() == ["langgraph.graph.invoke"] * 2


def test_config_passed_positionally_is_read(lg):
    """graph.invoke(input, config) is the usual call form. The thread id, and with it
    the session id, was only found when config was passed as a keyword."""
    config = {"configurable": {"thread_id": "thread-7"}, "recursion_limit": 9}

    lg.compile().invoke({"question": "q"}, config)

    (span,) = lg.spans()
    assert span.attributes["langgraph.thread_id"] == "thread-7"
    assert span.attributes["session.id"] == "thread-7"
    assert span.attributes["langgraph.session.id"] == "thread-7"
    assert span.attributes["gen_ai.conversation.id"] == "thread-7"
    assert span.attributes["langgraph.recursion_limit"] == 9
    assert span.attributes["gen_ai.agent.iteration_budget"] == 9


def test_unset_ids_in_the_config_are_not_recorded(lg):
    """LangGraph runs a subgraph with a config of its own in which checkpoint_id is
    present and None. That must not become the attribute value 'None'."""
    config = {"configurable": {"thread_id": None, "checkpoint_id": None}}

    lg.compile().invoke({"question": "q"}, config)

    (span,) = lg.spans()
    assert "langgraph.thread_id" not in span.attributes
    assert "langgraph.checkpoint_id" not in span.attributes
    assert span.attributes["session.id"] not in (None, "None", "")


@pytest.mark.parametrize("how", RUN_METHODS)
def test_node_work_is_a_child_of_the_graph_span(lg, how):
    def node_work(name):
        with lg.tracer.start_as_current_span("node." + name):
            pass

    lg.Pregel.on_node = node_work
    _run(lg.compile(), how)

    spans = {span.name: span for span in lg.spans()}
    assert sorted(spans) == ["langgraph.graph." + how, "node.first", "node.second"]
    graph_span = spans["langgraph.graph." + how]
    assert graph_span.parent is None
    for name in ("node.first", "node.second"):
        assert spans[name].parent.span_id == graph_span.context.span_id
    assert not _a_span_is_current()


def test_graph_runs_started_inside_a_node_nest(lg):
    """A subgraph, and a stream of the same graph from inside one of its own nodes."""
    outer, inner = lg.compile(), lg.compile()
    started = []

    def node_work(name):
        if not started:
            started.append(name)
            inner.invoke({"question": "inner"})
            list(outer.stream({"question": "again"}))

    lg.Pregel.on_node = node_work
    outer.invoke({"question": "outer"})

    by_question = {span.attributes["langgraph.input.question"]: span for span in lg.spans()}
    assert len(lg.spans()) == 3
    assert by_question["outer"].name == "langgraph.graph.invoke"
    assert by_question["outer"].parent is None
    for question, name in (("inner", "invoke"), ("again", "stream")):
        assert by_question[question].name == "langgraph.graph." + name
        assert by_question[question].parent.span_id == by_question["outer"].context.span_id


def test_concurrent_ainvoke_runs_are_separate_traces(lg):
    """The span is opened when the coroutine runs, in the task that runs it. Opened when
    ainvoke() was called, the first run's span became the parent of the second, and both
    were still current for the caller after they had ended."""
    graph = lg.compile()

    async def main():
        await asyncio.gather(graph.ainvoke({"question": "1"}), graph.ainvoke({"question": "2"}))
        return _a_span_is_current()

    left_current = asyncio.run(main())

    spans = lg.spans()
    assert [span.name for span in spans] == ["langgraph.graph.ainvoke"] * 2
    assert all(span.parent is None for span in spans)
    assert len({span.context.trace_id for span in spans}) == 2
    assert left_current is False


def test_ainvoke_run_by_another_task_leaves_no_current_span(lg):
    """asyncio.run(graph.ainvoke(...)) creates the coroutine here and runs it in a new
    task. The finished span stayed current here and adopted every later run."""
    graph = lg.compile()

    asyncio.run(graph.ainvoke({"question": "q"}))
    assert not _a_span_is_current()

    graph.invoke({"question": "q"})
    first, second = lg.spans()
    assert second.parent is None
    assert second.context.trace_id != first.context.trace_id


@pytest.mark.parametrize("how", ["stream", "astream"])
def test_stream_span_is_current_for_the_graph_not_for_the_consumer(lg, how):
    seen_by_nodes = []
    lg.Pregel.on_node = lambda name: seen_by_nodes.append(
        getattr(trace.get_current_span(), "name", None)
    )
    graph = lg.compile()
    seen_by_consumer = []

    if how == "stream":
        for _ in graph.stream({"question": "q"}):
            seen_by_consumer.append(_a_span_is_current())
    else:

        async def consume():
            async for _ in graph.astream({"question": "q"}):
                seen_by_consumer.append(_a_span_is_current())

        asyncio.run(consume())

    assert seen_by_nodes == ["langgraph.graph." + how] * 2
    assert seen_by_consumer == [False, False]


def test_stream_never_iterated_starts_nothing(lg):
    """stream() does no work until it is iterated. Its span used to start, and become
    current, at the call: never iterated, it never ended."""
    lg.compile().stream({"question": "q"})
    gc.collect()

    assert not _a_span_is_current()
    assert lg.names() == []


def test_stream_stopped_early_ends_the_span(lg):
    chunks = lg.compile().stream({"question": "q"})
    next(chunks)
    assert lg.names() == []  # the run is still open
    assert not _a_span_is_current()

    chunks.close()

    assert lg.names() == ["langgraph.graph.stream"]
    assert lg.spans()[0].status.status_code is StatusCode.UNSET
    assert not _a_span_is_current()


def test_astream_abandoned_without_aclose_ends_the_span(lg):
    graph = lg.compile()

    async def main():
        async for _ in graph.astream({"question": "q"}):
            break
        gc.collect()
        for _ in range(3):  # the loop closes the abandoned generator in a task of its own
            await asyncio.sleep(0)
        return _a_span_is_current()

    assert asyncio.run(main()) is False
    assert lg.names() == ["langgraph.graph.astream"]


@pytest.mark.parametrize("how", RUN_METHODS)
def test_node_error_marks_the_graph_span(lg, how):
    with pytest.raises(ValueError, match="node exploded"):
        _run(lg.compile(), how, {"question": "q", "fail_at": "second"})

    (span,) = lg.spans()
    assert span.name == "langgraph.graph." + how
    assert span.status.status_code is StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]
    assert not _a_span_is_current()


def test_budget_frames_do_not_outlive_the_run(lg):
    """A graph span opens an agent-budget frame. Streamed and async runs left theirs on
    the context-local stack: one more frame per run, for as long as the context lived."""
    graph = lg.compile()

    def depth():
        return len(agent_budget._BUDGET_STACK.get())

    before = depth()
    graph.invoke({"question": "q"})
    list(graph.stream({"question": "q"}))
    assert depth() == before

    async def main():
        inside = depth()
        await graph.ainvoke({"question": "q"})
        await graph.ainvoke({"question": "q"})
        async for _ in graph.astream({"question": "q"}):
            pass
        return depth() - inside

    assert asyncio.run(main()) == 0


@pytest.mark.parametrize("how", RUN_METHODS)
def test_session_and_user_extractors_apply_to_every_run_method(lg, how):
    """Streamed runs are traced outside the base span wrapper and must not lose what
    that wrapper adds to the other two."""
    config = OTelConfig()
    config.session_id_extractor = lambda instance, args, kwargs: "session-from-extractor"
    config.user_id_extractor = lambda instance, args, kwargs: "user-42"
    lg.instrumentor.config = config

    _run(lg.compile(), how)

    (span,) = lg.spans()
    assert span.attributes["session.id"] == "session-from-extractor"
    assert span.attributes["gen_ai.conversation.id"] == "session-from-extractor"
    assert span.attributes["user.id"] == "user-42"


@pytest.mark.parametrize("how", ["stream", "astream"])
def test_failing_extractors_do_not_cost_the_stream_or_its_span(lg, how):
    def broken(instance, args, kwargs):
        raise RuntimeError("extractor is broken")

    config = OTelConfig()
    config.session_id_extractor = broken
    config.user_id_extractor = broken
    lg.instrumentor.config = config

    assert len(_run(lg.compile(), how)) == 2
    (span,) = lg.spans()
    assert span.status.status_code is StatusCode.OK
    assert "user.id" not in span.attributes


def test_running_requests_gauge_is_balanced_for_streams(lg):
    server_metrics = MagicMock()
    target = "genai_otel.instrumentors.langgraph_instrumentor.get_server_metrics"
    with patch(target, return_value=server_metrics):
        graph = lg.compile()
        list(graph.stream({"question": "q"}))  # read to the end
        stopped_early = graph.stream({"question": "q"})
        next(stopped_early)
        stopped_early.close()
        with pytest.raises(ValueError):
            list(graph.stream({"question": "q", "fail_at": "first"}))
        graph.stream({"question": "q"})  # never iterated: never started

    assert server_metrics.increment_requests_running.call_count == 3
    assert server_metrics.decrement_requests_running.call_count == 3


@pytest.mark.parametrize("how", ["stream", "astream"])
def test_stream_is_delivered_when_its_span_cannot_be_started(lg, how):
    lg.instrumentor.tracer = MagicMock()
    lg.instrumentor.tracer.start_span.side_effect = RuntimeError("tracer is broken")

    chunks = _run(lg.compile(), how)

    assert [list(chunk) for chunk in chunks] == [["first"], ["second"]]
    assert lg.names() == []
    assert not _a_span_is_current()


def test_second_instrumentor_does_not_stack_wrappers(lg):
    again = LangGraphInstrumentor()
    again.tracer = lg.tracer
    again.instrument(OTelConfig())

    lg.compile().invoke({"question": "q"})

    assert lg.names() == ["langgraph.graph.invoke"]


def test_instrumenting_again_traces_with_the_new_instrumentor(lg):
    """uninstrument() then instrument() builds new instrumentors on a new provider.

    The wrappers stay on the Pregel class, so they have to follow the instrumentor
    that is current. Bound to the first one, they kept writing to its tracer - a
    provider that has been shut down - and LangGraph went dark.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    again = LangGraphInstrumentor()
    again.tracer = provider.get_tracer("second")
    again.instrument(OTelConfig())

    graph = lg.compile()
    graph.invoke({"question": "q"})
    list(graph.stream({"question": "q"}))

    assert [span.name for span in exporter.get_finished_spans()] == [
        "langgraph.graph.invoke",
        "langgraph.graph.stream",
    ]
    assert lg.names() == []


@pytest.mark.parametrize("how", ["stream", "astream"])
def test_stream_span_is_parented_where_the_stream_was_requested(lg, how):
    """The caller asked for the run inside its own span, even if it reads it later."""
    graph = lg.compile()
    with lg.tracer.start_as_current_span("handler") as handler:
        chunks = getattr(graph, how)({"question": "q"})

    if how == "stream":
        list(chunks)
    else:

        async def consume():
            return [chunk async for chunk in chunks]

        asyncio.run(consume())

    (span,) = [s for s in lg.spans() if s.name == "langgraph.graph." + how]
    assert span.parent is not None
    assert span.parent.span_id == handler.get_span_context().span_id


def test_ainvoke_closed_outside_its_task_does_not_raise(lg):
    """A pending run finalised from another context cannot reset its own markers."""
    graph = lg.compile()
    coroutine = graph.ainvoke({"question": "q"})
    # Start it in one context, close it from another.
    otel_context_copy = __import__("contextvars").copy_context()
    try:
        otel_context_copy.run(coroutine.send, None)
    except StopIteration:
        pass
    coroutine.close()


def test_not_instrumented_adds_no_spans(lg):
    """A wrapper whose instrumentor is switched off must be a plain pass-through."""
    graph = lg.compile()
    lg.instrumentor._instrumented = False

    assert graph.invoke({"question": "q"})["answer"] == "done"
    assert len(list(graph.stream({"question": "q"}))) == 2
    assert asyncio.run(graph.ainvoke({"question": "q"}))["answer"] == "done"
    assert len(_run(graph, "astream")) == 2
    assert lg.names() == []


# ---------------------------------------------------------------------------
# Node spans. The node runner is a LangGraph internal, so its handling is tested
# against stand-ins here and against the real library below when it is installed.
# ---------------------------------------------------------------------------


def _task(name="plan_node", step=1, triggers=("branch:to:plan_node",), config=None):
    if config is None:
        config = {"metadata": {"langgraph_step": step, "langgraph_node": name, "user": "x"}}
    return types.SimpleNamespace(name=name, config=config, triggers=list(triggers))


@pytest.fixture
def node_tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = LangGraphInstrumentor()
    instrumentor.tracer = provider.get_tracer("test")
    instrumentor.config = OTelConfig()
    instrumentor._instrumented = True
    return instrumentor, exporter


def test_node_span_attributes(node_tracing):
    instrumentor, exporter = node_tracing
    result = instrumentor._run_node(lambda task, policy: "ran", None, (_task(), None), {})

    assert result == "ran"
    (span,) = exporter.get_finished_spans()
    assert span.name == "langgraph.node.plan_node"
    attrs = span.attributes
    assert attrs["langgraph.node"] == "plan_node"
    assert attrs["langgraph.step"] == 1
    assert list(attrs["langgraph.triggers"]) == ["branch:to:plan_node"]
    assert attrs["graph.node.id"] == "plan_node"
    assert attrs["openinference.span.kind"] == "CHAIN"
    # Only the node keys reach the metadata attribute, not the run's own metadata.
    assert json.loads(attrs["metadata"]) == {"langgraph_node": "plan_node", "langgraph_step": 1}


def test_node_span_without_graph_attributes(node_tracing):
    instrumentor, exporter = node_tracing
    instrumentor.config = OTelConfig(enable_agent_graph_attributes=False)
    instrumentor._run_node(lambda task, policy: None, None, (_task(), None), {})

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["langgraph.node"] == "plan_node"
    for key in ("graph.node.id", "openinference.span.kind", "metadata"):
        assert key not in attrs


def test_node_span_tolerates_missing_metadata(node_tracing):
    instrumentor, exporter = node_tracing
    task = types.SimpleNamespace(name="respond", config=None, triggers=None)
    instrumentor._run_node(lambda task: None, None, (task,), {})

    attrs = exporter.get_finished_spans()[0].attributes
    assert "langgraph.step" not in attrs
    assert "langgraph.triggers" not in attrs
    assert json.loads(attrs["metadata"]) == {"langgraph_node": "respond"}


@pytest.mark.parametrize("name", ["__start__", "", None])
def test_internal_and_unnamed_tasks_get_no_span(node_tracing, name):
    instrumentor, exporter = node_tracing
    assert instrumentor._run_node(lambda task: "ok", None, (_task(name=name),), {}) == "ok"
    assert exporter.get_finished_spans() == ()


def test_node_task_passed_by_keyword(node_tracing):
    instrumentor, exporter = node_tracing
    instrumentor._run_node(lambda task=None: None, None, (), {"task": _task()})
    assert exporter.get_finished_spans()[0].name == "langgraph.node.plan_node"


def test_node_span_skipped_when_not_instrumented(node_tracing):
    instrumentor, exporter = node_tracing
    instrumentor._instrumented = False
    assert instrumentor._run_node(lambda task: 1, None, (_task(),), {}) == 1
    assert exporter.get_finished_spans() == ()


def test_node_span_preparation_failure_still_runs_the_node(node_tracing):
    instrumentor, exporter = node_tracing
    with patch.object(instrumentor, "_node_span", side_effect=RuntimeError("boom")):
        assert instrumentor._run_node(lambda task: 1, None, (_task(),), {}) == 1
    assert exporter.get_finished_spans() == ()


def test_node_error_is_recorded_and_raised(node_tracing):
    instrumentor, exporter = node_tracing

    def fail(task):
        raise ValueError("node exploded")

    with pytest.raises(ValueError):
        instrumentor._run_node(fail, None, (_task(),), {})
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "ValueError"
    assert span.events[0].name == "exception"


def test_node_error_recording_failure_is_swallowed():
    span = MagicMock()
    span.set_status.side_effect = RuntimeError("boom")
    LangGraphInstrumentor._record_node_error(span, ValueError("x"))


def test_node_error_without_langgraph_errors_module():
    span = MagicMock()
    with patch.dict("sys.modules", {"langgraph.errors": None}):
        LangGraphInstrumentor._record_node_error(span, ValueError("x"))
    span.set_status.assert_called_once()


def test_graph_bubble_up_is_not_an_error():
    class GraphBubbleUp(Exception):
        pass

    class GraphInterrupt(GraphBubbleUp):
        pass

    errors = types.ModuleType("langgraph.errors")
    errors.GraphBubbleUp = GraphBubbleUp
    span = MagicMock()
    with patch.dict("sys.modules", {"langgraph.errors": errors}):
        LangGraphInstrumentor._record_node_error(span, GraphInterrupt())
    span.set_status.assert_not_called()
    span.set_attribute.assert_called_once_with("langgraph.node.interrupted", True)


def test_async_node_span(node_tracing):
    instrumentor, exporter = node_tracing

    async def run(task, policy, stream=False):
        await asyncio.sleep(0)
        return "ran"

    assert asyncio.run(instrumentor._arun_node(run, None, (_task(), None), {})) == "ran"
    assert exporter.get_finished_spans()[0].name == "langgraph.node.plan_node"


def test_async_node_error(node_tracing):
    instrumentor, exporter = node_tracing

    async def fail(task):
        raise ValueError("node exploded")

    with pytest.raises(ValueError):
        asyncio.run(instrumentor._arun_node(fail, None, (_task(),), {}))
    assert exporter.get_finished_spans()[0].status.status_code is StatusCode.ERROR


def test_async_untraced_node_passes_through(node_tracing):
    instrumentor, exporter = node_tracing

    async def run(task):
        return "ok"

    assert asyncio.run(instrumentor._arun_node(run, None, (_task(name="__start__"),), {})) == "ok"
    assert exporter.get_finished_spans() == ()


def _fake_runner(name="langgraph.pregel._runner"):
    module = types.ModuleType(name)
    module.run_with_retry = lambda task, policy=None: "sync"

    async def arun_with_retry(task, policy=None, stream=False):
        return "async"

    module.arun_with_retry = arun_with_retry
    return module


def test_wrap_node_runner_wraps_once():
    import wrapt

    module = _fake_runner()
    with (
        patch(
            "genai_otel.instrumentors.langgraph_instrumentor._module_available", return_value=True
        ),
        patch("importlib.import_module", return_value=module),
    ):
        LangGraphInstrumentor._wrap_node_runner(wrapt)
        wrapped = module.run_with_retry
        LangGraphInstrumentor._wrap_node_runner(wrapt)

    assert isinstance(wrapped, wrapt.FunctionWrapper)
    assert module.run_with_retry is wrapped
    assert isinstance(module.arun_with_retry, wrapt.FunctionWrapper)


def test_wrap_node_runner_falls_back_to_older_module():
    import wrapt

    old = _fake_runner("langgraph.pregel.runner")

    def available(name):
        return name == "langgraph.pregel.runner"

    with (
        patch(
            "genai_otel.instrumentors.langgraph_instrumentor._module_available",
            side_effect=available,
        ),
        patch("importlib.import_module", return_value=old),
    ):
        LangGraphInstrumentor._wrap_node_runner(wrapt)
    assert isinstance(old.run_with_retry, wrapt.FunctionWrapper)


def test_wrap_node_runner_without_a_runner_is_a_no_op():
    import wrapt

    no_runner = types.ModuleType("langgraph.pregel._runner")
    with (
        patch(
            "genai_otel.instrumentors.langgraph_instrumentor._module_available", return_value=True
        ),
        patch("importlib.import_module", side_effect=[ImportError("moved"), no_runner]),
    ):
        LangGraphInstrumentor._wrap_node_runner(wrapt)  # must not raise
    assert not hasattr(no_runner, "run_with_retry")


def test_graph_span_is_the_root_node():
    instrumentor = LangGraphInstrumentor()
    instrumentor.config = OTelConfig()
    named = types.SimpleNamespace(name="orchestrator")
    attrs = instrumentor._extract_graph_attributes(named, ({},), {}, None)
    assert attrs["graph.node.id"] == "orchestrator"
    assert attrs["openinference.span.kind"] == "CHAIN"

    unnamed = instrumentor._extract_graph_attributes(object(), ({},), {}, None)
    assert unnamed["graph.node.id"] == "LangGraph"

    instrumentor.config = OTelConfig(enable_agent_graph_attributes=False)
    disabled = instrumentor._extract_graph_attributes(named, ({},), {}, None)
    assert "graph.node.id" not in disabled


@pytest.fixture
def real_langgraph():
    """Real LangGraph, instrumented, with the agent graph processor in the pipeline."""
    pytest.importorskip("langgraph.graph")
    from genai_otel.agent_graph import AgentGraphSpanProcessor
    from genai_otel.instrumentors import langgraph_instrumentor as module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AgentGraphSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    clean = otel_context.attach(otel_context.Context())
    instrumentor = LangGraphInstrumentor()
    instrumentor.tracer = provider.get_tracer("test")
    instrumentor.instrument(OTelConfig())
    try:
        yield exporter
    finally:
        module._ACTIVE = None
        otel_context.detach(clean)


def _orchestrator():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        n: int

    graph = StateGraph(State)
    graph.add_node("plan_node", lambda s: {"n": 0})
    graph.add_node("execute", lambda s: {"n": s["n"] + 1})
    graph.add_node("validate", lambda s: {})
    graph.add_node("respond", lambda s: {})
    graph.add_edge(START, "plan_node")
    graph.add_edge("plan_node", "execute")
    graph.add_edge("execute", "validate")
    graph.add_conditional_edges("validate", lambda s: "execute" if s["n"] < 2 else "respond")
    graph.add_edge("respond", END)
    return graph.compile(name="orchestrator")


@pytest.mark.parametrize("how", ["invoke", "stream", "ainvoke"])
def test_real_langgraph_nodes_form_the_agent_graph(real_langgraph, how):
    app = _orchestrator()
    if how == "invoke":
        app.invoke({"n": 0})
    elif how == "stream":
        list(app.stream({"n": 0}))
    else:
        asyncio.run(app.ainvoke({"n": 0}))

    spans = sorted(real_langgraph.get_finished_spans(), key=lambda s: s.start_time)
    root = spans[0]
    assert root.attributes["graph.node.id"] == "orchestrator"
    assert "graph.node.parent_id" not in root.attributes

    nodes = [s for s in spans if s.name.startswith("langgraph.node.")]
    assert [s.attributes["graph.node.id"] for s in nodes] == [
        "plan_node",
        "execute",
        "validate",
        "execute",
        "validate",
        "respond",
    ]
    assert [s.attributes["langgraph.step"] for s in nodes] == [1, 2, 3, 4, 5, 6]
    for node in nodes:
        assert node.parent.span_id == root.context.span_id
        assert node.attributes["graph.node.parent_id"] == "orchestrator"
        assert (
            json.loads(node.attributes["metadata"])["langgraph_node"]
            == node.attributes["langgraph.node"]
        )


def test_real_langgraph_node_children_nest_under_the_node(real_langgraph):
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from genai_otel.instrumentors import langgraph_instrumentor as module

    class State(TypedDict):
        n: int

    # Same provider as the node spans, standing in for an LLM call the node makes.
    tracer = module._ACTIVE.tracer

    def plan(state):
        with tracer.start_as_current_span("llm.call"):
            return {}

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", END)
    graph.compile().invoke({"n": 0})

    by_name = {s.name: s for s in real_langgraph.get_finished_spans()}
    assert by_name["llm.call"].parent.span_id == by_name["langgraph.node.plan"].context.span_id


def test_real_langgraph_failing_node_and_interrupt(real_langgraph):
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    class State(TypedDict):
        n: int

    def boom(state):
        raise ValueError("node exploded")

    def ask(state):
        interrupt("approve?")
        return {}

    failing = StateGraph(State)
    failing.add_node("boom", boom)
    failing.add_edge(START, "boom")
    failing.add_edge("boom", END)
    with pytest.raises(ValueError):
        failing.compile().invoke({"n": 0})

    paused = StateGraph(State)
    paused.add_node("ask", ask)
    paused.add_edge(START, "ask")
    paused.add_edge("ask", END)
    paused.compile(checkpointer=InMemorySaver()).invoke(
        {"n": 0}, {"configurable": {"thread_id": "t1"}}
    )

    by_name = {s.name: s for s in real_langgraph.get_finished_spans()}
    assert by_name["langgraph.node.boom"].status.status_code is StatusCode.ERROR
    ask_span = by_name["langgraph.node.ask"]
    assert ask_span.status.status_code is StatusCode.UNSET
    assert ask_span.attributes["langgraph.node.interrupted"] is True


if __name__ == "__main__":
    unittest.main()
