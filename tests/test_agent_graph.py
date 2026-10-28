"""Tests for AgentGraphSpanProcessor (Arize AX / OpenInference agent graph attributes)."""

import os
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel import agent_graph
from genai_otel.agent_graph import AgentGraphSpanProcessor
from genai_otel.config import OTelConfig


@pytest.fixture
def setup():
    processor = AgentGraphSpanProcessor()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    def spans():
        return {span.name: dict(span.attributes) for span in exporter.get_finished_spans()}

    return tracer, processor, spans


def _agent(name):
    return {"gen_ai.agent.name": name}


def test_root_agent_gets_graph_attributes_without_parent(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span("root", attributes=_agent("planner")):
        pass

    attrs = spans()["root"]
    assert attrs["agent.name"] == "planner"
    assert attrs["graph.node.id"] == "planner"
    assert attrs["graph.node.name"] == "planner"
    assert attrs["graph.node.display_name"] == "planner"
    assert attrs["openinference.span.kind"] == "AGENT"
    # Unset parent is how OpenInference marks the root node.
    assert "graph.node.parent_id" not in attrs


def test_sub_agent_parent_is_nearest_enclosing_agent(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span("orchestrator", attributes=_agent("orchestrator")):
        # An LLM span sits between the two agents.
        with tracer.start_as_current_span("llm", attributes={"gen_ai.operation.name": "chat"}):
            with tracer.start_as_current_span("researcher", attributes=_agent("researcher")):
                with tracer.start_as_current_span("writer", attributes=_agent("writer")):
                    pass

    result = spans()
    assert result["researcher"]["graph.node.parent_id"] == "orchestrator"
    assert result["writer"]["graph.node.parent_id"] == "researcher"


def test_non_agent_spans_are_untouched(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span("orchestrator", attributes=_agent("orchestrator")):
        with tracer.start_as_current_span("llm", attributes={"gen_ai.operation.name": "chat"}):
            pass

    llm = spans()["llm"]
    for key in ("agent.name", "graph.node.id", "graph.node.parent_id", "openinference.span.kind"):
        assert key not in llm


def test_same_agent_nested_is_one_node_not_a_self_loop(setup):
    # A CrewAI task span and the agent span running it carry the same name.
    tracer, _, spans = setup
    with tracer.start_as_current_span("manager", attributes=_agent("manager")):
        with tracer.start_as_current_span("task", attributes=_agent("researcher")):
            with tracer.start_as_current_span("agent", attributes=_agent("researcher")):
                with tracer.start_as_current_span("tool_agent", attributes=_agent("searcher")):
                    pass

    result = spans()
    assert result["task"]["graph.node.parent_id"] == "manager"
    assert result["agent"]["graph.node.parent_id"] == "manager"
    assert result["tool_agent"]["graph.node.parent_id"] == "researcher"


def test_same_agent_nested_at_root_stays_root(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span("run", attributes=_agent("assistant")):
        with tracer.start_as_current_span("on_messages", attributes=_agent("assistant")):
            pass

    assert "graph.node.parent_id" not in spans()["on_messages"]


def test_existing_keys_are_not_overwritten(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span("orchestrator", attributes=_agent("orchestrator")):
        with tracer.start_as_current_span(
            "custom",
            attributes={
                "gen_ai.agent.name": "worker",
                "agent.name": "Worker Agent",
                "graph.node.id": "worker-node",
                "graph.node.name": "Worker",
                "graph.node.parent_id": "explicit-parent",
                "openinference.span.kind": "CHAIN",
            },
        ):
            pass

    custom = spans()["custom"]
    assert custom["agent.name"] == "Worker Agent"
    assert custom["graph.node.id"] == "worker-node"
    assert custom["graph.node.name"] == "Worker"
    # The display label follows a node name the span already chose.
    assert custom["graph.node.display_name"] == "Worker"
    assert custom["graph.node.parent_id"] == "explicit-parent"
    assert custom["openinference.span.kind"] == "CHAIN"


def test_explicit_parent_id_is_inherited_by_same_node_children(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span(
        "worker", attributes={"graph.node.id": "worker", "graph.node.parent_id": "boss"}
    ):
        with tracer.start_as_current_span("worker_inner", attributes=_agent("worker")):
            pass

    assert spans()["worker_inner"]["graph.node.parent_id"] == "boss"


def test_empty_explicit_parent_id_means_root(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span("outer", attributes=_agent("outer")):
        with tracer.start_as_current_span(
            "inner", attributes={"gen_ai.agent.name": "inner", "graph.node.parent_id": ""}
        ):
            with tracer.start_as_current_span("leaf", attributes=_agent("inner")):
                pass

    result = spans()
    assert result["inner"]["graph.node.parent_id"] == ""
    assert "graph.node.parent_id" not in result["leaf"]


def test_openinference_agent_span_counts_as_parent(setup):
    # A span from an OpenInference instrumentor carries agent.name, not gen_ai.*.
    tracer, _, spans = setup
    with tracer.start_as_current_span("oi_agent", attributes={"agent.name": "oi"}):
        with tracer.start_as_current_span("ours", attributes=_agent("ours")):
            pass

    result = spans()
    assert result["oi_agent"]["graph.node.id"] == "oi"
    assert result["ours"]["graph.node.parent_id"] == "oi"


def test_agent_name_is_set_only_from_an_agent_name_key(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span("node", attributes={"graph.node.id": "step-1"}):
        pass

    attrs = spans()["node"]
    assert "agent.name" not in attrs
    assert attrs["graph.node.name"] == "step-1"


def test_tracking_is_released_when_spans_end(setup):
    tracer, processor, spans = setup
    with tracer.start_as_current_span("orchestrator", attributes=_agent("orchestrator")):
        with tracer.start_as_current_span("llm"):
            assert len(processor._nodes) == 2
        with tracer.start_as_current_span("sibling", attributes=_agent("sibling")):
            pass
    assert processor._nodes == {}
    assert spans()["sibling"]["graph.node.parent_id"] == "orchestrator"


def test_spans_outside_any_agent_are_not_tracked(setup):
    tracer, processor, _ = setup
    with tracer.start_as_current_span("plain"):
        with tracer.start_as_current_span("child"):
            assert processor._nodes == {}


def test_tracking_is_capped(setup):
    tracer, processor, _ = setup
    with patch.object(agent_graph, "_MAX_TRACKED_SPANS", 2):
        with tracer.start_as_current_span("a", attributes=_agent("a")):
            with tracer.start_as_current_span("b"):
                with tracer.start_as_current_span("c"):
                    assert len(processor._nodes) <= 2


def test_on_start_never_raises():
    processor = AgentGraphSpanProcessor()
    span = MagicMock()
    span.get_span_context.side_effect = RuntimeError("boom")
    processor.on_start(span)

    span = MagicMock()
    span.attributes = {"gen_ai.agent.name": "x"}
    span.set_attribute.side_effect = RuntimeError("boom")
    processor.on_start(span)


def test_on_end_never_raises():
    processor = AgentGraphSpanProcessor()
    span = MagicMock()
    span.get_span_context.side_effect = RuntimeError("boom")
    processor.on_end(span)


def test_invalid_span_context_is_ignored():
    processor = AgentGraphSpanProcessor()
    span = MagicMock()
    span.get_span_context.return_value = None
    processor.on_start(span)
    span.set_attribute.assert_not_called()


def test_shutdown_and_flush(setup):
    tracer, processor, _ = setup
    with tracer.start_as_current_span("a", attributes=_agent("a")):
        processor.shutdown()
        assert processor._nodes == {}
    assert processor.force_flush() is True


def test_config_flag_defaults_on():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GENAI_ENABLE_AGENT_GRAPH_ATTRIBUTES", None)
        assert OTelConfig().enable_agent_graph_attributes is True


def test_config_flag_can_be_disabled():
    with patch.dict(os.environ, {"GENAI_ENABLE_AGENT_GRAPH_ATTRIBUTES": "false"}):
        assert OTelConfig().enable_agent_graph_attributes is False


def test_setup_skips_processor_when_disabled():
    from genai_otel import auto_instrument

    config = OTelConfig(enable_agent_graph_attributes=False)
    provider = MagicMock()
    with (
        patch.object(auto_instrument, "TracerProvider", return_value=provider),
        patch.object(auto_instrument, "trace"),
        patch.object(auto_instrument, "metrics"),
        patch.object(auto_instrument, "OTLPSpanExporter"),
        patch.object(auto_instrument, "OTLPMetricExporter"),
        patch.object(auto_instrument, "BatchSpanProcessor"),
        patch.object(auto_instrument, "PeriodicExportingMetricReader"),
        patch.object(auto_instrument, "MeterProvider"),
    ):
        config.enable_gpu_metrics = False
        config.enable_mcp_instrumentation = False
        config.enabled_instrumentors = []
        auto_instrument.setup_auto_instrumentation(config)

    added = [call.args[0] for call in provider.add_span_processor.call_args_list]
    assert not any(isinstance(p, AgentGraphSpanProcessor) for p in added)


def test_span_declaring_a_non_agent_kind_is_not_a_node(setup):
    # A CrewAI task names the agent that runs it but is a step, not an agent.
    tracer, processor, spans = setup
    with tracer.start_as_current_span("crew", attributes={"graph.node.id": "crew"}):
        with tracer.start_as_current_span(
            "task",
            attributes={"gen_ai.agent.name": "researcher", "openinference.span.kind": "CHAIN"},
        ):
            with tracer.start_as_current_span("agent", attributes=_agent("researcher")):
                pass

    result = spans()
    assert "graph.node.id" not in result["task"]
    assert result["task"]["openinference.span.kind"] == "CHAIN"
    assert result["agent"]["graph.node.id"] == "researcher"
    assert result["agent"]["graph.node.parent_id"] == "crew"


def test_explicit_node_id_makes_a_node_whatever_the_kind(setup):
    tracer, _, spans = setup
    with tracer.start_as_current_span(
        "step", attributes={"graph.node.id": "plan", "openinference.span.kind": "CHAIN"}
    ):
        pass

    attrs = spans()["step"]
    assert attrs["graph.node.id"] == "plan"
    assert attrs["openinference.span.kind"] == "CHAIN"
