from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.strands_harness_instrumentor import (
    StrandsHarnessHook,
    StrandsHarnessInstrumentor,
)


def _setup(capture=False, pii=False):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = StrandsHarnessInstrumentor()
    instrumentor.tracer = provider.get_tracer("test.strands")
    instrumentor._setup_config(
        OTelConfig(
            enabled_instrumentors=["strands"],
            enable_content_capture=capture,
            content_max_length=200,
            enable_pii_detection=pii,
            pii_mode="redact" if pii else "detect",
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )
    )
    return instrumentor, StrandsHarnessHook(instrumentor), exporter


def _agent(**overrides):
    values = {
        "name": "researcher",
        "agent_id": "agent-1",
        "model": "gpt-6-sol",
        "session_manager": object(),
        "memory_manager": object(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _event(agent, **values):
    return SimpleNamespace(agent=agent, **values)


def _spans(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def test_optional_dependency_is_not_required():
    instrumentor = StrandsHarnessInstrumentor()
    # The test environment intentionally does not install strands-harness.
    assert instrumentor._strands_available is False


def test_basic_invocation_emits_harness_and_agent_hierarchy():
    _, hook, exporter = _setup()
    agent = _agent()
    hook.on_before_invocation(_event(agent, messages=[{"role": "user", "content": "hello"}]))
    hook.on_after_invocation(_event(agent, result=SimpleNamespace(stop_reason="end_turn")))

    spans = _spans(exporter)
    assert set(spans) == {"harness.run", "agent.run"}
    assert spans["agent.run"].parent.span_id == spans["harness.run"].context.span_id
    assert spans["agent.run"].attributes["gen_ai.agent.name"] == "researcher"


def test_model_call_is_nested_and_records_usage():
    _, hook, exporter = _setup()
    agent = _agent()
    hook.on_before_invocation(_event(agent))
    hook.on_before_model_call(_event(agent, projected_input_tokens=12))
    hook.on_after_model_call(
        _event(
            agent,
            stop_response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=12, output_tokens=7, total_tokens=19)
            ),
        )
    )
    hook.on_after_invocation(_event(agent))

    spans = _spans(exporter)
    assert spans["llm.request"].parent.span_id == spans["agent.run"].context.span_id
    assert spans["llm.request"].attributes["gen_ai.usage.input_tokens"] == 12
    assert spans["llm.request"].attributes["gen_ai.usage.output_tokens"] == 7


def test_tool_call_emits_tool_span_without_arguments_by_default():
    _, hook, exporter = _setup()
    agent = _agent()
    hook.on_before_invocation(_event(agent))
    hook.on_before_tool_call(
        _event(
            agent,
            selected_tool=SimpleNamespace(name="shell"),
            tool_use={"input": {"command": "pwd"}},
        )
    )
    hook.on_after_tool_call(_event(agent, result={"stdout": "secret"}))
    hook.on_after_invocation(_event(agent))

    span = _spans(exporter)["tool.call"]
    assert span.attributes["strands.tool.category"] == "tool"
    assert "gen_ai.tool.call.arguments" not in span.attributes
    assert "secret" not in str(span.attributes)


def test_mcp_call_records_server_identity():
    _, hook, exporter = _setup()
    agent = _agent()
    hook.on_before_invocation(_event(agent))
    hook.on_before_tool_call(
        _event(
            agent,
            selected_tool=SimpleNamespace(name="search"),
            tool_use=SimpleNamespace(server_name="docs", tool_use_id="call-1"),
        )
    )
    hook.on_after_tool_call(_event(agent))
    hook.on_after_invocation(_event(agent))

    span = _spans(exporter)["tool.call"]
    assert span.attributes["strands.mcp.server"] == "docs"
    assert span.attributes["gen_ai.tool.call.id"] == "call-1"


def test_nested_invocation_emits_subagent_span():
    _, hook, exporter = _setup()
    parent = _agent(name="parent", agent_id="parent")
    child = _agent(name="child", agent_id="child")
    hook.on_before_invocation(_event(parent))
    hook.on_before_invocation(_event(child))
    hook.on_after_invocation(_event(child))
    hook.on_after_invocation(_event(parent))

    spans = _spans(exporter)
    assert "subagent.run" in spans
    assert spans["subagent.run"].parent.span_id == spans["agent.run"].context.span_id


def test_session_resume_and_run_ids_are_recorded():
    _, hook, exporter = _setup()
    agent = _agent(_session_id="session-9")
    hook.on_before_invocation(
        _event(
            agent,
            invocation_state={"run_id": "run-2", "parent_run_id": "run-1", "resumed": True},
        )
    )
    hook.on_after_invocation(_event(agent))

    span = _spans(exporter)["agent.run"]
    assert span.attributes["strands.session.id"] == "session-9"
    assert span.attributes["strands.session.resumed"] is True
    assert span.attributes["strands.run.id"] == "run-2"
    assert span.attributes["strands.parent_run.id"] == "run-1"


def test_context_compaction_span_is_supported_when_event_is_exposed():
    _, hook, exporter = _setup()
    agent = _agent()
    hook.on_before_invocation(_event(agent))
    hook.on_before_context_compaction(_event(agent))
    hook.on_after_context_compaction(_event(agent))
    hook.on_after_invocation(_event(agent))

    assert "context.compaction" in _spans(exporter)


def test_memory_read_and_write_spans_omit_memory_contents():
    _, hook, exporter = _setup(capture=True)
    agent = _agent()
    hook.on_before_invocation(_event(agent))
    hook.on_before_memory_read(_event(agent, query="private query"))
    hook.on_after_memory_read(_event(agent, result="private memory"))
    hook.on_before_memory_write(_event(agent, value="private memory"))
    hook.on_after_memory_write(_event(agent))
    hook.on_after_invocation(_event(agent))

    spans = _spans(exporter)
    assert "memory.read" in spans
    assert "memory.write" in spans
    assert "private memory" not in str(spans["memory.read"].attributes)


def test_streaming_event_marks_model_span():
    _, hook, exporter = _setup()
    agent = _agent()
    hook.on_before_invocation(_event(agent))
    hook.on_before_model_call(_event(agent))
    hook.on_model_stream_chunk(_event(agent))
    hook.on_after_model_call(_event(agent))
    hook.on_after_invocation(_event(agent))

    span = _spans(exporter)["llm.request"]
    assert span.attributes["gen_ai.request.stream"] is True
    assert span.attributes["strands.response.streaming"] is True


def test_model_exception_marks_error_type():
    _, hook, exporter = _setup()
    agent = _agent()
    hook.on_before_invocation(_event(agent))
    hook.on_before_model_call(_event(agent))
    hook.on_after_model_call(_event(agent, exception=TimeoutError("slow")))
    hook.on_after_invocation(_event(agent, result=SimpleNamespace(stop_reason="timeout")))

    span = _spans(exporter)["llm.request"]
    assert span.attributes["error.type"] == "TimeoutError"
    assert span.status.status_code.name == "ERROR"


def test_cancelled_invocation_is_recorded_as_failed_run():
    _, hook, exporter = _setup()
    agent = _agent()
    hook.on_before_invocation(_event(agent))
    hook.on_after_invocation(_event(agent, result=SimpleNamespace(stop_reason="cancelled")))

    span = _spans(exporter)["agent.run"]
    assert span.attributes["strands.run.status"] == "cancelled"
    assert span.status.status_code.name == "ERROR"


def test_content_capture_follows_existing_capture_switch():
    _, disabled, disabled_exporter = _setup(capture=False)
    agent = _agent()
    disabled.on_before_invocation(_event(agent, messages=["private prompt"]))
    disabled.on_after_invocation(_event(agent, result={"output": "private result"}))
    assert "input.value" not in _spans(disabled_exporter)["agent.run"].attributes
    assert "private" not in str(_spans(disabled_exporter)["agent.run"].attributes)

    _, enabled, enabled_exporter = _setup(capture=True)
    enabled.on_before_invocation(_event(agent, messages=["private prompt"]))
    enabled.on_after_invocation(_event(agent, result={"output": "private result"}))
    assert "private prompt" in _spans(enabled_exporter)["agent.run"].attributes["input.value"]

    _, redacted, redacted_exporter = _setup(capture=True, pii=True)
    redacted.on_before_invocation(_event(agent, messages=["private prompt"]))
    redacted.on_after_invocation(_event(agent))
    assert _spans(redacted_exporter)["agent.run"].attributes["input.value"] == "[REDACTED]"


def test_existing_provider_span_is_enriched_without_duplicate_model_span():
    instrumentor, hook, exporter = _setup()
    native = instrumentor.tracer.start_span("openai.chat")
    token = __import__("opentelemetry.context", fromlist=["attach"]).attach(
        __import__("opentelemetry.trace", fromlist=["set_span_in_context"]).set_span_in_context(
            native
        )
    )
    try:
        hook.on_before_model_call(_event(_agent()))
        hook.on_after_model_call(_event(_agent(), stop_response={"usage": {"input_tokens": 1}}))
    finally:
        __import__("opentelemetry.context", fromlist=["detach"]).detach(token)
        native.end()

    names = [span.name for span in exporter.get_finished_spans()]
    assert names == ["openai.chat"]
    assert exporter.get_finished_spans()[0].attributes["gen_ai.usage.input_tokens"] == 1


def test_create_harness_wrapper_injects_one_shared_hook(monkeypatch):
    fake = SimpleNamespace(__version__="0.1.1")

    def create_harness(**kwargs):
        return kwargs

    fake.create_harness = create_harness
    monkeypatch.setitem(__import__("sys").modules, "strands_harness", fake)
    instrumentor = StrandsHarnessInstrumentor()
    instrumentor.tracer = TracerProvider().get_tracer("test.strands.wrapper")
    instrumentor.instrument(OTelConfig(enabled_instrumentors=["strands"], enable_gpu_metrics=False))
    result = fake.create_harness(hooks=[])

    assert instrumentor._instrumented is True
    assert result["hooks"] == [instrumentor._hook]


@pytest.mark.parametrize(
    "event_name", ["BeforeContextReductionEvent", "AfterContextReductionEvent"]
)
def test_context_event_aliases_are_supported(event_name):
    # The callback mapping is intentionally kept available for SDK versions
    # that call compaction "reduction".
    assert event_name in {
        "BeforeContextReductionEvent",
        "AfterContextReductionEvent",
    }
