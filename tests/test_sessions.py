"""Tests for genai_otel.session() and SessionSpanProcessor."""

import asyncio
import os
import threading
import types
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import genai_otel
from genai_otel.config import OTelConfig
from genai_otel.sessions import SessionSpanProcessor, current_session_id, current_user_id, session


@pytest.fixture
def setup():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    clean = otel_context.attach(otel_context.Context())

    def spans():
        return {span.name: dict(span.attributes) for span in exporter.get_finished_spans()}

    yield tracer, spans
    otel_context.detach(clean)


def test_session_is_exported_from_the_package():
    assert genai_otel.session is session


def test_spans_in_a_session_block_carry_its_ids(setup):
    tracer, spans = setup
    with session("conv-1", user_id="u-7"):
        with tracer.start_as_current_span("llm"):
            pass

    attrs = spans()["llm"]
    assert attrs["session.id"] == "conv-1"
    assert attrs["gen_ai.conversation.id"] == "conv-1"
    assert attrs["user.id"] == "u-7"


def test_user_id_is_optional(setup):
    tracer, spans = setup
    with session("conv-1"):
        with tracer.start_as_current_span("llm"):
            pass

    assert "user.id" not in spans()["llm"]


def test_spans_outside_a_session_are_untouched(setup):
    tracer, spans = setup
    with session("conv-1"):
        pass
    with tracer.start_as_current_span("later"):
        pass

    for key in ("session.id", "gen_ai.conversation.id", "user.id"):
        assert key not in spans()["later"]


def test_framework_session_reaches_every_child_span(setup):
    # A LangGraph run resolves its session from thread_id; its nodes and the LLM
    # calls inside them used to carry none.
    tracer, spans = setup
    with tracer.start_as_current_span(
        "graph", attributes={"session.id": "thread-9", "user.id": "u-1"}
    ):
        with tracer.start_as_current_span("node"):
            with tracer.start_as_current_span("llm"):
                pass

    result = spans()
    for name in ("node", "llm"):
        assert result[name]["session.id"] == "thread-9"
        assert result[name]["gen_ai.conversation.id"] == "thread-9"
        assert result[name]["user.id"] == "u-1"
    # The span that owns the session also gets the conversation id.
    assert result["graph"]["gen_ai.conversation.id"] == "thread-9"


def test_session_block_wins_over_the_parent_span(setup):
    tracer, spans = setup
    with tracer.start_as_current_span("outer", attributes={"session.id": "old"}):
        with session("new"):
            with tracer.start_as_current_span("inner"):
                pass

    assert spans()["inner"]["session.id"] == "new"


def test_existing_keys_are_never_overwritten(setup):
    tracer, spans = setup
    with session("conv-1", user_id="u-7"):
        with tracer.start_as_current_span(
            "own",
            attributes={
                "session.id": "mine",
                "gen_ai.conversation.id": "thread",
                "user.id": "me",
            },
        ):
            pass

    attrs = spans()["own"]
    assert attrs["session.id"] == "mine"
    assert attrs["gen_ai.conversation.id"] == "thread"
    assert attrs["user.id"] == "me"


def test_own_session_sets_the_missing_conversation_id(setup):
    tracer, spans = setup
    with session("ctx"):
        with tracer.start_as_current_span("own", attributes={"session.id": "mine"}):
            pass

    assert spans()["own"]["gen_ai.conversation.id"] == "mine"


def test_nested_blocks_restore_the_outer_session(setup):
    tracer, spans = setup
    with session("outer"):
        with session("inner"):
            with tracer.start_as_current_span("a"):
                pass
        with tracer.start_as_current_span("b"):
            pass

    result = spans()
    assert result["a"]["session.id"] == "inner"
    assert result["b"]["session.id"] == "outer"


def test_session_block_restores_context_on_error(setup):
    with pytest.raises(ValueError):
        with session("conv-1"):
            raise ValueError("boom")
    assert current_session_id() is None


def test_async_tasks_keep_their_own_session(setup):
    tracer, spans = setup

    async def turn(conversation):
        with session(conversation):
            await asyncio.sleep(0)
            with tracer.start_as_current_span(f"llm-{conversation}"):
                await asyncio.sleep(0)

    async def main():
        await asyncio.gather(turn("a"), turn("b"))

    asyncio.run(main())
    result = spans()
    assert result["llm-a"]["session.id"] == "a"
    assert result["llm-b"]["session.id"] == "b"


def test_thread_without_copied_context_has_no_session(setup):
    tracer, spans = setup
    with session("conv-1"):
        worker = threading.Thread(
            target=lambda: tracer.start_as_current_span("bg").__enter__().end()
        )
        worker.start()
        worker.join()

    assert "session.id" not in spans()["bg"]


def test_openinference_context_keys_are_honoured(setup):
    # using_session() / using_user() store plain "session.id" / "user.id" keys.
    tracer, spans = setup
    ctx = otel_context.set_value("session.id", "oi-session")
    ctx = otel_context.set_value("user.id", "oi-user", ctx)
    token = otel_context.attach(ctx)
    try:
        assert current_session_id() == "oi-session"
        assert current_user_id() == "oi-user"
        with tracer.start_as_current_span("llm"):
            pass
    finally:
        otel_context.detach(token)

    attrs = spans()["llm"]
    assert attrs["session.id"] == "oi-session"
    assert attrs["user.id"] == "oi-user"


def test_ids_are_stored_as_strings(setup):
    tracer, spans = setup
    with session(42, user_id=7):
        with tracer.start_as_current_span("llm"):
            pass

    attrs = spans()["llm"]
    assert attrs["session.id"] == "42"
    assert attrs["user.id"] == "7"


def test_on_start_never_raises():
    span = MagicMock()
    span.attributes = {}
    span.set_attribute.side_effect = RuntimeError("boom")
    with session("conv-1"):
        SessionSpanProcessor().on_start(span)


def test_processor_lifecycle_methods():
    processor = SessionSpanProcessor()
    processor.on_end(MagicMock())
    processor.shutdown()
    assert processor.force_flush() is True


def test_config_flag_defaults_on():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GENAI_ENABLE_SESSION_PROPAGATION", None)
        assert OTelConfig().enable_session_propagation is True


def test_config_flag_can_be_disabled():
    with patch.dict(os.environ, {"GENAI_ENABLE_SESSION_PROPAGATION": "false"}):
        assert OTelConfig().enable_session_propagation is False


# ---------------------------------------------------------------------------
# Framework precedence: a session() block beats each framework's fallback, but an
# id the application passed in the call itself still wins.
# ---------------------------------------------------------------------------


def test_langgraph_prefers_the_session_block_over_thread_id():
    from genai_otel.instrumentors.langgraph_instrumentor import LangGraphInstrumentor

    instrumentor = LangGraphInstrumentor()
    instrumentor.config = OTelConfig()
    config = {"configurable": {"thread_id": "thread-1"}}
    with session("conv-1"):
        attrs = instrumentor._extract_graph_attributes(object(), ({}, config), {}, None)
        explicit = instrumentor._extract_graph_attributes(
            object(), ({"session_id": "from-input"}, config), {}, None
        )
    assert attrs["session.id"] == "conv-1"
    assert attrs["gen_ai.conversation.id"] == "conv-1"
    assert explicit["session.id"] == "from-input"

    outside = instrumentor._extract_graph_attributes(object(), ({}, config), {}, None)
    assert outside["session.id"] == "thread-1"


def test_crewai_prefers_the_session_block_over_the_crew_id():
    from genai_otel.instrumentors.crewai_instrumentor import CrewAIInstrumentor

    instrumentor = CrewAIInstrumentor()
    instrumentor.config = OTelConfig()
    crew = types.SimpleNamespace(id="crew-uuid", agents=[], tasks=[])
    with session("conv-1"):
        attrs = instrumentor._extract_crew_attributes(crew, (), {})
        explicit = instrumentor._extract_crew_attributes(
            crew, (), {"inputs": {"session_id": "from-input"}}
        )
    assert attrs["session.id"] == "conv-1"
    assert explicit["session.id"] == "from-input"
    assert instrumentor._extract_crew_attributes(crew, (), {})["session.id"] == "crew-uuid"


def test_langchain_prefers_the_session_block_over_a_generated_id():
    from genai_otel.instrumentors.langchain_instrumentor import LangChainInstrumentor

    instrumentor = LangChainInstrumentor()
    instrumentor.config = OTelConfig()
    with session("conv-1"):
        assert instrumentor._resolve_session_id(None, ({},), {}) == "conv-1"
        assert (
            instrumentor._resolve_session_id(None, ({},), {"session_id": "from-call"})
            == "from-call"
        )


# ---------------------------------------------------------------------------
# Across services: W3C baggage.
# ---------------------------------------------------------------------------


@pytest.fixture
def baggage_keys():
    """Restore the module's baggage setting after a test changes it."""
    from genai_otel import sessions

    saved = sessions._baggage_keys
    yield sessions.configure_baggage
    sessions._baggage_keys = saved


def _propagator():
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    return CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, ("session.id",)),
        ("", ()),
        (" user.id , session.id, bogus", ("session.id", "user.id")),
        ("user.id", ("user.id",)),
    ],
)
def test_parse_baggage_keys(raw, expected):
    from genai_otel.sessions import parse_baggage_keys

    assert parse_baggage_keys(raw) == expected


def test_config_reads_baggage_keys():
    with patch.dict(os.environ, {"GENAI_SESSION_BAGGAGE_KEYS": "session.id,user.id"}):
        assert OTelConfig().session_baggage_keys == ["session.id", "user.id"]
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GENAI_SESSION_BAGGAGE_KEYS", None)
        assert OTelConfig().session_baggage_keys == ["session.id"]


def test_session_goes_out_as_baggage_but_user_is_opt_in(baggage_keys):
    from opentelemetry import baggage

    baggage_keys(["session.id"])
    with session("conv-1", user_id="u-7"):
        assert baggage.get_baggage("session.id") == "conv-1"
        assert baggage.get_baggage("user.id") is None

    baggage_keys(["session.id", "user.id"])
    with session("conv-1", user_id="u-7"):
        assert baggage.get_baggage("user.id") == "u-7"

    baggage_keys([])
    with session("conv-1", user_id="u-7"):
        assert baggage.get_all() == {}


def test_session_crosses_a_service_boundary(setup, baggage_keys):
    """Caller injects, callee extracts: the callee's spans carry the session."""
    tracer, spans = setup
    baggage_keys(["session.id", "user.id"])
    carrier = {}
    with session("conv-1", user_id="u-7"):
        with tracer.start_as_current_span("client"):
            _propagator().inject(carrier)
    assert "conv-1" in carrier["baggage"]

    # The receiving service: a fresh context, rebuilt from the headers only.
    server_ctx = _propagator().extract(carrier, context=otel_context.Context())
    token = otel_context.attach(server_ctx)
    try:
        with tracer.start_as_current_span("server"):
            with tracer.start_as_current_span("downstream_llm"):
                pass
    finally:
        otel_context.detach(token)

    result = spans()
    for name in ("server", "downstream_llm"):
        assert result[name]["session.id"] == "conv-1"
        assert result[name]["gen_ai.conversation.id"] == "conv-1"
        assert result[name]["user.id"] == "u-7"


def test_inbound_baggage_outside_the_keys_is_ignored(setup, baggage_keys):
    from opentelemetry import baggage

    tracer, spans = setup
    baggage_keys(["session.id"])
    ctx = baggage.set_baggage("session.id", "conv-1", otel_context.Context())
    ctx = baggage.set_baggage("user.id", "spoofed", ctx)
    token = otel_context.attach(ctx)
    try:
        assert current_user_id() is None
        with tracer.start_as_current_span("server"):
            pass
    finally:
        otel_context.detach(token)

    attrs = spans()["server"]
    assert attrs["session.id"] == "conv-1"
    assert "user.id" not in attrs

    baggage_keys([])
    token = otel_context.attach(ctx)
    try:
        assert current_session_id() is None
    finally:
        otel_context.detach(token)


def test_local_session_block_wins_over_inbound_baggage(baggage_keys):
    from opentelemetry import baggage

    baggage_keys(["session.id"])
    token = otel_context.attach(baggage.set_baggage("session.id", "upstream"))
    try:
        assert current_session_id() == "upstream"
        with session("local"):
            assert current_session_id() == "local"
    finally:
        otel_context.detach(token)


def _install_propagators(environ):
    """Run the propagator setup with ``environ``; return the patched set_global_textmap."""
    from genai_otel.auto_instrument import _install_propagators as install

    with (
        patch.dict(os.environ, environ),
        patch("opentelemetry.propagate.set_global_textmap") as set_textmap,
    ):
        if "OTEL_PROPAGATORS" not in environ:
            os.environ.pop("OTEL_PROPAGATORS", None)
        install()
    return set_textmap


def test_setup_installs_tracecontext_and_baggage_propagators():
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    set_textmap = _install_propagators({})
    set_textmap.assert_called_once()
    fields = set_textmap.call_args.args[0].fields
    assert TraceContextTextMapPropagator().fields <= fields
    assert W3CBaggagePropagator().fields <= fields


def test_setup_respects_otel_propagators():
    set_textmap = _install_propagators({"OTEL_PROPAGATORS": "tracecontext"})
    set_textmap.assert_not_called()
