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
def baggage_config():
    """Set the send/accept lists for a test, and restore them after."""
    from genai_otel import sessions

    saved = (sessions._send_keys, sessions._accept_keys)
    yield sessions.configure_baggage
    sessions._send_keys, sessions._accept_keys = saved


def _propagator():
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    from genai_otel.sessions import SessionBaggagePropagator

    return CompositePropagator([TraceContextTextMapPropagator(), SessionBaggagePropagator()])


def _received(carrier):
    """The context a receiving service rebuilds from ``carrier`` alone."""
    return _propagator().extract(carrier, context=otel_context.Context())


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, ()),
        ("", ()),
        (" user.id , session.id, bogus", ("session.id", "user.id")),
        ("user.id", ("user.id",)),
    ],
)
def test_parse_baggage_keys(raw, expected):
    from genai_otel.sessions import parse_baggage_keys

    assert parse_baggage_keys(raw) == expected


def test_config_reads_both_lists_and_defaults_to_none():
    env = {
        "GENAI_SESSION_BAGGAGE_KEYS": "session.id,user.id",
        "GENAI_SESSION_BAGGAGE_ACCEPT": "session.id",
    }
    with patch.dict(os.environ, env):
        config = OTelConfig()
    assert config.session_baggage_keys == ["session.id", "user.id"]
    assert config.session_baggage_accept == ["session.id"]

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GENAI_SESSION_BAGGAGE_KEYS", None)
        os.environ.pop("GENAI_SESSION_BAGGAGE_ACCEPT", None)
        config = OTelConfig()
    assert config.session_baggage_keys == []
    assert config.session_baggage_accept == []


def test_nothing_crosses_by_default(baggage_config):
    from opentelemetry import baggage

    from genai_otel.sessions import baggage_enabled

    baggage_config()
    assert not baggage_enabled()
    carrier = {}
    with session("conv-1", user_id="u-7"):
        assert baggage.get_all() == {}
        _propagator().inject(carrier)
    assert "baggage" not in carrier


def test_sent_ids_follow_the_send_list(baggage_config):
    baggage_config(send=["session.id"])
    carrier = {}
    with session("conv-1", user_id="u-7"):
        _propagator().inject(carrier)
    assert carrier["baggage"] == "session.id=conv-1"

    baggage_config(send=["session.id", "user.id"])
    carrier = {}
    with session("conv-1", user_id="u-7"):
        _propagator().inject(carrier)
    assert "session.id=conv-1" in carrier["baggage"]
    assert "user.id=u-7" in carrier["baggage"]


def test_session_crosses_to_a_service_that_accepts_it(setup, baggage_config):
    tracer, spans = setup
    baggage_config(send=["session.id", "user.id"], accept=["session.id", "user.id"])
    carrier = {}
    with session("conv-1", user_id="u-7"):
        _propagator().inject(carrier)

    token = otel_context.attach(_received(carrier))
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


def test_a_service_that_does_not_accept_ignores_inbound_baggage(setup, baggage_config):
    # An edge service: browsers can send any baggage header they like.
    tracer, spans = setup
    baggage_config(send=["session.id"])
    ctx = _received({"baggage": "session.id=spoofed,user.id=someone-else"})
    assert ctx is not None
    token = otel_context.attach(ctx)
    try:
        with tracer.start_as_current_span("server"):
            pass
    finally:
        otel_context.detach(token)

    attrs = spans()["server"]
    assert "session.id" not in attrs
    assert "user.id" not in attrs


def test_inbound_ids_outside_the_accept_list_are_dropped_on_extract(baggage_config):
    from opentelemetry import baggage

    baggage_config(accept=["session.id"])
    ctx = _received({"baggage": "session.id=conv-1,user.id=spoofed,tenant=acme"})
    assert baggage.get_baggage("session.id", ctx) == "conv-1"
    assert baggage.get_baggage("user.id", ctx) is None
    # Application baggage is not ours to filter.
    assert baggage.get_baggage("tenant", ctx) == "acme"


@pytest.mark.parametrize("value", ["x" * 257, "bad%0Aid"])
def test_invalid_inbound_ids_are_dropped(baggage_config, value):
    from opentelemetry import baggage

    from genai_otel.sessions import inbound_session_id

    baggage_config(accept=["session.id"])
    ctx = _received({"baggage": f"session.id={value}"})
    assert baggage.get_baggage("session.id", ctx) is None
    # And if stock W3C baggage put it in the context, it still is not used.
    raw = baggage.set_baggage("session.id", "x" * 257, otel_context.Context())
    assert inbound_session_id(raw) is None


def test_inbound_ids_are_not_forwarded_unless_sent(baggage_config):
    # Accepted from upstream for our own spans, but not passed on to every service
    # and third-party API this one calls.
    baggage_config(accept=["session.id", "user.id"], send=["session.id"])
    token = otel_context.attach(_received({"baggage": "session.id=conv-1,user.id=u-7,tenant=a"}))
    try:
        carrier = {}
        _propagator().inject(carrier)
    finally:
        otel_context.detach(token)
    assert "user.id" not in carrier["baggage"]
    assert "session.id=conv-1" in carrier["baggage"]
    assert "tenant=a" in carrier["baggage"]


def test_session_block_replaces_inherited_ids_in_baggage(baggage_config):
    from opentelemetry import baggage

    baggage_config(send=["session.id", "user.id"], accept=["session.id", "user.id"])
    token = otel_context.attach(_received({"baggage": "session.id=upstream,user.id=up-user"}))
    try:
        with session("local"):
            assert baggage.get_baggage("session.id") == "local"
            # This block names no user, so the caller's must not ride along.
            assert baggage.get_baggage("user.id") is None
    finally:
        otel_context.detach(token)


def test_inbound_session_ranks_below_the_services_own(setup, baggage_config):
    tracer, spans = setup
    baggage_config(accept=["session.id"])
    token = otel_context.attach(_received({"baggage": "session.id=upstream"}))
    try:
        with tracer.start_as_current_span("own", attributes={"session.id": "mine"}):
            with tracer.start_as_current_span("child"):
                pass
        with session("local"):
            with tracer.start_as_current_span("in_block"):
                pass
        with tracer.start_as_current_span("bare"):
            pass
    finally:
        otel_context.detach(token)

    result = spans()
    assert result["child"]["session.id"] == "mine"
    assert result["in_block"]["session.id"] == "local"
    assert result["bare"]["session.id"] == "upstream"


def test_frameworks_use_inbound_session_only_as_a_last_resort(baggage_config):
    from genai_otel.instrumentors.crewai_instrumentor import CrewAIInstrumentor
    from genai_otel.instrumentors.langchain_instrumentor import LangChainInstrumentor
    from genai_otel.instrumentors.langgraph_instrumentor import LangGraphInstrumentor

    baggage_config(accept=["session.id"])
    graph = LangGraphInstrumentor()
    graph.config = OTelConfig()
    crew_instrumentor = CrewAIInstrumentor()
    crew_instrumentor.config = OTelConfig()
    chain = LangChainInstrumentor()
    chain.config = OTelConfig()
    crew = types.SimpleNamespace(id="crew-uuid", agents=[], tasks=[])

    token = otel_context.attach(_received({"baggage": "session.id=upstream"}))
    try:
        thread = {"configurable": {"thread_id": "thread-1"}}
        assert (
            graph._extract_graph_attributes(object(), ({}, thread), {}, None)["session.id"]
            == "thread-1"
        )
        assert graph._extract_graph_attributes(object(), ({},), {}, None)["session.id"] == (
            "upstream"
        )
        # The crew id is CrewAI's own fallback, not the application's choice.
        assert crew_instrumentor._extract_crew_attributes(crew, (), {})["session.id"] == (
            "upstream"
        )
        assert chain._resolve_session_id(None, ({},), {}) == "upstream"
        assert chain._resolve_session_id(None, ({},), {"session_id": "call"}) == "call"
    finally:
        otel_context.detach(token)


def _install_propagators(environ, send=(), accept=()):
    """Run the propagator setup with ``environ``; return the patched set_global_textmap."""
    from genai_otel.auto_instrument import _install_propagators as install

    config = OTelConfig()
    config.session_baggage_keys = list(send)
    config.session_baggage_accept = list(accept)
    with (
        patch.dict(os.environ, environ),
        patch("opentelemetry.propagate.set_global_textmap") as set_textmap,
    ):
        if "OTEL_PROPAGATORS" not in environ:
            os.environ.pop("OTEL_PROPAGATORS", None)
        install(config)
    return set_textmap


def test_setup_keeps_tracecontext_only_when_baggage_is_off(baggage_config):
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    set_textmap = _install_propagators({})
    assert isinstance(set_textmap.call_args.args[0], TraceContextTextMapPropagator)


def test_setup_adds_session_baggage_when_a_service_opts_in(baggage_config):
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    from genai_otel import sessions

    set_textmap = _install_propagators({}, send=["session.id"])
    propagator = set_textmap.call_args.args[0]
    assert TraceContextTextMapPropagator().fields <= propagator.fields
    assert "baggage" in propagator.fields
    assert sessions._send_keys == ("session.id",)
    assert sessions._accept_keys == ()


def test_setup_respects_otel_propagators(baggage_config):
    set_textmap = _install_propagators({"OTEL_PROPAGATORS": "tracecontext"}, send=["session.id"])
    set_textmap.assert_not_called()


def test_setup_turns_baggage_off_with_session_propagation(baggage_config):
    from genai_otel import sessions
    from genai_otel.auto_instrument import _install_propagators as install

    config = OTelConfig(enable_session_propagation=False)
    config.session_baggage_keys = ["session.id"]
    with patch("opentelemetry.propagate.set_global_textmap"), patch.dict(os.environ, {}):
        os.environ.pop("OTEL_PROPAGATORS", None)
        install(config)
    assert sessions._send_keys == ()
