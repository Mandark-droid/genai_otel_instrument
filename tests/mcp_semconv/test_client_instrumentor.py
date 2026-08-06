"""Tests for the MCP client instrumentor's span emission and wrapping."""

import asyncio

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.mcp_semconv import (
    CommerceAttributes,
    MCPAttributes,
    MCPClientInstrumentor,
    expected_tool,
    mcp_session,
)
from genai_otel.mcp_semconv.client_instrumentor import get_call_context


class _TextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _CallToolResult:
    def __init__(self, content=None, is_error=False, structured=None):
        self.content = content or []
        self.isError = is_error
        self.structuredContent = structured


class _FakeClientSession:
    """Minimal stand-in for ``mcp.client.session.ClientSession``."""

    def __init__(self, result=None, error=None):
        self._result = result if result is not None else _CallToolResult()
        self._error = error
        self.calls = []

    async def call_tool(self, name, arguments=None, **kwargs):
        self.calls.append((name, arguments))
        if self._error is not None:
            raise self._error
        return self._result


@pytest.fixture
def tracer_provider():
    """A local tracer provider. The OTel global is deliberately left alone."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_MEMORY_EXPORTER))
    _MEMORY_EXPORTER.clear()
    return provider


#: One exporter instance, cleared per test by the fixture above.
_MEMORY_EXPORTER = InMemorySpanExporter()


@pytest.fixture
def exporter(tracer_provider):
    """The in-memory exporter fed by :func:`tracer_provider`."""
    return _MEMORY_EXPORTER


@pytest.fixture
def instrumentor(registry, tracer_provider):
    """An instrumentor whose tracer writes into the in-memory exporter."""
    made = MCPClientInstrumentor(registry=registry)
    made.tracer = tracer_provider.get_tracer(__name__)
    return made


def _invoke(instrumentor, session, name, arguments=None):
    """Drive the wrapped call_tool the way the MCP client would."""
    wrapper = instrumentor._wrap_call_tool(_FakeClientSession.call_tool)
    return asyncio.run(wrapper(session, name, arguments))


class TestSpanEmission:
    """One span per callTool, carrying the mcp.* attribution."""

    def test_emits_one_span_per_call(self, instrumentor, exporter):
        session = _FakeClientSession()
        _invoke(instrumentor, session, "food_search_restaurants", {"query": "pizza"})
        assert len(exporter.get_finished_spans()) == 1

    def test_span_is_named_for_the_server_and_unprefixed_tool(self, instrumentor, exporter):
        _invoke(instrumentor, _FakeClientSession(), "food_search_restaurants")
        assert exporter.get_finished_spans()[0].name == "mcp.call_tool food.search_restaurants"

    def test_span_name_omits_server_when_it_cannot_be_attributed(self, instrumentor, exporter):
        # get_addresses exists on both food and instamart, so the server is
        # genuinely unknown - the name must not invent one.
        _invoke(instrumentor, _FakeClientSession(), "get_addresses")
        assert exporter.get_finished_spans()[0].name == "mcp.call_tool get_addresses"

    def test_span_carries_prefix_stripped_attribution(self, instrumentor, exporter):
        _invoke(instrumentor, _FakeClientSession(), "food_search_restaurants")
        attributes = exporter.get_finished_spans()[0].attributes
        assert attributes[MCPAttributes.TOOL] == "search_restaurants"
        assert attributes[MCPAttributes.SERVER] == "food"
        assert attributes[MCPAttributes.TOOL_RAW_NAME] == "food_search_restaurants"
        assert attributes[MCPAttributes.STAGE] == "Discover"
        assert attributes[MCPAttributes.BEHAVIOUR] == "read-only"

    def test_underlying_call_still_receives_the_raw_name(self, instrumentor):
        session = _FakeClientSession()
        _invoke(instrumentor, session, "food_search_restaurants", {"query": "pizza"})
        # Stripping is a telemetry concern only - the wire call is untouched.
        assert session.calls == [("food_search_restaurants", {"query": "pizza"})]

    def test_result_is_returned_unchanged(self, instrumentor):
        expected = _CallToolResult(structured={"ok": True})
        result = _invoke(instrumentor, _FakeClientSession(result=expected), "search_menu")
        assert result is expected


class TestSessionContext:
    """Session id, ground truth and user hashing flow from the ambient context."""

    def test_session_id_lands_on_the_span(self, instrumentor, exporter):
        with mcp_session("sess-42"):
            _invoke(instrumentor, _FakeClientSession(), "search_menu")
        attributes = exporter.get_finished_spans()[0].attributes
        assert attributes[MCPAttributes.SESSION_ID] == "sess-42"

    def test_user_id_is_hashed_on_the_span(self, instrumentor, exporter):
        with mcp_session("sess-42", user_id="+919812345678"):
            _invoke(instrumentor, _FakeClientSession(), "search_menu")
        attributes = exporter.get_finished_spans()[0].attributes
        assert attributes[MCPAttributes.USER_ID_HASH].startswith("sha256:")
        assert "+919812345678" not in str(dict(attributes))

    def test_ground_truth_scores_the_prefixed_call_correctly(self, instrumentor, exporter):
        with mcp_session("sess-42"):
            with expected_tool("search_restaurants"):
                _invoke(instrumentor, _FakeClientSession(), "food_search_restaurants")
        attributes = exporter.get_finished_spans()[0].attributes
        assert attributes[MCPAttributes.TOOL_SELECTION_CORRECT] is True

    def test_context_is_restored_after_the_block(self):
        assert get_call_context() is None
        with mcp_session("sess-42"):
            assert get_call_context().session_id == "sess-42"
        assert get_call_context() is None

    def test_expected_tool_preserves_the_surrounding_session(self):
        with mcp_session("sess-42", requested_resolution="refund"):
            with expected_tool("search_menu"):
                context = get_call_context()
                assert context.session_id == "sess-42"
                assert context.requested_resolution == "refund"
                assert context.expected_tool == "search_menu"

    def test_identifiers_learned_in_one_call_apply_to_the_next(self, instrumentor, exporter):
        result = _CallToolResult(structured={"restaurants": [{"restaurantId": "R-7"}]})
        with mcp_session("sess-hallucination"):
            _invoke(instrumentor, _FakeClientSession(result=result), "search_restaurants")
            _invoke(
                instrumentor,
                _FakeClientSession(),
                "get_restaurant_menu",
                {"restaurantId": "R-7"},
            )
            _invoke(
                instrumentor,
                _FakeClientSession(),
                "get_restaurant_menu",
                {"restaurantId": "R-INVENTED"},
            )
        spans = exporter.get_finished_spans()
        assert spans[1].attributes[MCPAttributes.IDENTIFIER_HALLUCINATED] is False
        assert spans[2].attributes[MCPAttributes.IDENTIFIER_HALLUCINATED] is True

    def test_placement_attempts_increment_within_a_session(self, instrumentor, exporter):
        with mcp_session("sess-duplicate"):
            _invoke(instrumentor, _FakeClientSession(), "place_food_order")
            _invoke(instrumentor, _FakeClientSession(), "place_food_order")
        spans = exporter.get_finished_spans()
        assert spans[0].attributes[CommerceAttributes.ORDER_PLACEMENT_ATTEMPT] == 1
        assert spans[1].attributes[CommerceAttributes.ORDER_PLACEMENT_ATTEMPT] == 2


class TestErrorHandling:
    """Errors are attributed, re-raised, and never swallowed."""

    def test_raised_exception_propagates(self, instrumentor):
        session = _FakeClientSession(error=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            _invoke(instrumentor, session, "search_menu")

    def test_raised_exception_marks_the_span_error(self, instrumentor, exporter):
        with pytest.raises(RuntimeError):
            _invoke(instrumentor, _FakeClientSession(error=RuntimeError("boom")), "search_menu")
        span = exporter.get_finished_spans()[0]
        assert span.status.status_code.name == "ERROR"
        assert span.attributes[MCPAttributes.ERROR_MESSAGE_RAW] == "boom"

    def test_http_status_is_captured_from_a_transport_error(self, instrumentor, exporter):
        class _Response:
            status_code = 429

        class _Shed(Exception):
            response = _Response()

        with pytest.raises(_Shed):
            _invoke(instrumentor, _FakeClientSession(error=_Shed("rate limited")), "search_menu")
        assert exporter.get_finished_spans()[0].attributes[MCPAttributes.ERROR_HTTP_STATUS] == 429

    def test_tool_level_error_result_marks_the_span_error(self, instrumentor, exporter):
        result = _CallToolResult(content=[_TextBlock("coupon not applicable")], is_error=True)
        _invoke(instrumentor, _FakeClientSession(result=result), "apply_food_coupon")
        span = exporter.get_finished_spans()[0]
        assert span.status.status_code.name == "ERROR"
        assert span.attributes[MCPAttributes.ERROR_MESSAGE_RAW] == "coupon not applicable"

    def test_instrumentation_failure_does_not_break_the_call(self, instrumentor, monkeypatch):
        # If attribute building throws, the underlying call must still succeed.
        monkeypatch.setattr(
            "genai_otel.mcp_semconv.client_instrumentor.build_call_attributes",
            lambda **kwargs: (_ for _ in ()).throw(ValueError("instrumentation bug")),
        )
        expected = _CallToolResult(structured={"ok": True})
        assert _invoke(instrumentor, _FakeClientSession(result=expected), "search_menu") is expected


class TestInstrumentLifecycle:
    """instrument() / uninstrument() against the real ClientSession class."""

    def test_instrument_wraps_and_uninstrument_restores(self, registry):
        pytest.importorskip("mcp", reason="mcp SDK requires Python 3.10+")
        from mcp.client.session import ClientSession

        original = ClientSession.call_tool
        made = MCPClientInstrumentor(registry=registry)
        made.instrument(OTelConfig(service_name="test"))
        try:
            assert ClientSession.call_tool is not original
            assert getattr(ClientSession, "_genai_otel_mcp_instrumented", False) is True
        finally:
            made.uninstrument()
        assert ClientSession.call_tool is original

    def test_double_instrument_does_not_stack_wrappers(self, registry):
        pytest.importorskip("mcp", reason="mcp SDK requires Python 3.10+")
        from mcp.client.session import ClientSession

        original = ClientSession.call_tool
        first = MCPClientInstrumentor(registry=registry)
        first.instrument(OTelConfig(service_name="test"))
        wrapped_once = ClientSession.call_tool
        second = MCPClientInstrumentor(registry=registry)
        second.instrument(OTelConfig(service_name="test"))
        try:
            assert ClientSession.call_tool is wrapped_once
        finally:
            first.uninstrument()
        assert ClientSession.call_tool is original

    def test_registry_can_be_attached_after_construction(self, registry):
        made = MCPClientInstrumentor()
        assert made.registry is None
        made.set_registry(registry)
        assert made.registry.candidate_count == 35

    def test_mcp_tool_calls_report_no_token_usage(self, registry):
        # Token accounting belongs to the LLM instrumentor that chose the tool.
        assert MCPClientInstrumentor(registry=registry)._extract_usage(object()) is None

    def test_registered_in_the_instrumentor_catalogue(self):
        from genai_otel.auto_instrument import INSTRUMENTORS

        assert INSTRUMENTORS["mcp_client"] is MCPClientInstrumentor

    def test_not_enabled_by_default(self):
        # Opt-in: the instrumentor is most useful with a schema map attached,
        # which only the application can supply.
        from genai_otel.config import DEFAULT_INSTRUMENTORS

        assert "mcp_client" not in DEFAULT_INSTRUMENTORS
