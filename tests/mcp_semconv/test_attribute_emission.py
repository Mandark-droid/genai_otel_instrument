"""Tests for MCP span attribute emission."""

import pytest

from genai_otel.mcp_semconv import (
    CommerceAttributes,
    MCPAttributes,
    MCPCallContext,
    MCPSessionState,
    build_call_attributes,
    build_error_attributes,
    build_result_attributes,
    compute_cart_hash,
    extract_cart,
    extract_result_payload,
    hash_identifier,
)
from genai_otel.mcp_semconv.privacy import HASH_SALT_ENV, reset_salt_cache


class _TextBlock:
    """Stand-in for an MCP ``TextContent`` block."""

    def __init__(self, text):
        self.type = "text"
        self.text = text


class _CallToolResult:
    """Stand-in for an MCP ``CallToolResult``."""

    def __init__(self, content=None, is_error=False, structured=None):
        self.content = content or []
        self.isError = is_error
        self.structuredContent = structured


class TestToolIdentityAttributes:
    """mcp.tool / mcp.server / mcp.tool.raw_name."""

    def test_emits_core_identity(self, registry):
        attributes, _ = build_call_attributes("food_search_restaurants", registry=registry)
        assert attributes[MCPAttributes.TOOL] == "search_restaurants"
        assert attributes[MCPAttributes.SERVER] == "food"
        assert attributes[MCPAttributes.TOOL_RAW_NAME] == "food_search_restaurants"
        assert attributes[MCPAttributes.TOOL_PREFIX_STRIPPED] is True

    def test_emits_schema_metadata(self, registry):
        attributes, _ = build_call_attributes("instamart_checkout", registry=registry)
        assert attributes[MCPAttributes.STAGE] == "Order"
        assert attributes[MCPAttributes.BEHAVIOUR] == "mutating"
        assert attributes[MCPAttributes.IDEMPOTENT] is False

    def test_read_only_tool_is_idempotent(self, registry):
        attributes, _ = build_call_attributes("food_get_food_cart", registry=registry)
        assert attributes[MCPAttributes.IDEMPOTENT] is True

    def test_candidate_count_is_the_surface_size(self, registry):
        attributes, _ = build_call_attributes("search_menu", registry=registry)
        assert attributes[MCPAttributes.TOOL_SELECTION_CANDIDATE_COUNT] == 35

    def test_candidate_count_can_be_overridden(self, registry):
        context = MCPCallContext(candidate_count=14)
        attributes, _ = build_call_attributes("search_menu", registry=registry, context=context)
        assert attributes[MCPAttributes.TOOL_SELECTION_CANDIDATE_COUNT] == 14

    def test_works_without_a_registry(self):
        attributes, resolution = build_call_attributes("food_search_restaurants")
        # No schema map, so no stage/behaviour - but the call is still named.
        assert attributes[MCPAttributes.TOOL] == "food_search_restaurants"
        assert MCPAttributes.STAGE not in attributes
        assert resolution.known is False

    def test_ambiguous_tool_omits_server_but_keeps_agreed_metadata(self, registry):
        attributes, _ = build_call_attributes("get_addresses", registry=registry)
        assert MCPAttributes.SERVER not in attributes
        assert attributes[MCPAttributes.STAGE] == "Discover"


class TestToolSelectionScoring:
    """Ground-truth comparison must use the unprefixed name."""

    def test_prefixed_call_scores_correct_against_unprefixed_ground_truth(self, registry):
        # This is the regression that matters: comparing the raw name
        # 'food_search_restaurants' against ground truth 'search_restaurants'
        # would mark a correct call wrong and silently zero the whole run.
        context = MCPCallContext(expected_tool="search_restaurants")
        attributes, _ = build_call_attributes(
            "food_search_restaurants", registry=registry, context=context
        )
        assert attributes[MCPAttributes.TOOL_SELECTION_CORRECT] is True
        assert attributes[MCPAttributes.TOOL_SELECTION_EXPECTED] == "search_restaurants"

    def test_genuinely_wrong_tool_scores_incorrect(self, registry):
        context = MCPCallContext(expected_tool="search_restaurants")
        attributes, _ = build_call_attributes(
            "instamart_search_products", registry=registry, context=context
        )
        assert attributes[MCPAttributes.TOOL_SELECTION_CORRECT] is False

    def test_correctness_absent_when_no_ground_truth_supplied(self, registry):
        attributes, _ = build_call_attributes("search_menu", registry=registry)
        assert MCPAttributes.TOOL_SELECTION_CORRECT not in attributes


class TestPrivacy:
    """Hash identifiers at rest; log the session id, not the bodies."""

    def test_session_id_is_recorded(self, registry):
        context = MCPCallContext(session_id="sess-abc")
        attributes, _ = build_call_attributes("search_menu", registry=registry, context=context)
        assert attributes[MCPAttributes.SESSION_ID] == "sess-abc"

    def test_session_id_also_emitted_under_the_conventional_keys(self, registry):
        """MCP spans must be joinable with the LLM spans from the same run.

        `mcp.session_id` alone shares no key with `session.id` /
        `gen_ai.conversation.id`, which is what BaseInstrumentor sets — so a
        session could not be reassembled from its own telemetry and tool calls
        fragmented into ~1.5-span "sessions". Regression test for #11.
        """
        context = MCPCallContext(session_id="sess-abc")
        attributes, _ = build_call_attributes("search_menu", registry=registry, context=context)
        assert attributes["session.id"] == "sess-abc"
        assert attributes["gen_ai.conversation.id"] == "sess-abc"

    def test_no_session_keys_when_no_session_declared(self, registry):
        """Absent a session, emit nothing rather than an empty or invented id."""
        attributes, _ = build_call_attributes("search_menu", registry=registry)
        assert "session.id" not in attributes
        assert "gen_ai.conversation.id" not in attributes

    def test_user_identifier_is_hashed_never_raw(self, registry):
        phone = "+919812345678"
        context = MCPCallContext(session_id="s", user_id=phone)
        attributes, _ = build_call_attributes("search_menu", registry=registry, context=context)
        digest = attributes[MCPAttributes.USER_ID_HASH]
        assert digest.startswith("sha256:")
        assert phone not in str(attributes)

    def test_request_body_is_never_written_to_the_span(self, registry):
        arguments = {"query": "biryani for Ramesh", "note": "call me on 9812345678"}
        session = MCPSessionState("s")
        attributes, _ = build_call_attributes(
            "search_menu", arguments=arguments, registry=registry, session_state=session
        )
        serialised = str(attributes)
        assert "biryani" not in serialised
        assert "9812345678" not in serialised

    def test_hashing_is_salted_and_stable_within_a_process(self):
        assert hash_identifier("abc") == hash_identifier("abc")
        assert hash_identifier("abc") != hash_identifier("abd")

    def test_configured_salt_changes_the_digest(self, monkeypatch):
        reset_salt_cache()
        monkeypatch.setenv(HASH_SALT_ENV, "salt-one")
        first = hash_identifier("abc")
        monkeypatch.setenv(HASH_SALT_ENV, "salt-two")
        second = hash_identifier("abc")
        assert first != second
        reset_salt_cache()

    def test_empty_identifier_hashes_to_none(self):
        assert hash_identifier(None) is None
        assert hash_identifier("   ") is None


class TestIdentifierHallucination:
    """An id the session never saw in a response."""

    def test_identifier_from_a_prior_response_is_not_hallucinated(self, registry):
        session = MCPSessionState("s")
        session.observe_response({"restaurants": [{"restaurantId": "R-100"}]})
        attributes, _ = build_call_attributes(
            "get_restaurant_menu",
            arguments={"restaurantId": "R-100"},
            registry=registry,
            session_state=session,
        )
        assert attributes[MCPAttributes.IDENTIFIER_HALLUCINATED] is False

    def test_invented_identifier_is_flagged(self, registry):
        session = MCPSessionState("s")
        session.observe_response({"restaurants": [{"restaurantId": "R-100"}]})
        attributes, _ = build_call_attributes(
            "get_restaurant_menu",
            arguments={"restaurantId": "R-999"},
            registry=registry,
            session_state=session,
        )
        assert attributes[MCPAttributes.IDENTIFIER_HALLUCINATED] is True
        assert attributes[MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS] == ["restaurantId"]

    def test_only_keys_are_recorded_never_the_invented_value(self, registry):
        session = MCPSessionState("s")
        session.observe_response({"ok": True})
        attributes, _ = build_call_attributes(
            "get_restaurant_menu",
            arguments={"restaurantId": "R-SECRET-999"},
            registry=registry,
            session_state=session,
        )
        assert "R-SECRET-999" not in str(attributes)

    def test_seeded_user_supplied_identifier_is_exempt(self, registry):
        session = MCPSessionState("s", seed_identifiers=["ORD-777"])
        attributes, _ = build_call_attributes(
            "track_food_order",
            arguments={"orderId": "ORD-777"},
            registry=registry,
            session_state=session,
        )
        assert attributes[MCPAttributes.IDENTIFIER_HALLUCINATED] is False

    def test_non_identifier_arguments_are_not_checked(self, registry):
        # A free-text query or a user-typed coupon code is legitimately absent
        # from every prior response and must not be treated as hallucinated.
        session = MCPSessionState("s")
        attributes, _ = build_call_attributes(
            "search_menu",
            arguments={"query": "paneer", "couponCode": "WELCOME50"},
            registry=registry,
            session_state=session,
        )
        assert attributes[MCPAttributes.IDENTIFIER_HALLUCINATED] is False

    def test_nested_identifiers_are_discovered(self):
        session = MCPSessionState("s")
        learned = session.observe_response(
            {"data": {"cart": {"items": [{"itemId": "I-1"}, {"itemId": "I-2"}]}}}
        )
        assert learned == 2
        hallucinated, keys = session.check_request_identifiers({"itemId": "I-2"})
        assert hallucinated is False
        assert keys == ()

    def test_batch_of_identifiers_under_one_key_is_harvested(self):
        # {"orderIds": ["O-1", "O-2"]} is a batch of identifiers, not a nested
        # structure - each element must be learned individually.
        session = MCPSessionState("s")
        assert session.observe_response({"orderIds": ["O-1", "O-2"]}) == 2
        assert session.check_request_identifiers({"orderId": "O-2"})[0] is False
        assert session.check_request_identifiers({"orderId": "O-3"})[0] is True

    def test_retention_cap_disables_detection_rather_than_guessing(self):
        session = MCPSessionState("s", max_identifiers=3)
        session.observe_response({"ids": ["a", "b", "c", "d"]})
        # Past the cap the observed set is known-incomplete, so the check must
        # decline instead of manufacturing a false positive.
        hallucinated, keys = session.check_request_identifiers({"orderId": "never-seen"})
        assert hallucinated is False
        assert keys == ()


class TestCartAndPlacement:
    """Cart hashing and duplicate-order detection."""

    def test_cart_hash_is_order_independent(self):
        first = {"items": [{"itemId": "A", "quantity": 1}, {"itemId": "B", "quantity": 2}]}
        second = {"items": [{"itemId": "B", "quantity": 2}, {"itemId": "A", "quantity": 1}]}
        assert compute_cart_hash(first) == compute_cart_hash(second)

    def test_cart_hash_changes_with_contents(self):
        first = {"items": [{"itemId": "A", "quantity": 1}]}
        second = {"items": [{"itemId": "A", "quantity": 2}]}
        assert compute_cart_hash(first) != compute_cart_hash(second)

    def test_empty_cart_hashes_to_none(self):
        assert compute_cart_hash(None) is None
        assert compute_cart_hash({}) is None

    def test_first_placement_is_attempt_one(self, registry):
        session = MCPSessionState("s")
        session.note_cart({"items": [{"itemId": "A", "quantity": 1}]})
        attributes, _ = build_call_attributes(
            "place_food_order", registry=registry, session_state=session
        )
        assert attributes[CommerceAttributes.ORDER_PLACEMENT_ATTEMPT] == 1

    def test_second_placement_on_same_cart_is_the_duplicate_signal(self, registry):
        session = MCPSessionState("s")
        session.note_cart({"items": [{"itemId": "A", "quantity": 1}]})
        build_call_attributes("place_food_order", registry=registry, session_state=session)
        attributes, _ = build_call_attributes(
            "place_food_order", registry=registry, session_state=session
        )
        assert attributes[CommerceAttributes.ORDER_PLACEMENT_ATTEMPT] == 2

    def test_read_only_call_does_not_increment_placement(self, registry):
        session = MCPSessionState("s")
        build_call_attributes("get_food_cart", registry=registry, session_state=session)
        assert session.placement_attempts() == 0

    def test_extract_cart_finds_nested_cart(self):
        assert extract_cart({"cart": {"items": [{"itemId": "A"}]}}) == {"items": [{"itemId": "A"}]}
        assert extract_cart({"items": [{"itemId": "A"}]}) == {"items": [{"itemId": "A"}]}
        assert extract_cart({"restaurants": []}) is None


class TestResultAndErrorAttributes:
    """Result payload extraction and mcp.error.* mapping."""

    def test_extracts_structured_content(self):
        result = _CallToolResult(structured={"orderId": "O-1"})
        assert extract_result_payload(result) == {"orderId": "O-1"}

    def test_extracts_json_text_content(self):
        result = _CallToolResult(content=[_TextBlock('{"orderId": "O-1"}')])
        assert extract_result_payload(result) == {"orderId": "O-1"}

    def test_extracts_plain_text_content(self):
        result = _CallToolResult(content=[_TextBlock("not json")])
        assert extract_result_payload(result) == "not json"

    def test_empty_result_payload_is_none(self):
        assert extract_result_payload(_CallToolResult()) is None
        assert extract_result_payload(None) is None

    def test_successful_placement_marks_order_placed(self, registry):
        session = MCPSessionState("s")
        _, resolution = build_call_attributes(
            "place_food_order", registry=registry, session_state=session
        )
        result = _CallToolResult(structured={"orderId": "O-1"})
        attributes = build_result_attributes(result, resolution, session)
        assert attributes[CommerceAttributes.ORDER_PLACED] is True

    def test_errored_result_does_not_mark_order_placed(self, registry):
        session = MCPSessionState("s")
        _, resolution = build_call_attributes(
            "place_food_order", registry=registry, session_state=session
        )
        result = _CallToolResult(content=[_TextBlock("upstream exploded")], is_error=True)
        attributes = build_result_attributes(result, resolution, session)
        assert CommerceAttributes.ORDER_PLACED not in attributes
        assert attributes[MCPAttributes.ERROR_MESSAGE_RAW] == "upstream exploded"

    def test_response_identifiers_are_learned_for_later_calls(self, registry):
        session = MCPSessionState("s")
        _, resolution = build_call_attributes(
            "search_restaurants", registry=registry, session_state=session
        )
        build_result_attributes(
            _CallToolResult(structured={"restaurants": [{"restaurantId": "R-5"}]}),
            resolution,
            session,
        )
        hallucinated, _ = session.check_request_identifiers({"restaurantId": "R-5"})
        assert hallucinated is False

    def test_jsonrpc_code_extracted_from_mcp_error(self):
        class _ErrorData:
            code = -32602
            message = "Invalid params: cartId not found"

        class _McpError(Exception):
            error = _ErrorData()

        attributes = build_error_attributes(_McpError("Invalid params: cartId not found"))
        assert attributes[MCPAttributes.ERROR_JSONRPC_CODE] == -32602
        assert attributes[MCPAttributes.ERROR_MESSAGE_RAW] == "Invalid params: cartId not found"

    def test_http_status_extracted_from_transport_error(self):
        class _Response:
            status_code = 503

        class _HTTPStatusError(Exception):
            response = _Response()

        attributes = build_error_attributes(_HTTPStatusError("Service Unavailable"))
        assert attributes[MCPAttributes.ERROR_HTTP_STATUS] == 503

    def test_error_message_is_recorded_verbatim_for_clustering(self):
        message = "Coupon SAVE20 is not applicable on this order (min value Rs.299)"
        attributes = build_error_attributes({"message": message})
        assert attributes[MCPAttributes.ERROR_MESSAGE_RAW] == message

    def test_no_error_source_yields_no_attributes(self):
        assert build_error_attributes(None) == {}


class TestFailureIsolation:
    """Instrumentation must never break the call it observes."""

    @pytest.mark.parametrize("arguments", [None, "", 0, [], {}])
    def test_falsy_arguments_are_tolerated(self, registry, arguments):
        attributes, _ = build_call_attributes(
            "search_menu",
            arguments=arguments,
            registry=registry,
            session_state=MCPSessionState("s"),
        )
        assert attributes[MCPAttributes.TOOL] == "search_menu"

    def test_unserialisable_cart_still_hashes(self):
        class _Opaque:
            pass

        assert compute_cart_hash({"items": [{"itemId": _Opaque()}]}) is not None
