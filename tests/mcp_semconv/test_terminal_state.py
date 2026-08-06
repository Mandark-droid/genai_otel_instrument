"""Tests for the deterministic terminal-state classifier.

Every one of the seven states is reachable from a synthetic span set, and the
distinction that matters most - ``BLOCKED_NO_TOOL`` versus
``ABANDONED_AGENT_FAULT`` - is pinned from both directions.
"""

import pytest

from genai_otel.mcp_semconv import (
    CommerceAttributes,
    GuardrailAttributes,
    MCPAttributes,
    NoToolResolution,
    TerminalState,
    TerminalStateClassifier,
    classify_session,
)


def span(tool, server=None, status=None, **attributes):
    """Build a synthetic span in the mapping form the classifier accepts."""
    payload = {MCPAttributes.TOOL: tool}
    if server:
        payload[MCPAttributes.SERVER] = server
    payload.update(attributes)
    return {"name": "mcp.call_tool " + tool, "attributes": payload, "status": status}


def ok(tool, **attributes):
    return span(tool, **attributes)


def failed(tool, message="upstream error", **attributes):
    attributes.setdefault(MCPAttributes.ERROR_MESSAGE_RAW, message)
    return span(tool, status="ERROR", **attributes)


def placed(tool="place_food_order", **attributes):
    attributes.setdefault(CommerceAttributes.ORDER_PLACED, True)
    attributes.setdefault(CommerceAttributes.ORDER_PLACEMENT_ATTEMPT, 1)
    return span(tool, **attributes)


# ----------------------------------------------------------------------
# Every state is reachable
# ----------------------------------------------------------------------


class TestAllStatesReachable:
    """One synthetic session per terminal state."""

    def test_completed_success(self, registry):
        session = [
            ok("search_restaurants"),
            ok("get_restaurant_menu"),
            ok("update_food_cart", **{MCPAttributes.BEHAVIOUR: "mutating"}),
            placed(),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.COMPLETED_SUCCESS
        assert result.order_placed is True
        assert result.confident is True

    def test_completed_degraded_on_price_mismatch(self, registry):
        session = [
            ok("get_food_cart"),
            placed(
                **{
                    CommerceAttributes.PRICE_QUOTED: 449,
                    CommerceAttributes.PRICE_CHARGED: 519,
                }
            ),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.COMPLETED_DEGRADED
        assert result.rule == "degraded.price_mismatch"

    def test_completed_degraded_on_missed_coupon(self, registry):
        session = [
            ok("fetch_food_coupons", **{CommerceAttributes.COUPON_REQUESTED: True}),
            placed(),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.COMPLETED_DEGRADED
        assert result.rule == "degraded.coupon_missed"

    def test_completed_degraded_on_wrong_variant(self, registry):
        session = [
            ok("search_menu", **{CommerceAttributes.VARIANT_MATCH: False}),
            placed(),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.COMPLETED_DEGRADED
        assert result.rule == "degraded.wrong_variant"

    def test_completed_degraded_on_duplicate_placement(self, registry):
        session = [
            ok("get_food_cart"),
            placed(**{CommerceAttributes.ORDER_PLACEMENT_ATTEMPT: 2}),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.COMPLETED_DEGRADED
        assert result.rule == "degraded.duplicate_placement"

    def test_abandoned_recoverable(self, registry):
        session = [
            ok("search_restaurants"),
            failed("update_food_cart", "item temporarily unavailable"),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.ABANDONED_RECOVERABLE
        assert result.rule == "recoverable.unretried_failure"

    def test_abandoned_agent_fault(self, registry):
        session = [
            ok("search_restaurants"),
            failed(
                "get_restaurant_menu",
                **{
                    MCPAttributes.IDENTIFIER_HALLUCINATED: True,
                    MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS: ["restaurantId"],
                },
            ),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.ABANDONED_AGENT_FAULT
        assert result.rule == "agent_fault.hallucinated_identifier"

    def test_blocked_no_tool(self, registry):
        session = [ok("get_food_orders"), ok("track_food_order")]
        result = classify_session(session, registry=registry, requested_resolution="cancel_order")
        assert result.state is TerminalState.BLOCKED_NO_TOOL
        assert result.resolution is NoToolResolution.CANCEL_ORDER

    def test_blocked_by_guardrail(self, registry):
        session = [
            ok("search_restaurants"),
            span(
                "place_food_order",
                **{
                    GuardrailAttributes.DECISION: "block",
                    GuardrailAttributes.RULE: "spend-cap",
                },
            ),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.BLOCKED_BY_GUARDRAIL

    def test_blocked_by_platform(self, registry):
        session = [
            ok("search_restaurants"),
            failed(
                "place_food_order", "Service Unavailable", **{MCPAttributes.ERROR_HTTP_STATUS: 503}
            ),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.BLOCKED_BY_PLATFORM

    def test_every_state_is_covered_by_this_suite(self):
        # Guards against a state being added without a reachability test.
        covered = {
            TerminalState.COMPLETED_SUCCESS,
            TerminalState.COMPLETED_DEGRADED,
            TerminalState.ABANDONED_RECOVERABLE,
            TerminalState.ABANDONED_AGENT_FAULT,
            TerminalState.BLOCKED_NO_TOOL,
            TerminalState.BLOCKED_BY_GUARDRAIL,
            TerminalState.BLOCKED_BY_PLATFORM,
        }
        assert covered == set(TerminalState)


# ----------------------------------------------------------------------
# The distinction that matters
# ----------------------------------------------------------------------


class TestNoToolVersusAgentFault:
    """ "No tool could have succeeded" must never be confused with "the agent failed"."""

    #: Identical failure shape, differing only in which resolution was asked for.
    FAILING_SESSION = [
        ok("get_food_orders"),
        failed(
            "get_food_order_details",
            **{
                MCPAttributes.IDENTIFIER_HALLUCINATED: True,
                MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS: ["orderId"],
            },
        ),
    ]

    def test_same_session_is_no_tool_when_the_surface_lacks_the_capability(self, registry):
        result = classify_session(
            self.FAILING_SESSION, registry=registry, requested_resolution="refund"
        )
        assert result.state is TerminalState.BLOCKED_NO_TOOL
        assert result.resolution is NoToolResolution.REFUND

    def test_same_session_is_agent_fault_when_the_capability_exists(self, registry):
        # order_history_lookup IS satisfied (get_orders / get_food_orders), so
        # the failure is the agent's, not the surface's.
        result = classify_session(
            self.FAILING_SESSION,
            registry=registry,
            requested_resolution="order_history_lookup",
        )
        assert result.state is TerminalState.ABANDONED_AGENT_FAULT
        assert result.rule == "agent_fault.hallucinated_identifier"

    def test_no_tool_still_records_the_agent_signals_as_evidence(self, registry):
        result = classify_session(
            self.FAILING_SESSION, registry=registry, requested_resolution="refund"
        )
        rules = {item.rule for item in result.evidence}
        assert "no_tool.agent_signals_present_but_not_decisive" in rules
        assert result.signals["hallucinated_span_indexes"] == [1]

    @pytest.mark.parametrize(
        "resolution",
        [
            "cancel_order",
            "refund",
            "replacement",
            "missing_item_claim",
            "modify_placed_order",
            "reschedule_booking",
            "cancel_booking",
            "escalate_to_human",
            "address_edit_mid_cart",
        ],
    )
    def test_each_unsupported_resolution_sub_classifies(self, registry, resolution):
        result = classify_session(
            [ok("report_error")], registry=registry, requested_resolution=resolution
        )
        assert result.state is TerminalState.BLOCKED_NO_TOOL
        assert result.resolution is NoToolResolution(resolution)

    def test_the_one_supported_resolution_is_not_a_gap(self, registry):
        result = classify_session(
            [ok("get_orders")], registry=registry, requested_resolution="order_history_lookup"
        )
        assert result.state is not TerminalState.BLOCKED_NO_TOOL

    def test_available_capability_never_called_is_recoverable(self, registry):
        # The tool existed and would have worked; the agent just never used it.
        result = classify_session(
            [ok("search_restaurants")],
            registry=registry,
            requested_resolution="order_history_lookup",
        )
        assert result.state is TerminalState.ABANDONED_RECOVERABLE
        assert result.rule == "recoverable.capability_available_unused"

    def test_no_tool_cannot_be_decided_without_a_surface_description(self):
        # Declining is correct here - claiming a gap with no knowledge of the
        # surface would be a fabricated finding.
        result = classify_session([ok("report_error")], requested_resolution="refund")
        assert result.state is not TerminalState.BLOCKED_NO_TOOL

    def test_explicit_capability_list_works_without_a_registry(self):
        result = classify_session(
            [ok("report_error")],
            requested_resolution="refund",
            available_capabilities=["order_history_lookup"],
        )
        assert result.state is TerminalState.BLOCKED_NO_TOOL


# ----------------------------------------------------------------------
# Precedence
# ----------------------------------------------------------------------


class TestRulePrecedence:
    """The documented ordering, pinned so it cannot drift silently."""

    def test_guardrail_outranks_everything(self, registry):
        session = [
            span(
                "place_food_order",
                **{
                    GuardrailAttributes.DECISION: "deny",
                    MCPAttributes.IDENTIFIER_HALLUCINATED: True,
                    MCPAttributes.ERROR_HTTP_STATUS: 503,
                },
            )
        ]
        result = classify_session(session, registry=registry, requested_resolution="refund")
        assert result.state is TerminalState.BLOCKED_BY_GUARDRAIL

    def test_steer_is_not_a_block(self, registry):
        # A steer is a redirect with advisory guidance, not enforcement.
        session = [placed(**{GuardrailAttributes.DECISION: "steer"})]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.COMPLETED_SUCCESS

    def test_platform_outranks_the_surface_gap(self, registry):
        # A cancellation request that hit a 503 is a transient platform failure,
        # not evidence that the tool surface is missing something.
        session = [failed("report_error", **{MCPAttributes.ERROR_HTTP_STATUS: 503})]
        result = classify_session(session, registry=registry, requested_resolution="cancel_order")
        assert result.state is TerminalState.BLOCKED_BY_PLATFORM

    def test_surface_gap_outranks_a_completed_order(self, registry):
        # The order went through, then the user asked to cancel and nothing
        # could. The terminal condition is the gap - but the completed order is
        # still recorded rather than lost.
        session = [placed(), ok("track_food_order")]
        result = classify_session(session, registry=registry, requested_resolution="cancel_order")
        assert result.state is TerminalState.BLOCKED_NO_TOOL
        assert result.order_placed is True

    def test_agent_fault_outranks_recoverable(self, registry):
        session = [
            failed(
                "get_food_order_details",
                **{
                    MCPAttributes.IDENTIFIER_HALLUCINATED: True,
                    MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS: ["orderId"],
                },
            )
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.ABANDONED_AGENT_FAULT

    def test_recovered_failure_does_not_drive_the_outcome(self, registry):
        # A 503 the session recovered from must not classify it as blocked.
        session = [
            failed("search_restaurants", **{MCPAttributes.ERROR_HTTP_STATUS: 503}),
            ok("search_restaurants"),
            placed(),
        ]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.COMPLETED_SUCCESS


# ----------------------------------------------------------------------
# Cart drift
# ----------------------------------------------------------------------


class TestCartDrift:
    """Drift is an *unexplained* hash change, not any hash change."""

    #: Realistic attribute sets, matching what the instrumentor actually emits.
    CART_READ = {MCPAttributes.STAGE: "Cart", MCPAttributes.BEHAVIOUR: "read-only"}
    CART_WRITE = {MCPAttributes.STAGE: "Cart", MCPAttributes.BEHAVIOUR: "mutating"}
    PLACEMENT = {MCPAttributes.STAGE: "Order", MCPAttributes.BEHAVIOUR: "mutating"}

    def test_normal_cart_building_is_not_drift(self, registry):
        # Each hash change is explained by the cart mutation that caused it.
        session = [
            ok("get_food_cart", **dict(self.CART_READ, **{CommerceAttributes.CART_HASH: "h1"})),
            ok("update_food_cart", **dict(self.CART_WRITE, **{CommerceAttributes.CART_HASH: "h2"})),
            ok("update_food_cart", **dict(self.CART_WRITE, **{CommerceAttributes.CART_HASH: "h3"})),
            placed(**dict(self.PLACEMENT, **{CommerceAttributes.CART_HASH: "h3"})),
        ]
        result = classify_session(session, registry=registry)
        assert result.signals["cart_drift"] is False
        assert result.state is TerminalState.COMPLETED_SUCCESS

    def test_unexplained_hash_change_is_drift(self, registry):
        # The cart moved between the read and the placement with no cart
        # mutation in between: the cart ordered is not the cart confirmed.
        session = [
            ok("get_food_cart", **dict(self.CART_READ, **{CommerceAttributes.CART_HASH: "h1"})),
            placed(**dict(self.PLACEMENT, **{CommerceAttributes.CART_HASH: "h9"})),
        ]
        result = classify_session(session, registry=registry)
        assert result.signals["cart_drift"] is True
        assert result.state is TerminalState.COMPLETED_DEGRADED
        assert result.rule == "degraded.cart_drift"

    def test_placement_does_not_excuse_a_cart_change(self, registry):
        # Placement is a mutating call, but it mutates the order, not the cart.
        # If it were allowed to explain a cart change this drift would vanish.
        session = [
            ok("get_food_cart", **dict(self.CART_READ, **{CommerceAttributes.CART_HASH: "h1"})),
            ok("get_food_cart", **dict(self.CART_READ, **{CommerceAttributes.CART_HASH: "h1"})),
            placed(**dict(self.PLACEMENT, **{CommerceAttributes.CART_HASH: "h2"})),
        ]
        result = classify_session(session, registry=registry)
        assert result.signals["cart_drift"] is True
        assert result.signals["cart_drift_span_indexes"] == [2]

    def test_cart_mutation_between_readings_explains_the_change(self, registry):
        # The mutating call reported no hash of its own, but it still explains
        # the change observed by the next read.
        session = [
            ok("get_food_cart", **dict(self.CART_READ, **{CommerceAttributes.CART_HASH: "h1"})),
            ok("update_food_cart", **self.CART_WRITE),
            ok("get_food_cart", **dict(self.CART_READ, **{CommerceAttributes.CART_HASH: "h2"})),
            placed(**dict(self.PLACEMENT, **{CommerceAttributes.CART_HASH: "h2"})),
        ]
        result = classify_session(session, registry=registry)
        assert result.signals["cart_drift"] is False


# ----------------------------------------------------------------------
# Evidence and auditability
# ----------------------------------------------------------------------


class TestEvidence:
    """Every classification must be auditable."""

    def test_evidence_names_the_span_and_attributes_that_drove_it(self, registry):
        session = [
            ok("search_restaurants"),
            failed(
                "get_restaurant_menu",
                **{
                    MCPAttributes.IDENTIFIER_HALLUCINATED: True,
                    MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS: ["restaurantId"],
                },
            ),
        ]
        result = classify_session(session, registry=registry)
        evidence = result.evidence[0]
        assert evidence.span_index == 1
        assert evidence.span_name == "mcp.call_tool get_restaurant_menu"
        assert evidence.attributes[MCPAttributes.IDENTIFIER_HALLUCINATED] is True

    def test_explain_renders_an_audit_trail(self, registry):
        result = classify_session(
            [ok("report_error")], registry=registry, requested_resolution="refund"
        )
        rendered = result.explain()
        assert "BLOCKED_NO_TOOL" in rendered
        assert "resolution=refund" in rendered
        assert "no tool in the 35-tool registration surface" in rendered

    def test_signals_are_reported_even_when_not_decisive(self, registry):
        result = classify_session([placed()], registry=registry)
        assert result.signals["span_count"] == 1
        assert result.signals["order_placed"] is True

    def test_classification_is_deterministic(self, registry):
        session = [ok("search_restaurants"), failed("update_food_cart")]
        first = classify_session(session, registry=registry)
        second = classify_session(session, registry=registry)
        assert first.state is second.state
        assert first.rule == second.rule


class TestFallbackAndInputForms:
    """Defaulting is explicit, and span inputs are format-agnostic."""

    def test_empty_session_defaults_with_confident_false(self, registry):
        result = classify_session([], registry=registry)
        assert result.state is TerminalState.ABANDONED_RECOVERABLE
        assert result.confident is False
        assert result.rule == "fallback.no_terminal_signal"

    def test_browsing_session_that_just_stops_defaults(self, registry):
        result = classify_session(
            [ok("search_restaurants"), ok("get_restaurant_menu")], registry=registry
        )
        assert result.state is TerminalState.ABANDONED_RECOVERABLE
        assert result.confident is False

    def test_reads_requested_resolution_from_span_attributes(self, registry):
        session = [ok("report_error", **{CommerceAttributes.REQUESTED_RESOLUTION: "refund"})]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.BLOCKED_NO_TOOL
        assert result.resolution is NoToolResolution.REFUND

    def test_explicit_argument_overrides_the_span_attribute(self, registry):
        session = [ok("report_error", **{CommerceAttributes.REQUESTED_RESOLUTION: "refund"})]
        result = classify_session(
            session, registry=registry, requested_resolution="order_history_lookup"
        )
        assert result.state is not TerminalState.BLOCKED_NO_TOOL

    def test_accepts_object_spans_not_just_mappings(self, registry):
        class _Span:
            name = "mcp.call_tool place_food_order"
            attributes = {
                MCPAttributes.TOOL: "place_food_order",
                CommerceAttributes.ORDER_PLACED: True,
            }
            status = None

        result = classify_session([_Span()], registry=registry)
        assert result.state is TerminalState.COMPLETED_SUCCESS

    def test_string_boolean_attributes_are_tolerated(self, registry):
        session = [span("place_food_order", **{CommerceAttributes.ORDER_PLACED: "true"})]
        result = classify_session(session, registry=registry)
        assert result.state is TerminalState.COMPLETED_SUCCESS

    def test_unknown_resolution_name_yields_no_enum_but_still_blocks(self, registry):
        result = classify_session(
            [ok("report_error")],
            registry=registry,
            requested_resolution="teleport_the_order",
        )
        assert result.state is TerminalState.BLOCKED_NO_TOOL
        assert result.resolution is None

    def test_classifier_reports_the_surface_gaps(self, registry):
        classifier = TerminalStateClassifier(registry=registry)
        missing = classifier.missing_capabilities()
        assert "cancel_order" in missing
        assert "order_history_lookup" not in missing
