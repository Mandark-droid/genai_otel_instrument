"""Semantic-convention attribute names for MCP (Model Context Protocol) client spans.

These names are additive to the OpenTelemetry GenAI conventions and to the
attributes emitted by OpenInference's ``smolagents`` / ``mcp`` instrumentors.
Nothing here duplicates an existing OpenInference key: OpenInference records
*what* a tool call was (``tool.name``, ``input.value``, ``output.value``), while
this namespace records *which MCP server answered it, what kind of operation it
was, and whether the call was correct* - the attribution needed to measure an
MCP tool-calling agent.

Two namespaces are defined:

``mcp.*``
    Protocol-level and tool-selection attribution.

``commerce.*``
    Domain outcome signals for commerce-style tool surfaces (cart drift,
    duplicate order placement, terminal state).

All names are stable string constants so that downstream consumers (dashboards,
the terminal-state classifier, trace queries) never hardcode a literal.
"""

from typing import Final


class MCPAttributes:
    """Attribute names in the ``mcp.*`` namespace."""

    #: Upstream MCP server that served the call (e.g. ``food``, ``instamart``).
    SERVER: Final[str] = "mcp.server"

    #: Tool name with any composite-proxy prefix removed (e.g. ``search_restaurants``).
    #: Ground-truth comparison must use this key, never the raw name.
    TOOL: Final[str] = "mcp.tool"

    #: Tool name exactly as it appeared on the wire, prefix included
    #: (e.g. ``food_search_restaurants``). Kept so the raw registration
    #: surface stays auditable.
    TOOL_RAW_NAME: Final[str] = "mcp.tool.raw_name"

    #: True when a composite-proxy prefix was stripped to derive ``mcp.tool``.
    TOOL_PREFIX_STRIPPED: Final[str] = "mcp.tool.prefix_stripped"

    #: Journey stage from the tool schema (Discover / Cart / Order / Track /
    #: Find / Reserve / Manage / Support).
    STAGE: Final[str] = "mcp.stage"

    #: ``read-only`` or ``mutating``, from the tool schema.
    BEHAVIOUR: Final[str] = "mcp.behaviour"

    #: Whether repeating this call is safe.
    IDEMPOTENT: Final[str] = "mcp.idempotent"

    #: Support-correlation key. The single most important attribute for
    #: joining an agent session to a downstream support conversation.
    SESSION_ID: Final[str] = "mcp.session_id"

    #: Verbatim upstream error text, for error clustering. Never parsed or
    #: normalised - clustering depends on the exact wording.
    ERROR_MESSAGE_RAW: Final[str] = "mcp.error.message_raw"

    #: HTTP status of the failed transport call, when one is available.
    ERROR_HTTP_STATUS: Final[str] = "mcp.error.http_status"

    #: JSON-RPC error code of the failed call, when one is available.
    ERROR_JSONRPC_CODE: Final[str] = "mcp.error.jsonrpc_code"

    #: How many tools were in the registration surface the agent chose from.
    TOOL_SELECTION_CANDIDATE_COUNT: Final[str] = "mcp.tool_selection.candidate_count"

    #: Whether the selected tool matched ground truth. Only set when ground
    #: truth was supplied - absence means "not evaluated", not "incorrect".
    TOOL_SELECTION_CORRECT: Final[str] = "mcp.tool_selection.correct"

    #: The ground-truth tool name this call was compared against.
    TOOL_SELECTION_EXPECTED: Final[str] = "mcp.tool_selection.expected"

    #: Capability the user asked for, when the harness supplies it. Drives
    #: BLOCKED_NO_TOOL sub-classification.
    TOOL_SELECTION_REQUESTED_CAPABILITY: Final[str] = "mcp.tool_selection.requested_capability"

    #: True when the request carried an identifier that appeared in no prior
    #: response in this session (and was not seeded as user-supplied).
    IDENTIFIER_HALLUCINATED: Final[str] = "mcp.identifier.hallucinated"

    #: Which argument keys triggered ``IDENTIFIER_HALLUCINATED``. Keys only -
    #: the offending values are never recorded.
    IDENTIFIER_HALLUCINATED_KEYS: Final[str] = "mcp.identifier.hallucinated_keys"

    #: Salted hash of the end-user identifier. The raw identifier is never
    #: written to a span.
    USER_ID_HASH: Final[str] = "mcp.user.id_hash"


class CommerceAttributes:
    """Attribute names in the ``commerce.*`` namespace."""

    #: Stable hash of cart contents, for drift detection between the cart the
    #: agent confirmed and the cart it ordered.
    CART_HASH: Final[str] = "commerce.cart_hash"

    #: 1-based ordinal of an order-placement call against the same cart hash.
    #: A value of 2 or more is the duplicate-order signal.
    ORDER_PLACEMENT_ATTEMPT: Final[str] = "commerce.order.placement_attempt"

    #: Terminal state of the session (see :mod:`genai_otel.mcp_semconv.terminal_state`).
    TERMINAL_STATE: Final[str] = "commerce.terminal_state"

    #: For BLOCKED_NO_TOOL, the resolution the user actually requested.
    TERMINAL_STATE_RESOLUTION: Final[str] = "commerce.terminal_state.resolution"

    #: Human-readable rule name that produced the terminal state.
    TERMINAL_STATE_REASON: Final[str] = "commerce.terminal_state.reason"

    #: Resolution the user requested, supplied by the harness / task definition.
    REQUESTED_RESOLUTION: Final[str] = "commerce.requested_resolution"

    #: Set true on a span that successfully placed an order or booking.
    ORDER_PLACED: Final[str] = "commerce.order.placed"

    #: Price shown to the user before placement.
    PRICE_QUOTED: Final[str] = "commerce.price.quoted"

    #: Price actually charged at placement. A mismatch is a degradation signal.
    PRICE_CHARGED: Final[str] = "commerce.price.charged"

    #: A coupon was requested for this order.
    COUPON_REQUESTED: Final[str] = "commerce.coupon.requested"

    #: A coupon was actually applied. Requested-but-not-applied is degradation.
    COUPON_APPLIED: Final[str] = "commerce.coupon.applied"

    #: False when the ordered variant differs from the requested variant.
    VARIANT_MATCH: Final[str] = "commerce.variant.match"


class GuardrailAttributes:
    """Attribute names read (not written) by the classifier to detect enforcement.

    These are emitted by whichever enforcement layer is in front of the agent;
    the classifier only consumes them.
    """

    #: ``allow`` | ``block`` | ``steer``.
    DECISION: Final[str] = "traceguard.decision"

    #: Generic boolean form used by non-TraceGuard enforcement layers.
    BLOCKED: Final[str] = "guardrail.blocked"

    #: Identifier of the rule that fired.
    RULE: Final[str] = "traceguard.rule"


#: Behaviour values.
BEHAVIOUR_READ_ONLY: Final[str] = "read-only"
BEHAVIOUR_MUTATING: Final[str] = "mutating"

#: The journey stages present in the supported tool schemas.
STAGES: Final[frozenset] = frozenset(
    {
        "Discover",
        "Cart",
        "Order",
        "Track",
        "Find",
        "Reserve",
        "Manage",
        "Support",
    }
)
