"""MCP semantic conventions and terminal-state classification.

Two capabilities live here, both aimed at measuring an MCP tool-calling agent:

**MCP attribution** (:mod:`~genai_otel.mcp_semconv.client_instrumentor`) - one
span per ``callTool`` carrying which server answered, what kind of operation it
was, whether the tool choice was correct, and whether the identifiers passed
were real. This is additive to OpenInference: OpenInference records that a tool
was called, this records what the call *was*.

**Terminal-state classification** (:mod:`~genai_otel.mcp_semconv.terminal_state`)
- a deterministic, rule-based reduction of a session's spans to exactly one of
seven outcomes, with the evidence that produced it. No LLM judgement.

Typical use::

    from genai_otel.mcp_semconv import (
        MCPClientInstrumentor,
        MCPToolRegistry,
        classify_session,
        mcp_session,
    )

    registry = MCPToolRegistry.from_directory("docs/demos/swiggy-mcp-schemas")
    registry.mark_placement(["book_table"])          # stage Reserve, but a placement

    instrumentor = MCPClientInstrumentor(registry=registry)
    instrumentor.instrument(config)

    with mcp_session("sess-123", user_id=phone, requested_resolution="cancel_order"):
        agent.run(task)

    result = classify_session(collected_spans, registry=registry)
    print(result.explain())
"""

from .attributes import (
    BEHAVIOUR_MUTATING,
    BEHAVIOUR_READ_ONLY,
    STAGES,
    CommerceAttributes,
    GuardrailAttributes,
    MCPAttributes,
)
from .client_instrumentor import (
    MCPCallContext,
    MCPClientInstrumentor,
    build_call_attributes,
    build_error_attributes,
    build_result_attributes,
    expected_tool,
    extract_cart,
    extract_result_payload,
    get_call_context,
    mcp_session,
    reset_call_context,
    set_call_context,
)
from .privacy import HASH_SALT_ENV, hash_identifier, hash_user_fields
from .session_tracker import (
    MCPSessionRegistry,
    MCPSessionState,
    canonicalise_cart,
    compute_cart_hash,
)
from .terminal_state import (
    Classification,
    Evidence,
    NoToolResolution,
    TerminalState,
    TerminalStateClassifier,
    classify_session,
)
from .tool_registry import (
    DEFAULT_CAPABILITY_PATTERNS,
    KNOWN_CAPABILITIES,
    MCPToolRegistry,
    ToolMetadata,
    ToolResolution,
)

__all__ = [
    # Attributes
    "MCPAttributes",
    "CommerceAttributes",
    "GuardrailAttributes",
    "BEHAVIOUR_READ_ONLY",
    "BEHAVIOUR_MUTATING",
    "STAGES",
    # Registry
    "MCPToolRegistry",
    "ToolMetadata",
    "ToolResolution",
    "DEFAULT_CAPABILITY_PATTERNS",
    "KNOWN_CAPABILITIES",
    # Instrumentation
    "MCPClientInstrumentor",
    "MCPCallContext",
    "mcp_session",
    "expected_tool",
    "get_call_context",
    "set_call_context",
    "reset_call_context",
    "build_call_attributes",
    "build_result_attributes",
    "build_error_attributes",
    "extract_result_payload",
    "extract_cart",
    # Session state
    "MCPSessionRegistry",
    "MCPSessionState",
    "compute_cart_hash",
    "canonicalise_cart",
    # Privacy
    "hash_identifier",
    "hash_user_fields",
    "HASH_SALT_ENV",
    # Terminal state
    "TerminalState",
    "NoToolResolution",
    "TerminalStateClassifier",
    "Classification",
    "Evidence",
    "classify_session",
]
