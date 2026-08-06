"""MCP client instrumentor - one span per ``callTool``.

Why this exists alongside OpenInference
---------------------------------------
OpenInference's ``mcp`` instrumentor is a context-propagation shim: it threads
W3C trace context through the JSON-RPC ``_meta`` field so client and server
spans join up. It does not describe the call. OpenInference's ``smolagents``
instrumentor describes the *agent's* tool call (``tool.name``, ``input.value``,
``output.value``) but knows nothing about MCP - not which server answered, not
whether the call mutates state, not whether the identifier the agent passed was
real.

This instrumentor fills exactly that gap and nothing else. It wraps
``mcp.client.session.ClientSession.call_tool``, which is the one choke point
every MCP tool call passes through regardless of the driving framework, and
emits the ``mcp.*`` / ``commerce.*`` attribution defined in
:mod:`genai_otel.mcp_semconv.attributes`.

Privacy
-------
Request and response bodies are never written to a span by this module. The
session id is recorded (it is the support-correlation key); user identifiers are
hashed. Only *keys* are recorded for hallucinated identifiers, never values.
"""

import contextlib
import contextvars
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from ..instrumentors.base import BaseInstrumentor
from .attributes import CommerceAttributes, MCPAttributes
from .privacy import hash_identifier
from .session_tracker import MCPSessionRegistry, MCPSessionState, compute_cart_hash
from .tool_registry import MCPToolRegistry, ToolResolution

logger = logging.getLogger(__name__)


@dataclass
class MCPCallContext:
    """Ambient context the harness supplies for the current agent session.

    None of this is inferable from the MCP protocol itself, so it is set by
    whatever drives the agent (the eval harness, the application) rather than
    guessed.
    """

    session_id: Optional[str] = None
    #: Ground-truth tool for the current step, when known.
    expected_tool: Optional[str] = None
    #: Capability the user asked for. Drives BLOCKED_NO_TOOL sub-classification.
    requested_resolution: Optional[str] = None
    #: Overrides the registry's tool count when the surface offered to the model
    #: differs from the full registry.
    candidate_count: Optional[int] = None
    #: Raw end-user identifier. Hashed before it reaches a span.
    user_id: Optional[str] = None
    #: Identifiers the user supplied up front, exempt from hallucination checks.
    seed_identifiers: Tuple[str, ...] = field(default_factory=tuple)


_call_context: "contextvars.ContextVar[Optional[MCPCallContext]]" = contextvars.ContextVar(
    "genai_otel_mcp_call_context", default=None
)


def get_call_context() -> Optional[MCPCallContext]:
    """Return the active MCP call context, if any."""
    return _call_context.get()


def set_call_context(context: Optional[MCPCallContext]) -> "contextvars.Token":
    """Set the active MCP call context. Returns a token for :func:`reset_call_context`."""
    return _call_context.set(context)


def reset_call_context(token: "contextvars.Token") -> None:
    """Restore a previous MCP call context."""
    _call_context.reset(token)


@contextlib.contextmanager
def mcp_session(
    session_id: str,
    user_id: Optional[str] = None,
    requested_resolution: Optional[str] = None,
    seed_identifiers: Sequence[str] = (),
    candidate_count: Optional[int] = None,
) -> Iterator[MCPCallContext]:
    """Scope a block of agent work to one MCP session.

    Example::

        with mcp_session("sess-123", user_id=phone, requested_resolution="cancel_order"):
            agent.run(task)
    """
    context = MCPCallContext(
        session_id=session_id,
        user_id=user_id,
        requested_resolution=requested_resolution,
        seed_identifiers=tuple(seed_identifiers),
        candidate_count=candidate_count,
    )
    token = set_call_context(context)
    try:
        yield context
    finally:
        reset_call_context(token)


@contextlib.contextmanager
def expected_tool(name: Optional[str]) -> Iterator[None]:
    """Declare the ground-truth tool for the next call(s) inside this block."""
    current = get_call_context()
    if current is None:
        context = MCPCallContext(expected_tool=name)
    else:
        context = MCPCallContext(
            session_id=current.session_id,
            expected_tool=name,
            requested_resolution=current.requested_resolution,
            candidate_count=current.candidate_count,
            user_id=current.user_id,
            seed_identifiers=current.seed_identifiers,
        )
    token = set_call_context(context)
    try:
        yield
    finally:
        reset_call_context(token)


# ----------------------------------------------------------------------
# Payload helpers
# ----------------------------------------------------------------------


def _maybe_json(text: str) -> Any:
    """Parse ``text`` as JSON, returning the raw string when it is not JSON."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def extract_result_payload(result: Any) -> Any:
    """Extract a Python payload from an MCP ``CallToolResult``.

    Handles the structured-content form and the text-content list form, and
    tolerates plain dicts so the helper is usable on replayed traces.
    """
    if result is None:
        return None
    if isinstance(result, (Mapping, list)):
        return result

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured

    content = getattr(result, "content", None)
    if not content:
        return None

    parsed: List[Any] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, Mapping):
            text = block.get("text")
        if text is not None:
            parsed.append(_maybe_json(text))
    if not parsed:
        return None
    return parsed[0] if len(parsed) == 1 else parsed


def extract_cart(payload: Any) -> Any:
    """Find a cart object inside a tool response payload."""
    if isinstance(payload, Mapping):
        for key in ("cart", "cartDetails", "cart_details", "data"):
            value = payload.get(key)
            if isinstance(value, Mapping) and any(
                k in value for k in ("items", "cartItems", "cart_items", "lineItems", "line_items")
            ):
                return value
        if any(k in payload for k in ("items", "cartItems", "cart_items")):
            return payload
    return None


def _is_error_result(result: Any) -> bool:
    """Whether an MCP ``CallToolResult`` reports a tool-level error."""
    flag = getattr(result, "isError", None)
    if flag is None and isinstance(result, Mapping):
        flag = result.get("isError", result.get("is_error"))
    if flag is None:
        flag = getattr(result, "is_error", None)
    return bool(flag)


def _error_text(payload: Any) -> Optional[str]:
    """Best-effort verbatim error text from an error payload."""
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("message", "error", "detail", "reason"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, Mapping):
                nested = value.get("message")
                if isinstance(nested, str) and nested:
                    return nested
    if isinstance(payload, list) and payload:
        return _error_text(payload[0])
    return str(payload)


def build_error_attributes(source: Any) -> Dict[str, Any]:
    """Extract ``mcp.error.*`` attributes from an exception or error payload.

    The message is recorded verbatim - error clustering depends on the exact
    upstream wording, so it is never normalised or truncated here.
    """
    attributes: Dict[str, Any] = {}
    if source is None:
        return attributes

    # JSON-RPC error data, as carried by mcp.shared.exceptions.McpError.
    error_data = getattr(source, "error", None)
    code = getattr(error_data, "code", None)
    if code is None:
        code = getattr(source, "code", None)
    if code is None and isinstance(source, Mapping):
        nested = source.get("error")
        if isinstance(nested, Mapping):
            code = nested.get("code")
        if code is None:
            code = source.get("code")
    if isinstance(code, bool):
        code = None
    if isinstance(code, int):
        attributes[MCPAttributes.ERROR_JSONRPC_CODE] = code

    # HTTP status, as carried by httpx.HTTPStatusError and friends.
    status = None
    response = getattr(source, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(source, "status_code", None)
    if status is None and isinstance(source, Mapping):
        status = source.get("status_code") or source.get("httpStatus")
    if isinstance(status, bool):
        status = None
    if isinstance(status, int):
        attributes[MCPAttributes.ERROR_HTTP_STATUS] = status

    message = None
    if isinstance(source, BaseException):
        message = getattr(error_data, "message", None) or str(source)
    else:
        message = _error_text(source)
    if message:
        attributes[MCPAttributes.ERROR_MESSAGE_RAW] = message

    return attributes


# ----------------------------------------------------------------------
# Attribute builders
# ----------------------------------------------------------------------


def build_span_name(resolution: ToolResolution, fallback: Any = None) -> str:
    """Build the span name for one ``callTool``.

    ``mcp.call_tool {server}.{tool}`` when the server is known, falling back to
    ``mcp.call_tool {tool}`` when it is not (an unmapped tool, or a name present
    on several servers). The server is part of the name deliberately: the same
    tool name exists on more than one server in a composite surface, so a
    name without it collapses distinct operations into one entry in every
    latency and error-rate breakdown.
    """
    tool = resolution.tool or (str(fallback) if fallback else "")
    if resolution.server:
        return "mcp.call_tool {0}.{1}".format(resolution.server, tool)
    return "mcp.call_tool {0}".format(tool)


def build_call_attributes(
    raw_name: str,
    arguments: Any = None,
    registry: Optional[MCPToolRegistry] = None,
    session_state: Optional[MCPSessionState] = None,
    context: Optional[MCPCallContext] = None,
    server_hint: Optional[str] = None,
) -> Tuple[Dict[str, Any], ToolResolution]:
    """Build the pre-call ``mcp.*`` attributes for one ``callTool``.

    Args:
        raw_name: Tool name as it arrived on the wire, prefix included.
        arguments: Tool arguments, used only for identifier checking. Never
            recorded on the span.
        registry: Tool metadata registry.
        session_state: Session state for hallucination detection.
        context: Ambient harness context (session id, ground truth).
        server_hint: Server known from the transport.

    Returns:
        Tuple of the attribute dict and the tool resolution.
    """
    if registry is not None:
        resolution = registry.resolve(raw_name, server_hint=server_hint)
    else:
        resolution = ToolResolution(raw_name=raw_name, tool=raw_name, server=server_hint)

    attributes: Dict[str, Any] = {
        MCPAttributes.TOOL: resolution.tool,
        MCPAttributes.TOOL_RAW_NAME: resolution.raw_name,
    }
    if resolution.prefix_stripped:
        attributes[MCPAttributes.TOOL_PREFIX_STRIPPED] = True
    if resolution.server:
        attributes[MCPAttributes.SERVER] = resolution.server
    if resolution.stage:
        attributes[MCPAttributes.STAGE] = resolution.stage
    if resolution.behaviour:
        attributes[MCPAttributes.BEHAVIOUR] = resolution.behaviour
    if resolution.idempotent is not None:
        attributes[MCPAttributes.IDEMPOTENT] = resolution.idempotent

    # Tool-selection attribution.
    candidate_count = context.candidate_count if context is not None else None
    if candidate_count is None and registry is not None:
        candidate_count = registry.candidate_count
    if candidate_count:
        attributes[MCPAttributes.TOOL_SELECTION_CANDIDATE_COUNT] = candidate_count

    if context is not None:
        if context.session_id:
            attributes[MCPAttributes.SESSION_ID] = context.session_id
            # Also emit the conventional keys. `mcp.session_id` alone made MCP
            # spans unjoinable with the LLM spans from the same agent run, which
            # set `session.id` / `gen_ai.conversation.id` via
            # BaseInstrumentor's session_id_extractor. With no key in common, a
            # session cannot be reassembled from its own telemetry: on a real
            # run 10 spans carried session.id and 450+ tool calls carried none,
            # so 552 spans grouped into 209 "sessions" of ~1.5 spans each where
            # eight runs actually happened -- and the fragments look plausible,
            # so the error is silent. Additive: anything already reading
            # `mcp.session_id` is unaffected.
            # https://github.com/Mandark-droid/genai_otel_instrument/issues/11
            attributes["session.id"] = context.session_id
            attributes["gen_ai.conversation.id"] = context.session_id
        if context.expected_tool:
            # Compared against the unprefixed name. Comparing against the raw
            # name here is the bug that silently scores every run 0%.
            attributes[MCPAttributes.TOOL_SELECTION_EXPECTED] = context.expected_tool
            attributes[MCPAttributes.TOOL_SELECTION_CORRECT] = (
                resolution.tool == context.expected_tool
            )
        if context.requested_resolution:
            attributes[MCPAttributes.TOOL_SELECTION_REQUESTED_CAPABILITY] = (
                context.requested_resolution
            )
        if context.user_id:
            hashed = hash_identifier(context.user_id)
            if hashed:
                attributes[MCPAttributes.USER_ID_HASH] = hashed

    # Identifier hallucination: an id in the request that appeared in no prior
    # response and was not seeded as user-supplied.
    if session_state is not None and arguments:
        hallucinated, keys = session_state.check_request_identifiers(arguments)
        attributes[MCPAttributes.IDENTIFIER_HALLUCINATED] = hallucinated
        if keys:
            attributes[MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS] = list(keys)

    # Cart drift + duplicate placement.
    if session_state is not None:
        cart_hash = session_state.cart_hash
        if arguments:
            argument_cart = extract_cart(arguments)
            if argument_cart is not None:
                cart_hash = compute_cart_hash(argument_cart) or cart_hash
        if cart_hash:
            attributes[CommerceAttributes.CART_HASH] = cart_hash
        if resolution.placement:
            attributes[CommerceAttributes.ORDER_PLACEMENT_ATTEMPT] = session_state.note_placement(
                cart_hash
            )

    return attributes, resolution


def build_result_attributes(
    result: Any,
    resolution: ToolResolution,
    session_state: Optional[MCPSessionState] = None,
) -> Dict[str, Any]:
    """Build post-call attributes and update session state from a tool result."""
    attributes: Dict[str, Any] = {}
    payload = extract_result_payload(result)

    if _is_error_result(result):
        attributes.update(build_error_attributes(payload))
        return attributes

    if session_state is not None and payload is not None:
        # Everything the session has legitimately seen becomes non-hallucinated
        # for subsequent calls.
        session_state.observe_response(payload)
        cart = extract_cart(payload)
        if cart is not None:
            cart_hash = session_state.note_cart(cart)
            if cart_hash:
                attributes[CommerceAttributes.CART_HASH] = cart_hash

    if resolution.placement:
        attributes[CommerceAttributes.ORDER_PLACED] = True

    return attributes


class MCPClientInstrumentor(BaseInstrumentor):
    """Instrumentor emitting one span per MCP ``callTool``."""

    def __init__(
        self,
        registry: Optional[MCPToolRegistry] = None,
        session_registry: Optional[MCPSessionRegistry] = None,
    ):
        """Create the instrumentor.

        Args:
            registry: Tool metadata registry. Without one, spans still carry
                the tool name, server hint and error attribution, but no
                stage/behaviour/idempotency.
            session_registry: Session state store. A private one is created by
                default.
        """
        super().__init__()
        self.registry = registry
        self.session_registry = session_registry or MCPSessionRegistry()
        self._mcp_available = False
        self._original_call_tool = None
        self._check_availability()

    def _check_availability(self) -> None:
        """Check whether the MCP client SDK is installed."""
        try:
            import mcp.client.session  # noqa: F401

            self._mcp_available = True
            logger.debug("MCP client SDK detected and available for instrumentation")
        except ImportError:
            logger.debug("MCP client SDK not installed, instrumentation will be skipped")
            self._mcp_available = False

    def set_registry(self, registry: MCPToolRegistry) -> None:
        """Attach or replace the tool metadata registry."""
        self.registry = registry

    def instrument(self, config: OTelConfig) -> None:
        """Wrap ``ClientSession.call_tool``.

        Args:
            config: The OpenTelemetry configuration object.
        """
        if not self._mcp_available:
            logger.debug("Skipping MCP client instrumentation - library not available")
            return

        self.config = config

        try:
            from mcp.client.session import ClientSession

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(ClientSession, "_genai_otel_mcp_instrumented", False) is True:
                logger.debug("MCP client already instrumented, skipping")
                self._instrumented = True
                return

            original = ClientSession.call_tool
            self._original_call_tool = original
            ClientSession.call_tool = self._wrap_call_tool(original)

            try:
                ClientSession._genai_otel_mcp_instrumented = True
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                pass

            self._instrumented = True
            logger.info("MCP client instrumentation enabled")

        except Exception as e:  # noqa: BLE001
            logger.error("Failed to instrument MCP client: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def uninstrument(self) -> None:
        """Restore the original ``ClientSession.call_tool``."""
        if not self._instrumented or self._original_call_tool is None:
            return
        try:
            from mcp.client.session import ClientSession

            ClientSession.call_tool = self._original_call_tool
            ClientSession._genai_otel_mcp_instrumented = False
            self._instrumented = False
            logger.info("MCP client instrumentation removed")
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to uninstrument MCP client: %s", e)

    def _resolve_session_state(
        self, context: Optional[MCPCallContext]
    ) -> Optional[MCPSessionState]:
        """Get session state for the active context, if a session id is known."""
        if context is None or not context.session_id:
            return None
        return self.session_registry.get(
            context.session_id, seed_identifiers=context.seed_identifiers
        )

    def _wrap_call_tool(self, original):
        """Build the async wrapper around ``ClientSession.call_tool``."""
        instrumentor = self

        async def call_tool_wrapper(session_self, name, *args, **kwargs):
            arguments = kwargs.get("arguments")
            if arguments is None and args:
                arguments = args[0]

            context = get_call_context()
            session_state = instrumentor._resolve_session_state(context)
            server_hint = getattr(session_self, "_genai_otel_server_name", None)

            try:
                attributes, resolution = build_call_attributes(
                    raw_name=name,
                    arguments=arguments,
                    registry=instrumentor.registry,
                    session_state=session_state,
                    context=context,
                    server_hint=server_hint,
                )
            except Exception as e:  # noqa: BLE001
                # Instrumentation must never break the call it observes.
                logger.debug("Failed to build MCP call attributes: %s", e)
                attributes, resolution = {}, ToolResolution(raw_name=str(name), tool=str(name))

            span_name = build_span_name(resolution, fallback=name)
            span = instrumentor.tracer.start_span(span_name, attributes=attributes)

            with trace.use_span(span, end_on_exit=True, record_exception=False):
                try:
                    result = await original(session_self, name, *args, **kwargs)
                except BaseException as exc:
                    try:
                        span.set_attributes(build_error_attributes(exc))
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Failed to record MCP error attributes: %s", e)
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

                try:
                    span.set_attributes(build_result_attributes(result, resolution, session_state))
                    if _is_error_result(result):
                        span.set_status(Status(StatusCode.ERROR))
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to record MCP result attributes: %s", e)

                return result

        # Preserve the wrapped signature for introspection-based callers.
        try:
            call_tool_wrapper.__name__ = getattr(original, "__name__", "call_tool")
            call_tool_wrapper.__doc__ = getattr(original, "__doc__", None)
            call_tool_wrapper.__wrapped__ = original
            if not inspect.iscoroutinefunction(original):
                logger.debug("ClientSession.call_tool is not a coroutine function")
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
            pass

        return call_tool_wrapper

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """MCP tool calls carry no token usage.

        Token accounting for the model that *chose* the tool belongs to the LLM
        instrumentor for that provider, not here.
        """
        return None
