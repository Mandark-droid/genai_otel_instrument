"""Fast/slow routing telemetry: router metadata and decision-to-call links (issue #49).

In fast/slow routing a small decision model handles most steps, and a router sends a
request to a large model only when it judges the step needs it. Measuring whether
that pays off needs two facts no span carried:

1. **Why the router chose the model.** Routers return it in response headers
   (LiteLLM's proxy: ``x-litellm-model-id``, ``x-litellm-model-group``,
   ``x-litellm-attempted-fallbacks``...), and which headers differs per router.
   :func:`record_response_headers` copies an allowlist of them onto the model-call
   span as ``http.response.header.<name>``. The allowlist is empty by default, and a
   fixed denylist of credential headers wins over it.
2. **Which decision caused which call.** :func:`escalation_from` links every model
   call made inside it to a decision span, as an OpenTelemetry span link (not a
   parent change). With ``GENAI_LINK_DECISIONS=true``, the next model call after a
   decision in the same context is linked automatically, marked as inferred.

Each link is also mirrored as two span attributes on the linked call,
``genai.decision.link.span_id`` and ``genai.decision.link.inferred``, because a
trace store may keep a link's ids but drop the link's own attributes.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import SpanContext

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Response headers
# --------------------------------------------------------------------------- #

HEADER_ATTRIBUTE_PREFIX = "http.response.header."
MAX_HEADERS = 32
MAX_HEADER_VALUE = 256

_DENY_EXACT = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
    }
)
# Substrings that mark a credential. Deliberately not "token" or "key": that would
# block x-ratelimit-remaining-tokens and LiteLLM's x-litellm-key-* spend headers.
_DENY_PARTS = ("auth", "secret", "password", "cookie")

_warned: set = set()


def denied_header(name: str) -> bool:
    """Whether a header may never be recorded, whatever the allowlist says."""
    lowered = name.strip().lower()
    return lowered in _DENY_EXACT or any(part in lowered for part in _DENY_PARTS)


def parse_header_allowlist(*values: Any) -> Tuple[str, ...]:
    """The allowlist from comma-separated strings or sequences, lowercased, deduplicated,
    with denied names removed (each logged once)."""
    names: List[str] = []
    for value in values:
        if not value:
            continue
        items = value.split(",") if isinstance(value, str) else list(value)
        for item in items:
            name = str(item).strip().lower()
            if not name or name in names:
                continue
            if denied_header(name):
                if name not in _warned:
                    _warned.add(name)
                    logger.warning(
                        "Response header %r is a credential header and is never recorded, "
                        "even when listed in GENAI_CAPTURE_RESPONSE_HEADERS",
                        name,
                    )
                continue
            names.append(name)
    return tuple(names)


def record_response_headers(span: Any, headers: Any, allowlist: Sequence[str]) -> None:
    """Copy the allowlisted ``headers`` onto ``span``. Never raises."""
    if not allowlist or headers is None:
        return
    try:
        recorded = 0
        for name in allowlist:
            if recorded >= MAX_HEADERS:
                break
            getter = getattr(headers, "get_list", None)
            values = getter(name) if callable(getter) else None
            if values is None:
                value = headers.get(name) if hasattr(headers, "get") else None
                values = [value] if value is not None else []
            values = [str(v)[:MAX_HEADER_VALUE] for v in values if v is not None]
            if values:
                span.set_attribute(HEADER_ATTRIBUTE_PREFIX + name, values)
                recorded += 1
    except Exception as e:  # noqa: BLE001 - telemetry must never break the call
        logger.debug("Could not record response headers: %s", e)


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #

#: Operations that are one call to a model. Agent, tool and pipeline spans are not.
MODEL_CALL_OPERATIONS = frozenset(
    {
        "chat",
        "text_completion",
        "generate",
        "generate_content",
        "responses",
        "embeddings",
        "speech_to_text",
        "text_to_speech",
    }
)
_OUR_SCOPE_PREFIX = "genai_otel"


def is_our_model_call(span: Any) -> bool:
    """A recording span of this library's that is one model call."""
    try:
        if not span.is_recording():
            return False
        scope = getattr(span, "instrumentation_scope", None)
        if not str(getattr(scope, "name", "") or "").startswith(_OUR_SCOPE_PREFIX):
            return False
        attributes = getattr(span, "attributes", None) or {}
        return attributes.get("gen_ai.operation.name") in MODEL_CALL_OPERATIONS
    except Exception:  # noqa: BLE001
        return False


def served_by_another_model(requested: Any, served: Any) -> bool:
    """Whether a router served a different model from the one requested.

    A snapshot of the requested model is not another model: ``gpt-4o`` served as
    ``gpt-4o-2024-08-06`` is the same model; ``auto`` served as ``gpt-5.5`` is not.
    """
    if not isinstance(requested, str) or not isinstance(served, str):
        return False
    req, srv = requested.strip().lower(), served.strip().lower()
    if not req or not srv or req == srv:
        return False
    # A provider prefix on the request ("openai/gpt-4o") does not make it another model.
    req_tail = req.rsplit("/", 1)[-1]
    return not (srv.startswith(req) or srv.startswith(req_tail))


# --------------------------------------------------------------------------- #
# Decision -> call links
# --------------------------------------------------------------------------- #

_EXPLICIT: contextvars.ContextVar[Tuple[SpanContext, ...]] = contextvars.ContextVar(
    "genai_otel_escalation_from", default=()
)


class _Pending:
    """The last decision made in a context, waiting for the call it causes."""

    __slots__ = ("span_context", "consumed")

    def __init__(self, span_context: SpanContext):
        self.span_context = span_context
        self.consumed = False


_LAST_DECISION: contextvars.ContextVar[Optional[_Pending]] = contextvars.ContextVar(
    "genai_otel_last_decision", default=None
)


def _span_context_of(handle: Any) -> Optional[SpanContext]:
    if isinstance(handle, SpanContext):
        return handle if handle.is_valid else None
    for attr in ("span_context",):
        value = getattr(handle, attr, None)
        if isinstance(value, SpanContext) and value.is_valid:
            return value
    get = getattr(handle, "get_span_context", None)
    if callable(get):
        value = get()
        if isinstance(value, SpanContext) and value.is_valid:
            return value
    return None


@contextlib.contextmanager
def escalation_from(*handles: Any) -> Iterator[None]:
    """Link every model call made inside the block to the decision(s) that caused it.

    ``handles`` are what ``decision_span`` returns (its recorder), a span, a
    ``SpanContext``, or :func:`last_decision` for a decision made through an SDK.
    The link is added to model-call spans only (chat, completions, embeddings,
    speech), never to agent, tool or HTTP spans, so an agent span and its chat child
    do not both count as the escalation. Parentage is unchanged.
    """
    contexts = tuple(c for c in (_span_context_of(h) for h in handles) if c is not None)
    token = _EXPLICIT.set(_EXPLICIT.get() + contexts)
    try:
        yield
    finally:
        _EXPLICIT.reset(token)


def last_decision() -> Optional[SpanContext]:
    """The span context of the most recent decision made in this context, if any."""
    pending = _LAST_DECISION.get()
    return pending.span_context if pending is not None else None


def note_decision(span_context: SpanContext) -> None:
    """Record that a decision just ended here (called by the decision span paths)."""
    if span_context is not None and span_context.is_valid:
        _LAST_DECISION.set(_Pending(span_context))


def _infer_enabled() -> bool:
    return os.getenv("GENAI_LINK_DECISIONS", "false").strip().lower() in ("1", "true", "yes")


class DecisionLinkSpanProcessor(SpanProcessor):
    """Adds decision links to model-call spans as they start, and notes decisions as they end."""

    def __init__(self, infer: Optional[bool] = None):
        self._infer = _infer_enabled() if infer is None else bool(infer)

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            if not is_our_model_call(span):
                return
            explicit = _EXPLICIT.get()
            links: List[Tuple[SpanContext, bool]] = [(c, False) for c in explicit]
            if not explicit and self._infer:
                pending = _LAST_DECISION.get()
                if pending is not None and not pending.consumed:
                    pending.consumed = True
                    links.append((pending.span_context, True))
            if not links:
                return
            for context, inferred in links:
                span.add_link(
                    context, {"genai.link.type": "decision", "genai.link.inferred": inferred}
                )
            first, inferred = links[0]
            span.set_attribute("genai.decision.link.span_id", f"{first.span_id:016x}")
            span.set_attribute("genai.decision.link.inferred", inferred)
            if len(links) > 1:
                span.set_attribute(
                    "genai.decision.link.span_ids", [f"{c.span_id:016x}" for c, _ in links]
                )
        except Exception as e:  # noqa: BLE001 - telemetry must never break the call
            logger.debug("Could not add decision links: %s", e)

    def on_end(self, span: Any) -> None:
        try:
            attributes = getattr(span, "attributes", None) or {}
            if attributes.get("genai.decision.protocol") is None:
                return
            note_decision(span.get_span_context())
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not note decision: %s", e)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
