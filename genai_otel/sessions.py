"""Session tracking: one ``session.id`` across every span of a conversation.

Backends group traces into sessions by the ``session.id`` span attribute (Arize AX
reads exactly that key). Instrumentors resolve it per span from their own call, so
it used to stop at the framework span: the LangGraph run had one, its nodes and the
LLM calls inside them did not.

Two pieces close that gap:

* ``session()`` declares the session for a block of code. Every span started inside
  it, auto-instrumented LLM spans included, gets ``session.id``,
  ``gen_ai.conversation.id`` and, when given, ``user.id``.
* ``SessionSpanProcessor`` fills those keys on each span that lacks them, from the
  ``session()`` block it runs in or else from its parent span. A session an
  instrumentor resolves for a framework span (a LangGraph ``thread_id``, a CrewAI
  crew id) therefore reaches every span beneath it with no application code.

Across services the ids travel as W3C baggage. ``session()`` puts them in the
context's baggage, every instrumented outbound call (HTTP, MCP) carries it, and the
receiving service's spans read it back. Only the keys in ``baggage_keys`` cross a
process boundary in either direction: ``session.id`` by default, since baggage goes
to every outbound call, third-party APIs included. ``user.id`` is opt-in.

A key a span already carries is never overwritten.
"""

import contextlib
import logging
import os
from typing import Iterable, Iterator, Optional, Tuple

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import Span, SpanProcessor

logger = logging.getLogger(__name__)

SESSION_ID = "session.id"
USER_ID = "user.id"
CONVERSATION_ID = "gen_ai.conversation.id"

_SESSION_KEY = otel_context.create_key("genai_otel.session.id")
_USER_KEY = otel_context.create_key("genai_otel.user.id")

# OpenInference's using_session() / using_user() store the ids under these plain
# context keys. Reading them keeps those blocks working for our spans without
# importing openinference.
_OPENINFERENCE_SESSION_KEY = "session.id"
_OPENINFERENCE_USER_KEY = "user.id"

BAGGAGE_KEYS_ENV = "GENAI_SESSION_BAGGAGE_KEYS"
_DEFAULT_BAGGAGE_KEYS = (SESSION_ID,)
_SUPPORTED_BAGGAGE_KEYS = (SESSION_ID, USER_ID)


def parse_baggage_keys(raw: Optional[str]) -> Tuple[str, ...]:
    """The ``GENAI_SESSION_BAGGAGE_KEYS`` value as keys: unset means the default,
    empty means none, and unknown names are ignored."""
    if raw is None:
        return _DEFAULT_BAGGAGE_KEYS
    keys = [key.strip() for key in raw.split(",")]
    return tuple(key for key in _SUPPORTED_BAGGAGE_KEYS if key in keys)


_baggage_keys: Tuple[str, ...] = parse_baggage_keys(os.getenv(BAGGAGE_KEYS_ENV))


def configure_baggage(keys: Iterable[str]) -> None:
    """Set which of ``session.id`` / ``user.id`` cross service boundaries."""
    global _baggage_keys
    _baggage_keys = tuple(key for key in _SUPPORTED_BAGGAGE_KEYS if key in set(keys))


@contextlib.contextmanager
def session(session_id: str, user_id: Optional[str] = None) -> Iterator[None]:
    """Attach ``session_id`` (and ``user_id``) to every span started in the block.

    Use one id per conversation, the same for each of its turns::

        with genai_otel.session(conversation_id, user_id=user.id):
            graph.invoke(state)

    Works in sync and async code; the ids follow the OpenTelemetry context. The ids
    named in ``GENAI_SESSION_BAGGAGE_KEYS`` also go out as baggage, so services this
    block calls tag their spans with the same session.
    """
    ctx = otel_context.set_value(_SESSION_KEY, str(session_id))
    if SESSION_ID in _baggage_keys:
        ctx = baggage.set_baggage(SESSION_ID, str(session_id), ctx)
    if user_id is not None:
        ctx = otel_context.set_value(_USER_KEY, str(user_id), ctx)
        if USER_ID in _baggage_keys:
            ctx = baggage.set_baggage(USER_ID, str(user_id), ctx)
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


def _from_context(key, plain_key: str, ctx: Optional[otel_context.Context] = None) -> Optional[str]:
    value = otel_context.get_value(key, ctx)
    if not value:
        value = otel_context.get_value(plain_key, ctx)
    if not value and plain_key in _baggage_keys:
        # An upstream service's session() block, carried in by the request.
        value = baggage.get_baggage(plain_key, ctx)
    return str(value) if value else None


def current_session_id(ctx: Optional[otel_context.Context] = None) -> Optional[str]:
    """The session of an enclosing ``session()`` block, here or upstream, if any."""
    return _from_context(_SESSION_KEY, _OPENINFERENCE_SESSION_KEY, ctx)


def current_user_id(ctx: Optional[otel_context.Context] = None) -> Optional[str]:
    """The user of an enclosing ``session()`` block, here or upstream, if any."""
    return _from_context(_USER_KEY, _OPENINFERENCE_USER_KEY, ctx)


def _parent_ids(parent_context) -> Tuple[Optional[str], Optional[str]]:
    """Session and user ids on the live parent span, read at the child's start."""
    parent = trace.get_current_span(parent_context)
    attributes = getattr(parent, "attributes", None)
    if not attributes:
        return None, None
    session_id = attributes.get(SESSION_ID)
    user_id = attributes.get(USER_ID)
    return (
        str(session_id) if session_id else None,
        str(user_id) if user_id else None,
    )


class SessionSpanProcessor(SpanProcessor):
    """Give every span the session and user of its ``session()`` block or parent."""

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        try:
            attributes = span.attributes or {}
            has_session = SESSION_ID in attributes
            has_user = USER_ID in attributes
            if has_session and has_user and CONVERSATION_ID in attributes:
                return

            session_id = current_session_id(parent_context)
            user_id = current_user_id(parent_context)
            if session_id is None or user_id is None:
                parent_session, parent_user = _parent_ids(parent_context)
                session_id = session_id or parent_session
                user_id = user_id or parent_user

            if not has_session and session_id:
                span.set_attribute(SESSION_ID, session_id)
            session_id = attributes.get(SESSION_ID) or session_id
            if session_id and CONVERSATION_ID not in attributes:
                span.set_attribute(CONVERSATION_ID, str(session_id))
            if not has_user and user_id:
                span.set_attribute(USER_ID, user_id)
        except Exception as e:  # noqa: BLE001 - session ids must never break a span
            logger.debug("Failed to add session attributes: %s", e)

    def on_end(self, span) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
