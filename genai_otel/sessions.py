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

Across services the ids can travel as W3C baggage. Both directions are opt-in per
service, because baggage reaches every outbound call (third-party APIs included)
and inbound baggage is whatever the caller chose to send:

* ``GENAI_SESSION_BAGGAGE_KEYS``: the ids a service sends. ``session()`` puts them in
  the context's baggage and every instrumented outbound call carries them.
* ``GENAI_SESSION_BAGGAGE_ACCEPT``: the ids a service takes from inbound baggage.
  Set it only where callers are trusted, never on a service browsers reach. An
  accepted id ranks below every id the service resolves itself.

``SessionBaggagePropagator`` enforces both lists on the wire: our two keys are
dropped on inject unless sent and on extract unless accepted. Other baggage passes
through untouched.

A key a span already carries is never overwritten.
"""

import contextlib
import logging
import os
from typing import Iterable, Iterator, Optional, Set, Tuple

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.textmap import (
    CarrierT,
    Getter,
    Setter,
    TextMapPropagator,
    default_getter,
    default_setter,
)
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
BAGGAGE_ACCEPT_ENV = "GENAI_SESSION_BAGGAGE_ACCEPT"
_SUPPORTED_BAGGAGE_KEYS = (SESSION_ID, USER_ID)

# An accepted id is an attribute value chosen by another process. Anything longer
# or containing control characters is dropped rather than recorded.
_MAX_BAGGAGE_ID_LENGTH = 256


def parse_baggage_keys(raw: Optional[str]) -> Tuple[str, ...]:
    """A ``GENAI_SESSION_BAGGAGE_*`` value as keys. Unset or empty means none, and
    unknown names are ignored."""
    if not raw:
        return ()
    keys = {key.strip() for key in raw.split(",")}
    return tuple(key for key in _SUPPORTED_BAGGAGE_KEYS if key in keys)


_send_keys: Tuple[str, ...] = parse_baggage_keys(os.getenv(BAGGAGE_KEYS_ENV))
_accept_keys: Tuple[str, ...] = parse_baggage_keys(os.getenv(BAGGAGE_ACCEPT_ENV))


def _known(keys: Iterable[str]) -> Tuple[str, ...]:
    wanted: Set[str] = set(keys)
    return tuple(key for key in _SUPPORTED_BAGGAGE_KEYS if key in wanted)


def configure_baggage(send: Iterable[str] = (), accept: Iterable[str] = ()) -> None:
    """Set which of ``session.id`` / ``user.id`` a service sends and accepts as baggage."""
    global _send_keys, _accept_keys
    _send_keys = _known(send)
    _accept_keys = _known(accept)


def baggage_enabled() -> bool:
    """Whether any session id crosses a process boundary, in either direction."""
    return bool(_send_keys or _accept_keys)


def _valid_id(value) -> Optional[str]:
    if not isinstance(value, str) or not value or len(value) > _MAX_BAGGAGE_ID_LENGTH:
        return None
    if not value.isprintable():
        return None
    return value


def _without(ctx: otel_context.Context, keys: Iterable[str]) -> otel_context.Context:
    for key in keys:
        if baggage.get_baggage(key, ctx) is not None:
            ctx = baggage.remove_baggage(key, ctx)
    return ctx


class SessionBaggagePropagator(TextMapPropagator):
    """W3C baggage, with ``session.id`` / ``user.id`` limited to the configured keys.

    On inject, our keys not in ``GENAI_SESSION_BAGGAGE_KEYS`` are left out, so an id
    a caller sent us is not forwarded on to every service and API we call. On
    extract, our keys not in ``GENAI_SESSION_BAGGAGE_ACCEPT``, or not a valid id, are
    dropped. All other baggage is propagated as the stock propagator would.
    """

    def __init__(self) -> None:
        self._w3c = W3CBaggagePropagator()

    def extract(
        self,
        carrier: CarrierT,
        context: Optional[otel_context.Context] = None,
        getter: Getter[CarrierT] = default_getter,
    ) -> otel_context.Context:
        ctx = self._w3c.extract(carrier, context=context, getter=getter)
        drop = [key for key in _SUPPORTED_BAGGAGE_KEYS if key not in _accept_keys]
        drop += [
            key
            for key in _accept_keys
            if baggage.get_baggage(key, ctx) is not None
            and _valid_id(baggage.get_baggage(key, ctx)) is None
        ]
        return _without(ctx, drop)

    def inject(
        self,
        carrier: CarrierT,
        context: Optional[otel_context.Context] = None,
        setter: Setter[CarrierT] = default_setter,
    ) -> None:
        ctx = context if context is not None else otel_context.get_current()
        ctx = _without(ctx, [key for key in _SUPPORTED_BAGGAGE_KEYS if key not in _send_keys])
        self._w3c.inject(carrier, context=ctx, setter=setter)

    @property
    def fields(self) -> Set[str]:
        return self._w3c.fields


@contextlib.contextmanager
def session(session_id: str, user_id: Optional[str] = None) -> Iterator[None]:
    """Attach ``session_id`` (and ``user_id``) to every span started in the block.

    Use one id per conversation, the same for each of its turns::

        with genai_otel.session(conversation_id, user_id=user.id):
            graph.invoke(state)

    Works in sync and async code; the ids follow the OpenTelemetry context. The ids
    named in ``GENAI_SESSION_BAGGAGE_KEYS`` also go out as baggage, so services this
    block calls can tag their spans with the same session.
    """
    ctx = otel_context.set_value(_SESSION_KEY, str(session_id))
    # An id this block does not set must not ride along from an outer block or a
    # caller, so our keys are replaced or removed, never left as they were.
    ctx = _without(ctx, _SUPPORTED_BAGGAGE_KEYS)
    if SESSION_ID in _send_keys:
        ctx = baggage.set_baggage(SESSION_ID, str(session_id), ctx)
    if user_id is not None:
        ctx = otel_context.set_value(_USER_KEY, str(user_id), ctx)
        if USER_ID in _send_keys:
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
    return str(value) if value else None


def current_session_id(ctx: Optional[otel_context.Context] = None) -> Optional[str]:
    """The session declared by an enclosing ``session()`` block, if any."""
    return _from_context(_SESSION_KEY, _OPENINFERENCE_SESSION_KEY, ctx)


def current_user_id(ctx: Optional[otel_context.Context] = None) -> Optional[str]:
    """The user declared by an enclosing ``session()`` block, if any."""
    return _from_context(_USER_KEY, _OPENINFERENCE_USER_KEY, ctx)


def span_session_id(ctx: Optional[otel_context.Context] = None) -> Optional[str]:
    """The ``session.id`` already on the current span, if any.

    An instrumentor resolving a session for a new span checks this before it
    generates one, so a call made inside a session keeps it.
    """
    session_id, _ = _parent_ids(ctx)
    return session_id


def _inbound(key: str, ctx: Optional[otel_context.Context]) -> Optional[str]:
    if key not in _accept_keys:
        return None
    return _valid_id(baggage.get_baggage(key, ctx))


def inbound_session_id(ctx: Optional[otel_context.Context] = None) -> Optional[str]:
    """A session an upstream service sent as baggage, when this service accepts it.

    The lowest-ranked source: use it only when the service has no session of its own.
    """
    return _inbound(SESSION_ID, ctx)


def inbound_user_id(ctx: Optional[otel_context.Context] = None) -> Optional[str]:
    """A user an upstream service sent as baggage, when this service accepts it."""
    return _inbound(USER_ID, ctx)


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


def _session_propagator_installed() -> bool:
    from opentelemetry import propagate

    textmap = propagate.get_global_textmap()
    propagators = getattr(textmap, "_propagators", None) or [textmap]
    return any(isinstance(p, SessionBaggagePropagator) for p in propagators)


def warn_if_propagator_replaced() -> bool:
    """Log once when session baggage is configured but our propagator is not global.

    Code that calls ``set_global_textmap`` after ``instrument()`` replaces it. With
    stock W3C baggage in its place, any inbound ``session.id`` / ``user.id`` is
    forwarded on unfiltered; with trace context alone, no session crosses at all.
    Returns whether it warned.
    """
    if not baggage_enabled() or os.getenv("OTEL_PROPAGATORS"):
        return False
    try:
        if _session_propagator_installed():
            return False
    except Exception:  # noqa: BLE001 - a diagnostic must never break a span
        return False
    logger.warning(
        "GENAI_SESSION_BAGGAGE_KEYS/_ACCEPT is set, but the global propagator was "
        "replaced after genai_otel.instrument(). session.id / user.id baggage is not "
        "filtered and may not cross at all. Remove the later set_global_textmap() call, "
        "or include genai_otel.sessions.SessionBaggagePropagator in it."
    )
    return True


class SessionSpanProcessor(SpanProcessor):
    """Give every span the session and user of its ``session()`` block or parent."""

    def __init__(self) -> None:
        # Checked at the first span rather than at setup: an application that
        # overrides the propagator does so right after instrument() returns.
        self._propagator_checked = False

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        if not self._propagator_checked:
            self._propagator_checked = True
            warn_if_propagator_replaced()
        try:
            attributes = span.attributes or {}
            has_session = SESSION_ID in attributes
            has_user = USER_ID in attributes
            if has_session and has_user and CONVERSATION_ID in attributes:
                return

            # A session() block here, then the parent span, then an id an upstream
            # service sent, which ranks last.
            session_id = current_session_id(parent_context)
            user_id = current_user_id(parent_context)
            if session_id is None or user_id is None:
                parent_session, parent_user = _parent_ids(parent_context)
                session_id = session_id or parent_session
                user_id = user_id or parent_user
            session_id = session_id or inbound_session_id(parent_context)
            user_id = user_id or inbound_user_id(parent_context)

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
