"""Voice frameworks' own spans, made to follow this library's rules and shape (issue #48).

Pipecat and LiveKit Agents trace themselves through the global tracer provider, which
is this library's once ``instrument()`` has run. This processor:

* applies the content rule: with content capture off, Pipecat's content keys
  (messages, transcripts, spoken text, system instructions, tool definitions) are
  removed from its spans before they are exported. LiveKit's are withheld by LiveKit
  itself once ``LIVEKIT_TELEMETRY_ALLOW_PII=false`` (the LiveKit instrumentor sets that
  when the application has not);
* normalises LiveKit's speaking spans (``user_speaking`` / ``agent_speaking``) into
  the ``voice.turn.*`` shape, and puts the conversation totals on ``agent_session``.

It must be registered ahead of the export processors, because it edits spans in
``on_end`` the way the LiteLLM enrichment processor does.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from opentelemetry.sdk.trace import SpanProcessor

from .manual import dead_air_threshold_ms
from .shape import AGENT, USER, Turn, conversation_attributes, turn_attributes

logger = logging.getLogger(__name__)

#: Pipecat attributes that carry what was said or written. Removed with capture off.
PIPECAT_CONTENT_KEYS = frozenset(
    {
        "input",
        "output",
        "transcript",
        "text",
        "gen_ai.system_instructions",
        "tools.definitions",
        "tools",
    }
)
_PIPECAT_SCOPE = "pipecat"
_LIVEKIT_SCOPE = "livekit"
_LIVEKIT_SPEAKERS = {"user_speaking": USER, "agent_speaking": AGENT}
_LIVEKIT_SESSION = "agent_session"
_FAR_FUTURE = 2**63 - 1


def _scope(span: Any) -> str:
    scope = getattr(span, "instrumentation_scope", None)
    return str(getattr(scope, "name", "") or "")


def _set(span: Any, key: str, value: Any) -> None:
    attributes = getattr(span, "_attributes", None)
    if attributes is None:
        return
    try:
        attributes[key] = value
        return
    except (TypeError, AttributeError):
        pass  # an ended span's attributes are read-only through the public API
    try:
        inner = getattr(attributes, "_dict", None)
        if inner is not None:
            inner[key] = value
    except Exception as e:  # noqa: BLE001
        logger.debug("Cannot set %s on %s: %s", key, getattr(span, "name", "?"), e)


def _delete(span: Any, key: str) -> None:
    attributes = getattr(span, "_attributes", None)
    if attributes is None:
        return
    try:
        # BoundedAttributes keeps its items in ``_dict``; an ended span's mapping is
        # read-only through its public API, so the item is removed there.
        inner = getattr(attributes, "_dict", None)
        if inner is not None and key in inner:
            del inner[key]
    except Exception as e:  # noqa: BLE001
        logger.debug("Cannot remove %s from %s: %s", key, getattr(span, "name", "?"), e)


class _Session:
    """LiveKit speaking intervals of one trace (one agent session)."""

    def __init__(self) -> None:
        self.open: Dict[int, Turn] = {}
        self.ended: List[Turn] = []


class VoiceSpanProcessor(SpanProcessor):
    """Content rule for Pipecat spans; voice turn shape for LiveKit spans."""

    def __init__(self, capture_content: bool, livekit: bool = True, pipecat: bool = True):
        self._capture = bool(capture_content)
        self._livekit = livekit
        self._pipecat = pipecat
        self._sessions: Dict[int, _Session] = {}
        self._lock = threading.Lock()

    # -- LiveKit -----------------------------------------------------------

    def _session(self, trace_id: int) -> _Session:
        with self._lock:
            session = self._sessions.get(trace_id)
            if session is None:
                session = self._sessions[trace_id] = _Session()
            return session

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            if not self._livekit or not _scope(span).startswith(_LIVEKIT_SCOPE):
                return
            speaker = _LIVEKIT_SPEAKERS.get(getattr(span, "name", ""))
            if speaker is None:
                return
            ctx = span.get_span_context()
            start = getattr(span, "start_time", None) or 0
            self._session(ctx.trace_id).open[ctx.span_id] = Turn(speaker, start, _FAR_FUTURE)
        except Exception as e:  # noqa: BLE001 - telemetry must never break the call
            logger.debug("voice processor on_start failed: %s", e)

    def on_end(self, span: Any) -> None:
        try:
            scope = _scope(span)
            if self._pipecat and scope.startswith(_PIPECAT_SCOPE) and not self._capture:
                for key in PIPECAT_CONTENT_KEYS & set((span.attributes or {}).keys()):
                    _delete(span, key)
            if self._livekit and scope.startswith(_LIVEKIT_SCOPE):
                self._livekit_end(span)
        except Exception as e:  # noqa: BLE001
            logger.debug("voice processor on_end failed: %s", e)

    def _livekit_end(self, span: Any) -> None:
        ctx = span.get_span_context()
        name = getattr(span, "name", "")
        if name in _LIVEKIT_SPEAKERS:
            session = self._session(ctx.trace_id)
            turn = session.open.pop(ctx.span_id, None)
            if turn is None:
                turn = Turn(_LIVEKIT_SPEAKERS[name], span.start_time or 0, span.end_time or 0)
            turn.start_ns = span.start_time or turn.start_ns
            turn.end_ns = span.end_time or turn.start_ns
            session.ended.append(turn)
            # Everything known so far: ended turns, and open ones as still speaking.
            known = session.ended + list(session.open.values())
            attrs = turn_attributes(known)[len(session.ended) - 1]
            for key, value in attrs.items():
                _set(span, key, value)
        elif name == _LIVEKIT_SESSION:
            with self._lock:
                session = self._sessions.pop(ctx.trace_id, None)
            turns = session.ended if session else []
            _set(span, "voice.framework", "livekit")
            totals = conversation_attributes(
                turns, dead_air_threshold_ms(), span.start_time, span.end_time
            )
            for key, value in totals.items():
                _set(span, key, value)

    def shutdown(self) -> None:
        with self._lock:
            self._sessions.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


_INSTALLED: Optional[VoiceSpanProcessor] = None


def install(
    tracer_provider: Any,
    capture_content: bool,
    pipecat: Optional[bool] = None,
    livekit: Optional[bool] = None,
) -> Optional[VoiceSpanProcessor]:
    """The one processor for ``tracer_provider``, ahead of its export processors.

    Each framework instrumentor turns on its own part; the other parts are untouched.
    """
    global _INSTALLED
    if _INSTALLED is None:
        add = getattr(tracer_provider, "add_span_processor", None)
        if not callable(add):
            return None
        processor = VoiceSpanProcessor(capture_content, livekit=False, pipecat=False)
        # Ahead of the exporters: the multi-processor runs processors in order, and
        # the export processors were added when instrument() set the provider up.
        active = getattr(tracer_provider, "_active_span_processor", None)
        processors = getattr(active, "_span_processors", None)
        prepended = False
        if isinstance(processors, tuple):
            try:
                with active._lock:  # noqa: SLF001
                    active._span_processors = (processor,) + processors  # noqa: SLF001
                prepended = True
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not prepend the voice processor: %s", e)
        if not prepended:
            add(processor)
        _INSTALLED = processor
    if pipecat is not None:
        _INSTALLED._pipecat = bool(pipecat)  # noqa: SLF001
    if livekit is not None:
        _INSTALLED._livekit = bool(livekit)  # noqa: SLF001
    return _INSTALLED
