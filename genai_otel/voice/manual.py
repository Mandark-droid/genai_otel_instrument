"""``voice_conversation()``: voice turn telemetry for an application's own audio loop.

For voice agents built without Pipecat or LiveKit Agents, the same ``voice.*`` shape
those integrations emit::

    with genai_otel.voice_conversation(conversation_id=call_id) as conv:
        with conv.turn("user"):            # from the first to the last user audio
            ...
        with conv.turn("agent"):           # from the first to the last agent audio
            ...
        # or, when the audio loop knows the times only afterwards:
        conv.add_turn("user", start=t0, end=t1)            # seconds since the epoch
        conv.add_turn("agent", start=t2, end=t3, interrupted=True)

Turns are recorded as ``voice.turn`` spans, children of the ``voice.conversation``
span, when the conversation closes: only then is everything about each turn known
(a later turn may overlap an earlier one). Timing only: no audio and no text.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any, Iterator, List, Optional

from opentelemetry import trace

from .shape import (
    AGENT,
    DEFAULT_DEAD_AIR_MS,
    SPEAKERS,
    Turn,
    conversation_attributes,
    turn_attributes,
)

logger = logging.getLogger(__name__)

_TRACER_NAME = "genai_otel.voice"


def _ns(value: Any) -> Optional[int]:
    """Seconds (float) or nanoseconds (int > 1e15) since the epoch, as nanoseconds."""
    if value is None:
        return None
    if isinstance(value, int) and value > 10**15:
        return value
    return int(float(value) * 1_000_000_000)


def dead_air_threshold_ms() -> int:
    try:
        return int(os.getenv("GENAI_VOICE_DEAD_AIR_MS", str(DEFAULT_DEAD_AIR_MS)))
    except ValueError:
        return DEFAULT_DEAD_AIR_MS


class VoiceConversation:
    """An open conversation: collects turns, emits them and the totals on close."""

    def __init__(self, conversation_id: Optional[str], framework: str, tracer: Any, attributes):
        self.turns: List[Turn] = []
        self._tracer = tracer
        self.span = tracer.start_span("voice.conversation", attributes=dict(attributes or {}))
        if conversation_id:
            self.span.set_attribute("voice.conversation.id", str(conversation_id))
        self.span.set_attribute("voice.framework", framework)
        self._start_ns = time.time_ns()
        self._closed = False

    @property
    def last_user_turn(self) -> Optional[int]:
        users = [i for i, t in enumerate(self.turns) if t.speaker != AGENT]
        return users[-1] if users else None

    def add_turn(
        self,
        speaker: str,
        start: Any,
        end: Any,
        interrupted: Optional[bool] = None,
        answers: Optional[int] = None,
    ) -> int:
        """Record a turn whose times are known. Returns its index (for ``answers=``)."""
        if speaker not in SPEAKERS:
            raise ValueError(f"speaker must be one of {SPEAKERS}, not {speaker!r}")
        start_ns, end_ns = _ns(start), _ns(end)
        if start_ns is None or end_ns is None or end_ns < start_ns:
            raise ValueError("a turn needs a start and an end, and must not end before it starts")
        self.turns.append(Turn(speaker, start_ns, end_ns, interrupted, answers))
        return len(self.turns) - 1

    @contextlib.contextmanager
    def turn(
        self, speaker: str, interrupted: Optional[bool] = None, answers: Optional[int] = None
    ) -> Iterator[None]:
        """A turn timed by the block: enter at the first audio, leave at the last."""
        start = time.time_ns()
        try:
            yield
        finally:
            self.add_turn(speaker, start, time.time_ns(), interrupted, answers)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        end_ns = time.time_ns()
        try:
            parent = trace.set_span_in_context(self.span)
            for turn, attrs in zip(self.turns, turn_attributes(self.turns)):
                span = self._tracer.start_span(
                    "voice.turn", context=parent, start_time=turn.start_ns, attributes=attrs
                )
                span.end(end_time=turn.end_ns)
            for key, value in conversation_attributes(
                self.turns, dead_air_threshold_ms(), self._start_ns, end_ns
            ).items():
                self.span.set_attribute(key, value)
        except Exception as e:  # noqa: BLE001 - telemetry must never break the call
            logger.debug("Could not record voice turns: %s", e)
        finally:
            self.span.end(end_time=end_ns)


@contextlib.contextmanager
def voice_conversation(
    conversation_id: Optional[str] = None, attributes: Optional[dict] = None
) -> Iterator[VoiceConversation]:
    """Open a ``voice.conversation`` span; turns added inside are emitted when it closes."""
    conv = VoiceConversation(conversation_id, "manual", trace.get_tracer(_TRACER_NAME), attributes)
    try:
        yield conv
    finally:
        conv.close()
