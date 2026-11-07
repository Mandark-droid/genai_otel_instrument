"""Pipecat: who spoke when, from the speaking frames a pipeline already carries.

Pipecat's own tracing records one ``turn`` span per exchange (user and bot together).
This observer records the parties separately, as ``voice.turn`` spans under a
``voice.conversation`` span, from ``UserStartedSpeakingFrame`` /
``UserStoppedSpeakingFrame`` and ``BotStartedSpeakingFrame`` /
``BotStoppedSpeakingFrame``. It reads only those frames and the pipeline's start and
end frames, and returns at once for every other frame.

A frame is observed every time a processor passes it on, and Pipecat pushes some
speaking frames both upstream and downstream, so a start counts only when that party
was not already speaking, and a stop only when it was.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .manual import VoiceConversation
from .shape import AGENT, USER

logger = logging.getLogger(__name__)


def make_observer(tracer: Any, conversation_id: Optional[str] = None) -> Any:
    """A Pipecat ``BaseObserver`` emitting the voice turn shape. Imports Pipecat lazily."""
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        CancelFrame,
        EndFrame,
        StartFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.observers.base_observer import BaseObserver

    starts = {UserStartedSpeakingFrame: USER, BotStartedSpeakingFrame: AGENT}
    stops = {UserStoppedSpeakingFrame: USER, BotStoppedSpeakingFrame: AGENT}
    watched = tuple(starts) + tuple(stops) + (StartFrame, EndFrame, CancelFrame)

    class VoiceTurnObserver(BaseObserver):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self._conv: Optional[VoiceConversation] = None
            self._speaking: dict = {}

        def _conversation(self) -> VoiceConversation:
            if self._conv is None:
                self._conv = VoiceConversation(conversation_id, "pipecat", tracer, None)
            return self._conv

        async def on_push_frame(self, data: Any) -> None:
            frame = getattr(data, "frame", None)
            if not isinstance(frame, watched):
                return
            try:
                now = time.time_ns()
                kind = type(frame)
                if isinstance(frame, StartFrame):
                    self._conversation()
                elif kind in starts:
                    speaker = starts[kind]
                    if speaker not in self._speaking:
                        self._conversation()
                        self._speaking[speaker] = now
                elif kind in stops:
                    speaker = stops[kind]
                    started = self._speaking.pop(speaker, None)
                    if started is not None:
                        self._conversation().add_turn(speaker, started, max(now, started))
                elif self._conv is not None:  # EndFrame / CancelFrame
                    for speaker, started in list(self._speaking.items()):
                        self._conv.add_turn(speaker, started, max(now, started))
                    self._speaking.clear()
                    self._conv.close()
                    self._conv = None
            except Exception as e:  # noqa: BLE001 - telemetry must never break the pipeline
                logger.debug("voice turn observer failed: %s", e)

    return VoiceTurnObserver()
