"""The voice turn shape and its arithmetic: who spoke when, and what that adds up to.

A speaker turn is the AUDIBLE interval of one party's speech: from the first audio
to the last. Everything else follows from the intervals:

* ``voice.turn.overlap_ms``: time this turn overlapped the other party's speech;
* ``voice.turn.interrupted``: this turn was cut off by the other party. On a user
  turn, the agent interrupted; on an agent turn, the user barged in. Reported by the
  framework when it knows (``source=framework``), else derived: the other party
  started during the turn and the turn ended while they were still speaking
  (``source=derived``). Overlap that does not end the turn (a backchannel "mm-hm")
  is overlap, not an interruption;
* ``voice.turn.response_latency_ms`` on an agent turn: its start minus the end of
  the user turn it answers (the last user turn that started before it). Negative
  when the agent started before the user had finished;
* conversation totals: talk time per party, the agent's share, silence, dead air
  (gaps longer than a threshold), turn counts, interruptions and barge-ins.

All times are nanoseconds since the epoch, as OpenTelemetry spans use. Pure
functions: no spans, no clocks, so the arithmetic is tested on its own.
"""

from __future__ import annotations

import dataclasses
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

USER = "user"
AGENT = "agent"
SPEAKERS = (USER, AGENT)

DEFAULT_DEAD_AIR_MS = 2000

_NS_PER_MS = 1_000_000


@dataclasses.dataclass
class Turn:
    """One party's audible speech, from first to last audio."""

    speaker: str
    start_ns: int
    end_ns: int
    #: What the framework said, when it knows: True / False. None = derive it.
    interrupted: Optional[bool] = None
    #: The user turn an agent turn answers, by index into the conversation's turns.
    answers: Optional[int] = None
    #: Framework-measured delays to carry through (``end_of_turn_delay_ms`` ...).
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return max(self.end_ns - self.start_ns, 0)


def _ms(ns: float) -> float:
    return round(ns / _NS_PER_MS, 3)


def _overlap(a: Turn, b: Turn) -> int:
    return max(0, min(a.end_ns, b.end_ns) - max(a.start_ns, b.start_ns))


def _union(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _cut_by(turn: Turn, other: Turn) -> bool:
    """``other`` started during ``turn`` and ``turn`` ended while ``other`` still spoke."""
    return turn.start_ns < other.start_ns < turn.end_ns and turn.end_ns < other.end_ns


def turn_attributes(turns: Sequence[Turn]) -> List[Dict[str, Any]]:
    """The ``voice.turn.*`` attributes of each turn, in the order given."""
    order = sorted(range(len(turns)), key=lambda i: (turns[i].start_ns, turns[i].end_ns))
    index_of = {i: n for n, i in enumerate(order)}
    out: List[Dict[str, Any]] = [{} for _ in turns]
    for i, turn in enumerate(turns):
        others = [t for t in turns if t.speaker != turn.speaker]
        attrs: Dict[str, Any] = {
            "voice.turn.speaker": turn.speaker,
            "voice.turn.index": index_of[i],
            "voice.turn.duration_ms": _ms(turn.duration_ns),
        }
        overlap = sum(_overlap(turn, o) for o in others)
        attrs["voice.turn.overlap_ms"] = _ms(overlap)
        if turn.interrupted is not None:
            attrs["voice.turn.interrupted"] = bool(turn.interrupted)
            attrs["voice.turn.interrupted.source"] = "framework"
        else:
            attrs["voice.turn.interrupted"] = any(_cut_by(turn, o) for o in others)
            attrs["voice.turn.interrupted.source"] = "derived"
        if turn.speaker == AGENT:
            answered = _answered(turns, i)
            if answered is not None:
                attrs["voice.turn.answers_turn"] = index_of[answered]
                attrs["voice.turn.response_latency_ms"] = _ms(
                    turn.start_ns - turns[answered].end_ns
                )
        for key, value in turn.extra.items():
            attrs[f"voice.turn.{key}"] = value
        out[i] = attrs
    return out


def _answered(turns: Sequence[Turn], agent_index: int) -> Optional[int]:
    turn = turns[agent_index]
    if turn.answers is not None and 0 <= turn.answers < len(turns):
        return turn.answers
    candidates = [
        (t.start_ns, i)
        for i, t in enumerate(turns)
        if t.speaker == USER and t.start_ns < turn.start_ns
    ]
    return max(candidates)[1] if candidates else None


def conversation_attributes(
    turns: Sequence[Turn],
    dead_air_ms: int = DEFAULT_DEAD_AIR_MS,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> Dict[str, Any]:
    """The conversation's ``voice.*`` totals.

    Silence is counted between the first and the last speech; with ``start_ns`` /
    ``end_ns`` the conversation's own bounds are used for its duration only, so a
    long ring before anyone speaks is not dead air.
    """
    attrs: Dict[str, Any] = {
        "voice.turn_count.user": sum(1 for t in turns if t.speaker == USER),
        "voice.turn_count.agent": sum(1 for t in turns if t.speaker == AGENT),
    }
    if not turns:
        return attrs
    per_turn = turn_attributes(turns)
    talk = {
        s: sum(e - b for b, e in _union([(t.start_ns, t.end_ns) for t in turns if t.speaker == s]))
        for s in SPEAKERS
    }
    attrs["voice.user_talk_ms"] = _ms(talk[USER])
    attrs["voice.agent_talk_ms"] = _ms(talk[AGENT])
    both = talk[USER] + talk[AGENT]
    if both:
        attrs["voice.agent_talk_share"] = round(talk[AGENT] / both, 4)
    speech = _union([(t.start_ns, t.end_ns) for t in turns])
    gaps = [b2 - e1 for (_, e1), (b2, _) in zip(speech, speech[1:]) if b2 > e1]
    threshold = dead_air_ms * _NS_PER_MS
    dead = [g for g in gaps if g > threshold]
    attrs["voice.silence_ms"] = _ms(sum(gaps))
    attrs["voice.dead_air_ms"] = _ms(sum(dead))
    attrs["voice.dead_air_count"] = len(dead)
    attrs["voice.dead_air_max_ms"] = _ms(max(dead)) if dead else 0.0
    attrs["voice.dead_air_threshold_ms"] = dead_air_ms
    attrs["voice.interruption_count"] = sum(
        1 for t, a in zip(turns, per_turn) if t.speaker == USER and a["voice.turn.interrupted"]
    )
    attrs["voice.barge_in_count"] = sum(
        1 for t, a in zip(turns, per_turn) if t.speaker == AGENT and a["voice.turn.interrupted"]
    )
    first = start_ns if start_ns is not None else speech[0][0]
    last = end_ns if end_ns is not None else speech[-1][1]
    attrs["voice.conversation.duration_ms"] = _ms(max(last - first, 0))
    latencies = [
        a["voice.turn.response_latency_ms"]
        for a in per_turn
        if "voice.turn.response_latency_ms" in a
    ]
    if latencies:
        attrs["voice.response_latency_ms.max"] = max(latencies)
        attrs["voice.response_latency_ms.median"] = round(statistics.median(latencies), 3)
    return attrs
