"""Voice turn shape (issue #48): the interval arithmetic and the manual helper.

Each conversation below has planted timings (an agent interruption, a barge-in and a
long silence), and the emitted turns must reproduce them.
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.voice import manual
from genai_otel.voice.shape import AGENT, USER, Turn, conversation_attributes, turn_attributes

S = 1_000_000_000  # ns per second
T0 = 1_700_000_000 * S


def _t(speaker, start_s, end_s, **kw):
    return Turn(speaker, T0 + int(start_s * S), T0 + int(end_s * S), **kw)


# A planted call:
#   0.0- 2.0  user
#   2.6- 5.0  agent answers (600 ms); the user barges in at 4.2 and the agent stops at 5.0
#   4.2- 6.0  user (the barge-in)
#   6.8- 9.0  agent answers (800 ms); the user's "mm-hm" at 7.5-7.8 is overlap, not a cut
#   7.5- 7.8  user backchannel
#   9.5-11.5  user
#  11.0-12.0  agent starts at 11.0 while the user is still speaking: it cuts the user off
#  16.5-17.0  user, after 4.5 s of silence (dead air)
CALL = [
    _t(USER, 0.0, 2.0),
    _t(AGENT, 2.6, 5.0),
    _t(USER, 4.2, 6.0),
    _t(AGENT, 6.8, 9.0),
    _t(USER, 7.5, 7.8),
    _t(USER, 9.5, 11.5),
    _t(AGENT, 11.0, 12.0),
    _t(USER, 16.5, 17.0),
]


def test_turn_attributes_reproduce_the_planted_timings():
    a = turn_attributes(CALL)
    assert a[1]["voice.turn.answers_turn"] == 0
    assert a[1]["voice.turn.response_latency_ms"] == pytest.approx(600)
    # The user barged in on the agent's first answer.
    assert a[1]["voice.turn.interrupted"] is True
    assert a[1]["voice.turn.interrupted.source"] == "derived"
    assert a[1]["voice.turn.overlap_ms"] == pytest.approx(800)
    # The backchannel overlapped the agent but did not stop it.
    assert a[3]["voice.turn.response_latency_ms"] == pytest.approx(800)
    assert a[3]["voice.turn.overlap_ms"] == pytest.approx(300)
    assert a[3]["voice.turn.interrupted"] is False
    assert a[4]["voice.turn.interrupted"] is False
    # The agent cut the user off, and started before the user had finished.
    assert a[5]["voice.turn.interrupted"] is True
    assert a[6]["voice.turn.interrupted"] is False
    assert a[6]["voice.turn.response_latency_ms"] == pytest.approx(-500)
    assert a[0]["voice.turn.interrupted"] is False


def test_conversation_totals():
    c = conversation_attributes(CALL, dead_air_ms=2000)
    assert c["voice.turn_count.user"] == 5
    assert c["voice.turn_count.agent"] == 3
    assert c["voice.interruption_count"] == 1  # the agent cut the user off once
    assert c["voice.barge_in_count"] == 1  # the user cut the agent off once
    assert c["voice.dead_air_count"] == 1
    assert c["voice.dead_air_ms"] == pytest.approx(4500)
    assert c["voice.dead_air_max_ms"] == pytest.approx(4500)
    # Gaps: 2.0->2.6, 6.0->6.8, 9.0->9.5, 12.0->16.5
    assert c["voice.silence_ms"] == pytest.approx(600 + 800 + 500 + 4500)
    assert c["voice.user_talk_ms"] == pytest.approx(2000 + 1800 + 300 + 2000 + 500)
    assert c["voice.agent_talk_ms"] == pytest.approx(2400 + 2200 + 1000)
    assert c["voice.agent_talk_share"] == pytest.approx(5600 / 12200, abs=1e-4)
    assert c["voice.response_latency_ms.max"] == pytest.approx(800)


def test_framework_flags_win_over_derivation():
    turns = [_t(USER, 0, 1), _t(AGENT, 1.5, 3, interrupted=True)]
    a = turn_attributes(turns)
    assert a[1]["voice.turn.interrupted"] is True
    assert a[1]["voice.turn.interrupted.source"] == "framework"


def test_dead_air_threshold_boundary():
    turns = [_t(USER, 0, 1), _t(AGENT, 3, 4)]  # exactly 2 s
    assert conversation_attributes(turns, dead_air_ms=2000)["voice.dead_air_count"] == 0
    assert conversation_attributes(turns, dead_air_ms=1999)["voice.dead_air_count"] == 1


def test_negative_latency_when_the_agent_started_early():
    a = turn_attributes([_t(USER, 0, 2), _t(AGENT, 1.5, 3)])
    assert a[1]["voice.turn.response_latency_ms"] == pytest.approx(-500)


def test_empty_conversation():
    assert conversation_attributes([]) == {"voice.turn_count.user": 0, "voice.turn_count.agent": 0}


# --------------------------------------------------------------------------- manual helper


@pytest.fixture
def exporter(monkeypatch):
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(manual.trace, "get_tracer", lambda name: provider.get_tracer(name))
    return exp


def test_manual_helper_emits_the_planted_call(exporter):
    with manual.voice_conversation(conversation_id="call-1") as conv:
        for t in CALL:
            conv.add_turn(t.speaker, t.start_ns, t.end_ns)
    spans = exporter.get_finished_spans()
    (conv_span,) = [s for s in spans if s.name == "voice.conversation"]
    turns = sorted((s for s in spans if s.name == "voice.turn"), key=lambda s: s.start_time)
    assert len(turns) == 8
    assert all(s.parent.span_id == conv_span.context.span_id for s in turns)
    for span, planted in zip(turns, CALL):
        assert span.start_time == planted.start_ns
        assert span.end_time == planted.end_ns
        assert span.attributes["voice.turn.speaker"] == planted.speaker
    a = dict(conv_span.attributes)
    assert a["voice.conversation.id"] == "call-1"
    assert a["voice.framework"] == "manual"
    assert a["voice.barge_in_count"] == 1
    assert a["voice.interruption_count"] == 1
    assert a["voice.dead_air_count"] == 1
    # Timing only: no gen_ai operation, so a backend never counts these as model calls.
    assert all("gen_ai.operation.name" not in s.attributes for s in spans)


def test_manual_turn_blocks_time_themselves(exporter):
    with manual.voice_conversation() as conv:
        with conv.turn(USER):
            pass
        with conv.turn(AGENT, answers=conv.last_user_turn):
            pass
    turns = [s for s in exporter.get_finished_spans() if s.name == "voice.turn"]
    assert [
        s.attributes["voice.turn.speaker"] for s in sorted(turns, key=lambda s: s.start_time)
    ] == [
        USER,
        AGENT,
    ]


def test_manual_helper_refuses_bad_turns(exporter):
    with manual.voice_conversation() as conv:
        with pytest.raises(ValueError):
            conv.add_turn("caller", 1.0, 2.0)
        with pytest.raises(ValueError):
            conv.add_turn(USER, 2.0, 1.0)


def test_dead_air_threshold_from_the_environment(exporter, monkeypatch):
    monkeypatch.setenv("GENAI_VOICE_DEAD_AIR_MS", "500")
    with manual.voice_conversation() as conv:
        conv.add_turn(USER, 1.0, 2.0)
        conv.add_turn(AGENT, 2.6, 3.0)
    (conv_span,) = [s for s in exporter.get_finished_spans() if s.name == "voice.conversation"]
    assert conv_span.attributes["voice.dead_air_count"] == 1
    assert conv_span.attributes["voice.dead_air_threshold_ms"] == 500
