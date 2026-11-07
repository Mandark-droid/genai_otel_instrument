"""Voice frameworks' own spans (#48): LiveKit turn shape, Pipecat content rule, observer.

The processor is exercised with spans named and scoped as LiveKit Agents 1.8 and
Pipecat 1.15 emit them (read from those releases), so these run without either
framework installed. The Pipecat observer test runs a real pipeline when Pipecat is
installed.
"""

import asyncio
import os
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.voice import processor as voice_processor
from genai_otel.voice.processor import VoiceSpanProcessor

S = 1_000_000_000
T0 = 1_700_000_000 * S


def _provider(proc):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(proc)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _at(seconds):
    return T0 + int(seconds * S)


def test_livekit_speaking_spans_get_the_turn_shape_and_the_session_its_totals():
    proc = VoiceSpanProcessor(capture_content=False, livekit=True, pipecat=False)
    provider, exporter = _provider(proc)
    tracer = provider.get_tracer("livekit-agents")
    from opentelemetry import trace

    session = tracer.start_span("agent_session", start_time=_at(0))
    ctx = trace.set_span_in_context(session)
    # user 0-2; agent 2.5-5 (user barges in at 4.0, agent stops at 5 while user speaks);
    # user 4.0-6.0; 4 s of silence; user 10-11.
    user1 = tracer.start_span("user_speaking", context=ctx, start_time=_at(0))
    user1.end(end_time=_at(2))
    agent1 = tracer.start_span("agent_speaking", context=ctx, start_time=_at(2.5))
    user2 = tracer.start_span("user_speaking", context=ctx, start_time=_at(4.0))
    agent1.end(end_time=_at(5))
    user2.end(end_time=_at(6))
    user3 = tracer.start_span("user_speaking", context=ctx, start_time=_at(10))
    user3.end(end_time=_at(11))
    session.end(end_time=_at(12))

    spans = {s.name: s for s in exporter.get_finished_spans() if s.name != "user_speaking"}
    users = [s for s in exporter.get_finished_spans() if s.name == "user_speaking"]
    agent = spans["agent_speaking"]
    assert agent.attributes["voice.turn.speaker"] == "agent"
    assert agent.attributes["voice.turn.response_latency_ms"] == pytest.approx(500)
    # The user had started (4.0) and was still speaking when the agent stopped (5.0).
    assert agent.attributes["voice.turn.interrupted"] is True
    assert agent.attributes["voice.turn.overlap_ms"] == pytest.approx(1000)
    assert all(u.attributes["voice.turn.speaker"] == "user" for u in users)
    totals = spans["agent_session"].attributes
    assert totals["voice.framework"] == "livekit"
    assert totals["voice.barge_in_count"] == 1
    assert totals["voice.dead_air_count"] == 1
    assert totals["voice.dead_air_ms"] == pytest.approx(4000)
    assert totals["voice.turn_count.user"] == 3


def test_pipecat_content_is_removed_with_capture_off_only():
    for capture in (False, True):
        proc = VoiceSpanProcessor(capture_content=capture, livekit=False, pipecat=True)
        provider, exporter = _provider(proc)
        tracer = provider.get_tracer("pipecat")
        with tracer.start_as_current_span(
            "llm",
            attributes={
                "input": "[secret prompt]",
                "output": "secret answer",
                "transcript": "secret words",
                "gen_ai.system_instructions": "be secret",
                "gen_ai.request.model": "gpt-5.5",
                "gen_ai.usage.input_tokens": 12,
            },
        ):
            pass
        (span,) = exporter.get_finished_spans()
        a = dict(span.attributes)
        assert a["gen_ai.request.model"] == "gpt-5.5"
        assert a["gen_ai.usage.input_tokens"] == 12
        for key in ("input", "output", "transcript", "gen_ai.system_instructions"):
            assert (key in a) is capture, (capture, key)


def test_other_libraries_spans_are_untouched():
    proc = VoiceSpanProcessor(capture_content=False, livekit=True, pipecat=True)
    provider, exporter = _provider(proc)
    with provider.get_tracer("my_app").start_as_current_span("x", attributes={"input": "keep"}):
        pass
    (span,) = exporter.get_finished_spans()
    assert span.attributes["input"] == "keep"


@pytest.mark.parametrize(
    "capture,preset,expected", [(False, None, "false"), (False, "true", "true"), (True, None, None)]
)
def test_livekit_pii_switch_follows_ours_unless_set(monkeypatch, capture, preset, expected):
    from genai_otel.instrumentors import voice_instrumentors as vi

    monkeypatch.delenv("LIVEKIT_TELEMETRY_ALLOW_PII", raising=False)
    if preset is not None:
        monkeypatch.setenv("LIVEKIT_TELEMETRY_ALLOW_PII", preset)
    monkeypatch.setattr(voice_processor, "_INSTALLED", None)
    with patch.object(vi, "_version", lambda dist: (1, 8)):
        inst = vi.LiveKitAgentsInstrumentor()
        inst.instrument(OTelConfig(enable_content_capture=capture))
    assert os.environ.get("LIVEKIT_TELEMETRY_ALLOW_PII") == expected


def test_not_installed_frameworks_do_nothing(monkeypatch):
    from genai_otel.instrumentors import voice_instrumentors as vi

    monkeypatch.setattr(voice_processor, "_INSTALLED", None)
    with patch.object(vi, "_version", lambda dist: None):
        vi.PipecatInstrumentor().instrument(OTelConfig())
        vi.LiveKitAgentsInstrumentor().instrument(OTelConfig())
    assert voice_processor._INSTALLED is None


def test_pipecat_observer_reproduces_planted_turns(monkeypatch):
    pytest.importorskip("pipecat")
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        EndFrame,
        StartFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )

    from genai_otel.voice.pipecat_observer import make_observer

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observer = make_observer(provider.get_tracer("genai_otel.voice"), "conv-1")

    class Pushed:
        def __init__(self, frame):
            self.frame = frame

    script = [
        (0.0, StartFrame()),
        (0.0, UserStartedSpeakingFrame()),
        (0.3, UserStoppedSpeakingFrame()),
        (0.45, BotStartedSpeakingFrame()),
        (0.45, BotStartedSpeakingFrame()),  # the same start seen twice (up + downstream)
        (0.65, UserStartedSpeakingFrame()),
        (0.8, BotStoppedSpeakingFrame()),
        (0.95, UserStoppedSpeakingFrame()),
        (1.0, EndFrame()),
    ]

    async def drive():
        t = 0.0
        for at, frame in script:
            await asyncio.sleep(at - t)
            t = at
            await observer.on_push_frame(Pushed(frame))

    asyncio.run(drive())
    turns = sorted(
        (s for s in exporter.get_finished_spans() if s.name == "voice.turn"),
        key=lambda s: s.start_time,
    )
    assert [t.attributes["voice.turn.speaker"] for t in turns] == ["user", "agent", "user"]
    durations = [(t.end_time - t.start_time) / S for t in turns]
    assert durations == pytest.approx([0.3, 0.35, 0.3], abs=0.05)
    assert turns[1].attributes["voice.turn.interrupted"] is True  # the user barged in
    (conv,) = [s for s in exporter.get_finished_spans() if s.name == "voice.conversation"]
    assert conv.attributes["voice.framework"] == "pipecat"
    assert conv.attributes["voice.barge_in_count"] == 1
