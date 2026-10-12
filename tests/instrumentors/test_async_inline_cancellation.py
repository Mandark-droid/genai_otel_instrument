"""An async caller that is cancelled while it waits for the detectors.

In inline mode an async caller awaits a worker thread that runs the detectors.
Cancelled (a timeout, a client that went away), the caller stopped waiting and
ended the span; the worker thread went on and wrote its findings to a span that
had already been exported, without them. The span is left to the evaluation
queue instead, which ends it when the detectors are done.

The stand-in detector is held and released by the test, so nothing below depends
on how long anything takes.
"""

import asyncio
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors import base
from genai_otel.instrumentors.base import BaseInstrumentor
from genai_otel.instrumentors.litellm_latency_instrumentor import LiteLLMLatencyInstrumentor
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

CARD = "4111 1111 1111 1111"
MESSAGES = [{"role": "user", "content": "My card number is {0}.".format(CARD)}]
WAIT = 30


class _HeldDetector:
    """Stands in for the PII detector; every call waits until the test releases it."""

    def __init__(self):
        self._state = threading.Condition()
        self._released = False
        self.started = 0

    def detect(self, text):
        with self._state:
            self.started += 1
            self._state.notify_all()
            assert self._state.wait_for(lambda: self._released, timeout=WAIT), "never released"
        return SimpleNamespace(
            has_pii=True,
            entity_counts={"CREDIT_CARD": 1},
            entities=[CARD],
            score=0.9,
            redacted_text=None,
            blocked=False,
        )

    def wait_until_started(self, count=1):
        with self._state:
            assert self._state.wait_for(lambda: self.started >= count, timeout=WAIT)

    def release(self):
        with self._state:
            self._released = True
            self._state.notify_all()


def _chat_result():
    message = SimpleNamespace(content="Noted.", role="assistant", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=2, total_tokens=14)
    return SimpleNamespace(id="chatcmpl-1", model="m", choices=[choice], usage=usage)


async def _provider(*args, **kwargs):
    return _chat_result()


def _openai(provider):
    instrumentor = OpenAIInstrumentor()
    instrumentor.config = OTelConfig(service_name="t", evaluation_mode="inline")
    with patch("opentelemetry.trace.get_tracer", return_value=provider.get_tracer("test")):
        wrapped = instrumentor._create_async_span_wrapper(
            span_name="openai.chat.completion",
            extract_attributes=instrumentor._extract_openai_attributes,
        )(_provider)
    return wrapped


def _litellm(provider):
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        instrumentor = LiteLLMLatencyInstrumentor()
    instrumentor.config = OTelConfig(service_name="t", evaluation_mode="inline")
    instrumentor.tracer = provider.get_tracer("test")
    instrumentor._instrumented = True
    return instrumentor._wrap_async(_provider, "litellm.acompletion")


@pytest.fixture(params=[_openai, _litellm], ids=["openai", "litellm"])
def cancelled_call(request, monkeypatch):
    """Make one async call and cancel the caller once the detector has started."""
    for name in (
        "_toxicity_detector",
        "_bias_detector",
        "_prompt_injection_detector",
        "_restricted_topics_detector",
        "_hallucination_detector",
    ):
        monkeypatch.setattr(BaseInstrumentor, name, None)
    monkeypatch.setattr(BaseInstrumentor, "_blocking_active", False)
    monkeypatch.setattr(BaseInstrumentor, "_blocking_policies", frozenset())
    detector = _HeldDetector()
    monkeypatch.setattr(BaseInstrumentor, "_pii_detector", detector)
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", base._EvaluationQueue())

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    wrapped = request.param(provider)

    async def run():
        with patch("opentelemetry.trace.get_tracer", return_value=tracer):
            task = asyncio.ensure_future(wrapped(model="m", messages=MESSAGES))
            # Not on the loop's own thread: the detector is what is being waited for.
            await asyncio.get_running_loop().run_in_executor(None, detector.wait_until_started)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    yield SimpleNamespace(run=run, detector=detector, spans=exporter.get_finished_spans)
    detector.release()
    assert base.drain_evaluations(WAIT)


def test_the_cancelled_caller_gets_its_cancellation_at_once(cancelled_call):
    asyncio.run(cancelled_call.run())  # returns while the detector is still held


def test_the_span_is_exported_with_its_evaluation_once_the_detectors_are_done(
    cancelled_call, caplog
):
    with caplog.at_level(logging.WARNING, logger="opentelemetry"):
        asyncio.run(cancelled_call.run())
        # Exported when the detectors are done, not when the caller left.
        assert cancelled_call.spans() == ()
        cancelled_call.detector.release()
        assert base.drain_evaluations(WAIT)

    (span,) = cancelled_call.spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert span.attributes["evaluation.completed"] is True
    # Nothing was written to the span after it had ended, and it ended once.
    assert [record.getMessage() for record in caplog.records] == []
