"""Evaluation detectors and block policies apply to AsyncOpenAI calls.

The async client has its own span wrapper, and it called neither hook: an
application on ``AsyncOpenAI`` got no evaluation attributes at all, and a block
policy let every request through. Detectors are CPU work, so on this path they
run on a worker thread rather than holding the event loop.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors.base import BaseInstrumentor, drain_evaluations
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "My card number is 4111 1111 1111 1111."},
]


class _PIIDetector:
    """Stands in for the PII detector; records what it was shown and on which thread."""

    def __init__(self, has_pii):
        self.has_pii = has_pii
        self.seen = []
        self.threads = []

    def detect(self, text):
        self.seen.append(text)
        self.threads.append(threading.get_ident())
        return SimpleNamespace(
            has_pii=self.has_pii,
            entity_counts={"CREDIT_CARD": 1} if self.has_pii else {},
            entities=["4111 1111 1111 1111"] if self.has_pii else [],
            score=0.9 if self.has_pii else 0.0,
            redacted_text=None,
            blocked=False,
        )


def _chat_result():
    message = SimpleNamespace(content="Noted.", role="assistant", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=2, total_tokens=14)
    return SimpleNamespace(id="chatcmpl-1", model="gpt-4o-mini", choices=[choice], usage=usage)


@pytest.fixture
def evaluation_state():
    """Detector slots are class-level: start from none configured, whatever an
    earlier test left behind, and put everything back afterwards."""
    cleared = {
        "_pii_detector": None,
        "_toxicity_detector": None,
        "_bias_detector": None,
        "_prompt_injection_detector": None,
        "_restricted_topics_detector": None,
        "_hallucination_detector": None,
        "_blocking_active": False,
        "_blocking_policies": frozenset(),
    }
    saved = {name: getattr(BaseInstrumentor, name) for name in cleared}
    for name, value in cleared.items():
        setattr(BaseInstrumentor, name, value)
    yield
    for name, value in saved.items():
        setattr(BaseInstrumentor, name, value)


@pytest.fixture
def harness():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    calls = []

    async def create(*args, **kwargs):
        calls.append(kwargs)
        return _chat_result()

    instrumentor = OpenAIInstrumentor()
    instrumentor.config = OTelConfig(service_name="t")
    with patch("opentelemetry.trace.get_tracer", return_value=provider.get_tracer(__name__)):
        wrapped = instrumentor._create_async_span_wrapper(
            span_name="openai.chat.completion",
            extract_attributes=instrumentor._extract_openai_attributes,
        )(create)

        def run(**kwargs):
            loop_thread = {}

            async def main():
                loop_thread["id"] = threading.get_ident()
                return await wrapped(model="gpt-4o-mini", messages=MESSAGES, **kwargs)

            try:
                return asyncio.run(main()), loop_thread["id"]
            finally:
                run.loop_thread = loop_thread.get("id")

        def spans():
            # The detectors run after the call has returned, and the span is
            # exported when they are done.
            assert drain_evaluations(30)
            return exporter.get_finished_spans()

        yield SimpleNamespace(run=run, calls=calls, spans=spans)


def test_async_call_is_evaluated(evaluation_state, harness):
    detector = _PIIDetector(has_pii=True)
    BaseInstrumentor._pii_detector = detector

    harness.run()

    (span,) = harness.spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert span.attributes["evaluation.pii.prompt.entity_count"] == 1
    # The user's message is evaluated, not only the first one.
    assert "4111 1111 1111 1111" in detector.seen[0]


def test_async_detectors_stay_off_the_event_loop(evaluation_state, harness):
    """A detector takes tens to hundreds of milliseconds on a long prompt. Run on
    the loop, that stalls every other task the application has in flight."""
    detector = _PIIDetector(has_pii=False)
    BaseInstrumentor._pii_detector = detector

    _, loop_thread = harness.run()
    assert drain_evaluations(30)

    assert detector.threads, "the detector did not run"
    assert loop_thread not in detector.threads


def test_async_block_policy_stops_the_request(evaluation_state, harness):
    detector = _PIIDetector(has_pii=True)
    BaseInstrumentor._pii_detector = detector
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True

    with pytest.raises(PolicyViolationError):
        harness.run()

    assert harness.calls == [], "the request reached the provider despite the block policy"
    (span,) = harness.spans()
    assert span.attributes["evaluation.blocked"] is True
    assert span.status.status_code.name == "ERROR"
    assert harness.run.loop_thread not in detector.threads


def test_async_block_policy_lets_a_clean_request_through(evaluation_state, harness):
    BaseInstrumentor._pii_detector = _PIIDetector(has_pii=False)
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True

    result, _ = harness.run()

    assert result.choices[0].message.content == "Noted."
    assert len(harness.calls) == 1


def test_async_call_without_detectors_starts_no_worker(evaluation_state, harness):
    """The common case pays nothing: no detectors, no thread hop."""
    with patch("genai_otel.instrumentors.base._off_the_event_loop", new=MagicMock()) as offload:
        harness.run()

    offload.assert_not_called()
    (span,) = harness.spans()
    assert not [key for key in span.attributes if key.startswith("evaluation.")]


def test_a_detector_that_fails_does_not_fail_the_call(evaluation_state, harness):
    class _Broken:
        def detect(self, text):
            raise RuntimeError("model not loaded")

    BaseInstrumentor._pii_detector = _Broken()

    result, _ = harness.run()

    assert result.choices[0].message.content == "Noted."
