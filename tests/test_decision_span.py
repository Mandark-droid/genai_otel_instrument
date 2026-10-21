"""genai_otel.decision_span: in-process decision models recorded by hand."""

import asyncio
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import genai_otel
from genai_otel import decision as manual
from genai_otel.config import OTelConfig
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors.base import BaseInstrumentor

QUESTIONS = {
    "next_tool": {"type": "choice", "criteria": {"search": None, "refund": None, "escalate": None}}
}
ANSWERS = {
    "next_tool": {
        "type": "choice",
        "choice": "refund",
        "confidence": 0.64,
        "probabilities": {"refund": 0.7, "search": 0.2, "escalate": 0.1},
    }
}


@pytest.fixture
def spans(monkeypatch):
    """Point the helper at an in-memory exporter, with a fresh configuration per test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(manual, "_INSTRUMENTOR", None)
    monkeypatch.setattr(manual, "_CONFIG", None)

    def configure(**config):
        manual.configure(OTelConfig(content_max_length=0, **config))
        manual._instrumentor().tracer = provider.get_tracer("test")
        return exporter

    return configure


def _only(exporter):
    (span,) = exporter.get_finished_spans()
    return span


def test_exported_from_the_package():
    assert genai_otel.decision_span is manual.decision_span
    assert "decision_span" in genai_otel.__all__


def test_context_manager_with_capture_off(spans):
    exporter = spans(enable_content_capture=False)
    with genai_otel.decision_span(model="laya-router-421m", questions=QUESTIONS, state="hi") as d:
        d.record(ANSWERS)
    span = _only(exporter)
    attrs = dict(span.attributes)

    assert span.name == "decision"
    assert attrs["openinference.span.kind"] == "DECISION"
    assert attrs["genai.decision.protocol"] == "manual"
    assert attrs["decision.system"] == "custom"
    assert attrs["decision.model_name"] == "laya-router-421m"
    assert attrs["genai.decision.option_count"] == 3
    assert "genai.decision.choice" not in attrs
    assert "input.value" not in attrs
    # No usage recorded: no tokens and no cost rather than zeros.
    assert not any("token" in key for key in attrs)
    assert "gen_ai.usage.cost.total" not in attrs
    assert span.status.status_code == StatusCode.OK


def test_context_manager_with_capture_on_and_usage(spans):
    exporter = spans(enable_content_capture=True)
    with genai_otel.decision_span(
        model="d1", questions=QUESTIONS, provider="liquid", state="I want my money back"
    ) as d:
        d.record(ANSWERS, usage={"input_tokens": 1000, "output_tokens": 0}, model="d1")
    attrs = dict(_only(exporter).attributes)

    assert attrs["genai.decision.choice"] == "refund"
    assert attrs["genai.decision.confidence"] == 0.64
    assert attrs["genai.decision.margin"] == pytest.approx(0.5)
    assert attrs["decision.provider"] == "liquid"
    assert attrs["decision.token_count.input"] == 1000
    assert attrs["gen_ai.usage.cost.decision"] == pytest.approx(1000 * 0.02 / 1e6)
    assert json.loads(attrs["input.value"])["state"] == "I want my money back"


def test_openai_shaped_questions_and_answers(spans):
    exporter = spans(enable_content_capture=True)
    with genai_otel.decision_span(
        model="m", questions=[{"type": "predicate", "name": "spam"}], system="openai"
    ) as d:
        d.record([{"type": "predicate", "probability": 0.3}])
    attrs = dict(_only(exporter).attributes)
    assert attrs["genai.decision.probability"] == 0.3
    assert tuple(attrs["genai.decision.question.names"]) == ("spam",)
    assert attrs["decision.system"] == "openai"


def test_an_exception_ends_the_span_as_an_error(spans):
    exporter = spans()
    with pytest.raises(RuntimeError):
        with genai_otel.decision_span(model="m", questions=QUESTIONS):
            raise RuntimeError("model crashed")
    span = _only(exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[0].name == "exception"


def test_a_failure_without_an_exception(spans):
    exporter = spans()
    with genai_otel.decision_span(model="m", questions=QUESTIONS) as d:
        d.fail("rules engine timed out", "timeout")
    span = _only(exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "timeout"


def test_nothing_recorded_still_proves_a_decision_ran(spans):
    exporter = spans()
    with genai_otel.decision_span(model="m", questions=QUESTIONS):
        pass
    attrs = dict(_only(exporter).attributes)
    assert attrs["genai.decision.question_count"] == 1
    assert "genai.decision.refusal_count" not in attrs


def test_decorator_records_the_return_value(spans):
    exporter = spans(enable_content_capture=True)

    @genai_otel.decision_span(model="clef-flash", questions=QUESTIONS, name="route")
    def route(message):
        return ANSWERS

    assert route("hello") is ANSWERS
    assert route("again") is ANSWERS
    first, second = exporter.get_finished_spans()
    assert first.name == "route"
    assert first.attributes["genai.decision.choice"] == "refund"
    # Self-hosted weights: no list price.
    assert "gen_ai.usage.cost.total" not in first.attributes
    assert second.context.span_id != first.context.span_id


def test_decorator_takes_a_whole_response(spans):
    exporter = spans(enable_content_capture=True)

    @genai_otel.decision_span(model="jev", questions=QUESTIONS)
    def decide():
        return {"answers": ANSWERS, "usage": {"input_tokens": 10, "output_tokens": 0}}

    decide()
    assert _only(exporter).attributes["decision.token_count.input"] == 10


def test_decorator_ignores_a_return_value_that_is_not_an_answer(spans):
    exporter = spans(enable_content_capture=True)

    @genai_otel.decision_span(model="m", questions=QUESTIONS)
    def decide():
        return "refund"

    decide()
    assert "genai.decision.choice" not in _only(exporter).attributes


def test_decorator_lets_the_function_record_itself(spans):
    exporter = spans(enable_content_capture=True)
    helper = genai_otel.decision_span(model="m", questions=QUESTIONS)

    with helper as d:
        d.record(ANSWERS)
    assert _only(exporter).attributes["genai.decision.choice"] == "refund"


def test_async_decorator(spans):
    exporter = spans(enable_content_capture=True)

    @genai_otel.decision_span(model="m", questions=QUESTIONS)
    async def decide():
        await asyncio.sleep(0)
        return ANSWERS

    assert asyncio.run(decide()) is ANSWERS
    assert _only(exporter).attributes["genai.decision.choice"] == "refund"


def test_the_input_is_screened_by_the_detectors(spans):
    from types import SimpleNamespace

    from genai_otel.config import EVALUATION_MODE_INLINE

    seen = []

    def detect(text):
        seen.append(text)
        return SimpleNamespace(has_pii=False, entity_counts={}, entities=[], score=0.0)

    BaseInstrumentor._pii_detector = SimpleNamespace(detect=detect)
    exporter = spans(evaluation_mode=EVALUATION_MODE_INLINE)
    with genai_otel.decision_span(model="m", questions=QUESTIONS, state="call 555-0100") as d:
        d.record(ANSWERS)
    assert seen and "555-0100" in seen[0]
    assert _only(exporter).attributes["evaluation.skipped_reason"] == "decision_span"


def test_a_block_policy_stops_the_decision(spans):
    from types import SimpleNamespace

    def detect(text):
        return SimpleNamespace(has_pii=True, entity_counts={"PHONE_NUMBER": 1}, score=0.9)

    BaseInstrumentor._pii_detector = SimpleNamespace(detect=detect)
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True
    exporter = spans()
    ran = []
    with pytest.raises(PolicyViolationError):
        with genai_otel.decision_span(model="m", questions=QUESTIONS, state="call 555-0100"):
            ran.append(True)
    assert ran == []
    assert _only(exporter).status.status_code == StatusCode.ERROR


def test_defaults_come_from_the_environment_before_instrument(monkeypatch):
    monkeypatch.setattr(manual, "_INSTRUMENTOR", None)
    monkeypatch.setattr(manual, "_CONFIG", None)
    monkeypatch.setenv("GENAI_ENABLE_CONTENT_CAPTURE", "true")
    assert manual._instrumentor()._capture() is True
    # instrument() later points the existing helper at its configuration.
    manual.configure(OTelConfig(enable_content_capture=False))
    assert manual._instrumentor()._capture() is False


def test_background_evaluation_ends_the_span_after_the_detectors(spans):
    from types import SimpleNamespace

    from genai_otel.instrumentors.base import drain_evaluations

    BaseInstrumentor._pii_detector = SimpleNamespace(
        detect=lambda text: SimpleNamespace(has_pii=False, entity_counts={}, entities=[], score=0)
    )
    exporter = spans()  # background is the default evaluation mode
    with genai_otel.decision_span(model="m", questions=QUESTIONS, state="hello") as d:
        d.record(ANSWERS)
    assert drain_evaluations(timeout=10)
    attrs = _only(exporter).attributes
    assert attrs["evaluation.skipped_reason"] == "decision_span"
    assert attrs["evaluation.pii.prompt.detected"] is False
