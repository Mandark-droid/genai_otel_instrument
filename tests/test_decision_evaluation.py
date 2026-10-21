"""The evaluation span processor and enriching exporter on decision spans."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from genai_otel.evaluation.span_processor import EvaluationSpanProcessor, is_decision_span


class TestEvaluationProcessor:
    def test_is_decision_span(self):
        assert is_decision_span({"openinference.span.kind": "DECISION"})
        assert is_decision_span({"openinference.span.kind": "decision"})
        assert is_decision_span({"gen_ai.request.type": "decision"})
        assert is_decision_span({"genai.decision.protocol": "manual"})
        assert not is_decision_span({"openinference.span.kind": "LLM"})
        assert not is_decision_span({})
        assert not is_decision_span(None)

    def test_no_response_text_on_a_decision_span(self):
        processor = EvaluationSpanProcessor()
        attrs = {"openinference.span.kind": "DECISION", "output.value": '{"answers": []}'}
        assert processor._extract_response(attrs) is None
        assert processor._extract_response({"output.value": "text"}) == "text"

    def test_on_end_screens_the_input_and_says_why_the_response_was_not(self):
        processor = EvaluationSpanProcessor()
        seen = []
        processor.pii_config = SimpleNamespace(enabled=True)
        processor.pii_detector = MagicMock()
        processor._check_pii = lambda span, prompt, response, setter: seen.append(
            (prompt, response)
        )
        span = SimpleNamespace(
            name="openai.decisions",
            attributes={
                "openinference.span.kind": "DECISION",
                "input.value": "call me on 555-0100",
                "output.value": '{"answers": []}',
            },
            events=(),
        )
        span._attributes = dict(span.attributes)
        processor.on_end(span)
        assert seen and seen[0][1] is None
        assert span._attributes["evaluation.skipped_reason"] == "decision_span"
