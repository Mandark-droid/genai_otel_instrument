"""Outcome and provenance metadata must stay bounded and content-free."""

import pytest

from genai_otel.evaluation.telemetry import apply_evaluation_telemetry, configuration_fingerprint


class _Span:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_success_and_content_handling_metadata_do_not_copy_content():
    span = _Span()
    source = "private prompt test@example.com"
    attrs = {
        "gen_ai.request.first_message": source,
        "evaluation.pii.prompt.detected": True,
        "evaluation.pii.prompt.redacted": "private prompt [REDACTED]",
        "evaluation.truncated": True,
    }

    apply_evaluation_telemetry(span, attrs, {"pii"}, source, None, "chat")

    assert span.attributes["evaluation.outcome"] == "completed"
    assert span.attributes["evaluation.pii.outcome"] == "completed"
    assert span.attributes["evaluation.content.outcomes"] == [
        "captured",
        "truncated",
        "redacted",
    ]
    assert span.attributes["evaluation.provenance.sdk.name"] == "genai-otel-instrument"
    assert (
        span.attributes["evaluation.provenance.detector.pii.implementation"]
        == "genai_otel.evaluation.pii_detector.PIIDetector"
    )
    assert len(span.attributes["evaluation.config.fingerprint"]) == 64
    assert all(source not in str(value) for value in span.attributes.values())


def test_missing_input_and_unsupported_operation_are_not_reported_as_success():
    no_input = _Span()
    apply_evaluation_telemetry(no_input, {}, {"pii", "hallucination"}, None, None)
    assert no_input.attributes["evaluation.pii.outcome"] == "no_applicable_input"
    assert no_input.attributes["evaluation.outcome"] == "no_applicable_input"

    unsupported = _Span()
    apply_evaluation_telemetry(
        unsupported,
        {"gen_ai.operation.name": "embeddings"},
        {"hallucination"},
        None,
        None,
        "embeddings",
    )
    assert unsupported.attributes["evaluation.hallucination.outcome"] == "unsupported_operation"
    assert unsupported.attributes["evaluation.outcome"] == "unsupported_operation"


def test_skipped_and_error_outcomes_take_precedence_over_success():
    skipped = _Span()
    apply_evaluation_telemetry(
        skipped,
        {"evaluation.skipped": "overloaded", "evaluation.pii.prompt.detected": True},
        {"pii"},
        "prompt",
        None,
    )
    assert skipped.attributes["evaluation.pii.outcome"] == "skipped"
    assert skipped.attributes["evaluation.outcome"] == "skipped"

    failed = _Span()
    apply_evaluation_telemetry(
        failed,
        {"evaluation.pii.error": "RuntimeError"},
        {"pii"},
        "prompt",
        None,
    )
    assert failed.attributes["evaluation.pii.outcome"] == "error"
    assert failed.attributes["evaluation.outcome"] == "error"


def test_pre_call_rejection_is_distinct_from_post_call_detector_annotation():
    rejected = _Span()
    apply_evaluation_telemetry(
        rejected,
        {"evaluation.blocked": True, "evaluation.pii.prompt.blocked": True},
        {"pii"},
        "prompt",
        None,
    )
    assert "rejected" in rejected.attributes["evaluation.content.outcomes"]

    annotation_only = _Span()
    apply_evaluation_telemetry(
        annotation_only,
        {"evaluation.pii.prompt.blocked": True},
        {"pii"},
        "prompt",
        None,
    )
    assert "rejected" not in annotation_only.attributes["evaluation.content.outcomes"]


def test_fingerprint_is_stable_and_only_depends_on_enabled_detector_names():
    assert configuration_fingerprint({"pii", "bias"}) == configuration_fingerprint(
        ["bias", "pii", "pii"]
    )
    assert configuration_fingerprint({"pii"}) != configuration_fingerprint({"bias"})


@pytest.mark.parametrize(
    ("detector", "scope"),
    [
        ("pii", "prompt"),
        ("toxicity", "prompt"),
        ("bias", "prompt"),
        ("prompt_injection", "prompt"),
        ("restricted_topics", "prompt"),
        ("hallucination", "response"),
    ],
)
def test_each_detector_reports_completed_only_for_its_supported_text_scope(detector, scope):
    span = _Span()
    attrs = {f"evaluation.{detector}.{scope}.detected": False}
    prompt = "input" if scope == "prompt" else None
    response = "output" if scope == "response" else None

    apply_evaluation_telemetry(span, attrs, {detector}, prompt, response)

    assert span.attributes[f"evaluation.{detector}.outcome"] == "completed"
