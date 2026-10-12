"""The evaluation code logs how much text it read, never the text.

Content capture is off by default, and a prompt has no business in a log file
whatever that setting says: the first characters of every prompt used to be
logged at INFO by the code that was looking for personal data in it.
"""

import logging
from types import SimpleNamespace

import pytest

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor

NAME = "Margaret Thornbury"
CARD = "4111 1111 1111 1111"
PROMPT = "Card {0} belongs to {1}, please charge it.".format(CARD, NAME)
RESPONSE = "Charged the card of {0} as requested.".format(NAME)
# Pieces of the texts; none may appear in any log record.
SECRETS = (NAME, "Margaret", "Thornbury", CARD, "4111", "belongs", "Charged", "requested")


class _Span:
    """Stands in for a span: has attributes and takes more."""

    name = "test.chat"

    def __init__(self, attributes=None):
        self.attributes = dict(attributes or {})

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def is_recording(self):
        return True


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


def _answer(**fields):
    def detect(text, **kwargs):
        return SimpleNamespace(**fields)

    return SimpleNamespace(detect=detect)


@pytest.fixture
def every_detector(monkeypatch):
    """All six detectors configured, each finding something."""
    found = {
        "_pii_detector": _answer(
            has_pii=True,
            entity_counts={"CREDIT_CARD": 1, "PERSON": 1},
            entities=[{"type": "CREDIT_CARD"}, {"type": "PERSON"}],
            score=0.9,
            redacted_text="Card ******** belongs to ********",
            blocked=False,
        ),
        "_toxicity_detector": _answer(
            is_toxic=True, max_score=0.9, toxic_categories=["insult"], scores={"insult": 0.9}
        ),
        "_bias_detector": _answer(
            has_bias=True,
            max_score=0.5,
            detected_biases=["gender"],
            bias_scores={"gender": 0.5},
            patterns_matched={"gender": ["women are"]},
        ),
        "_prompt_injection_detector": _answer(
            is_injection=True,
            injection_score=0.6,
            injection_types=["jailbreak"],
            patterns_matched={"jailbreak": ["developer mode"]},
        ),
        "_restricted_topics_detector": _answer(
            has_restricted_topic=True,
            max_score=0.5,
            detected_topics=["violence"],
            topic_scores={"violence": 0.5},
        ),
        "_hallucination_detector": _answer(
            has_hallucination=True,
            hallucination_score=0.7,
            citation_count=0,
            hedge_words_count=0,
            factual_claim_count=1,
            hallucination_indicators=["absolute_statements"],
            unsupported_claims=[],
        ),
    }
    for slot, detector in found.items():
        monkeypatch.setattr(BaseInstrumentor, slot, detector)
    monkeypatch.setattr(BaseInstrumentor, "_blocking_active", False)
    monkeypatch.setattr(BaseInstrumentor, "_blocking_policies", frozenset())


@pytest.fixture
def logged(caplog):
    """Everything the library logs, at every level."""
    caplog.set_level(logging.DEBUG, logger="genai_otel")

    def messages():
        return [record.getMessage() for record in caplog.records]

    return messages


def _instrumentor():
    instrumentor = _Instrumentor()
    instrumentor.config = OTelConfig(service_name="t", evaluation_mode="inline")
    return instrumentor


def _result():
    message = SimpleNamespace(content=RESPONSE, role="assistant", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _leaks(messages):
    return [(secret, message) for message in messages for secret in SECRETS if secret in message]


def test_evaluating_a_call_logs_none_of_its_text(every_detector, logged):
    span = _Span({"gen_ai.operation.name": "chat"})
    request = {"messages": [{"role": "user", "content": PROMPT}]}

    _instrumentor()._run_evaluation_checks(span, (), request, _result())

    assert span.attributes["evaluation.completed"] is True
    assert logged(), "the evaluation pass logged nothing at all, so this proves nothing"
    assert _leaks(logged()) == []
    # What it does say is how much it read.
    assert any("%d characters" % len(PROMPT) in message for message in logged())


@pytest.mark.parametrize(
    "first_message",
    [
        '{"role": "user", "content": "%s"}' % PROMPT,  # as JSON
        "{'role': 'user', 'content': '%s'}" % PROMPT,  # as an older release wrote it
        '{"role": "user", "content": "%s' % PROMPT[:30],  # cut off by a length limit
        PROMPT,  # not a message at all
    ],
    ids=["json", "python-dict", "truncated", "raw"],
)
def test_a_prompt_read_back_from_the_span_is_not_logged_either(
    every_detector, logged, first_message
):
    """A wrapper that passes no ``messages`` leaves the prompt to be read from
    the span, and each way of reading it used to log what it found."""
    span = _Span({"gen_ai.request.first_message": first_message})
    instrumentor = _instrumentor()

    prompt, _ = instrumentor._evaluation_texts(span, {}, None)
    instrumentor._evaluate_texts(span, prompt, None)

    assert prompt and prompt.startswith("Card 4111")
    assert logged()
    assert _leaks(logged()) == []


def test_the_block_check_logs_none_of_the_prompt(every_detector, logged, monkeypatch):
    monkeypatch.setattr(BaseInstrumentor, "_blocking_active", True)
    monkeypatch.setattr(
        BaseInstrumentor, "_blocking_policies", frozenset({"pii", "bias", "toxicity"})
    )
    span = _Span()

    with pytest.raises(Exception):
        _instrumentor()._enforce_precall_policy(
            span, {"messages": [{"role": "user", "content": PROMPT}]}, "chat"
        )

    assert span.attributes["evaluation.blocked"] is True
    assert _leaks(logged()) == []
