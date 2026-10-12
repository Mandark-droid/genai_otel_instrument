"""How much text the detectors read, and how a span says that it was cut.

A detector's cost grows with the text. Past a limit the detectors read the end of
a prompt (the newest turns of a conversation) and the start of a response. A span
whose text was cut carries markers saying so; text is never cut silently.
"""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.evaluation.bias_detector import BiasDetector
from genai_otel.evaluation.config import BiasConfig, HallucinationConfig
from genai_otel.evaluation.hallucination_detector import HallucinationDetector
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors import base
from genai_otel.instrumentors.base import BaseInstrumentor

CARD = "4111 1111 1111 1111"
FILLER = "The statement is generated a few days before the end of each billing cycle. "
WAIT = 30


def _clean(found=False):
    return SimpleNamespace(
        has_pii=found,
        entity_counts={"CREDIT_CARD": 1} if found else {},
        entities=[CARD] if found else [],
        score=0.9 if found else 0.0,
        redacted_text=None,
        blocked=False,
    )


class _TextDetector:
    """Stands in for a detector that takes text and nothing else."""

    def __init__(self):
        self.seen = []

    def detect(self, text):
        self.seen.append(text)
        return _clean(CARD in text)


class _CappingDetector:
    """Stands in for the PII detector, which can limit its own model."""

    accepts_model_chars = True

    def __init__(self):
        self.calls = []

    def detect(self, text, language="en", model_chars=0):
        self.calls.append((text, model_chars))
        result = _clean(CARD in text)
        result.model_analyzed_length = model_chars - 3 if model_chars else None
        return result


class _ResponseDetector:
    """Stands in for the hallucination detector: reads the response and a context."""

    def __init__(self):
        self.calls = []

    def detect(self, text, context=None):
        self.calls.append((text, context))
        return SimpleNamespace(
            has_hallucination=False,
            hallucination_score=0.0,
            citation_count=0,
            hedge_words_count=0,
            factual_claim_count=0,
            hallucination_indicators=[],
            unsupported_claims=[],
        )


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


@pytest.fixture
def detectors():
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
    assert base.drain_evaluations(WAIT)
    for name, value in saved.items():
        setattr(BaseInstrumentor, name, value)


@pytest.fixture
def chat():
    """Make one chat call through the shared wrapper; returns its finished span."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    calls = []

    def call(prompt, response="Noted.", **config):
        instrumentor = _Instrumentor()
        instrumentor.config = OTelConfig(service_name="t", **config)
        instrumentor._instrumented = True
        instrumentor.tracer = provider.get_tracer("test")

        def create(*args, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content=response, role="assistant", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

        wrapped = instrumentor.create_span_wrapper(
            "test.chat", lambda i, a, k: {"gen_ai.operation.name": "chat"}
        )(create)
        try:
            wrapped(model="m", messages=[{"role": "user", "content": prompt}])
        finally:
            assert base.drain_evaluations(WAIT)
        return exporter.get_finished_spans()[-1]

    call.calls = calls
    call.spans = exporter.get_finished_spans
    return call


def _markers(span):
    return {
        key: value
        for key, value in span.attributes.items()
        if key == "evaluation.truncated" or key.endswith("length")
    }


LONG = FILLER * 40  # about 3,000 characters


# --- prompts ----------------------------------------------------------------


def test_the_default_limit_is_1500_characters(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _TextDetector()

    span = chat(LONG)

    assert detector.seen == [LONG[-1500:]]
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(LONG),
        "evaluation.prompt.analyzed_length": 1500,
    }


def test_the_end_of_a_long_prompt_is_what_gets_read(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _TextDetector()
    prompt = "My card number is {0}. ".format(CARD) + FILLER * 20 + "What is my balance?"

    span = chat(prompt, evaluation_max_chars=200)

    assert detector.seen == [prompt[-200:]]
    assert detector.seen[0].endswith("What is my balance?")
    assert span.attributes["evaluation.pii.prompt.detected"] is False
    assert span.attributes["evaluation.truncated"] is True
    assert span.attributes["evaluation.prompt.length"] == len(prompt)
    assert span.attributes["evaluation.prompt.analyzed_length"] == 200


def test_a_prompt_within_the_limit_is_read_whole_and_carries_no_marker(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _TextDetector()
    prompt = FILLER * 3

    span = chat(prompt, evaluation_max_chars=len(prompt))

    assert detector.seen == [prompt]
    assert _markers(span) == {}


def test_zero_means_no_limit(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _TextDetector()

    span = chat(LONG, evaluation_max_chars=0)

    assert detector.seen == [LONG]
    assert _markers(span) == {}


def test_every_prompt_detector_reads_the_same_part(detectors, chat):
    readers = {}
    for slot in (
        "_toxicity_detector",
        "_bias_detector",
        "_prompt_injection_detector",
        "_restricted_topics_detector",
    ):
        readers[slot] = _TextDetector()
        setattr(BaseInstrumentor, slot, readers[slot])
    # Each stand-in answers in the PII shape; give the base what it reads from each.
    for reader in readers.values():
        reader.detect = _answering(reader)

    chat(LONG, evaluation_max_chars=300)

    assert {tuple(reader.seen) for reader in readers.values()} == {(LONG[-300:],)}


def _answering(reader):
    def detect(text):
        reader.seen.append(text)
        return SimpleNamespace(
            is_toxic=False,
            has_bias=False,
            is_injection=False,
            has_restricted_topic=False,
        )

    return detect


# --- PII: the patterns still cover the whole prompt -----------------------------


def test_the_pii_detector_gets_the_whole_prompt_and_the_limit(detectors, chat):
    """Its patterns then cover what its model does not read: a card number
    anywhere in a long prompt is still found."""
    BaseInstrumentor._pii_detector = detector = _CappingDetector()
    prompt = "My card number is {0}. ".format(CARD) + FILLER * 20

    span = chat(prompt, evaluation_max_chars=200)

    assert detector.calls == [(prompt, 200)]
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert span.attributes["evaluation.truncated"] is True
    assert span.attributes["evaluation.prompt.length"] == len(prompt)
    assert span.attributes["evaluation.prompt.analyzed_length"] == 200
    # What the model read, and that the patterns read all of it.
    assert span.attributes["evaluation.pii.prompt.model_analyzed_length"] == 197
    assert span.attributes["evaluation.pii.prompt.pattern_analyzed_length"] == len(prompt)


def test_a_short_prompt_goes_to_the_pii_detector_as_before(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _CappingDetector()

    span = chat("My card number is {0}.".format(CARD))

    assert detector.calls == [("My card number is {0}.".format(CARD), 0)]
    assert _markers(span) == {}


# --- responses ----------------------------------------------------------------


def test_the_start_of_a_long_response_is_what_gets_read(detectors, chat):
    BaseInstrumentor._hallucination_detector = detector = _ResponseDetector()
    response = "In 2019 exactly 97.3% agreed. " + FILLER * 20

    span = chat("Short question?", response=response, evaluation_max_chars=200)

    assert detector.calls == [(response[:200], "Short question?")]
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.response.length": len(response),
        "evaluation.response.analyzed_length": 200,
    }


def test_a_long_prompt_used_as_context_is_cut_and_marked_too(detectors, chat):
    BaseInstrumentor._hallucination_detector = detector = _ResponseDetector()

    span = chat(LONG, response="Short answer.", evaluation_max_chars=200)

    assert detector.calls == [("Short answer.", LONG[-200:])]
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(LONG),
        "evaluation.prompt.analyzed_length": 200,
    }


def test_a_long_response_nobody_reads_is_not_marked(detectors, chat):
    """Nothing was cut if nothing read it."""
    BaseInstrumentor._pii_detector = _TextDetector()

    span = chat("Short question?", response=FILLER * 40, evaluation_max_chars=200)

    assert _markers(span) == {}


def test_both_sides_cut_are_both_marked(detectors, chat):
    BaseInstrumentor._hallucination_detector = _ResponseDetector()
    response = FILLER * 30

    span = chat(LONG, response=response, evaluation_max_chars=100)

    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(LONG),
        "evaluation.prompt.analyzed_length": 100,
        "evaluation.response.length": len(response),
        "evaluation.response.analyzed_length": 100,
    }


# --- the pre-call block check -------------------------------------------------


def _block_on(policy):
    BaseInstrumentor._blocking_policies = frozenset({policy})
    BaseInstrumentor._blocking_active = True


def test_the_block_check_reads_the_same_part_of_the_prompt(detectors, chat):
    # Toxicity: a model-based detector, which the limit is there for. A block
    # decision by a pattern-based one reads the whole prompt; see
    # test_block_policy_coverage.py.
    BaseInstrumentor._toxicity_detector = detector = _TextDetector()
    detector.detect = lambda text: (
        detector.seen.append(text),
        SimpleNamespace(is_toxic="blocked phrase" in text),
    )[1]
    _block_on("toxicity")
    prompt = FILLER * 20 + "This ends with the blocked phrase."

    with pytest.raises(PolicyViolationError):
        chat(prompt, evaluation_max_chars=200)

    assert chat.calls == []
    assert detector.seen == [prompt[-200:]]
    (span,) = chat.spans()
    assert span.attributes["evaluation.blocked"] is True
    # A request refused on part of its prompt says which part was read.
    assert span.attributes["evaluation.truncated"] is True
    assert span.attributes["evaluation.prompt.length"] == len(prompt)
    assert span.attributes["evaluation.prompt.analyzed_length"] == 200


def test_the_block_check_gives_the_pii_detector_the_whole_prompt(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _CappingDetector()
    _block_on("pii")
    prompt = "My card number is {0}. ".format(CARD) + FILLER * 20

    with pytest.raises(PolicyViolationError):
        chat(prompt, evaluation_max_chars=200)

    assert chat.calls == []
    assert detector.calls == [(prompt, 200)]
    (span,) = chat.spans()
    assert span.attributes["evaluation.pii.prompt.blocked"] is True
    assert span.attributes["evaluation.pii.prompt.model_analyzed_length"] == 197
    assert span.attributes["evaluation.pii.prompt.pattern_analyzed_length"] == len(prompt)


def test_a_prompt_that_passes_the_block_check_is_asked_about_the_same_text_again(detectors, chat):
    """Same text both times, so the detector can answer the second from memory."""
    BaseInstrumentor._toxicity_detector = detector = _TextDetector()
    detector.detect = lambda text: (detector.seen.append(text), SimpleNamespace(is_toxic=False))[1]
    _block_on("toxicity")

    chat(LONG, evaluation_max_chars=200)

    assert detector.seen == [LONG[-200:], LONG[-200:]]


# --- with the real pattern-based detectors: one analysis per call ---------------


def test_with_a_block_policy_the_prompt_is_analysed_once_not_twice(detectors, chat):
    """The block check analyses the prompt before the request is sent. The pass
    that annotates the span afterwards used to analyse it all over again."""
    detector = BiasDetector(BiasConfig(enabled=True, block_on_detection=True))
    BaseInstrumentor._bias_detector = detector
    _block_on("bias")
    caller = threading.get_ident()
    work = []
    analyse = detector._check_bias_type

    def counted(text, bias_type):
        work.append(threading.get_ident())
        return analyse(text, bias_type)

    with patch.object(detector, "_check_bias_type", side_effect=counted):
        span = chat("Please summarise the attached meeting notes.")

    one_pass = len(detector.config.bias_types)
    assert len(work) == one_pass
    assert set(work) == {caller}  # the block check, on the caller's thread
    assert span.attributes["evaluation.bias.prompt.detected"] is False
    assert span.attributes["evaluation.completed"] is True


def test_a_repeated_request_is_analysed_once(detectors, chat):
    detector = HallucinationDetector(HallucinationConfig(enabled=True))
    BaseInstrumentor._hallucination_detector = detector
    count = detector._count_factual_claims

    with patch.object(detector, "_count_factual_claims", side_effect=count) as work:
        first = chat("What happened?", response="In 2019 exactly 97.3% agreed.")
        second = chat("What happened?", response="In 2019 exactly 97.3% agreed.")

    assert work.call_count == 1
    assert (
        second.attributes["evaluation.hallucination.response.score"]
        == first.attributes["evaluation.hallucination.response.score"]
    )


# --- PII in redact mode: nothing is cut for it -----------------------------------
#
# A redacted copy is exported in place of the prompt, so the model reads all of
# the prompt whatever the limit. The span must not say that PII detection read
# part of it, because it did not.


class _RedactingDetector(_CappingDetector):
    """Stands in for the PII detector in redact mode."""

    reads_whole_text = True


def test_a_redacting_pii_detector_is_given_the_whole_prompt_and_no_limit(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _RedactingDetector()
    prompt = "My card number is {0}. ".format(CARD) + FILLER * 20

    span = chat(prompt, evaluation_max_chars=200)

    assert detector.calls == [(prompt, 0)]
    assert span.attributes["evaluation.pii.prompt.detected"] is True


def test_a_span_does_not_say_truncated_when_only_a_redacting_pii_detector_read_it(detectors, chat):
    BaseInstrumentor._pii_detector = _RedactingDetector()

    span = chat("My card number is {0}. ".format(CARD) + FILLER * 20, evaluation_max_chars=200)

    assert _markers(span) == {}


def test_another_detector_s_cut_is_still_marked_next_to_a_redacting_pii_detector(detectors, chat):
    BaseInstrumentor._pii_detector = _RedactingDetector()
    BaseInstrumentor._toxicity_detector = reader = _TextDetector()
    reader.detect = _answering(reader)
    prompt = "My card number is {0}. ".format(CARD) + FILLER * 20

    span = chat(prompt, evaluation_max_chars=200)

    assert reader.seen == [prompt[-200:]]
    # The cut is the toxicity detector's. Nothing says the PII model read less.
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(prompt),
        "evaluation.prompt.analyzed_length": 200,
        "evaluation.toxicity.prompt.analyzed_length": 200,
    }


def test_a_block_check_by_a_detector_that_reads_everything_marks_nothing(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _RedactingDetector()
    _block_on("pii")
    prompt = FILLER * 20 + "Nothing to see here."

    span = chat(prompt, evaluation_max_chars=200)

    assert detector.calls[0] == (prompt, 0)
    assert _markers(span) == {}


def test_the_same_holds_when_detectors_run_inline(detectors, chat):
    BaseInstrumentor._pii_detector = detector = _RedactingDetector()
    prompt = "My card number is {0}. ".format(CARD) + FILLER * 20

    span = chat(prompt, evaluation_max_chars=200, evaluation_mode="inline")

    assert detector.calls == [(prompt, 0)]
    assert _markers(span) == {}


# --- toxicity: a limit of its own ---------------------------------------------------
#
# The local toxicity model is the slowest detector by far, so by default it reads
# less than the others: the last 500 characters of the prompt. The span says so
# whenever that is not the whole prompt.


def _toxicity_reader():
    reader = _TextDetector()
    reader.detect = _answering(reader)
    BaseInstrumentor._toxicity_detector = reader
    return reader


def test_toxicity_reads_the_last_500_characters_by_default(detectors, chat):
    reader = _toxicity_reader()

    span = chat(LONG)

    assert reader.seen == [LONG[-500:]]
    # Nothing read 1500 characters, so nothing says that anything did.
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(LONG),
        "evaluation.toxicity.prompt.analyzed_length": 500,
    }


def test_a_prompt_the_general_limit_did_not_cut_is_still_marked_when_toxicity_cut_it(
    detectors, chat
):
    reader = _toxicity_reader()
    prompt = (FILLER * 20)[:1000]

    span = chat(prompt)

    assert reader.seen == [prompt[-500:]]
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": 1000,
        "evaluation.toxicity.prompt.analyzed_length": 500,
    }


def test_a_prompt_within_the_toxicity_limit_carries_no_marker(detectors, chat):
    reader = _toxicity_reader()
    prompt = (FILLER * 20)[:500]

    span = chat(prompt)

    assert reader.seen == [prompt]
    assert _markers(span) == {}


def test_toxicity_and_another_detector_each_read_their_own_part(detectors, chat):
    reader = _toxicity_reader()
    BaseInstrumentor._bias_detector = other = _TextDetector()
    other.detect = _answering(other)

    span = chat(LONG)

    assert reader.seen == [LONG[-500:]]
    assert other.seen == [LONG[-1500:]]
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(LONG),
        "evaluation.prompt.analyzed_length": 1500,
        "evaluation.toxicity.prompt.analyzed_length": 500,
    }


@pytest.mark.parametrize(
    "general, toxicity, read",
    [
        (1500, 500, 500),  # the defaults
        (1500, 300, 300),
        (200, 500, 200),  # the smaller of the two
        (0, 500, 500),  # no general limit does not lift the toxicity limit
        (1500, 0, 1500),  # no limit of its own: the general one applies
        (0, 0, None),  # no limit at all
    ],
)
def test_the_toxicity_limit_is_the_smaller_of_its_own_and_the_general_one(
    detectors, chat, general, toxicity, read
):
    reader = _toxicity_reader()

    span = chat(LONG, evaluation_max_chars=general, toxicity_max_chars=toxicity)

    if read is None:
        assert reader.seen == [LONG]
        assert _markers(span) == {}
    else:
        assert reader.seen == [LONG[-read:]]
        assert span.attributes["evaluation.truncated"] is True
        assert span.attributes["evaluation.prompt.length"] == len(LONG)
        assert span.attributes["evaluation.toxicity.prompt.analyzed_length"] == read


def test_toxicity_reading_the_general_part_is_marked_under_both_names(detectors, chat):
    reader = _toxicity_reader()

    span = chat(LONG, toxicity_max_chars=0)

    assert reader.seen == [LONG[-1500:]]
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(LONG),
        "evaluation.prompt.analyzed_length": 1500,
        "evaluation.toxicity.prompt.analyzed_length": 1500,
    }


def test_the_toxicity_block_check_reads_the_part_the_later_pass_reads(detectors, chat):
    """Same text both times, so the detector answers the second from memory."""
    reader = _toxicity_reader()
    _block_on("toxicity")

    span = chat(LONG)

    assert reader.seen == [LONG[-500:], LONG[-500:]]
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(LONG),
        "evaluation.toxicity.prompt.analyzed_length": 500,
    }


def test_a_request_toxicity_refused_says_how_much_of_it_was_read(detectors, chat):
    BaseInstrumentor._toxicity_detector = detector = _TextDetector()
    detector.detect = lambda text: (
        detector.seen.append(text),
        SimpleNamespace(is_toxic="blocked phrase" in text),
    )[1]
    _block_on("toxicity")
    prompt = FILLER * 20 + "This ends with the blocked phrase."

    with pytest.raises(PolicyViolationError):
        chat(prompt)

    assert chat.calls == []
    assert detector.seen == [prompt[-500:]]
    (span,) = chat.spans()
    assert span.attributes["evaluation.blocked"] is True
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(prompt),
        "evaluation.toxicity.prompt.analyzed_length": 500,
    }


def test_the_toxicity_limit_holds_when_detectors_run_inline(detectors, chat):
    reader = _toxicity_reader()

    span = chat(LONG, evaluation_mode="inline")

    assert reader.seen == [LONG[-500:]]
    assert span.attributes["evaluation.toxicity.prompt.analyzed_length"] == 500


# --- a hardened profile: nothing is left unread ---------------------------------------


def _every_prompt_reader():
    readers = {}
    for slot in (
        "_toxicity_detector",
        "_bias_detector",
        "_prompt_injection_detector",
        "_restricted_topics_detector",
    ):
        readers[slot] = _TextDetector()
        readers[slot].detect = _answering(readers[slot])
        setattr(BaseInstrumentor, slot, readers[slot])
    BaseInstrumentor._pii_detector = readers["_pii_detector"] = _CappingDetector()
    return readers


@pytest.mark.parametrize("profile", ["strict", "bfsi", "bank"])
def test_under_a_hardened_profile_every_detector_reads_the_whole_prompt(detectors, chat, profile):
    readers = _every_prompt_reader()

    span = chat(LONG, profile=profile)

    pii = readers.pop("_pii_detector")
    assert pii.calls == [(LONG, 0)]  # the whole prompt, and no limit for the model
    assert {tuple(reader.seen) for reader in readers.values()} == {(LONG,)}
    assert _markers(span) == {}


def test_under_a_hardened_profile_a_limit_the_operator_set_still_applies(detectors, chat):
    readers = _every_prompt_reader()

    span = chat(LONG, profile="bank", evaluation_max_chars=200)

    pii = readers.pop("_pii_detector")
    assert pii.calls == [(LONG, 200)]
    # Toxicity has no limit of its own under the profile, so the general one is its limit.
    assert {tuple(reader.seen) for reader in readers.values()} == {(LONG[-200:],)}
    assert span.attributes["evaluation.prompt.analyzed_length"] == 200


def test_under_a_hardened_profile_a_toxicity_limit_the_operator_set_still_applies(detectors, chat):
    readers = _every_prompt_reader()

    span = chat(LONG, profile="bank", toxicity_max_chars=300)

    assert readers["_toxicity_detector"].seen == [LONG[-300:]]
    assert readers["_bias_detector"].seen == [LONG]
    assert _markers(span) == {
        "evaluation.truncated": True,
        "evaluation.prompt.length": len(LONG),
        "evaluation.toxicity.prompt.analyzed_length": 300,
    }
