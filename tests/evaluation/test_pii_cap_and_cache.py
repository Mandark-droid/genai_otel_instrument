"""PII detection on long text, its result cache, and the spaCy pipeline it runs.

The NER model is the expensive part and its cost grows with the text, so on a
long prompt it reads the end only. The patterns that need no model (card numbers,
e-mail addresses, national identifiers) are cheap and still cover all of it.

No model is loaded here: a stand-in plays the Presidio analyzer.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.evaluation.config import PIIConfig, PIIEntityType, PIIMode
from genai_otel.evaluation.pii_detector import PIIDetector

CARD = "4111 1111 1111 1111"
EMAIL = "jane.example@example.com"
NAME = "Jane Example"
FILLER = "The statement is generated a few days before the end of each billing cycle. "


class _Pipeline:
    """Stands in for a spaCy pipeline: a list of components that can be disabled."""

    def __init__(self, pipes, can_disable=True):
        self.pipe_names = list(pipes)
        self.disabled = []
        self._can_disable = can_disable

    def disable_pipe(self, name):
        if not self._can_disable:
            raise ValueError("cannot disable %s" % name)
        self.pipe_names.remove(name)
        self.disabled.append(name)


class _Analyzer:
    """Stands in for Presidio's AnalyzerEngine: reports each needle it is shown."""

    def __init__(self, needles=(), pipeline=None, fail_times=0):
        self.needles = list(needles)  # (entity type, text, score)
        self.seen = []
        self.fail_times = fail_times
        self.registry = MagicMock()
        self.nlp_engine = SimpleNamespace(
            nlp={"en": pipeline or _Pipeline(["tok2vec", "tagger", "parser", "ner"])}
        )

    def analyze(self, text, language, entities, score_threshold, allow_list):
        self.seen.append(text)
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("model not loaded")
        found = []
        for entity_type, needle, score in self.needles:
            start = text.find(needle)
            if start >= 0 and score >= score_threshold:
                found.append(
                    SimpleNamespace(
                        entity_type=entity_type, start=start, end=start + len(needle), score=score
                    )
                )
        return found


def _detector(needles=(), config=None, **analyzer_kwargs):
    with patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability"):
        detector = PIIDetector(config or PIIConfig(enabled=True))
    detector._analyzer = _Analyzer(needles, **analyzer_kwargs)
    detector._presidio_available = True
    return detector


def _found(result, entity_type):
    return [e for e in result.entities if e["type"] == entity_type]


# --- the result cache ---------------------------------------------------------


def test_text_already_analysed_is_not_analysed_again():
    detector = _detector([("PERSON", NAME, 0.85)])
    text = "Please write to %s about the invoice." % NAME

    first = detector.detect(text)
    second = detector.detect(text)

    assert len(detector._analyzer.seen) == 1
    assert second.has_pii is True
    assert second.entities == first.entities
    assert second.entity_counts == {"PERSON": 1}
    assert second.score == first.score


def test_a_remembered_result_still_carries_the_text_it_was_asked_about():
    detector = _detector()
    text = "Nothing personal in here."
    detector.detect(text)
    assert detector.detect(text).original_text == text


def test_the_cache_does_not_hold_on_to_the_text():
    detector = _detector()
    text = "A long internal memo that nobody else should keep a copy of."
    detector.detect(text)
    assert len(detector._cache) == 1
    assert text not in repr(detector._cache._entries)


def test_a_failed_analysis_is_not_remembered():
    """A detector that could not run answers "nothing found". Remembering that
    would keep answering it after the fault has cleared."""
    detector = _detector([("PERSON", NAME, 0.85)], fail_times=1)
    text = "Please write to %s." % NAME

    assert detector.detect(text).has_pii is False
    assert detector.detect(text).has_pii is True
    assert len(detector._analyzer.seen) == 2


def test_a_changed_setting_is_not_answered_from_the_cache():
    detector = _detector([("PERSON", NAME, 0.85)])
    text = "Please write to %s." % NAME
    assert detector.detect(text).has_pii is True

    detector.config.threshold = 0.9

    assert detector.detect(text).has_pii is False
    assert len(detector._analyzer.seen) == 2


def test_each_language_is_analysed_separately():
    detector = _detector()
    detector.detect("Bonjour.", language="en")
    detector.detect("Bonjour.", language="fr")
    assert len(detector._analyzer.seen) == 2


def test_the_cache_can_be_turned_off():
    detector = _detector(config=PIIConfig(enabled=True, cache_size=0))
    detector.detect("Same text.")
    detector.detect("Same text.")
    assert len(detector._analyzer.seen) == 2


def _strings_in(value, seen=None):
    """Every string reachable from ``value``."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for pair in value.items() for part in pair for s in _strings_in(part, seen)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [s for part in value for s in _strings_in(part, seen)]
    if hasattr(value, "__dict__"):
        return _strings_in(vars(value), seen)
    return []


@pytest.mark.parametrize("mode", [PIIMode.DETECT, PIIMode.REDACT, PIIMode.BLOCK])
def test_the_cache_keeps_nothing_that_was_found(mode):
    """An entry outlives its request. What was found in the text is the part of
    it least fit to keep: a name, a card number, the copy with them starred out."""
    detector = _detector(
        [("PERSON", NAME, 0.85), ("CREDIT_CARD", CARD, 1.0)],
        config=PIIConfig(enabled=True, mode=mode, content_max_length=0),
    )
    text = "Please charge %s for %s and write to her." % (CARD, NAME)

    detector.detect(text)

    kept = _strings_in(list(detector._cache._entries.values()))
    assert len(detector._cache) == 1
    for secret in (NAME, CARD, "Jane", "4111", "Please charge", "write to her"):
        assert not [s for s in kept if secret in s], secret


@pytest.mark.parametrize("mode", [PIIMode.DETECT, PIIMode.REDACT, PIIMode.BLOCK])
def test_a_remembered_result_is_rebuilt_from_the_text_it_is_asked_about(mode):
    detector = _detector(
        [("PERSON", NAME, 0.85), ("CREDIT_CARD", CARD, 1.0)],
        config=PIIConfig(enabled=True, mode=mode, content_max_length=0),
    )
    text = "Please charge %s for %s and write to her." % (CARD, NAME)

    first = detector.detect(text)
    second = detector.detect(text)

    assert len(detector._analyzer.seen) == 1
    assert second == first
    assert sorted(e["text"] for e in second.entities) == [CARD, NAME]
    if mode == PIIMode.REDACT:
        assert second.redacted_text == "Please charge ******** for ******** and write to her."


def test_what_a_caller_does_to_its_result_does_not_reach_the_next_caller():
    detector = _detector([("PERSON", NAME, 0.85)])
    text = "Please write to %s about the invoice." % NAME

    first = detector.detect(text)
    first.entities.clear()
    first.entity_counts.clear()
    second = detector.detect(text)
    second.entities[0]["type"] = "CHANGED"
    second.entity_counts["PERSON"] = 99
    third = detector.detect(text)

    assert len(detector._analyzer.seen) == 1
    assert [e["type"] for e in third.entities] == ["PERSON"]
    assert third.entity_counts == {"PERSON": 1}
    assert third.has_pii is True


def test_a_remembered_pattern_only_result_is_rebuilt_too():
    with patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability"):
        detector = PIIDetector(PIIConfig(enabled=True, mode=PIIMode.REDACT, content_max_length=0))
    detector._presidio_available = False
    text = "Card %s on file." % CARD

    first = detector.detect(text)
    kept = _strings_in(list(detector._cache._entries.values()))
    with patch.object(detector, "_fallback_detection", side_effect=AssertionError("ran again")):
        second = detector.detect(text)

    assert not [s for s in kept if "4111" in s]
    assert second == first
    assert CARD not in second.redacted_text


def test_pattern_only_detection_is_remembered_too():
    with patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability"):
        detector = PIIDetector(PIIConfig(enabled=True))
    detector._presidio_available = False
    text = "Card %s." % CARD
    first = detector.detect(text)
    with patch.object(detector, "_fallback_detection", side_effect=AssertionError("ran again")):
        second = detector.detect(text)
    assert second.entities == first.entities


# --- the spaCy pipeline ---------------------------------------------------------


def _detector_with_real_setup(analyzer):
    """Run the detector's own start-up against stand-in Presidio modules."""
    presidio_analyzer = MagicMock()
    presidio_analyzer.AnalyzerEngine = MagicMock(return_value=analyzer)
    modules = {"presidio_analyzer": presidio_analyzer, "presidio_anonymizer": MagicMock()}
    with patch.dict(sys.modules, modules):
        return PIIDetector(PIIConfig(enabled=True))


def test_the_dependency_parser_is_switched_off():
    """Nothing reads the parser's output; it only costs time on every text."""
    pipeline = _Pipeline(["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    detector = _detector_with_real_setup(_Analyzer(pipeline=pipeline))

    assert detector.is_available()
    assert pipeline.disabled == ["parser"]
    assert pipeline.pipe_names == ["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "ner"]


def test_a_pipeline_without_a_parser_is_left_alone():
    pipeline = _Pipeline(["tok2vec", "ner"])
    detector = _detector_with_real_setup(_Analyzer(pipeline=pipeline))

    assert detector.is_available()
    assert pipeline.disabled == []


def test_a_parser_that_cannot_be_disabled_does_not_stop_detection():
    pipeline = _Pipeline(["tok2vec", "parser", "ner"], can_disable=False)
    detector = _detector_with_real_setup(_Analyzer(pipeline=pipeline))

    assert detector.is_available()
    assert "parser" in pipeline.pipe_names


def test_an_nlp_engine_that_is_not_spacy_is_left_alone():
    analyzer = _Analyzer()
    analyzer.nlp_engine = SimpleNamespace()  # no spaCy pipelines to look at
    assert _detector_with_real_setup(analyzer).is_available()


# --- long text: the model reads the end, the patterns read everything -----------


def test_the_detector_says_it_can_limit_the_model():
    assert PIIDetector.accepts_model_chars is True


def test_text_within_the_limit_goes_to_the_model_whole():
    detector = _detector()
    text = FILLER * 3
    result = detector.detect(text, model_chars=len(text))
    assert detector._analyzer.seen == [text]
    assert result.model_analyzed_length is None


def test_no_limit_means_the_model_reads_everything():
    detector = _detector()
    text = FILLER * 50
    result = detector.detect(text)
    assert detector._analyzer.seen == [text]
    assert result.model_analyzed_length is None


def test_on_long_text_the_model_reads_only_the_end():
    detector = _detector()
    text = FILLER * 50

    result = detector.detect(text, model_chars=300)

    (seen,) = detector._analyzer.seen
    assert len(seen) <= 300
    assert text.endswith(seen)
    # It starts on a word, not in the middle of one.
    assert text[len(text) - len(seen) - 1].isspace()
    assert result.model_analyzed_length == len(seen)


def test_a_card_number_before_the_limit_is_still_found():
    detector = _detector()
    text = "My card is %s. " % CARD + FILLER * 50

    result = detector.detect(text, model_chars=300)

    (card,) = _found(result, "CREDIT_CARD")
    assert text[card["start"] : card["end"]] == CARD
    assert card["text"] == CARD
    assert result.has_pii is True
    assert result.entity_counts["CREDIT_CARD"] == 1
    assert CARD not in detector._analyzer.seen[0]


def test_an_email_address_before_the_limit_is_still_found():
    detector = _detector()
    text = "Reply to %s please. " % EMAIL + FILLER * 50
    (email,) = _found(detector.detect(text, model_chars=300), "EMAIL_ADDRESS")
    assert text[email["start"] : email["end"]] == EMAIL


def test_what_the_model_finds_is_reported_at_its_place_in_the_whole_text():
    detector = _detector([("PERSON", NAME, 0.85)])
    text = FILLER * 50 + "Yours, %s." % NAME

    result = detector.detect(text, model_chars=300)

    (person,) = _found(result, "PERSON")
    assert text[person["start"] : person["end"]] == NAME
    assert person["text"] == NAME


def test_a_name_before_the_limit_is_not_looked_for():
    """Names need the model, and the model reads the end only."""
    detector = _detector([("PERSON", NAME, 0.85)])
    text = "From %s. " % NAME + FILLER * 50
    result = detector.detect(text, model_chars=300)
    assert _found(result, "PERSON") == []
    assert result.model_analyzed_length is not None


def test_a_number_that_straddles_the_limit_is_found_once():
    detector = _detector()
    tail = "x" * 290
    text = FILLER * 20 + "card %s %s" % (CARD, tail)
    # The limit falls inside the card number.
    cut_inside_card = len(tail) + 1 + len(CARD) // 2

    result = detector.detect(text, model_chars=cut_inside_card)

    (card,) = _found(result, "CREDIT_CARD")
    assert text[card["start"] : card["end"]] == CARD


def test_patterns_do_not_report_again_what_the_model_part_covers():
    detector = _detector([("EMAIL_ADDRESS", EMAIL, 1.0)])
    text = FILLER * 50 + "Reply to %s." % EMAIL
    result = detector.detect(text, model_chars=300)
    assert len(_found(result, "EMAIL_ADDRESS")) == 1


def test_pattern_findings_respect_the_threshold():
    config = PIIConfig(
        enabled=True,
        threshold=0.92,
        entity_types={PIIEntityType.IP_ADDRESS, PIIEntityType.US_SSN},
    )
    detector = _detector(config=config)
    text = "Host 203.0.113.7, SSN 536-22-8726. " + FILLER * 50

    result = detector.detect(text, model_chars=300)

    assert _found(result, "IP_ADDRESS") == []  # the address pattern scores below 0.92
    assert len(_found(result, "US_SSN")) == 1


def test_pattern_findings_respect_the_allow_list():
    config = PIIConfig(enabled=True, allow_list=[EMAIL])
    detector = _detector(config=config)
    text = "Reply to %s or to other@example.org. " % EMAIL + FILLER * 50

    result = detector.detect(text, model_chars=300)

    assert [e["text"] for e in _found(result, "EMAIL_ADDRESS")] == ["other@example.org"]


def test_redaction_covers_the_whole_text():
    config = PIIConfig(enabled=True, mode=PIIMode.REDACT, content_max_length=0)
    detector = _detector([("CREDIT_CARD", CARD, 1.0), ("PERSON", NAME, 0.85)], config=config)
    text = "My card is %s. " % CARD + FILLER * 50 + "Yours, %s." % NAME

    result = detector.detect(text, model_chars=300)

    assert CARD not in result.redacted_text
    assert NAME not in result.redacted_text
    assert result.redacted_text.startswith("My card is ********. ")
    assert result.redacted_text.endswith("Yours, ********.")
    assert FILLER in result.redacted_text


# --- redact mode: the model reads everything -------------------------------------
#
# A redacted copy is exported in place of the text. Built from text the model did
# not read, it would carry every name in that text and still be called redacted.


def _redacting(needles=(), **config):
    return _detector(
        needles, config=PIIConfig(enabled=True, mode=PIIMode.REDACT, cache_size=0, **config)
    )


def test_in_redact_mode_the_model_reads_the_whole_text_whatever_the_limit():
    detector = _redacting([("PERSON", NAME, 0.85)])
    text = "From %s. " % NAME + FILLER * 50

    result = detector.detect(text, model_chars=300)

    assert detector._analyzer.seen == [text]
    assert result.model_analyzed_length is None
    assert NAME not in result.redacted_text
    assert result.redacted_text.startswith("From ********. ")


def test_redact_mode_says_that_it_reads_everything():
    assert _redacting().reads_whole_text is True
    assert _detector(config=PIIConfig(enabled=True, mode=PIIMode.DETECT)).reads_whole_text is False
    assert _detector(config=PIIConfig(enabled=True, mode=PIIMode.BLOCK)).reads_whole_text is False


def test_in_redact_mode_a_limit_does_not_make_a_second_cache_entry():
    detector = _detector(
        [("PERSON", NAME, 0.85)], config=PIIConfig(enabled=True, mode=PIIMode.REDACT)
    )
    text = "From %s. " % NAME + FILLER * 50

    detector.detect(text, model_chars=300)
    detector.detect(text)

    assert len(detector._analyzer.seen) == 1
    assert len(detector._cache) == 1


def test_a_redacted_copy_is_never_the_text_itself_when_the_anonymizer_fails():
    """It used to hand the text back unchanged, and that went out labelled redacted."""
    detector = _redacting([("PERSON", NAME, 0.85)], content_max_length=0)
    detector._anonymizer = MagicMock()
    detector._anonymizer.anonymize.side_effect = RuntimeError("operator table missing")
    text = "From %s, about the invoice." % NAME

    result = detector.detect(text)

    assert result.redacted_text == "From ********, about the invoice."


# --- the part the model does not read: what the patterns may report --------------
#
# Presidio validates what its patterns find: a card number has to pass the
# checksum, an e-mail address has to end in a real top-level domain, and a phone
# number scores below the default threshold unless the words around it say what
# it is. Unvalidated, the same patterns refuse order numbers and timestamps.


def _before_the_limit(record, **config):
    detector = _detector(config=PIIConfig(enabled=True, cache_size=0, **config))
    text = record + " " + FILLER * 50
    result = detector.detect(text, model_chars=300)
    return [(e["type"], e["text"], e["score"]) for e in result.entities]


@pytest.mark.parametrize(
    "number",
    [
        "5830219947120386",  # sixteen digits that fail the checksum: an order number
        "4111 1111 1111 1112",
        "9999 9999 9999 9995",  # passes the checksum; no card starts with a 9
        "0000 0000 0000 0000",
        "2223 0000 4840 0011",
    ],
)
def test_sixteen_digits_that_presidio_would_not_call_a_card_are_not_reported(number):
    found = _before_the_limit("order %s shipped" % number)
    assert [f for f in found if f[0] == "CREDIT_CARD"] == []


@pytest.mark.parametrize(
    "number", [CARD, "4111-1111-1111-1111", "4111111111111111", "5555 5555 5555 4444"]
)
def test_a_card_number_is_reported_with_the_score_of_a_validated_one(number):
    found = _before_the_limit("my card is %s" % number)
    assert ("CREDIT_CARD", number, 1.0) in found


@pytest.mark.parametrize("number", ["1763418822", "2048113377", "415-555-0132", "415.555.0132"])
def test_a_phone_shaped_number_is_not_reported_at_the_default_threshold(number):
    """Presidio gives a phone number 0.4 until the words around it support it,
    and those words need the model."""
    assert _before_the_limit("ts=%s level=info" % number) == []


def test_a_phone_shaped_number_is_reported_to_whoever_lowers_the_threshold():
    found = _before_the_limit("ts=1763418822 level=info", threshold=0.4)
    assert ("PHONE_NUMBER", "1763418822", 0.4) in found


@pytest.mark.parametrize(
    "number",
    [
        "000-12-3456",
        "666-12-3456",
        "123-45-6789",
        "078-05-1120",
        "111-11-1111",
        "536-00-8726",
        "536-22-0000",
    ],
)
def test_a_number_that_cannot_be_a_social_security_number_is_not_reported(number):
    found = _before_the_limit("reference %s" % number)
    assert [f for f in found if f[0] == "US_SSN"] == []


def test_a_social_security_number_is_reported():
    assert ("US_SSN", "536-22-8726", 0.95) in _before_the_limit("reference 536-22-8726")


@pytest.mark.parametrize("address", ["999.1.1.1", "256.100.50.25", "1.2.3.400"])
def test_dotted_numbers_that_are_not_an_ip_address_are_not_reported(address):
    found = _before_the_limit("build %s done" % address)
    assert [f for f in found if f[0] == "IP_ADDRESS"] == []


def test_an_ip_address_is_reported():
    assert ("IP_ADDRESS", "203.0.113.7", 0.9) in _before_the_limit("from 203.0.113.7 today")


class _Suffixes:
    """Stands in for the public suffix list: knows ``org`` and ``co.uk``."""

    def __init__(self, **options):
        self.options = options

    def __call__(self, address):
        domain = address.rsplit("@", 1)[-1]
        known = domain.endswith((".org", ".co.uk"))
        return SimpleNamespace(fqdn=domain if known else "")


@pytest.fixture
def suffix_list(monkeypatch):
    from genai_otel.evaluation import pii_detector

    created = []

    def extractor(**options):
        created.append(options)
        return _Suffixes(**options)

    monkeypatch.setitem(sys.modules, "tldextract", SimpleNamespace(TLDExtract=extractor))
    monkeypatch.setattr(pii_detector, "_SUFFIX_LIST", None)
    return created


@pytest.mark.parametrize("name", ["icon@2x.png", "svc@cluster.local", "root@localhost.localdomain"])
def test_a_name_with_an_at_sign_that_is_not_an_address_is_not_reported(suffix_list, name):
    assert _before_the_limit("see %s for details" % name) == []


def test_an_address_is_reported_with_the_score_of_a_validated_one(suffix_list):
    found = _before_the_limit("write to jane.roe@example.org or j@example.co.uk")
    assert found == [
        ("EMAIL_ADDRESS", "jane.roe@example.org", 1.0),
        ("EMAIL_ADDRESS", "j@example.co.uk", 1.0),
    ]


def test_the_suffix_list_is_read_from_the_package_never_fetched(suffix_list):
    """Looking an address up must not be what sends a request out of the process."""
    _before_the_limit("write to jane.roe@example.org")
    _before_the_limit("write to jane.roe@example.org")
    assert suffix_list == [{"cache_dir": None, "suffix_list_urls": ()}]


def test_without_a_suffix_list_an_address_is_reported_on_its_shape(monkeypatch):
    from genai_otel.evaluation import pii_detector

    monkeypatch.setitem(sys.modules, "tldextract", None)
    monkeypatch.setattr(pii_detector, "_SUFFIX_LIST", None)
    assert _before_the_limit("see icon@2x.png") == [("EMAIL_ADDRESS", "icon@2x.png", 0.9)]


def test_national_identifiers_are_reported_as_the_model_path_reports_them():
    """These recognisers are this library's own, so both paths already agree."""
    found = _before_the_limit("PAN ABCDE1234F, IFSC HDFC0001234, UPI rahul.k@oksbi, 9876543210")
    assert ("IN_PAN", "ABCDE1234F", 0.9) in found
    assert ("IN_IFSC", "HDFC0001234", 0.9) in found
    assert ("IN_UPI", "rahul.k@oksbi", 0.9) in found
    assert ("IN_PHONE", "9876543210", 0.8) in found


def test_without_presidio_the_patterns_report_what_they_always_did():
    """Validation mirrors Presidio; where Presidio is not installed there is
    nothing to mirror, and the fallback stays as it was."""
    with patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability"):
        detector = PIIDetector(PIIConfig(enabled=True, cache_size=0))
    detector._presidio_available = False

    result = detector.detect("order 5830219947120386, ts=1763418822, ref 123-45-6789")

    found = {(e["type"], e["text"], e["score"]) for e in result.entities}
    assert ("CREDIT_CARD", "5830219947120386", 0.85) in found
    assert ("PHONE_NUMBER", "1763418822", 0.8) in found
    assert ("US_SSN", "123-45-6789", 0.95) in found


def test_block_mode_blocks_on_a_finding_before_the_limit():
    config = PIIConfig(enabled=True, mode=PIIMode.BLOCK)
    detector = _detector(config=config)
    text = "My card is %s. " % CARD + FILLER * 50
    assert detector.detect(text, model_chars=300).blocked is True


def test_a_capped_result_is_remembered_separately_from_an_uncapped_one():
    detector = _detector([("PERSON", NAME, 0.85)])
    text = "From %s. " % NAME + FILLER * 50

    assert _found(detector.detect(text, model_chars=300), "PERSON") == []
    assert len(_found(detector.detect(text), "PERSON")) == 1
    detector.detect(text, model_chars=300)

    assert len(detector._analyzer.seen) == 2


def test_without_the_model_the_patterns_read_everything():
    with patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability"):
        detector = PIIDetector(PIIConfig(enabled=True))
    detector._presidio_available = False
    text = "My card is %s. " % CARD + FILLER * 50

    result = detector.detect(text, model_chars=300)

    assert len(_found(result, "CREDIT_CARD")) == 1
    assert result.model_analyzed_length is None


@pytest.mark.parametrize("model_chars", [0, -1, None])
def test_a_limit_that_is_not_positive_is_no_limit(model_chars):
    detector = _detector()
    text = FILLER * 50
    detector.detect(text, model_chars=model_chars)
    assert detector._analyzer.seen == [text]
