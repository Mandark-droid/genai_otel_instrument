"""The pattern-based detectors take time in proportion to the text, whatever it is.

A regular expression that retries from every position of a long run (a dotted
package list, a base64 blob, a table rule made of hyphens) takes time in
proportion to the square of the text and does not let any other thread run
meanwhile. A block policy reads the whole prompt, and so do the PII patterns, so
one long prompt of the wrong shape used to stop the process for seconds.

Two things are checked for every pattern that was rewritten: that a pass over
hostile text grows no faster than the text does, and that the pattern still
finds exactly what it found before. The expressions as they were are kept here
to compare against.
"""

import base64
import itertools
import random
import re
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.evaluation import pii_detector
from genai_otel.evaluation.bias_detector import BiasDetector
from genai_otel.evaluation.config import (
    BiasConfig,
    HallucinationConfig,
    PIIConfig,
    PIIEntityType,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
)
from genai_otel.evaluation.hallucination_detector import HallucinationDetector
from genai_otel.evaluation.pii_detector import PIIDetector
from genai_otel.evaluation.prompt_injection_detector import PromptInjectionDetector
from genai_otel.evaluation.restricted_topics_detector import RestrictedTopicsDetector

SMALL = 4000
FACTOR = 4


def _repeat(unit, n):
    return (unit * (n // len(unit) + 1))[:n]


def _blob(n):
    rng = random.Random(5)
    return base64.b64encode(bytes(rng.getrandbits(8) for _ in range(n))).decode()[:n]


# Text shapes that made at least one expression retry from every position.
SHAPES = {
    "one long word": lambda n: "a" * n,
    "one long number": lambda n: "7" * n,
    "spaces": lambda n: " " * n,
    "newlines": lambda n: "\n" * n,
    "a rule of hyphens": lambda n: "-" * n,
    "backticks": lambda n: "`" * n,
    "dotted names": lambda n: _repeat("pkg1.", n),
    "hyphenated ids": lambda n: _repeat("a1b2-", n),
    "underscores and dots": lambda n: _repeat("_.", n),
    "base64": _blob,
    "comma separated numbers": lambda n: _repeat("123,", n),
    "dotted numbers": lambda n: _repeat("12.", n),
    "amount then digits": lambda n: "$1,111." + "1" * n,
    "addresses without a domain": lambda n: _repeat("a@", n),
    "dotted names then an at sign": lambda n: _repeat("pkg1.", n - 2) + "@x",
    "underscores and dots then an at sign": lambda n: _repeat("_.", n - 2) + "@x",
    "an at sign after every run of punctuation": lambda n: _repeat("-1" * 120 + "@", n),
    "ignore then nothing to ignore": lambda n: "ignore " + _repeat("the ", n),
    "should then blank space": lambda n: "you should" + " " * n,
    "is then blank lines": lambda n: "a is" + "\n" * n,
    "claims in one endless sentence": lambda n: _repeat("in 2019 ", n),
    "different claims in one endless sentence": lambda n: " ".join(
        "in %d" % (1000 + i) for i in range(n // 8)
    ),
    "claims after the last full stop": lambda n: _repeat("fine. ", n // 2)
    + " ".join("in %d" % (1000 + i) for i in range(n // 16)),
    "names": lambda n: _repeat("John Smith ", n),
}


def _pii_patterns(text):
    with patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability"):
        detector = PIIDetector(PIIConfig(enabled=True, cache_size=0))
    return detector._scan_patterns(text)


def _hallucination(text):
    detector = HallucinationDetector(HallucinationConfig(enabled=True, cache_size=0))
    return detector.detect(text, context=text[:1500])


PASSES = {
    "pii patterns": _pii_patterns,
    "bias": BiasDetector(BiasConfig(enabled=True, cache_size=0)).detect,
    "prompt injection": PromptInjectionDetector(
        PromptInjectionConfig(enabled=True, cache_size=0)
    ).detect,
    "restricted topics": RestrictedTopicsDetector(
        RestrictedTopicsConfig(enabled=True, cache_size=0)
    ).detect,
    "hallucination": _hallucination,
}


# How many times each text is timed at most.
ROUNDS = 6


def _seconds(run, text):
    started = time.perf_counter()
    run(text)
    return time.perf_counter() - started


def _in_proportion(small, big):
    # Twice the proportional time, and a few milliseconds for timer noise on a
    # pass that is over almost before it starts.
    return big <= 2 * FACTOR * small + 0.03


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("name", sorted(PASSES))
def test_a_pass_grows_no_faster_than_the_text(name, shape):
    """Four times the text may take four times as long, not sixteen."""
    run, make = PASSES[name], SHAPES[shape]
    short, long = make(SMALL), make(SMALL * FACTOR)

    # The fastest of several runs of each, taken turn about so that whatever
    # else the machine is doing falls on both alike. A pass that grows faster
    # than its text is over the line on every run; one that was only
    # interrupted is not, so the timing stops as soon as it is under.
    small = _seconds(run, short)
    big = _seconds(run, long)
    for _ in range(ROUNDS - 1):
        if _in_proportion(small, big):
            break
        small = min(small, _seconds(run, short))
        big = min(big, _seconds(run, long))

    assert _in_proportion(small, big), "%s on %s: %.4f s for %d chars, %.4f s for %d" % (
        name,
        shape,
        small,
        SMALL,
        big,
        SMALL * FACTOR,
    )


def test_no_address_pattern_is_tried_from_every_place_a_word_starts():
    """An address is looked for from each "@" outwards. Tried from every place a
    word starts, as a search for the whole expression does, the two address
    patterns were the ones that cost most over a long text."""
    with patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability"):
        detector = PIIDetector(PIIConfig(enabled=True, cache_size=0))
    searched = []
    finditer = re.finditer

    def recording(pattern, string, *args, **kwargs):
        searched.append(pattern)
        return finditer(pattern, string, *args, **kwargs)

    text = "write to a.b@example.com or pay rahul@oksbi, or call 987-654-3210"
    with patch("genai_otel.evaluation.pii_detector.re.finditer", side_effect=recording):
        found, _ = detector._scan_patterns(text)

    assert searched
    assert not [pattern for pattern in searched if "@" in pattern]
    assert {entity["type"] for entity in found} == {"EMAIL_ADDRESS", "IN_UPI", "PHONE_NUMBER"}


def test_an_at_sign_does_not_make_a_run_of_punctuation_slow():
    """In a run of punctuation a word starts at every character. With an "@"
    anywhere in the text, each of them used to be read on from for as long as
    an address may be, which is hundreds of times the work of the run alone."""
    run = _repeat("_.", SMALL * FACTOR * 4)
    with_an_at_sign = run[:-2] + "@x"

    def close(alone, with_it):
        return with_it <= 3 * alone + 0.03

    alone = _seconds(_pii_patterns, run)
    with_it = _seconds(_pii_patterns, with_an_at_sign)
    for _ in range(ROUNDS - 1):
        if close(alone, with_it):
            break
        alone = min(alone, _seconds(_pii_patterns, run))
        with_it = min(with_it, _seconds(_pii_patterns, with_an_at_sign))

    assert close(alone, with_it), "%.4f s without an at sign, %.4f s with one" % (alone, with_it)


# --- the same findings as before ------------------------------------------------


def _corpus(pieces, count=2500, seed=3, longest=14):
    """Short texts glued together from ``pieces`` in every order chance allows."""
    rng = random.Random(seed)
    for _ in range(count):
        yield "".join(rng.choice(pieces) for _ in range(rng.randrange(1, longest)))


def _phrases(slots, count=2500, seed=3):
    """Texts built slot by slot, so that most come close to matching.

    One to three phrases each; now and then a slot is dropped or said twice,
    which is where an expression rewritten carelessly would start to differ.
    """
    rng = random.Random(seed)
    for _ in range(count):
        phrases = []
        for _ in range(rng.randrange(1, 4)):
            chosen = [rng.choice(options) for options in slots]
            if rng.random() < 0.3:
                del chosen[rng.randrange(len(chosen))]
            if rng.random() < 0.2:
                again = rng.randrange(len(chosen))
                chosen.insert(again, chosen[again])
            phrases.append("".join(chosen))
        yield rng.choice(["", " ", ". ", "x", "\n"]).join(phrases)


def _spans(pattern, text, flags=0):
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]


EMAIL_AS_RELEASED = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
UPI_AS_RELEASED = r"\b[\w.+-]+@(?:ok(?:hdfcbank|axis|icici|sbi|bizaxis)|paytm|ybl|upi|apl|ibl|axl|barodampay|mahb|cnrb|pnb|sib|federal|kotak|indus|citi|rbl|idbi|aubank|dlb|kvb|csbpay|idfcfirst|freecharge|slice|jupiter|fi|gpay|oksbi)\b"

ADDRESS_PIECES = [
    "jane",
    "roe",
    "a",
    "Z9",
    "x_y",
    "example",
    "mail",
    "org",
    "com",
    "c",
    "oksbi",
    "paytm",
    "okaxis",
    "ybl",
    "fi",
    "@",
    "@",
    ".",
    ".",
    "-",
    "+",
    "%",
    "_",
    "|",
    " ",
    " ",
    "\n",
    ",",
    "é",
    "卡",
    "1",
    "2024",
]


ADDRESS_SLOTS = [
    ["", " ", "<", "é", "x.", "-", "to:"],
    ["jane", "jane.roe", "j_r", "a", "first.last+tag", "9876543210", "Z9"],
    ["@", "@", "@", "@@", " @", ""],
    ["example", "mail.example", "sub-domain.example", "oksbi", "paytm", "ybl", "fi", "okaxis", "x"],
    ["", ".org", ".com", ".c", ".co.uk", ".ORG", ".o|g"],
    ["", " ", ".", ",", "é", "1", ">"],
]


def _address_texts(seed):
    yield from _corpus(ADDRESS_PIECES, seed=seed)
    yield from _phrases(ADDRESS_SLOTS, seed=seed)


def _scanner(entity_type):
    """Where the pattern scan finds ``entity_type`` in a text, as (start, end)."""
    with patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability"):
        detector = PIIDetector(PIIConfig(enabled=True, cache_size=0, entity_types={entity_type}))

    def spans(text):
        return [(e["start"], e["end"]) for e in detector._scan_patterns(text)[0]]

    return spans


EMAILS = _scanner(PIIEntityType.EMAIL_ADDRESS)
UPI_ADDRESSES = _scanner(PIIEntityType.IN_UPI)


def test_the_email_pattern_finds_what_it_found():
    checked = 0
    for text in _address_texts(3):
        assert EMAILS(text) == _spans(EMAIL_AS_RELEASED, text), text
        checked += bool(_spans(EMAIL_AS_RELEASED, text))
    assert checked > 500  # the corpus does contain addresses


def test_the_upi_pattern_finds_what_it_found():
    checked = 0
    for text in _address_texts(4):
        assert UPI_ADDRESSES(text) == _spans(UPI_AS_RELEASED, text), text
        checked += bool(_spans(UPI_AS_RELEASED, text))
    assert checked > 500


def test_the_upi_recogniser_given_to_presidio_finds_what_it_found():
    """The same expression is registered with Presidio, which reads the part
    of the text that the model does, or all of it when there is no limit."""
    presidio_analyzer = MagicMock()
    modules = {"presidio_analyzer": presidio_analyzer, "presidio_anonymizer": MagicMock()}
    with patch.dict(sys.modules, modules):
        PIIDetector(PIIConfig(enabled=True))
    (registered,) = [
        call.kwargs["regex"]
        for call in presidio_analyzer.Pattern.call_args_list
        if call.kwargs.get("name") == "upi"
    ]

    checked = 0
    for text in _address_texts(8):
        assert _spans(registered, text) == _spans(UPI_AS_RELEASED, text), text
        checked += bool(_spans(UPI_AS_RELEASED, text))
    assert checked > 500

    def run(text):
        return re.findall(registered, text)

    small = _seconds(run, _repeat("pkg1.", SMALL))
    big = _seconds(run, _repeat("pkg1.", SMALL * FACTOR))
    assert big <= 2 * FACTOR * small + 0.03


@pytest.mark.parametrize(
    "text",
    [
        "write to first.last+tag@mail.example.org today",
        "Reply-To:<jane_roe%dept@sub.example.co.uk>",
        "a@b.cd",
        ("x" * 64) + "@example.org",
        "no address here, only an @ sign and example.org",
    ],
)
def test_the_email_pattern_on_addresses_as_people_write_them(text):
    assert EMAILS(text) == _spans(EMAIL_AS_RELEASED, text)


@pytest.mark.parametrize(
    "text", ["pay rahul.k@oksbi today", "upi: shop-12@paytm, alt 9876543210@ybl", "me@fi"]
)
def test_the_upi_pattern_on_addresses_as_people_write_them(text):
    assert UPI_ADDRESSES(text) == _spans(UPI_AS_RELEASED, text)


# What the scan looks for: the released expressions with the part before the
# "@" read up to a fixed length.
EMAIL_AS_SEARCHED = EMAIL_AS_RELEASED.replace("]+@", "]{1,64}@")
UPI_AS_SEARCHED = UPI_AS_RELEASED.replace("]+@", "]{1,256}@")

# Runs longer than an address may be, of every kind of character one may have,
# and "@" with and without what makes an address of it.
LONG_ADDRESS_PIECES = [
    "a" * 70,
    "b." * 40,
    "_." * 140,
    "x" * 300,
    "-1" * 130,
    "é" * 70,
    "Z9" * 33,
    "+" * 65,
    "%",
    "jane",
    ".",
    "-",
    "_",
    " ",
    "\n",
    "卡",
    "@",
    "@",
    "@example.org",
    "@mail.example.co.uk ",
    "@oksbi",
    "@ybl ",
    "@paytm.",
    "@fi",
    "@x",
]


def _long_address_texts(seed):
    yield from _corpus(LONG_ADDRESS_PIECES, count=1500, seed=seed, longest=9)


@pytest.mark.parametrize(
    "found, searched, released",
    [
        (EMAILS, EMAIL_AS_SEARCHED, EMAIL_AS_RELEASED),
        (UPI_ADDRESSES, UPI_AS_SEARCHED, UPI_AS_RELEASED),
    ],
    ids=["email", "upi"],
)
def test_an_address_is_found_from_its_at_sign_where_the_expression_finds_it(
    found, searched, released
):
    """Looking from each "@" outwards finds what the expression finds when it is
    tried from every place a word starts, in long runs as in short texts."""
    assert searched != released
    matched = cut_short = 0
    for texts in (_address_texts(11), _long_address_texts(12)):
        for text in texts:
            expected = _spans(searched, text)
            assert found(text) == expected, text
            matched += bool(expected)
            # Where the fixed length made a difference.
            cut_short += expected != _spans(released, text)
    assert matched > 800
    assert cut_short > 20


def _every_text(tokens, longest):
    for length in range(longest + 1):
        for chosen in itertools.product(tokens, repeat=length):
            yield "".join(chosen)


@pytest.mark.parametrize(
    "pattern, tokens",
    [
        (r"\b[\w.+-]{1,3}@(?:ybl|fi)\b", ["a", ".", "-", "@", "ybl", " "]),
        (
            r"\b[A-Za-z0-9._%+-]{1,2}@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            ["a", ".", "@", "b.cd", " ", "1"],
        ),
    ],
    ids=["a fixed tail", "a tail of any length"],
)
def test_every_short_text_gives_the_matches_the_expression_gives(pattern, tokens):
    """Every text of up to six of the tokens, with an address so short that its
    longest length is reached all the time."""
    search = pii_detector._AddressSearch(pattern)
    matched = 0
    for text in _every_text(tokens, 6):
        expected = _spans(pattern, text)
        assert search.spans(text) == expected, text
        matched += bool(expected)
    assert matched > 1000


def test_an_expression_of_another_shape_is_not_taken_for_an_address():
    with pytest.raises(ValueError):
        pii_detector._AddressSearch(r"\b\d{3}-\d{2}-\d{4}\b")


def test_an_address_with_a_local_part_past_the_limit_is_still_reported():
    """The part before the @ is read up to the longest a mail system accepts.
    A longer one is still reported, from the last place a word starts in it."""
    text = "send it to " + "first.middle." * 8 + "last@example.org please"
    (found,) = EMAILS(text)
    assert text[found[0] : found[1]].endswith("last@example.org")


INJECTION_CONTEXT_SWITCHING_AS_RELEASED = [
    r"---+\s*(?:end|stop|ignore)",
    r"```+\s*(?:end|stop)",
    r"</(?:system|instruction|prompt)>",
    r"\[(?:END|STOP|IGNORE)\s+(?:SYSTEM|INSTRUCTION|PROMPT)\]",
    r"<\|(?:end|stop)(?:of)?(?:system|instruction|prompt)\|>",
]


def test_context_switching_finds_what_it_found():
    detector = PromptInjectionDetector(PromptInjectionConfig(enabled=True, cache_size=0))
    pieces = ["-", "--", "---", "`", "```", " ", "\n", "end", "stop", "ignore", "END", "x", "e"]
    found_some = 0
    for text in _corpus(pieces, longest=10):
        expected = [
            m.group()
            for pattern in INJECTION_CONTEXT_SWITCHING_AS_RELEASED
            for m in re.finditer(pattern, text, re.IGNORECASE)
        ]
        assert detector._check_injection_type(text, "context_switching")[1] == expected, text
        found_some += bool(expected)
    assert found_some > 100


FINANCIAL_ADVICE_AS_RELEASED = [
    r"should\s+I\s+(?:invest|buy|sell|trade)\s+(?:in\s+)?(?:stocks?|crypto|bitcoin|forex)",
    r"(?:you|we)\s+should\s+(?:definitely|absolutely)?\s*(?:buy|sell|invest\s+in|trade)\s+(?:these\s+)?(?:stocks?|crypto|bitcoin)",
    r"(?:best|good|top)\s+(?:stocks?|investments?|crypto)\s+to\s+(?:buy|invest|purchase)",
    r"(?:best|good|top)\s+investment\s+strategies",
    r"(?:what|which)\s+(?:stocks?|crypto|investments?)\s+should\s+(?:I|we)\s+(?:buy|invest\s+in)",
    r"(?:should|can)\s+(?:I|we)\s+(?:invest\s+in|buy|sell|trade)",
    r"financial\s+advice\s+(?:on|for|about)",
]


FINANCIAL_ADVICE_SLOTS = [
    ["you", "we", "they"],
    [" ", "  ", "\n", ""],
    ["should"],
    [" ", "  ", "\t", ""],
    ["", "definitely", "absolutely", "definitely ", "absolutely  ", "certainly "],
    ["", " "],
    ["buy", "sell", "invest in", "invest  in", "trade", "hold"],
    [" ", "  ", ""],
    ["", "these ", "these"],
    ["stocks", "stock", "crypto", "bitcoin", "bonds"],
]


def test_financial_advice_finds_what_it_found():
    detector = RestrictedTopicsDetector(RestrictedTopicsConfig(enabled=True, cache_size=0))
    found_some = 0
    for text in _phrases(FINANCIAL_ADVICE_SLOTS):
        expected = [
            m.group()
            for pattern in FINANCIAL_ADVICE_AS_RELEASED
            for m in re.finditer(pattern, text, re.IGNORECASE)
        ]
        assert detector._check_topic(text, "financial_advice")[1] == expected, text
        found_some += bool(expected)
    assert found_some > 300


# The hallucination heuristics as they were released, to compare against.

FACTUAL_CLAIMS_AS_RELEASED = {
    "specific_dates": [
        r"\b(?:in|on|during)\s+\d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    ],
    "specific_numbers": [
        r"\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b",
        r"\b(?:exactly|precisely|approximately)\s+\d+(?:,\d{3})*(?:\.\d+)?\b",
        r"\b\d{1,3}(?:,\d{3})+\b",
        r"\b\d+(?:\.\d+)?%",
    ],
    "specific_names": [
        r"\b(?:according\s+to|says|stated|claimed)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:said|wrote|published|discovered)\b",
    ],
    "statistics": [
        r"\b\d+(?:\.\d+)?%\s+of\b",
        r"\b(?:studies?|research|data)\s+(?:shows?|indicates?|suggests?)\s+that\b",
    ],
}


def _claims_as_released(text):
    claims = []
    for patterns in FACTUAL_CLAIMS_AS_RELEASED.values():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                sentence_pattern = r"[^.!?]*" + re.escape(match.group()) + r"[^.!?]*[.!?]"
                sentence_match = re.search(sentence_pattern, text, re.IGNORECASE)
                if sentence_match:
                    claims.append(sentence_match.group().strip())
    return claims


def _claim_count_as_released(text):
    return sum(
        len(re.findall(pattern, text, re.IGNORECASE))
        for patterns in FACTUAL_CLAIMS_AS_RELEASED.values()
        for pattern in patterns
    )


def _internal_contradiction_as_released(text):
    sentences = re.split(r"[.!?]+", text)
    if len(sentences) < 2:
        return False
    number_pattern = r"(\w+)\s+(?:is|are|was|were|reached?)\s+(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?)"
    subject_values = {}
    for sentence in sentences:
        for subject, value in re.findall(number_pattern, sentence.lower()):
            if subject in subject_values:
                if subject_values[subject] != value:
                    return True
            else:
                subject_values[subject] = value
    markers = [r"\bhowever\b", r"\bbut\b", r"\balthough\b", r"\bactually\b"]
    if any(re.search(marker, text.lower()) for marker in markers) and len(sentences) >= 2:
        numbers = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?", text)
        if len(numbers) >= 2 and len(set(numbers)) >= 2:
            return True
    return False


def _context_numbers_as_released(text):
    number_pattern = r"(\w+)\s+(?:is|are|was|were|reached?)\s+(?:exactly|approximately|about)?\s*(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?)"
    return [match.groups() for match in re.finditer(number_pattern, text.lower())]


CLAIM_PIECES = [
    "in",
    "on",
    "2019",
    "2020",
    "May",
    "5",
    "12",
    "1,234",
    "45.7",
    "$",
    "%",
    "million",
    "billion",
    "thousand",
    "exactly",
    "about",
    "approximately",
    "is",
    "are",
    "was",
    "reached",
    "revenue",
    "it",
    "of",
    "John",
    "Smith",
    "said",
    "says",
    "data",
    "shows",
    "that",
    "however",
    "IN",
    ",",
    ",",
    ".",
    ".",
    "!",
    "?",
    " ",
    " ",
    " ",
    "\n",
    "x",
    "é",
]


CLAIM_SLOTS = [
    ["in", "on", "during", "In", "at", "admin"],
    [" ", "  "],
    ["2019", "2020", "99", "20199"],
    [" ", ". ", ", ", "! ", ""],
    ["revenue", "John Smith said", "According to John Smith", "data shows that", ""],
    [" ", ""],
    ["$45.7 billion", "12.5%", "1,234", "exactly 5", "12.5% of", "3 million", "12,345,678", ""],
    [".", "!", "?", "", " and", ". ", "x"],
]

SUBJECT_NUMBER_SLOTS = [
    ["revenue", "it", "x", "5abc", "é", "costs"],
    [" ", "  ", "\n", ""],
    ["is", "are", "was", "were", "reach", "reached", "has"],
    [" ", "\n", "  ", ""],
    ["", "exactly", "about ", "approximately  ", "about", "roughly "],
    ["", " "],
    ["5", "45.7", "1,234", "3,000", "12", "7abc"],
    ["", " ", "  "],
    ["", "million", "billion", "thousand", "units"],
    ["", ".", " however", " but ", "!"],
]


@pytest.fixture(scope="module")
def hallucination():
    return HallucinationDetector(HallucinationConfig(enabled=True, cache_size=0))


def _claim_texts(seed):
    yield from _corpus(CLAIM_PIECES, seed=seed, longest=24)
    yield from _phrases(CLAIM_SLOTS, seed=seed)


def _subject_number_texts(seed):
    yield from _corpus(CLAIM_PIECES, seed=seed, longest=24)
    yield from _phrases(SUBJECT_NUMBER_SLOTS, seed=seed)


def test_factual_claims_are_counted_as_before(hallucination):
    counted = 0
    for text in _claim_texts(3):
        assert hallucination._count_factual_claims(text) == _claim_count_as_released(text), text
        counted += _claim_count_as_released(text)
    assert counted > 5000


def test_the_sentence_of_each_claim_is_the_one_reported_before(hallucination):
    reported = 0
    for text in _claim_texts(9):
        expected = _claims_as_released(text)
        assert hallucination._extract_factual_claims(text) == expected, text
        reported += len(expected)
    assert reported > 5000


@pytest.mark.parametrize(
    "text",
    [
        "Revenue was $45.7 billion in 2020. It grew 12.5% of last year? Yes! In 2020 again.",
        "no claims here. none at all.",
        "in 2019 and nothing after it",
        "First. Second in 2019 third. IN 2019 fourth!",
        "a 12.5% rise. then 12.5% again, and 12.5%. done",
        "x.in 2019.y",
        "",
    ],
)
def test_the_sentence_of_each_claim_on_ordinary_text(hallucination, text):
    assert hallucination._extract_factual_claims(text) == _claims_as_released(text)


def test_only_as_many_claims_are_looked_up_as_are_asked_for(hallucination):
    text = "Sales reached 1,234 units in 2019. Costs fell 12.5% in 2020. Staff grew in 2021."
    everything = hallucination._extract_factual_claims(text)
    assert len(everything) > 3
    assert hallucination._extract_factual_claims(text, limit=3) == everything[:3]


def test_internal_contradictions_are_found_as_before(hallucination):
    found = 0
    for text in _subject_number_texts(6):
        expected = _internal_contradiction_as_released(text)
        assert hallucination._check_internal_contradiction(text) is expected, text
        found += expected
    assert found > 300


def test_subject_and_number_pairs_are_read_as_before(hallucination):
    read = 0
    for text in _subject_number_texts(7):
        expected = _context_numbers_as_released(text)
        assert hallucination._subject_numbers_in_context(text.lower()) == expected, text
        read += len(expected)
    assert read > 1000


@pytest.mark.parametrize(
    "text",
    [
        "it is 5abc is 7",
        "x is 5 is 7 is 9",
        "revenue was exactly 45.7 million and costs were about5",
        "a is\n\n 3,000 thousand. b reached 2",
        "éa is 5",
    ],
)
def test_subject_and_number_pairs_where_one_runs_into_the_next(hallucination, text):
    assert hallucination._subject_numbers_in_context(text) == _context_numbers_as_released(text)
    plain = r"(\w+)\s+(?:is|are|was|were|reached?)\s+(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?)"
    assert hallucination._subject_numbers(text) == re.findall(plain, text)
