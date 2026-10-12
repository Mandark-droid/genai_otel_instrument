"""The local toxicity model reads all of a long text, not only the start of it.

The model reads a fixed number of word pieces in one pass. Given more, it used to
read the first pass-worth and drop the rest without a word, and in a long
conversation the first part is the oldest. A text longer than one pass is now
given to it in pieces, each of which fits, and the score of each category is the
highest any piece got.

No model is loaded here: stand-ins play the model and its tokenizer, and record
what they were given.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.evaluation import toxicity_detector
from genai_otel.evaluation.config import ToxicityConfig
from genai_otel.evaluation.toxicity_detector import ToxicityDetector

WINDOW = 40  # word pieces of text the stand-in model reads in one pass


class _Tokenizer:
    """A word piece per four characters of each word, and two markers around the text."""

    model_max_length = WINDOW + 2

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def tokenize(self, text):
        return [word[i : i + 4] for word in text.split() for i in range(0, len(word), 4)]


class _Model:
    """Scores a text by the marker words it can see, and keeps what it was shown.

    Like the real model it looks at the first window of word pieces only: what
    lies beyond is not scored at all.
    """

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self._reads = _Tokenizer()  # its own, so that a test can break the public one
        self.calls = []

    def _scores(self, text):
        visible = "".join(self._reads.tokenize(text)[:WINDOW])

        def score(marker, high):
            return high if marker in visible else 0.01

        return {
            "toxicity": score("VILE", 0.9),
            "severe_toxicity": 0.01,
            "obscene": 0.01,
            "threat": score("HARM", 0.85),
            "insult": score("FOOL", 0.8),
            "identity_attack": 0.01,
        }

    def predict(self, text):
        self.calls.append(text)
        if isinstance(text, str):
            return self._scores(text)
        each = [self._scores(item) for item in text]
        return {name: [scores[name] for scores in each] for name in each[0]}

    def pieces(self):
        """Every piece of text the model was given, in order."""
        out = []
        for call in self.calls:
            out.extend([call] if isinstance(call, str) else call)
        return out


def _detector(**config):
    with patch.object(ToxicityDetector, "_check_availability"):
        detector = ToxicityDetector(ToxicityConfig(enabled=True, **config))
    detector._detoxify_model = _Model()
    detector._detoxify_available = True
    return detector


def _words(count, **markers):
    """``count`` numbered words, so that every place in the text is unlike any other.

    ``markers`` puts a marker word in place of the word at a position.
    """
    words = ["w%04d" % number for number in range(count)]
    for marker, position in markers.items():
        words[position] = marker
    return " ".join(words)


def _covered(text, pieces):
    """How many characters of ``text`` lie in at least one of ``pieces``."""
    covered = [False] * len(text)
    for piece in pieces:
        start = text.index(piece)  # the words are numbered, so a piece occurs once
        for index in range(start, start + len(piece)):
            covered[index] = True
    return sum(covered)


LONG = _words(400)  # 800 word pieces: twenty windows


def test_a_short_text_is_still_given_whole_in_one_call():
    detector = _detector()
    text = _words(10)

    detector.detect(text)

    assert detector._detoxify_model.calls == [text]


def test_a_text_that_just_fits_is_one_call_too():
    detector = _detector()
    text = _words(WINDOW // 2)  # two word pieces a word: exactly one window
    assert len(detector._detoxify_model.tokenizer.tokenize(text)) == WINDOW

    detector.detect(text)

    assert detector._detoxify_model.calls == [text]


def test_every_character_of_a_long_text_is_in_some_piece():
    detector = _detector()

    detector.detect(LONG)

    pieces = detector._detoxify_model.pieces()
    assert len(pieces) > 1
    assert _covered(LONG, pieces) == len(LONG)


def test_no_piece_is_longer_than_the_model_reads():
    detector = _detector()

    detector.detect(LONG)

    tokenizer = detector._detoxify_model.tokenizer
    sizes = [len(tokenizer.tokenize(piece)) for piece in detector._detoxify_model.pieces()]
    assert max(sizes) <= WINDOW
    # And they are not wastefully small: a pass costs the same however full it is.
    assert len(sizes) <= 2 * (len(tokenizer.tokenize(LONG)) // WINDOW + 1)


def test_a_phrase_that_lies_across_a_cut_is_whole_in_one_piece():
    """Pieces overlap by a few words. Cut clean, a phrase on the boundary would
    be seen by no piece as a whole."""
    detector = _detector()

    detector.detect(LONG)

    pieces = detector._detoxify_model.pieces()
    words = LONG.split()
    for first in range(len(words) - 5):
        phrase = " ".join(words[first : first + 6])
        assert any(phrase in piece for piece in pieces), phrase


@pytest.mark.parametrize("position", [0, 199, 399], ids=["start", "middle", "end"])
def test_a_toxic_word_is_found_wherever_it_is(position):
    detector = _detector()

    result = detector.detect(_words(400, VILE=position))

    assert result.is_toxic is True
    assert result.scores["toxicity"] == 0.9
    assert result.toxic_categories == ["toxicity"]


def test_each_category_is_reported_at_the_highest_score_any_piece_gave_it():
    detector = _detector()
    text = _words(400, VILE=30, FOOL=220, HARM=395)

    result = detector.detect(text)

    assert result.scores["toxicity"] == 0.9
    assert result.scores["insult"] == 0.8
    assert result.scores["threat"] == 0.85
    assert result.scores["severe_toxicity"] == 0.01
    assert result.max_score == 0.9
    assert sorted(result.toxic_categories) == ["insult", "threat", "toxicity"]


def test_a_clean_long_text_is_clean():
    detector = _detector()

    result = detector.detect(LONG)

    assert result.is_toxic is False
    assert result.max_score == 0.01


def test_a_word_longer_than_the_model_reads_is_cut_up_too():
    detector = _detector()
    text = _words(30) + " " + "x" * 600 + "VILE " + _words(30)  # one word of 150 word pieces

    result = detector.detect(text)

    tokenizer = detector._detoxify_model.tokenizer
    pieces = detector._detoxify_model.pieces()
    assert max(len(tokenizer.tokenize(piece)) for piece in pieces) <= WINDOW
    assert sum(len(piece) for piece in pieces) >= len(text)
    assert result.is_toxic is True


def test_a_long_text_is_remembered_as_a_whole():
    """The key is the text that was asked about, not the pieces it was read in."""
    detector = _detector()
    text = _words(400, VILE=399)

    first = detector.detect(text)
    calls = len(detector._detoxify_model.calls)
    second = detector.detect(text)

    assert len(detector._detoxify_model.calls) == calls
    assert second.scores == first.scores
    assert len(detector._cache) == 1


def test_a_model_whose_window_cannot_be_told_is_given_the_text_as_before():
    detector = _detector()
    detector._detoxify_model.tokenizer = SimpleNamespace(model_max_length=10**30)

    detector.detect(LONG)

    assert detector._detoxify_model.calls == [LONG]


def test_a_tokenizer_that_fails_does_not_stop_detection(caplog):
    detector = _detector()

    def broken(text):
        raise RuntimeError("vocabulary not loaded")

    detector._detoxify_model.tokenizer.tokenize = broken

    result = detector.detect(_words(400, VILE=0))

    assert detector._detoxify_model.calls == [_words(400, VILE=0)]
    assert result.is_toxic is True


def test_the_batch_interface_reads_all_of_a_long_text_as_well():
    detector = _detector()
    short, long_text = _words(10), _words(400, VILE=399)

    clean, toxic = detector.analyze_batch([short, long_text])

    assert clean.is_toxic is False
    assert toxic.is_toxic is True


def test_perspective_is_sent_the_whole_text_in_one_request():
    """Only the local model reads in passes. The remote service is asked as before."""
    detector = _detector(use_perspective_api=True, perspective_api_key="not-a-real-key")
    detector._perspective_available = True
    detector._egress_allowed = True
    detector._perspective_client = MagicMock()
    analyze = detector._perspective_client.comments.return_value.analyze
    analyze.return_value.execute.return_value = {
        "attributeScores": {"TOXICITY": {"summaryScore": {"value": 0.2}}}
    }

    detector.detect(LONG)

    assert analyze.call_count == 1
    assert analyze.call_args.kwargs["body"]["comment"]["text"] == LONG
    assert detector._detoxify_model.calls == []


def test_the_pieces_are_scored_several_to_a_call():
    """Several pieces at once take about half the time of one call each. The
    number in a call is bounded, so that the memory a long text takes is too."""
    detector = _detector()

    detector.detect(LONG)

    calls = detector._detoxify_model.calls
    pieces = detector._detoxify_model.pieces()
    per_call = toxicity_detector._PIECES_PER_CALL
    assert all(isinstance(call, list) and 1 <= len(call) <= per_call for call in calls)
    assert len(calls) == -(-len(pieces) // per_call)


class _WideTokenizer(_Tokenizer):
    """A window as wide as the real model's."""

    model_max_length = 512


def test_pieces_are_much_smaller_than_the_model_could_read():
    """The model scores a piece as a whole. One toxic sentence among a few hundred
    word pieces of ordinary text is scored as ordinary text, so a piece as large
    as the window would find it only where it happened to come first."""
    detector = _detector()
    detector._detoxify_model.tokenizer = _WideTokenizer()
    text = _words(1200)  # 2400 word pieces: longer than the 510 of one pass

    detector.detect(text)

    tokenizer = detector._detoxify_model.tokenizer
    sizes = [len(tokenizer.tokenize(piece)) for piece in detector._detoxify_model.pieces()]
    assert max(sizes) <= toxicity_detector._PIECE_WORD_PIECES < 510
    assert _covered(text, detector._detoxify_model.pieces()) == len(text)


def test_a_text_that_fits_the_wide_window_is_still_one_call():
    detector = _detector()
    detector._detoxify_model.tokenizer = _WideTokenizer()
    text = _words(250)  # 500 word pieces

    detector.detect(text)

    assert detector._detoxify_model.calls == [text]


class _CallableTokenizer(_Tokenizer):
    """Like the real one, it can be called, and counts quietly only when told to."""

    def __init__(self):
        self.asked = []

    def __call__(self, text, **options):
        self.asked.append(options)
        return {"input_ids": list(range(len(self.tokenize(text))))}


def test_the_tokenizer_is_asked_how_long_a_text_is_without_its_warning():
    """Asked the plain way, the real tokenizer logs that so long a text will break
    the model. It will not: it is about to be divided."""
    detector = _detector()
    detector._detoxify_model.tokenizer = tokenizer = _CallableTokenizer()

    result = detector.detect(_words(400, VILE=399))

    assert result.is_toxic is True
    assert tokenizer.asked
    assert all(
        options == {"add_special_tokens": False, "truncation": False, "verbose": False}
        for options in tokenizer.asked
    )
