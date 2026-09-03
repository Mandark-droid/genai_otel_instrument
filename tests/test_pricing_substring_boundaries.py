"""Regression tests for unanchored substring matching in model-name resolution.

Two independent code paths resolved a model name by naked substring containment,
so a short token could capture an unrelated model and bill it at another model's
price. Both must now require the match to land on a *token boundary* - the
characters flanking the match have to be separators or the string edge.

1. ``_resolve_model_key`` (pricing-key lookup). The key ``o1`` is a substring of
   ``sa-o1-0k``, so every Sao10K fine-tune on DeepInfra/Together resolved to o1
   and was billed at $15/1M instead of roughly $0.10/1M.

2. ``_extract_param_count_from_model_name`` (local-model fallback). The generic
   size word ``mini`` is a substring of ``ge-mini``, so any Gemini model without
   an explicit pricing key was treated as a 0.02B local model and charged a
   fabricated price rather than being reported as unpriced. 96 Gemini ids in the
   models.dev/LiteLLM corpus hit this.

A fabricated price is worse than no price: it is indistinguishable from a real
one downstream, so the error never surfaces.
"""

import pytest

from genai_otel.cost_calculator import CostCalculator


@pytest.fixture(scope="module")
def calc():
    return CostCalculator()


# ---------------------------------------------------------------------------
# 1. Pricing-key lookup must not capture across a token boundary.
# ---------------------------------------------------------------------------

# (model a caller sends, pricing key that must NOT capture it)
MUST_NOT_CAPTURE = [
    ("Sao10K/L3-8B-Stheno-v3.2", "o1"),
    ("Sao10K/L3.1-70B-Euryale-v2.2", "o1"),
    ("sao10k/fimbulvetr-11b", "o1"),
    ("google/veo3.1", "o3"),
    ("google/veo3.1-fast", "o3"),
    ("deepgram/nova-2-automotive", "auto"),
    ("Devstral-2-123B-Instruct-2512-int4-AutoRound", "auto"),
    ("openai-gpt-52", "gpt-5"),
    ("openai-gpt-53-codex", "gpt-5"),
    ("github_copilot/gpt-41-copilot", "gpt-4"),
    ("github_copilot/claude-opus-41", "claude-opus-4"),
    ("mistralai/mistral-nemotron", "mistral-nemo"),
    ("mistral-large2", "mistral-large"),
]


@pytest.mark.parametrize("model,forbidden_key", MUST_NOT_CAPTURE)
def test_short_key_does_not_capture_unrelated_model(calc, model, forbidden_key):
    """A pricing key may not match in the middle of a longer alphanumeric run."""
    resolved = calc._normalize_model_name(model, "chat")
    assert resolved != forbidden_key, (
        f"{model!r} resolved to pricing key {forbidden_key!r}; the match is buried "
        f"inside a longer token, so this bills an unrelated model's price"
    )


def test_sao10k_is_not_billed_as_o1(calc):
    """The concrete mis-bill: an 8B Llama finetune charged at o1's frontier price."""
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 200_000}
    cost = calc.calculate_granular_cost("Sao10K/L3-8B-Stheno-v3.2", usage, "chat")
    # o1 is $15/1M in + $60/1M out => $27.00 for this usage. Anything near that
    # means the o1 key captured it again.
    assert (
        cost["total"] < 5.0
    ), f"charged ${cost['total']:.2f} for an 8B finetune - that is o1 pricing"


# Matches that are legitimately on a boundary must keep working.
MUST_STILL_RESOLVE = [
    ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ("anthropic.claude-3-7-sonnet-20240620-v1:0", "claude-3-7-sonnet"),
    ("us.anthropic.claude-opus-4-20250514-v1:0", "claude-opus-4"),
]


@pytest.mark.parametrize("model,expected_fragment", MUST_STILL_RESOLVE)
def test_boundary_matches_still_resolve(calc, model, expected_fragment):
    """Dated snapshots and cloud-prefixed ids still resolve to their family key."""
    resolved = calc._normalize_model_name(model, "chat")
    assert resolved is not None, f"{model!r} became unpriced"
    assert (
        expected_fragment in resolved.lower()
    ), f"{model!r} resolved to {resolved!r}, expected a {expected_fragment!r} key"


# ---------------------------------------------------------------------------
# 2. The local-model size heuristic must not read a size word out of the middle
#    of another word.
# ---------------------------------------------------------------------------

BURIED_SIZE_WORDS = [
    "gemini-3.8-flash",  # "mini" inside "gemini"
    "gemini-3.5-transcribe",
    "google/gemini-3-1-flash-tts",
    "gemini-embedding",
    "minimax-m2p1",  # "mini" inside "minimax"
    "stabilityai/stablediffusionxl",  # "xl" inside "sdxl"
    "tinyfish/search",  # "tiny" inside "tinyfish"
]


@pytest.mark.parametrize("model", BURIED_SIZE_WORDS)
def test_buried_size_word_does_not_fabricate_a_param_count(calc, model):
    """A size word buried in another word must not imply a parameter count."""
    assert calc._extract_param_count_from_model_name(model) is None, (
        f"{model!r} produced a parameter count from a size word buried inside "
        f"another word, which fabricates a local-model price"
    )


# Standalone size words and explicit size tokens must keep working.
REAL_SIZES = [
    ("gpt2-xl", 1.5),
    ("t5-small", 0.06),
    ("t5-xxl", 11.0),
    ("bert-large-uncased", 0.34),
    ("bert-base-uncased", 0.11),
    ("llama-2-7b", 7.0),
    ("llama3:70b", 70.0),
    ("smollm2:360m", 0.36),
    ("MMPO_Gemma_7b_gamma1.1", 7.0),
    ("smollm2-135M_pretrained_400k", 0.135),
]


@pytest.mark.parametrize("model,expected", REAL_SIZES)
def test_real_size_indicators_still_parse(calc, model, expected):
    """Explicit size tokens and standalone size words are unaffected."""
    assert calc._extract_param_count_from_model_name(model) == pytest.approx(expected)
