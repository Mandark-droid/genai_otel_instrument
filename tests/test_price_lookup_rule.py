"""The model price lookup rule.

A pricing key found inside a longer model name prices that name only if what follows the key
still names the same model. A new version, a 4-digit release tag or (for chat) another
modality is refused, so an unknown model reads as unpriced instead of silently billing an older
model's price. Among acceptable keys, one that is the whole rest of the name wins, then the
longest.
"""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.cost_calculator import CostCalculator, _remainder_rank

CHAT_USAGE = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}


@pytest.mark.parametrize(
    "key,remainder,expected",
    [
        ("claude-opus-5-5", "", 0),
        # Same-model suffixes.
        ("claude-sonnet-5-5", "-20260928", 1),
        ("gpt-4o", "-2024-08-06", 1),
        ("gemini-2.5-flash-preview", "-09-2025", 1),
        ("gemini-1.5-pro", "-002", 1),
        ("claude-3-haiku-20240307", "-v1:0", 1),
        ("claude-opus-5-5", "@default", 1),
        ("claude-opus-5", "[1m]", 1),
        ("gpt-4o", "-latest", 1),
        ("qwen3", ":latest", 1),
        ("mimo-v2.6-pro-ultraspeed", "-preview", 1),
        ("llama-3.1-70b", "-instruct", 1),
        ("qwen3.8-27b", "-fp8", 1),
        # A new version is refused.
        ("mistral-large", "-4", None),
        ("kimi-k3", ".1-pro", None),
        ("anthropic.claude-opus-5", "-5", None),
        ("gpt-realtime", "-2.1-mini", None),
        # ...but a patch release after a two-part version is the same model.
        ("mistral-large-4-0", "-8", 2),
        # A size tag is not a version.
        ("llama-4-scout", "-17b-16e-instruct", 2),
        # A 4-digit release tag names a different release; an ISO date does not.
        ("mistral-small", "-2603", None),
        ("deepseek-v4-flash", "-0731", None),
        ("pixtral-large", "-2502-v1:0", None),
        ("gpt-4o-mini", "-2024-07-18:acme::abc123", 2),
        # Tier words are accepted, weakest: they do not reliably mean another price.
        ("gpt-5.1-codex", "-max", 2),
        ("grok-4.1-fast", "-reasoning", 2),
    ],
)
def test_remainder_rank(key, remainder, expected):
    assert _remainder_rank(key, remainder, refuse_other_modality=True) == expected


@pytest.mark.parametrize(
    "remainder", ["-tts", "-image", "-preview-tts", "-realtime-preview", "-embedding"]
)
def test_another_modality_is_refused_only_for_chat(remainder):
    assert _remainder_rank("gemini-2.5-flash", remainder, refuse_other_modality=True) is None
    assert _remainder_rank("gemini-2.5-flash", remainder, refuse_other_modality=False) is not None


# --- the rule against a hand-built table, independent of the shipped prices -------------
TABLE = {
    "chat": {
        "anthropic.claude-opus-5": {"promptPrice": 0.005, "completionPrice": 0.025},
        "claude-opus-5-5": {"promptPrice": 0.004, "completionPrice": 0.02},
        "us.anthropic.claude-opus-5-5": {"promptPrice": 0.0044, "completionPrice": 0.022},
        "moonshotai/Kimi-K3": {"promptPrice": 0.003, "completionPrice": 0.015},
        "kimi-k3.1-pro": {"promptPrice": 0.002, "completionPrice": 0.01},
        "mistral-large": {"promptPrice": 0.002, "completionPrice": 0.006},
        "gemini-2.5-flash": {"promptPrice": 0.0003, "completionPrice": 0.0025},
        "gpt-4o": {"promptPrice": 0.0025, "completionPrice": 0.01},
        "o1": {"promptPrice": 0.015, "completionPrice": 0.06},
    },
    "images": {"gpt-image-2": {"low": {"1024x1024": 0.006}}},
}


@pytest.fixture()
def synthetic():
    with patch.object(CostCalculator, "_load_pricing", MagicMock()):
        calc = CostCalculator()
    calc.pricing_data = json.loads(json.dumps(TABLE))
    return calc


@pytest.mark.parametrize(
    "name,expected",
    [
        # F1: a longer provider-prefixed key of an OLDER model must not win.
        ("anthropic.claude-opus-5-5", "claude-opus-5-5"),
        ("moonshotai/kimi-k3.1-pro", "kimi-k3.1-pro"),
        # The most specific key that is the whole rest of the name wins (regional premium).
        ("bedrock/us.anthropic.claude-opus-5-5", "us.anthropic.claude-opus-5-5"),
        # Same-model suffixes keep the key.
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("azure/gpt-4o-2024-08-06", "gpt-4o"),
        # F2: a new version, release or modality is unpriced, not the predecessor's price.
        ("mistral-large-4", None),
        ("mistral-large-2512", None),
        ("gemini-2.5-flash-tts", None),
        ("gemini-2.5-flash-preview-tts", None),
        # The token-boundary rule still holds.
        ("sao10k-l3-8b", None),
    ],
)
def test_resolution(synthetic, name, expected):
    assert synthetic.resolve_model_key(name) == expected


def test_a_refused_name_is_reported_unpriced(synthetic):
    # "gpt-4o" + "-5" is a new version: no table price, and no size word to estimate from.
    assert synthetic.resolve_model_key("gpt-4o-5") is None
    assert synthetic.calculate_cost("gpt-4o-5", CHAT_USAGE, "chat") == 0.0
    assert synthetic.pricing_source("gpt-4o-5", "chat") == "unpriced"


def test_a_parameter_count_still_estimates(synthetic):
    assert synthetic.pricing_source("acme-llama:7b", "chat") == "estimated"


def test_a_new_image_model_does_not_borrow_another_models_per_image_price(synthetic):
    assert synthetic.resolve_model_key("gpt-image-2.6-sunburst", "image") is None
    assert synthetic.resolve_model_key("gpt-image-2", "image") == "gpt-image-2"


# --- the shipped table ------------------------------------------------------------------
@pytest.fixture(scope="module")
def shipped():
    return CostCalculator()


GATEWAY_PREFIXES = ["openrouter/", "bedrock/", "vertex_ai/", "azure/", "litellm_proxy/"]


def test_every_key_keeps_its_price_behind_a_gateway_prefix(shipped):
    chat = shipped.pricing_data["chat"]
    wrong = []
    for key, entry in chat.items():
        for prefix in GATEWAY_PREFIXES:
            got = shipped.resolve_model_key(prefix + key)
            other = chat.get(got, {}) if got else {}
            if (
                got is None
                or other.get("promptPrice") != entry.get("promptPrice")
                or other.get("completionPrice") != entry.get("completionPrice")
            ):
                wrong.append((prefix + key, got))
    assert not wrong, f"{len(wrong)} keys lose their price behind a gateway prefix: {wrong[:10]}"


# The corpus: 532 model names from first-party listings on models.dev (2026-11-06), each with
# the provider's listed price. A ratchet: a table or rule change may improve these numbers,
# never worsen them.
CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "model_price_lookup_corpus_2026-11-06.json").read_text(
        encoding="utf-8"
    )
)["names"]
MIN_RIGHT = 388
MAX_LOOKUP_ERRORS = 2

_ROUTE = re.compile(
    r"^((us|eu|apac|jp|au|in|ca|global|us-gov)[.-])?((anthropic|openai|meta|mistral|amazon|cohere|"
    r"deepseek|qwen|google|xai|moonshotai|minimax|zai|writer|ai21)[.-])"
)
_TRAILER = re.compile(
    r"(-v\d+(:\d+)?|@[a-z0-9._-]+|-\d{8}|-\d{4}-\d{2}-\d{2}|-\d{2}-\d{4}|-latest|-maas)$"
)


def _model_part(name):
    """The model a name refers to, without route prefixes and snapshot trailers."""
    part = _ROUTE.sub("", name.lower().split("/")[-1])
    while True:
        shorter = _TRAILER.sub("", part)
        if shorter == part:
            return re.sub(r"[._]", "-", part)
        part = shorter


def test_corpus_ratchet(shipped):
    chat = shipped.pricing_data["chat"]
    right, lookup_errors = 0, []
    for name, listed in CORPUS.items():
        key = shipped.resolve_model_key(name)
        if key is None:
            continue
        entry = chat[key]
        got_in, got_out = entry.get("promptPrice", 0) * 1000, entry.get("completionPrice", 0) * 1000
        want_in, want_out = listed["input_per_1m"], listed["output_per_1m"]
        if abs(got_in - want_in) <= 0.03 * want_in + 1e-9 and abs(got_out - want_out) <= (
            0.03 * want_out + 1e-9
        ):
            right += 1
        elif _model_part(key) != _model_part(name):
            lookup_errors.append((name, key))
    assert right >= MIN_RIGHT, f"right prices fell to {right} (ratchet {MIN_RIGHT})"
    assert (
        len(lookup_errors) <= MAX_LOOKUP_ERRORS
    ), f"names priced as another model: {lookup_errors}"


@pytest.mark.parametrize(
    "name,prompt_price,completion_price",
    [
        ("anthropic.claude-opus-5-5", 0.004, 0.02),
        ("xai/grok-4.7", 0.002, 0.006),
        ("grok-4.5", 0.002, 0.006),
        ("xai/grok-4.20-reasoning", 0.00125, 0.0025),
        ("grok-4.20-0309-reasoning", 0.00125, 0.0025),
        ("us.anthropic.claude-sonnet-5-5", 0.0022, 0.011),
        ("mistral-small-2603", 0.00015, 0.0006),
        ("gemini-3.5-flash-lite", 0.0003, 0.0025),
        ("DeepSeek-V4.1-Flash-1017", 0.00015, 0.0006),
    ],
)
def test_shipped_names_priced(shipped, name, prompt_price, completion_price):
    costs = shipped.calculate_granular_cost(
        name, {"prompt_tokens": 1000, "completion_tokens": 1000}, "chat"
    )
    assert costs["prompt"] == pytest.approx(prompt_price)
    assert costs["completion"] == pytest.approx(completion_price)


def test_sonnet_5_5_cache_reads_at_a_twentieth_of_input(shipped):
    # Anthropic prices Opus 5.5 and Sonnet 5.5 cache hits at 0.05x the input price.
    chat = shipped.pricing_data["chat"]
    assert chat["claude-sonnet-5-5"]["cacheReadPrice"] == pytest.approx(0.0001)
    assert chat["us.anthropic.claude-sonnet-5-5"]["cacheReadPrice"] == pytest.approx(0.00011)
    assert chat["claude-opus-5-5"]["cacheReadPrice"] == pytest.approx(0.0002)


@pytest.mark.parametrize("name", ["mistral-large-4-9", "gemini-2.5-flash-tts", "grok-9.9"])
def test_unknown_versions_and_modalities_get_no_table_price(shipped, name):
    # No key prices them. A size word in the name ("large") may still give an estimate,
    # which is reported as such, never as a table price.
    assert shipped.resolve_model_key(name) is None
    assert shipped.pricing_source(name, "chat") in ("unpriced", "estimated")


def test_span_lookup_and_key_scan_agree(shipped):
    # _resolve_model_key looks up the name's boundary-aligned substrings, and falls back to
    # scanning every key only for names with very many separators. Both must give one answer.
    names = list(CORPUS)
    for key in list(shipped.pricing_data["chat"])[::7]:
        names += [f"openrouter/{key}", f"{key}-20260101", f"{key}-2", f"{key}.1-pro", f"{key}-2512"]
        names += [f"{key}-tts", f"bedrock/us.{key}-v1:0", f"x-{key}"]
    for category in ("chat", "images", "audio"):
        for name in names:
            assert shipped._resolve_model_key(name, category) == shipped._resolve_model_key(
                name, category, _scan_only=True
            ), (category, name)
