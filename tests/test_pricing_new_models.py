"""Regression tests for newly added model pricing entries.

Unlike ``test_cost_calculator.py`` (which mocks ``_load_pricing`` and uses a
tiny hand-built table), these tests instantiate a real ``CostCalculator`` so
they load the shipped ``genai_otel/llm_pricing.json``. They guard against a
model entry being dropped, renamed, or mis-priced, and they exercise the
exact-match and longest-substring (dated-snapshot) lookup paths that the
provider-prefixed and snapshot aliases rely on.
"""

import pytest

from genai_otel.cost_calculator import CostCalculator


@pytest.fixture(scope="module")
def calc():
    return CostCalculator()


# (model_id, promptPrice, completionPrice) per 1k tokens
NEW_CHAT_MODELS = [
    ("claude-opus-4-8", 0.005, 0.025),
    ("claude-opus-4.8", 0.005, 0.025),
    ("claude-fable-5", 0.01, 0.05),
    ("gpt-5.5-mini", 0.0004, 0.0016),
    ("gpt-5.5-nano", 0.0001, 0.0004),
    ("gpt-5.5-pro", 0.03, 0.18),
    ("gemini-3.5-flash", 0.0015, 0.009),
    ("gemini/gemini-3.5-flash", 0.0015, 0.009),
    ("MiniMax-M3", 0.0003, 0.0012),
    ("minimax-m3", 0.0003, 0.0012),
    ("MiniMax-M3-highspeed", 0.0006, 0.0024),
    # --- June 2026 sweep (models.dev + vendor docs) ---
    ("claude-sonnet-5", 0.002, 0.01),
    ("anthropic.claude-sonnet-5", 0.002, 0.01),
    ("kimi-k2.7-code", 0.00095, 0.004),
    ("kimi-k2.7-code-highspeed", 0.0019, 0.008),
    ("moonshotai/kimi-k2.7-code", 0.00095, 0.004),
    ("north-mini-code-1-0", 0.0, 0.0),
    ("cohere/north-mini-code", 0.0, 0.0),
    ("nvidia/nemotron-3-ultra-550b-a55b", 0.0005, 0.0025),
    ("nemotron-3-ultra-550b-a55b", 0.0005, 0.0025),
    ("mimo-v2-flash", 0.00014, 0.00028),
    ("mimo-v2-omni", 0.00014, 0.00028),
    ("mimo-v2-pro", 0.000435, 0.00087),
    ("mimo-v2.5", 0.00014, 0.00028),
    ("mimo-v2.5-pro", 0.000435, 0.00087),
    ("mimo-v2.5-pro-ultraspeed", 0.001305, 0.00261),
    ("xiaomi/mimo-v2.5-pro", 0.000435, 0.00087),
    ("glm-5.2", 0.0014, 0.0044),
    ("zai/glm-5.2", 0.0014, 0.0044),
    ("qwen3.7-plus", 0.0005, 0.003),
    ("dashscope/qwen3.7-plus", 0.0005, 0.003),
    ("longcat-2.0", 0.00075, 0.00295),
    ("meituan/longcat-2.0", 0.00075, 0.00295),
    ("longcat-flash-chat", 0.0002, 0.0008),
    ("fugu-ultra", 0.005, 0.03),
    ("sakana/fugu-ultra", 0.005, 0.03),
]


@pytest.mark.parametrize("model,prompt_price,completion_price", NEW_CHAT_MODELS)
def test_new_chat_model_cost(calc, model, prompt_price, completion_price):
    """Each new model resolves and prices 1k prompt + 1k completion correctly."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    costs = calc.calculate_granular_cost(model, usage, "chat")
    assert costs["prompt"] == pytest.approx(prompt_price)
    assert costs["completion"] == pytest.approx(completion_price)
    assert costs["total"] == pytest.approx(prompt_price + completion_price)


# Dated/preview snapshot suffixes must route to the right variant via the
# longest-substring fallback, NOT collapse onto a shorter sibling key.
SNAPSHOT_ROUTING = [
    ("gpt-5.5-mini-2026-04-01", "gpt-5.5-mini"),
    ("gpt-5.5-nano-2026-04-01", "gpt-5.5-nano"),
    ("gpt-5.5-2026-04-01", "gpt-5.5"),
    ("claude-opus-4-8-20260514", "claude-opus-4-8"),
    ("claude-fable-5-20260601", "claude-fable-5"),
    ("claude-fable-5[1m]", "claude-fable-5"),
    ("gemini-3.5-flash-preview-05-19", "gemini-3.5-flash"),
    # June 2026 sweep: novel snapshots must route to the new family, not a
    # shorter sibling (e.g. sonnet-5 must NOT collapse onto sonnet-4-5, and the
    # MiMo UltraSpeed tier must NOT collapse onto the cheaper mimo-v2.5-pro).
    ("claude-sonnet-5-20260815", "claude-sonnet-5"),
    ("fugu-ultra-20260901", "fugu-ultra"),
    ("glm-5.2-2026-06-13", "glm-5.2"),
    ("kimi-k2.7-code-20260612", "kimi-k2.7-code"),
    ("mimo-v2.5-pro-ultraspeed-preview", "mimo-v2.5-pro-ultraspeed"),
]


@pytest.mark.parametrize("requested,expected_key", SNAPSHOT_ROUTING)
def test_snapshot_alias_routing(calc, requested, expected_key):
    assert calc._normalize_model_name(requested, "chat") == expected_key


# --- July / August 2026 sweep ------------------------------------------------
# The scheduled monthly refresh did not deliver these: the 2026-08-03 run of the
# routine fired but produced no branch, PR or draft release, so July's releases
# were never imported. Backfilled from models.dev (first-party + hyperscaler
# providers only).
JULY_AUGUST_2026_MODELS = [
    # July
    ("claude-opus-5", 0.005, 0.025),
    ("anthropic.claude-opus-5", 0.005, 0.025),
    ("kimi-k3", 0.003, 0.015),
    ("moonshotai/Kimi-K3", 0.003, 0.015),
    ("gemini-3.6-flash", 0.0015, 0.0075),
    ("gemini-3-6-flash", 0.0015, 0.0075),
    ("gemini-3-5-flash-lite", 0.0003, 0.0025),
    ("thinkingmachines/Inkling", 0.00187, 0.00468),
    ("tencent/Hy3", 0.00014, 0.00058),
    # August
    ("qwen3.8-max", 0.002, 0.006),
    ("muse-spark-1.2", 0.00125, 0.00425),
]


@pytest.mark.parametrize("model,prompt_price,completion_price", JULY_AUGUST_2026_MODELS)
def test_july_august_2026_model_cost(calc, model, prompt_price, completion_price):
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    costs = calc.calculate_granular_cost(model, usage, "chat")
    assert costs["prompt"] == pytest.approx(prompt_price)
    assert costs["completion"] == pytest.approx(completion_price)


# Vendors write "gpt-4.1"; callers and eval harnesses frequently send
# "gpt-4-1". A dotted key cannot match a dashed name as a substring, so before
# the punctuation aliases these fell through to a shorter, unrelated key -
# "gpt-4-1" landed on "gpt-4" and was billed at $30/1M instead of $2/1M.
# Each case below is a real mis-billing, so assert the price, not just the key.
PUNCTUATION_ALIASES = [
    ("gpt-4-1", 0.002, 0.008),
    ("gpt-4-1-mini", 0.0004, 0.0016),
    ("gpt-4-1-nano", 0.0001, 0.0004),
    ("gpt-5-1-codex-mini", 0.00025, 0.002),
    ("gpt-5-4-pro", 0.03, 0.18),
    ("glm-5-1", 0.0014, 0.0044),
]


@pytest.mark.parametrize("model,prompt_price,completion_price", PUNCTUATION_ALIASES)
def test_dashed_version_alias_does_not_fall_back_to_shorter_key(
    calc, model, prompt_price, completion_price
):
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    costs = calc.calculate_granular_cost(model, usage, "chat")
    assert costs["prompt"] == pytest.approx(prompt_price)
    assert costs["completion"] == pytest.approx(completion_price)


def test_gpt_4_1_is_not_billed_as_gpt_4(calc):
    """The specific regression above, stated as its own case."""
    assert calc._normalize_model_name("gpt-4-1", "chat") != "gpt-4"
    usage = {"prompt_tokens": 1000, "completion_tokens": 0}
    gpt4 = calc.calculate_granular_cost("gpt-4", usage, "chat")
    gpt41 = calc.calculate_granular_cost("gpt-4-1", usage, "chat")
    assert gpt41["prompt"] < gpt4["prompt"]


# Embeddings, rerankers and ASR models live in their own categories with their
# own value shapes. Several are also already keyed under "embeddings", so
# letting them into "chat" would both duplicate the entry and add keys to the
# chat substring index that no chat call should ever match.
NON_CHAT_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "mistral-embed",
    "gemini-embedding-001",
    "whisper-large-v3",
]


@pytest.mark.parametrize("model", NON_CHAT_MODELS)
def test_embedding_and_asr_models_are_not_in_the_chat_table(calc, model):
    assert model not in calc.pricing_data["chat"]


# Free-tier listings (NVIDIA NIM, Groq, zai *-flash, llama.com) quote 0/0 on
# models.dev. Importing those would have been actively harmful: the free-tier id
# is the LONGER key, so it would win the substring race and shadow the paid
# entry for the same family, silently reporting real spend as $0.00. They are
# skipped at import, so these names must still reach a paid price - or no price
# at all, which the pricing_source attribute reports honestly.
FREE_TIER_IDS_THAT_MUST_NOT_SHADOW = [
    "glm-4.5-flash",
    "glm-4.7-flash",
    "meta/llama-3.1-8b-instruct",
    "google/gemma-3n-e4b-it",
]


@pytest.mark.parametrize("model", FREE_TIER_IDS_THAT_MUST_NOT_SHADOW)
def test_free_tier_listings_do_not_shadow_paid_pricing(calc, model):
    key = calc._normalize_model_name(model, "chat")
    if key is None:
        return  # no price at all is acceptable; a bogus $0 price is not
    entry = calc.pricing_data["chat"][key]
    assert not (
        entry.get("promptPrice") == 0 and entry.get("completionPrice") == 0
    ), f"{model} resolved to {key}, a $0/$0 entry that would report real spend as free"


# --- 2026-08-13 additions -------------------------------------------------
# Muse Glimmer 30B, Nemotron 3.5 Lightning, and the Qwen3.8-2.4T alias.
# "2.4T" is Qwen3.8-Max's parameter count rather than a separate model, so the
# alias must land on the same price - a distinct entry would let the two drift.
AUGUST_2026_MODELS = [
    ("muse-glimmer-30b", 0.0003, 0.0012),
    ("muse-glimmer", 0.0003, 0.0012),
    ("meta/muse-glimmer-30b", 0.0003, 0.0012),
    ("nemotron-3.5-lightning", 0.00008, 0.0002),
    ("nemotron-3-5-lightning", 0.00008, 0.0002),
    ("nvidia/nemotron-3.5-lightning", 0.00008, 0.0002),
    ("qwen3.8-2.4t", 0.002, 0.006),
    ("qwen3.8-2-4t", 0.002, 0.006),
    ("Qwen/Qwen3.8-2.4T", 0.002, 0.006),
]


@pytest.mark.parametrize("model,prompt_price,completion_price", AUGUST_2026_MODELS)
def test_august_2026_models_priced(calc, model, prompt_price, completion_price):
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    granular = calc.calculate_granular_cost(model, usage, "chat")
    assert granular["total"] == pytest.approx(prompt_price + completion_price)


def test_qwen38_2_4t_matches_max(calc):
    """The alias must track qwen3.8-max exactly; drift would bill the same model
    two different ways depending on which id the caller happened to send."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    alias = calc.calculate_granular_cost("qwen3.8-2.4t", usage, "chat")["total"]
    canonical = calc.calculate_granular_cost("qwen3.8-max", usage, "chat")["total"]
    assert alias == pytest.approx(canonical)


@pytest.mark.parametrize(
    "model,expected_key",
    [
        # Must not be swallowed by a shorter nemotron sibling.
        ("nemotron-3.5-lightning", "nemotron-3.5-lightning"),
        ("nvidia/nemotron-3.5-lightning", "nvidia/nemotron-3.5-lightning"),
        # Must not resolve to the muse-spark family it was distilled from.
        ("muse-glimmer-30b", "muse-glimmer-30b"),
    ],
)
def test_august_2026_routing_not_shadowed(calc, model, expected_key):
    assert calc._normalize_model_name(model, "chat") == expected_key


def test_muse_glimmer_does_not_shadow_muse_spark(calc):
    """Adding muse-glimmer must leave the distilled-from family alone."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    spark = calc.calculate_granular_cost("muse-spark-1.2", usage, "chat")["total"]
    assert spark == pytest.approx(0.00125 + 0.00425)
