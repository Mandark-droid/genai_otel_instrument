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


# --- September 2026 monthly sweep (target month: August 2026) --------------
# GLM-5.3, GLM-5.3-Flash and Grok 4.6 were previously invisible: their names
# were only ever resolved via the longest-substring fallback, and the
# pre-existing shorter siblings ("glm-5", "grok-4") won that race, so every
# call silently billed at the OLD model's rate. Qwen3.8 Flash, Qwen3.8 27B,
# Gemini 3.7 Flash, Sakana Namazu and Solar Pro 4 had no entry at all.
SEPTEMBER_2026_SWEEP_MODELS = [
    ("glm-5.3", 0.0014, 0.0044),
    ("glm-5-3", 0.0014, 0.0044),
    ("zai/glm-5.3", 0.0014, 0.0044),
    ("THUDM/GLM-5.3", 0.0014, 0.0044),
    ("glm-5.3-flash", 0.000075, 0.00025),
    ("glm-5-3-flash", 0.000075, 0.00025),
    ("zai/glm-5.3-flash", 0.000075, 0.00025),
    ("grok-4.6", 0.002, 0.006),
    ("grok-4-6", 0.002, 0.006),
    ("xai/grok-4.6", 0.002, 0.006),
    ("xai.grok-4.6", 0.002, 0.006),
    ("us.xai.grok-4.6", 0.002, 0.006),
    ("qwen3.8-flash", 0.00015, 0.00047),
    ("qwen3-8-flash", 0.00015, 0.00047),
    ("dashscope/qwen3.8-flash", 0.00015, 0.00047),
    ("qwen3.8-27b", 0.0004, 0.003),
    ("qwen3-8-27b", 0.0004, 0.003),
    ("Qwen/Qwen3.8-27B", 0.0004, 0.003),
    ("gemini-3.7-flash", 0.00075, 0.00375),
    ("gemini-3-7-flash", 0.00075, 0.00375),
    ("gemini/gemini-3.7-flash", 0.00075, 0.00375),
    ("gemini-flash-latest", 0.00075, 0.00375),
    ("sakana-namazu", 0.00095, 0.004),
    ("sakana/sakana-namazu", 0.00095, 0.004),
    ("solar-pro4", 0.0003, 0.0012),
    ("upstage/solar-pro4", 0.0003, 0.0012),
]


@pytest.mark.parametrize("model,prompt_price,completion_price", SEPTEMBER_2026_SWEEP_MODELS)
def test_september_2026_sweep_model_cost(calc, model, prompt_price, completion_price):
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    costs = calc.calculate_granular_cost(model, usage, "chat")
    assert costs["prompt"] == pytest.approx(prompt_price)
    assert costs["completion"] == pytest.approx(completion_price)
    assert costs["total"] == pytest.approx(prompt_price + completion_price)


# Novel dated/preview snapshots must route to the new family via the
# longest-substring fallback rather than collapsing onto a shorter sibling.
SEPTEMBER_2026_SNAPSHOT_ROUTING = [
    ("glm-5.3-20260814", "glm-5.3"),
    ("glm-5.3-flash-20260826", "glm-5.3-flash"),
    ("grok-4.6-0812", "grok-4.6"),
    ("qwen3.8-flash-preview", "qwen3.8-flash"),
    ("gemini-3.7-flash-preview-08-13", "gemini-3.7-flash"),
]


@pytest.mark.parametrize("requested,expected_key", SEPTEMBER_2026_SNAPSHOT_ROUTING)
def test_september_2026_snapshot_alias_routing(calc, requested, expected_key):
    assert calc._normalize_model_name(requested, "chat") == expected_key


def test_glm_5_3_does_not_shadow_or_get_shadowed_by_glm_5(calc):
    """GLM-5.3 must resolve to its own price rather than the older GLM-5 (Feb
    2026) entry that previously shadowed it via a shorter substring match."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    glm5 = calc.calculate_granular_cost("glm-5", usage, "chat")["total"]
    glm53 = calc.calculate_granular_cost("glm-5.3", usage, "chat")["total"]
    assert glm5 == pytest.approx(0.001 + 0.0032)
    assert glm53 == pytest.approx(0.0014 + 0.0044)
    assert glm53 != pytest.approx(glm5)


def test_glm_5_3_flash_does_not_shadow_or_get_shadowed_by_glm_5_3(calc):
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    glm53 = calc.calculate_granular_cost("glm-5.3", usage, "chat")["total"]
    glm53flash = calc.calculate_granular_cost("glm-5.3-flash", usage, "chat")["total"]
    assert glm53flash == pytest.approx(0.000075 + 0.00025)
    assert glm53flash != pytest.approx(glm53)


def test_grok_4_6_does_not_shadow_or_get_shadowed_by_grok_4(calc):
    """Before this entry existed, 'grok-4.6' resolved to the older 'grok-4'
    key ($3/$15 per 1M) instead of xAI's actual $2/$6 per 1M rate."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    grok4 = calc.calculate_granular_cost("grok-4", usage, "chat")["total"]
    grok46 = calc.calculate_granular_cost("grok-4.6", usage, "chat")["total"]
    assert grok4 == pytest.approx(0.003 + 0.015)
    assert grok46 == pytest.approx(0.002 + 0.006)
    assert grok46 != pytest.approx(grok4)


def test_gemini_flash_latest_tracks_gemini_3_7_flash(calc):
    """gemini-flash-latest was repointed to Gemini 3.7 Flash on 2026-08-13; the
    stored price must track the new target, not the stale 2026-05-19 rate the
    alias previously carried."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    latest = calc.calculate_granular_cost("gemini-flash-latest", usage, "chat")["total"]
    flash37 = calc.calculate_granular_cost("gemini-3.7-flash", usage, "chat")["total"]
    assert latest == pytest.approx(flash37)


def test_qwen38_27b_is_not_shadowed_by_qwen38_max(calc):
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    max_ = calc.calculate_granular_cost("qwen3.8-max", usage, "chat")["total"]
    b27 = calc.calculate_granular_cost("qwen3.8-27b", usage, "chat")["total"]
    assert b27 == pytest.approx(0.0004 + 0.003)
    assert b27 != pytest.approx(max_)


def test_grok_imagine_image_2_0_price(calc):
    usage = {"size": "per_image", "quality": "standard", "n": 1}
    cost = calc.calculate_cost("grok-imagine-image-2.0", usage, "image")
    assert cost == pytest.approx(0.04)
    alias_cost = calc.calculate_cost("xai/grok-imagine-image-2.0", usage, "image")
    assert alias_cost == pytest.approx(0.04)


# --- deprecation registry -------------------------------------------------
# A deprecated model still has a real price and still bills, so it must stay
# "table" rather than looking unpriced. What the flag adds is a time signal:
# the id keeps working right up until the provider withdraws it.
DEPRECATED_MODELS = [
    ("moonshot/moonshot-v1-128k", "chat"),
    ("moonshot/moonshot-v1-8k", "chat"),
    ("assemblyai/slam-1", "audio"),
    ("deepgram/nova-2", "audio"),
    ("deepgram/enhanced", "audio"),
    ("gpt-3.5-turbo-0301", "chat"),
]

ACTIVE_MODELS = [
    ("gpt-4o-mini", "chat"),
    ("claude-opus-5", "chat"),
    ("deepgram/nova-3", "audio"),
    ("elevenlabs/scribe_v1", "audio"),
    ("muse-glimmer-30b", "chat"),
]


@pytest.mark.parametrize("model,call_type", DEPRECATED_MODELS)
def test_deprecated_models_report_a_reason(calc, model, call_type):
    reason = calc.deprecation(model, call_type)
    assert reason, f"{model} should be flagged deprecated"
    assert len(reason) > 20, "reason should say why and what to migrate to"


@pytest.mark.parametrize("model,call_type", ACTIVE_MODELS)
def test_active_models_are_not_flagged(calc, model, call_type):
    assert calc.deprecation(model, call_type) is None


@pytest.mark.parametrize("model,call_type", DEPRECATED_MODELS)
def test_deprecated_models_still_price(calc, model, call_type):
    """Deprecation must not zero the price - these still cost real money."""
    assert calc.pricing_source(model, call_type) == "table"


def test_deprecated_registry_is_not_a_pricing_category(calc):
    """The registry is keyed by pricing key, not model name. If it were indexed
    as a category, a call_type of 'deprecated' would resolve names against it."""
    assert calc.pricing_source("moonshot/moonshot-v1-8k", "deprecated") in (
        "table",
        "unpriced",
    )
    assert "deprecated" not in calc._exact_index


def test_deprecation_survives_alias_resolution(calc):
    """Lookup goes through the same alias path as pricing, so a prefixed id and
    the key it resolves to must agree."""
    assert calc.deprecation("deepgram/nova-2-phonecall", "audio")


def test_unknown_model_is_not_flagged(calc):
    assert calc.deprecation("some-model-that-does-not-exist-xyz", "chat") is None


# --- audio price units ----------------------------------------------------
# A bare number cannot say what it is per. Text-to-speech bills per character
# and transcription per second, and storing both as an undifferentiated float
# is what let 42 entries sit at per-minute against a per-second contract.
AUDIO_UNIT_CASES = [
    ("elevenlabs/scribe_v1", {"seconds": 3600}, 0.21996, "per_second"),
    ("eleven_multilingual_v2", {"characters": 1000}, 0.10, "per_1k_chars"),
    ("deepgram/nova-3", {"seconds": 60}, 0.0077, "per_second"),
    ("deepgram/aura-asteria-en", {"characters": 1000}, 0.015, "per_1k_chars"),
    ("assemblyai/best", {"seconds": 3600}, 0.21, "per_second"),
    ("gpt-4o-transcribe", {"tokens": 1000}, 0.0025, "per_1k_tokens"),
]


@pytest.mark.parametrize("model,usage,expected,unit", AUDIO_UNIT_CASES)
def test_audio_declared_units_price_correctly(calc, model, usage, expected, unit):
    assert calc.calculate_cost(model, usage, "audio") == pytest.approx(expected, rel=1e-4)
    key = calc._normalize_model_name(model, "audio")
    assert unit in calc.pricing_data["audio"][key], f"{model} should declare {unit}"


@pytest.mark.parametrize(
    "model,wrong_usage",
    [
        ("elevenlabs/scribe_v1", {"characters": 1000}),  # per-second model billed by character
        ("eleven_multilingual_v2", {"seconds": 3600}),  # per-character model billed by second
        ("deepgram/nova-3", {"characters": 5000}),
    ],
)
def test_audio_unit_mismatch_refuses_rather_than_guessing(calc, model, wrong_usage):
    """The whole point of declaring the unit. Converting seconds to characters is
    not possible, so a silent answer here would be a plausible wrong number -
    exactly the shape of the 60x error this replaces."""
    assert calc.calculate_cost(model, wrong_usage, "audio") == 0.0


def test_every_audio_entry_declares_a_unit_or_is_a_known_legacy(calc):
    """New entries must declare a unit. The remaining bare floats are vendors
    whose billing unit was never established, and are listed explicitly so the
    set cannot grow silently."""
    LEGACY_UNVERIFIED = {
        "cartesia/sonic-2",
        "cartesia/sonic-english",
        "cartesia/sonic-multilingual",
        "gemini-live-2.5-flash-preview-native-audio",
        "hume/evi-2",
        "playht/Play3.0-mini",
        "playht/PlayDialog",
    }
    undeclared = {k for k, v in calc.pricing_data["audio"].items() if not isinstance(v, dict)}
    assert (
        undeclared <= LEGACY_UNVERIFIED
    ), f"audio entries without a declared unit: {sorted(undeclared - LEGACY_UNVERIFIED)}"


def test_legacy_float_entries_still_price(calc):
    """Backwards compatibility: a bare number still works, including for custom
    pricing supplied by users, and keeps the old inferred-unit behaviour."""
    assert calc.calculate_cost("hume/evi-2", {"seconds": 60}, "audio") > 0


# --- prices_checked -------------------------------------------------------
# Records when a price was last confirmed against the vendor's own page. Absence
# means "never verified here", not "correct" - most of the table is inherited
# from an upstream aggregate, which this week's audit showed can be stale by a
# model generation, transposed between tiers, or off by 2.5x.
VENDOR_VERIFIED = [
    ("elevenlabs/scribe_v1", "audio"),
    ("eleven_multilingual_v2", "audio"),
    ("deepgram/nova-3", "audio"),
    ("deepgram/aura-asteria-en", "audio"),
    ("assemblyai/best", "audio"),
    ("fireworks/whisper-v3", "audio"),
    ("deepseek-v4-flash", "chat"),
    ("muse-glimmer-30b", "chat"),
]


@pytest.mark.parametrize("model,call_type", VENDOR_VERIFIED)
def test_verified_prices_carry_a_date(calc, model, call_type):
    from datetime import date

    stamped = calc.price_checked(model, call_type)
    assert stamped, f"{model} was verified against a vendor page and should say when"
    date.fromisoformat(stamped)  # must be a parseable ISO date


def test_unverified_prices_report_none_rather_than_a_guess(calc):
    """None is the honest answer for an inherited number. Stamping everything
    with today's date would make the audit worse than having no dates at all."""
    assert calc.price_checked("gpt-4o-mini", "chat") is None


def test_stale_prices_surfaces_the_unverified_majority(calc):
    stale = calc.stale_prices()
    never = [k for k, d in stale if d is None]
    assert len(never) > 1000, "most of the table has never been vendor-verified"
    assert "gpt-4o-mini" in never
    # Anything checked this week is not stale.
    assert not any(k == "elevenlabs/scribe_v1" for k, _ in stale)


def test_stale_prices_can_exclude_the_unverified(calc):
    aged_only = calc.stale_prices(include_unverified=False)
    assert all(d is not None for _, d in aged_only)


def test_stale_prices_honours_the_age_cutoff(calc):
    """A zero-day cutoff makes even this week's checks stale, which proves the
    date is actually being compared rather than ignored."""
    everything = calc.stale_prices(older_than_days=0, include_unverified=False)
    assert any(k == "elevenlabs/scribe_v1" for k, _ in everything)


def test_metadata_registries_are_not_pricing_categories(calc):
    """Both prices_checked and deprecated are keyed by pricing key. Indexing
    either as a category would let a matching call_type resolve against it."""
    for key in ("prices_checked", "deprecated"):
        assert key not in calc._exact_index
        assert key not in calc._substr_index
