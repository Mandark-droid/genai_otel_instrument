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
