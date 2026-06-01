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
    ("gpt-5.5-mini", 0.0004, 0.0016),
    ("gpt-5.5-nano", 0.0001, 0.0004),
    ("gpt-5.5-pro", 0.03, 0.18),
    ("gemini-3.5-flash", 0.0015, 0.009),
    ("gemini/gemini-3.5-flash", 0.0015, 0.009),
    ("MiniMax-M3", 0.0003, 0.0012),
    ("minimax-m3", 0.0003, 0.0012),
    ("MiniMax-M3-highspeed", 0.0006, 0.0024),
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
    ("gemini-3.5-flash-preview-05-19", "gemini-3.5-flash"),
]


@pytest.mark.parametrize("requested,expected_key", SNAPSHOT_ROUTING)
def test_snapshot_alias_routing(calc, requested, expected_key):
    assert calc._normalize_model_name(requested, "chat") == expected_key
