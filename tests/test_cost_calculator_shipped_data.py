"""The calculator handles every shape the shipped pricing file actually contains.

Found while generating a cross-language contract from v1.30.0:
- 9 embeddings entries are objects ({promptPrice, completionPrice, note}), not numbers,
  and 4 image entries are bare numbers, not quality/size tables; calculate_cost raised
  TypeError on all 13 and the span silently got no cost;
- the speech_to_text table (40 entries) had no call-type branch, so it was unreachable;
- pricing_source() looked embedding and image calls up in the CHAT table (dall-e-3
  reported "unpriced", text-embedding-3-small "estimated");
- usage with completion_tokens_details=None, or reasoning_tokens=None, raised TypeError;
- five chat keys exist twice differing only by case, so one of each pair is unreachable.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from genai_otel.cost_calculator import CostCalculator

PRICING = json.loads(
    (Path(__file__).parent.parent / "genai_otel" / "llm_pricing.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def calc():
    return CostCalculator()


def test_object_shaped_embedding_entries_are_priced(calc):
    # nvidia/NV-Embed-v2 is {"promptPrice": 5e-05, ...}: 2,000 tokens -> 2 x 5e-05
    assert calc.calculate_cost(
        "nvidia/NV-Embed-v2", {"prompt_tokens": 2000}, "embedding"
    ) == pytest.approx(1e-04)


def test_number_shaped_image_entries_are_a_flat_price_per_image(calc):
    assert calc.calculate_cost("recraft/recraftv3", {"n": 2}, "image") == pytest.approx(0.08)


def test_no_shipped_entry_makes_the_calculator_raise(calc):
    usages = {
        "chat": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
        "embedding": {"prompt_tokens": 1000},
        "image": {"n": 1, "size": "1024x1024", "quality": "standard"},
        "speech_to_text": {"prompt_tokens": 1000, "completion_tokens": 1000},
    }
    sections = {
        "chat": "chat",
        "embeddings": "embedding",
        "images": "image",
        "speech_to_text": "speech_to_text",
    }
    for section, call_type in sections.items():
        for model in PRICING[section]:
            calc.calculate_cost(model, usages[call_type], call_type)  # must not raise


def test_speech_to_text_is_reachable(calc):
    cost = calc.calculate_cost(
        "nvidia/parakeet-tdt-0.6b-v2",
        {"prompt_tokens": 1000, "completion_tokens": 1000},
        "speech_to_text",
    )
    assert cost == pytest.approx(0.0003)


def test_pricing_source_reads_the_right_table(calc):
    assert calc.pricing_source("dall-e-3", "image") == "table"
    assert calc.pricing_source("text-embedding-3-small", "embedding") == "table"
    assert calc.pricing_source("nvidia/parakeet-tdt-0.6b-v2", "speech_to_text") == "table"


@pytest.mark.parametrize("details", [None, {}, {"reasoning_tokens": None}])
def test_missing_reasoning_count_does_not_raise(calc, details):
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 100,
        "total_tokens": 200,
        "completion_tokens_details": details,
    }
    costs = calc.calculate_granular_cost("gpt-4o", usage, "chat")
    assert costs["total"] > 0
    assert costs["reasoning"] == 0.0


def test_no_chat_key_is_shadowed_by_a_case_twin():
    lowered = Counter(k.lower() for k in PRICING["chat"])
    assert [k for k, n in lowered.items() if n > 1] == []
