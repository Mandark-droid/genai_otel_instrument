"""Gemini thinking and cached tokens are counted.

Gemini 2.5+ bills thinking tokens as output but reports them in `thoughts_token_count`,
NOT inside `candidates_token_count`. Both instrumentors reported only the candidates, so a
thinking-heavy call under-counted its output tokens and its cost. Cached prompt tokens
(`cached_content_token_count`, part of the prompt count, billed at the cache rate) were not
reported at all. The google-genai SDK also leaves unset counts as None, not 0.
"""

from types import SimpleNamespace

import pytest

from genai_otel.instrumentors.google_ai_instrumentor import GoogleAIInstrumentor
from genai_otel.instrumentors.vertexai_instrumentor import VertexAIInstrumentor

INSTRUMENTORS = [GoogleAIInstrumentor, VertexAIInstrumentor]


def _response(**counts) -> SimpleNamespace:
    fields = {
        "prompt_token_count": None,
        "candidates_token_count": None,
        "total_token_count": None,
        "thoughts_token_count": None,
        "cached_content_token_count": None,
    }
    fields.update(counts)
    return SimpleNamespace(usage_metadata=SimpleNamespace(**fields))


@pytest.mark.parametrize("cls", INSTRUMENTORS)
def test_thinking_tokens_are_output_tokens(cls) -> None:
    usage = cls()._extract_usage(
        _response(
            prompt_token_count=20,
            candidates_token_count=10,
            thoughts_token_count=50,
            total_token_count=80,
        )
    )
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 60
    assert usage["total_tokens"] == 80
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 50}


@pytest.mark.parametrize("cls", INSTRUMENTORS)
def test_cached_prompt_tokens_are_reported(cls) -> None:
    usage = cls()._extract_usage(
        _response(
            prompt_token_count=100,
            candidates_token_count=5,
            cached_content_token_count=64,
            total_token_count=105,
        )
    )
    assert usage["cache_read_input_tokens"] == 64
    assert usage["prompt_tokens"] == 100  # cached tokens are part of the prompt count


@pytest.mark.parametrize("cls", INSTRUMENTORS)
def test_unset_counts_are_zero_not_none(cls) -> None:
    usage = cls()._extract_usage(_response(prompt_token_count=7, candidates_token_count=3))
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_a_thinking_call_costs_its_thinking() -> None:
    from genai_otel.cost_calculator import CostCalculator

    usage = GoogleAIInstrumentor()._extract_usage(
        _response(
            prompt_token_count=1000,
            candidates_token_count=100,
            thoughts_token_count=900,
            total_token_count=2000,
        )
    )
    calc = CostCalculator()
    with_thinking = calc.calculate_cost("gemini-2.5-pro", usage, "chat")
    without = calc.calculate_cost(
        "gemini-2.5-pro",
        {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
        "chat",
    )
    assert with_thinking > without * 3
