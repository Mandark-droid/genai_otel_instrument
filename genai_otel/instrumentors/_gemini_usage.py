"""Token usage from a Gemini ``usage_metadata`` object, shared by the Google AI and
Vertex AI instrumentors.

Gemini 2.5+ bills thinking tokens as output but reports them in ``thoughts_token_count``,
not inside ``candidates_token_count``; reading the candidates alone under-counted output
tokens and cost for every thinking call. They are added to the completion count (the
convention OpenAI set: output includes reasoning) and also reported as reasoning tokens.
``cached_content_token_count`` is the part of the prompt served from cache. The
google-genai SDK leaves unset counts as None, so every count defaults to 0.
"""

from typing import Any, Dict, Optional


def _count(usage: Any, snake: str, camel: str) -> int:
    value = getattr(usage, snake, None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        value = getattr(usage, camel, None)
    # Only real numbers count: an absent field (None) is 0, and so is anything else -
    # int() of an arbitrary object (a mock, a proto wrapper) must not invent tokens.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def gemini_usage(usage: Any) -> Optional[Dict[str, Any]]:
    """Canonical usage dict, or None when the response reported no tokens at all."""
    if usage is None:
        return None
    prompt = _count(usage, "prompt_token_count", "promptTokenCount")
    candidates = _count(usage, "candidates_token_count", "candidatesTokenCount")
    thoughts = _count(usage, "thoughts_token_count", "thoughtsTokenCount")
    cached = _count(usage, "cached_content_token_count", "cachedContentTokenCount")
    total = _count(usage, "total_token_count", "totalTokenCount")
    if not (prompt or candidates or thoughts or total):
        return None
    completion = candidates + thoughts
    result: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
    }
    if thoughts:
        result["completion_tokens_details"] = {"reasoning_tokens": thoughts}
    if cached:
        result["cache_read_input_tokens"] = cached
    return result
