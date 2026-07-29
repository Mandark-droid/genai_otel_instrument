"""Raw-response wrappers must be unwrapped before extraction.

Regression test for #10. LiteLLM calls the OpenAI SDK in raw-response mode so it
can read rate-limit headers, so ``chat.completions.create`` returns
``LegacyAPIResponse`` rather than a ``ChatCompletion``. That wrapper exposes no
``.usage``, so every extractor's ``hasattr(result, "usage")`` check was False and
usage, cost and finish-reason were dropped silently -- spans were created and
request attributes were correct, only the numbers were missing.

The existing suite could not catch this: it exercises the parsed SDK object
directly, which is precisely the shape LiteLLM never hands over.

https://github.com/Mandark-droid/genai_otel_instrument/issues/10
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from genai_otel.instrumentors.base import _unwrap_api_response


def _completion() -> SimpleNamespace:
    """The parsed object an extractor expects to receive."""
    return SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=5, total_tokens=13),
    )


class _RawResponse:
    """Stands in for ``openai._legacy_response.LegacyAPIResponse``.

    Deliberately has no ``usage`` attribute and yields the model only via
    ``parse()`` -- the two properties that made the real wrapper slip through.
    """

    def __init__(self, parsed, *, calls: list | None = None) -> None:
        self._parsed = parsed
        self._calls = calls if calls is not None else []

    def parse(self):
        self._calls.append(1)
        return self._parsed


def test_wrapper_is_unwrapped_to_the_parsed_model() -> None:
    parsed = _completion()
    out = _unwrap_api_response(_RawResponse(parsed))
    assert out is parsed
    assert out.usage.total_tokens == 13


def test_already_parsed_result_is_returned_untouched() -> None:
    """The common path must not pay for the uncommon one."""
    parsed = _completion()
    calls: list = []
    # Has .usage, so .parse() must never be consulted even if one exists.
    parsed.parse = lambda: calls.append(1)  # type: ignore[attr-defined]
    assert _unwrap_api_response(parsed) is parsed
    assert calls == []


def test_unparseable_wrapper_degrades_instead_of_raising() -> None:
    """Instrumentation must never raise into the caller's request."""

    class _Broken:
        def parse(self):
            raise RuntimeError("cannot deserialise")

    broken = _Broken()
    assert _unwrap_api_response(broken) is broken


@pytest.mark.parametrize("value", [None, 42, "text", object()])
def test_objects_without_parse_pass_through(value) -> None:
    assert _unwrap_api_response(value) is value


def test_parse_returning_none_falls_back_to_the_wrapper() -> None:
    """A wrapper that parses to nothing is still better than losing the result."""

    class _Empty:
        def parse(self):
            return None

    empty = _Empty()
    assert _unwrap_api_response(empty) is empty


def test_usage_extraction_sees_through_the_wrapper() -> None:
    """The guarantee that matters: usage is recoverable from a wrapped response.

    `_extract_usage` is the real seam -- `_record_result_metrics` calls it to set
    the `gen_ai.usage.*` attributes, whereas `_extract_response_attributes`
    carries only the response model. Asserting against the wrong seam would pass
    while the tokens stayed missing, which is the class of mistake that let this
    ship.
    """
    from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

    inst = OpenAIInstrumentor.__new__(OpenAIInstrumentor)

    # Straight at the wrapper: nothing recoverable — the original bug.
    assert not inst._extract_usage(_RawResponse(_completion()))

    # Unwrapped first, as `_record_result_metrics` now does.
    usage = inst._extract_usage(_unwrap_api_response(_RawResponse(_completion())))
    assert usage["total_tokens"] == 13
    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == 5


def test_response_model_also_survives_the_wrapper() -> None:
    from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

    inst = OpenAIInstrumentor.__new__(OpenAIInstrumentor)
    attrs = inst._extract_response_attributes(_unwrap_api_response(_RawResponse(_completion())))
    assert attrs.get("gen_ai.response.model") == "gpt-4o-mini"
