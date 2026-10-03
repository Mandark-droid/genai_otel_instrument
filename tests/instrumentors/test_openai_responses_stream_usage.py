"""A streamed Responses API call reports its usage.

A Responses stream ends with a ``response.completed`` event whose usage is at
``event.response.usage``, not ``event.usage``. The stream finalizer reads ``.usage`` from
the last chunk, so a streamed Responses call recorded no tokens and was never priced.
"""

from types import SimpleNamespace

from genai_otel.instrumentors.base import _StreamTiming
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor


def _usage(inp: int, out: int, reasoning: int = 0, cached: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=inp + out,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _event(kind: str, usage=None) -> SimpleNamespace:
    if kind.startswith("response.") and kind.split(".", 1)[1] in (
        "created",
        "in_progress",
        "completed",
        "incomplete",
        "failed",
    ):
        return SimpleNamespace(type=kind, response=SimpleNamespace(usage=usage))
    return SimpleNamespace(type=kind, delta="tok")


def _run(events) -> dict:
    inst = OpenAIInstrumentor()
    timing = _StreamTiming(start_time=0.0)
    for e in events:
        inst._accumulate_stream_usage(timing, e)
    return timing.usage


def test_usage_comes_from_the_completed_event() -> None:
    usage = _run(
        [
            _event("response.created"),
            _event("response.output_text.delta"),
            _event("response.output_text.delta"),
            _event("response.completed", _usage(12, 40, reasoning=8, cached=4)),
        ]
    )
    assert usage["prompt_tokens"] == 12
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 52
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 8}
    assert usage["cache_read_input_tokens"] == 4


def test_an_incomplete_response_still_reports_what_it_used() -> None:
    """max_output_tokens reached: the response is incomplete but was billed."""
    usage = _run(
        [_event("response.output_text.delta"), _event("response.incomplete", _usage(5, 9))]
    )
    assert usage["completion_tokens"] == 9


def test_a_chat_completions_stream_is_left_to_the_last_chunk() -> None:
    chunk = SimpleNamespace(choices=[], usage=None)
    assert _run([chunk, chunk]) is None


def test_events_without_usage_do_not_clear_it() -> None:
    usage = _run([_event("response.completed", _usage(3, 4)), _event("response.output_text.done")])
    assert usage["total_tokens"] == 7
