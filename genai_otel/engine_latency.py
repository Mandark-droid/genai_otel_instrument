"""Latency breakdown for self-hosted inference engines.

vLLM and SGLang independently converged on an identical ``gen_ai.latency.*``
span-attribute set -- ``vllm/tracing/utils.py`` and
``sglang/srt/observability/trace.py`` each define it, and each says in comments
that it mirrors the OTel GenAI conventions and adds the latency keys because
the spec lacks them. `semantic-conventions-genai#408
<https://github.com/open-telemetry/semantic-conventions-genai/issues/408>`_
proposes adopting that set.

This module emits those keys verbatim rather than inventing a parallel
vocabulary: an operator who runs vLLM's own OTLP tracing alongside this library
should get one set of names across both, not two that have to be joined.

Everything here is defensive. Engine internals change between releases, and a
missing or renamed timing field must degrade to "that phase was not measured"
rather than raise inside a wrapped inference call.
"""

import logging
from typing import Any, Dict, Optional

from .semconv import SemanticConvention as SC

logger = logging.getLogger(__name__)

# Timings below this are noise from clock granularity rather than real work.
_MIN_DURATION_SECONDS = 0.0


def _seconds(value: Any) -> Optional[float]:
    """Coerce a timing to a non-negative float, or None if it is not usable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    if value < _MIN_DURATION_SECONDS:
        return None
    return value


def _delta(end: Any, start: Any) -> Optional[float]:
    """Duration between two engine timestamps, or None if either is missing.

    Engines populate these timestamps progressively, so a request that errored
    or is still streaming legitimately has some of them unset. Returning None
    leaves the attribute off the span instead of recording a negative or
    nonsensical duration.
    """
    end_s, start_s = _seconds(end), _seconds(start)
    if end_s is None or start_s is None:
        return None
    duration = end_s - start_s
    return duration if duration >= 0 else None


def vllm_latency_attributes(metrics: Any) -> Dict[str, float]:
    """Derive the latency attribute set from a vLLM ``RequestMetrics``.

    ``RequestOutput.metrics`` carries absolute timestamps (``arrival_time``,
    ``first_scheduled_time``, ``first_token_time``, ``last_token_time``,
    ``finished_time``) plus a few pre-computed durations. The phase durations
    below are the differences vLLM's own tracing takes.

    ``model_forward_time`` is in **milliseconds** in vLLM while every other
    field is in seconds; it is converted here so the whole attribute set shares
    one unit.
    """
    if metrics is None:
        return {}

    def field(name: str) -> Any:
        return getattr(metrics, name, None)

    arrival = field("arrival_time")
    first_scheduled = field("first_scheduled_time")
    first_token = field("first_token_time")
    last_token = field("last_token_time")
    finished = field("finished_time")

    candidates = {
        SC.GEN_AI_LATENCY_TIME_IN_QUEUE: _seconds(field("time_in_queue")),
        SC.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: _delta(first_token, arrival),
        SC.GEN_AI_LATENCY_E2E: _delta(finished, arrival),
        SC.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: _delta(first_token, first_scheduled),
        SC.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: _delta(last_token, first_token),
        SC.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: _delta(last_token, first_scheduled),
        SC.GEN_AI_LATENCY_TIME_IN_SCHEDULER: _seconds(field("scheduler_time")),
        SC.GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE: _seconds(field("model_execute_time")),
    }

    # vLLM reports this one in milliseconds; normalise to seconds.
    forward_ms = _seconds(field("model_forward_time"))
    if forward_ms is not None:
        candidates[SC.GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD] = forward_ms / 1000.0

    return {k: v for k, v in candidates.items() if v is not None}


def sglang_latency_attributes(meta_info: Any) -> Dict[str, float]:
    """Derive latency attributes from an SGLang ``meta_info`` mapping.

    SGLang returns per-request metadata alongside the generated text. Field
    coverage varies by version, so each key is read independently and any that
    is absent simply does not produce an attribute.
    """
    if not isinstance(meta_info, dict):
        return {}

    candidates = {
        SC.GEN_AI_LATENCY_E2E: _seconds(meta_info.get("e2e_latency")),
        SC.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: _seconds(meta_info.get("ttft")),
        SC.GEN_AI_LATENCY_TIME_IN_QUEUE: _seconds(meta_info.get("queue_time")),
        SC.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: _seconds(meta_info.get("prefill_latency")),
        SC.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: _seconds(meta_info.get("decode_latency")),
    }
    return {k: v for k, v in candidates.items() if v is not None}


def llamacpp_latency_attributes(timings: Any) -> Dict[str, float]:
    """Derive latency attributes from a llama.cpp ``timings`` object.

    llama.cpp reports durations in **milliseconds** (``prompt_ms``,
    ``predicted_ms``), which map cleanly onto prefill and decode. They are
    converted to seconds so this attribute set matches the other engines'.
    """
    if not isinstance(timings, dict):
        return {}

    attrs: Dict[str, float] = {}
    prompt_ms = _seconds(timings.get("prompt_ms"))
    predicted_ms = _seconds(timings.get("predicted_ms"))

    if prompt_ms is not None:
        attrs[SC.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL] = prompt_ms / 1000.0
        # llama.cpp has no scheduler queue in-process, so prefill completing is
        # the first token becoming available.
        attrs[SC.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN] = prompt_ms / 1000.0
    if predicted_ms is not None:
        attrs[SC.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE] = predicted_ms / 1000.0
    if prompt_ms is not None and predicted_ms is not None:
        total = (prompt_ms + predicted_ms) / 1000.0
        attrs[SC.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE] = total
        attrs[SC.GEN_AI_LATENCY_E2E] = total

    return attrs


def apply_latency_attributes(attrs: Dict[str, Any], latency: Dict[str, float]) -> Dict[str, Any]:
    """Merge derived latency attributes into an attribute dict, in place."""
    for key, value in latency.items():
        attrs[key] = value
    return attrs
