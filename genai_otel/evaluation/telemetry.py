"""Safe, bounded provenance and outcome metadata for evaluation spans."""

import hashlib
import json
from typing import Iterable, Mapping, Optional

from genai_otel import __version__

DETECTOR_IMPLEMENTATIONS = {
    "pii": "genai_otel.evaluation.pii_detector.PIIDetector",
    "toxicity": "genai_otel.evaluation.toxicity_detector.ToxicityDetector",
    "bias": "genai_otel.evaluation.bias_detector.BiasDetector",
    "prompt_injection": "genai_otel.evaluation.prompt_injection_detector.PromptInjectionDetector",
    "restricted_topics": "genai_otel.evaluation.restricted_topics_detector.RestrictedTopicsDetector",
    "hallucination": "genai_otel.evaluation.hallucination_detector.HallucinationDetector",
}

OUTCOMES = frozenset(
    {"completed", "no_applicable_input", "unsupported_operation", "skipped", "error"}
)
CONTENT_OUTCOMES = ("captured", "omitted", "truncated", "redacted", "rejected", "error")
_PROMPT_AND_RESPONSE = frozenset({"pii", "toxicity", "bias", "restricted_topics"})


def configuration_fingerprint(detectors: Iterable[str]) -> str:
    """Hash only normalized detector enablement, never detector config values."""
    normalized = {"schema": 1, "enabled_detectors": sorted(set(detectors))}
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _set(target, key: str, value) -> None:
    if isinstance(target, dict):
        target[key] = value
        return
    try:
        target.set_attribute(key, value)
    except (AttributeError, RuntimeError, TypeError):
        attributes = getattr(target, "_attributes", None)
        if attributes is not None:
            try:
                attributes[key] = value
            except (AttributeError, TypeError):
                pass


def _detector_status(
    name: str,
    attributes: Mapping,
    prompt: Optional[str],
    response: Optional[str],
    operation: str,
) -> str:
    if attributes.get("evaluation.skipped"):
        return "skipped"
    if attributes.get(f"evaluation.{name}.error"):
        return "error"

    uses_prompt = name in _PROMPT_AND_RESPONSE or name == "prompt_injection"
    uses_response = name in _PROMPT_AND_RESPONSE or name == "hallucination"
    prompt_present = uses_prompt and isinstance(prompt, str) and bool(prompt.strip())
    response_present = uses_response and isinstance(response, str) and bool(response.strip())
    if not prompt_present and not response_present:
        is_decision = (
            str(attributes.get("openinference.span.kind", "")).upper() == "DECISION"
            or attributes.get("gen_ai.request.type") == "decision"
            or "genai.decision.protocol" in attributes
        )
        if operation in {"embedding", "embeddings", "rerank", "moderation"} or (
            is_decision and name == "hallucination"
        ):
            return "unsupported_operation"
        return "no_applicable_input"

    scope_names = []
    if prompt_present:
        scope_names.append("prompt")
    if response_present:
        scope_names.append("response")
    has_result = any(f"evaluation.{name}.{scope}.detected" in attributes for scope in scope_names)
    return "completed" if has_result else "skipped"


def apply_evaluation_telemetry(
    target,
    attributes: Mapping,
    detectors: Iterable[str],
    prompt: Optional[str],
    response: Optional[str],
    operation: str = "",
    events=(),
) -> None:
    """Write bounded outcomes and provenance without copying configuration or content."""
    enabled = sorted(set(detectors) & set(DETECTOR_IMPLEMENTATIONS))
    if not enabled:
        return

    _set(target, "evaluation.provenance.sdk.name", "genai-otel-instrument")
    _set(target, "evaluation.provenance.sdk.version", str(__version__))
    _set(target, "evaluation.config.fingerprint", configuration_fingerprint(enabled))
    for name in enabled:
        _set(
            target,
            f"evaluation.provenance.detector.{name}.implementation",
            DETECTOR_IMPLEMENTATIONS[name],
        )
        _set(target, f"evaluation.provenance.detector.{name}.version", str(__version__))

    statuses = {
        name: _detector_status(name, attributes, prompt, response, str(operation).lower())
        for name in enabled
    }
    for name, status in statuses.items():
        _set(target, f"evaluation.{name}.outcome", status if status in OUTCOMES else "error")

    if attributes.get("evaluation.skipped"):
        suite_status = "skipped"
    elif "error" in statuses.values():
        suite_status = "error"
    elif "completed" in statuses.values():
        suite_status = "completed"
    elif statuses and all(value == "unsupported_operation" for value in statuses.values()):
        suite_status = "unsupported_operation"
    elif "skipped" in statuses.values():
        suite_status = "skipped"
    else:
        suite_status = "no_applicable_input"
    _set(target, "evaluation.outcome", suite_status)

    captured = any(
        key.startswith(
            ("gen_ai.prompt", "gen_ai.completion", "llm.input_messages", "llm.output_messages")
        )
        or key in {"gen_ai.request.first_message", "gen_ai.response", "input.value", "output.value"}
        for key in attributes
    )
    if events:
        captured = captured or any(
            str(getattr(event, "name", "")).startswith("gen_ai.prompt") for event in events
        )
    content = {"captured"} if captured else {"omitted"}
    if attributes.get("evaluation.truncated"):
        content.add("truncated")
    if any(".redacted" in str(key) for key in attributes):
        content.add("redacted")
    if attributes.get("evaluation.blocked") or attributes.get("evaluation.content.rejected"):
        content.add("rejected")
    if attributes.get("evaluation.content.error"):
        content.add("error")
    _set(
        target,
        "evaluation.content.outcomes",
        [value for value in CONTENT_OUTCOMES if value in content],
    )
