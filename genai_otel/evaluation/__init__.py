"""Evaluation and safety features for GenAI observability.

This module provides opt-in evaluation metrics and safety guardrails:

- **PII Detection**: Detect and handle personally identifiable information
- **Toxicity Detection**: Monitor toxic or harmful content
- **Bias Detection**: Detect demographic and other biases
- **Prompt Injection Detection**: Protect against prompt injection attacks
- **Restricted Topics**: Block sensitive or inappropriate topics
- **Hallucination Detection**: Track factual accuracy and groundedness

All features are:
- Opt-in via configuration
- Zero-code for basic usage
- Extensible for custom implementations
- Compatible with existing instrumentation

Example:
    ```python
    from genai_otel import instrument

    # Enable PII detection
    instrument(
        enable_pii_detection=True,
        pii_mode="redact",
        pii_gdpr_mode=True
    )
    ```

Requirements:
    Install optional dependencies:
    ```bash
    pip install genai-otel-instrument[evaluation]
    ```
"""

from .config import (
    BiasConfig,
    HallucinationConfig,
    PIIConfig,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
    ToxicityConfig,
)

# Lazy imports for detector classes and span processor - these pull in heavy ML
# dependencies (spacy, torch, detoxify, presidio) that take 10-20s to import.
# Only loaded when actually accessed, not at package import time.
_LAZY_IMPORTS = {
    "BiasDetector": ".bias_detector",
    "BiasDetectionResult": ".bias_detector",
    "HallucinationDetector": ".hallucination_detector",
    "HallucinationResult": ".hallucination_detector",
    "PIIDetector": ".pii_detector",
    "PIIDetectionResult": ".pii_detector",
    "PromptInjectionDetector": ".prompt_injection_detector",
    "PromptInjectionResult": ".prompt_injection_detector",
    "RestrictedTopicsDetector": ".restricted_topics_detector",
    "RestrictedTopicsResult": ".restricted_topics_detector",
    "ToxicityDetector": ".toxicity_detector",
    "ToxicityDetectionResult": ".toxicity_detector",
    "EvaluationSpanProcessor": ".span_processor",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config classes
    "BiasConfig",
    "HallucinationConfig",
    "PIIConfig",
    "PromptInjectionConfig",
    "RestrictedTopicsConfig",
    "ToxicityConfig",
    # Detectors
    "BiasDetector",
    "BiasDetectionResult",
    "HallucinationDetector",
    "HallucinationResult",
    "PIIDetector",
    "PIIDetectionResult",
    "PromptInjectionDetector",
    "PromptInjectionResult",
    "RestrictedTopicsDetector",
    "RestrictedTopicsResult",
    "ToxicityDetector",
    "ToxicityDetectionResult",
    # Span processor
    "EvaluationSpanProcessor",
]
