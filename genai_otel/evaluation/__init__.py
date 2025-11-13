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
    PIIConfig,
    ToxicityConfig,
    BiasConfig,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
    HallucinationConfig,
)
from .pii_detector import PIIDetector, PIIDetectionResult
from .span_processor import EvaluationSpanProcessor

__all__ = [
    # Config classes
    "PIIConfig",
    "ToxicityConfig",
    "BiasConfig",
    "PromptInjectionConfig",
    "RestrictedTopicsConfig",
    "HallucinationConfig",
    # Detectors
    "PIIDetector",
    "PIIDetectionResult",
    # Span processor
    "EvaluationSpanProcessor",
]
