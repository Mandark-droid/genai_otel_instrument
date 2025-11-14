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

from .bias_detector import BiasDetectionResult, BiasDetector
from .config import (
    BiasConfig,
    HallucinationConfig,
    PIIConfig,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
    ToxicityConfig,
)
from .hallucination_detector import HallucinationDetector, HallucinationResult
from .pii_detector import PIIDetectionResult, PIIDetector
from .prompt_injection_detector import PromptInjectionDetector, PromptInjectionResult
from .restricted_topics_detector import RestrictedTopicsDetector, RestrictedTopicsResult
from .span_processor import EvaluationSpanProcessor
from .toxicity_detector import ToxicityDetectionResult, ToxicityDetector

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
