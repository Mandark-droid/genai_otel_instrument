"""OpenTelemetry Span Processor for evaluation and safety features.

This module provides a span processor that adds evaluation metrics and safety
checks to GenAI spans, including PII detection, toxicity detection, bias detection,
prompt injection detection, restricted topics, and hallucination detection.
"""

import logging
from typing import Optional

from opentelemetry import metrics
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import Status, StatusCode

from .config import (
    BiasConfig,
    HallucinationConfig,
    PIIConfig,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
    ToxicityConfig,
)
from .pii_detector import PIIDetector

logger = logging.getLogger(__name__)


class EvaluationSpanProcessor(SpanProcessor):
    """Span processor for evaluation and safety features.

    This processor analyzes GenAI spans and adds evaluation metrics and safety
    attributes. It runs checks on prompts and responses based on enabled features.

    Features:
        - PII Detection: Detect and redact personally identifiable information
        - Toxicity Detection: Monitor toxic or harmful content
        - Bias Detection: Detect demographic and other biases
        - Prompt Injection Detection: Protect against prompt injection attacks
        - Restricted Topics: Block sensitive or inappropriate topics
        - Hallucination Detection: Track factual accuracy and groundedness

    All features are opt-in and configured independently.
    """

    def __init__(
        self,
        pii_config: Optional[PIIConfig] = None,
        toxicity_config: Optional[ToxicityConfig] = None,
        bias_config: Optional[BiasConfig] = None,
        prompt_injection_config: Optional[PromptInjectionConfig] = None,
        restricted_topics_config: Optional[RestrictedTopicsConfig] = None,
        hallucination_config: Optional[HallucinationConfig] = None,
    ):
        """Initialize evaluation span processor.

        Args:
            pii_config: PII detection configuration
            toxicity_config: Toxicity detection configuration
            bias_config: Bias detection configuration
            prompt_injection_config: Prompt injection detection configuration
            restricted_topics_config: Restricted topics configuration
            hallucination_config: Hallucination detection configuration
        """
        super().__init__()

        # Store configurations
        self.pii_config = pii_config or PIIConfig()
        self.toxicity_config = toxicity_config or ToxicityConfig()
        self.bias_config = bias_config or BiasConfig()
        self.prompt_injection_config = prompt_injection_config or PromptInjectionConfig()
        self.restricted_topics_config = (
            restricted_topics_config or RestrictedTopicsConfig()
        )
        self.hallucination_config = hallucination_config or HallucinationConfig()

        # Initialize detectors
        self.pii_detector = None
        if self.pii_config.enabled:
            self.pii_detector = PIIDetector(self.pii_config)
            if not self.pii_detector.is_available():
                logger.warning(
                    "PII detector not available, PII detection will use fallback patterns"
                )

        # TODO: Initialize other detectors in future phases
        self.toxicity_detector = None
        self.bias_detector = None
        self.prompt_injection_detector = None
        self.restricted_topics_detector = None
        self.hallucination_detector = None

        # Initialize metrics
        meter = metrics.get_meter(__name__)

        # PII Detection Metrics
        self.pii_detection_counter = meter.create_counter(
            name="genai.evaluation.pii.detections",
            description="Number of PII detections in prompts and responses",
            unit="1",
        )

        self.pii_entity_counter = meter.create_counter(
            name="genai.evaluation.pii.entities",
            description="Number of PII entities detected by type",
            unit="1",
        )

        self.pii_blocked_counter = meter.create_counter(
            name="genai.evaluation.pii.blocked",
            description="Number of requests/responses blocked due to PII",
            unit="1",
        )

        # TODO: Add metrics for other evaluation features in future phases
        # self.toxicity_detection_counter = ...
        # self.bias_detection_counter = ...
        # self.prompt_injection_counter = ...
        # self.restricted_topics_counter = ...
        # self.hallucination_counter = ...

        logger.info("EvaluationSpanProcessor initialized with features:")
        logger.info("  - PII Detection: %s", self.pii_config.enabled)
        logger.info("  - Toxicity Detection: %s", self.toxicity_config.enabled)
        logger.info("  - Bias Detection: %s", self.bias_config.enabled)
        logger.info("  - Prompt Injection Detection: %s", self.prompt_injection_config.enabled)
        logger.info("  - Restricted Topics: %s", self.restricted_topics_config.enabled)
        logger.info("  - Hallucination Detection: %s", self.hallucination_config.enabled)

    def on_start(self, span: Span, parent_context=None) -> None:
        """Called when a span is started.

        For evaluation features, we primarily process on_end when we have the full
        prompt and response data.

        Args:
            span: The span that was started
            parent_context: Parent context (optional)
        """
        # Most evaluation happens on_end, but we can do prompt analysis here if needed
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended.

        This is where we perform evaluation and safety checks on the span's
        prompt and response data.

        Args:
            span: The span that ended
        """
        if not isinstance(span, Span):
            return

        try:
            # Extract prompt and response from span attributes
            attributes = dict(span.attributes) if span.attributes else {}

            prompt = self._extract_prompt(attributes)
            response = self._extract_response(attributes)

            # Run PII detection
            if self.pii_config.enabled and self.pii_detector:
                self._check_pii(span, prompt, response)

            # TODO: Add other checks in future phases
            # if self.toxicity_config.enabled and self.toxicity_detector:
            #     self._check_toxicity(span, prompt, response)
            #
            # if self.bias_config.enabled and self.bias_detector:
            #     self._check_bias(span, prompt, response)
            #
            # if self.prompt_injection_config.enabled and self.prompt_injection_detector:
            #     self._check_prompt_injection(span, prompt)
            #
            # if self.restricted_topics_config.enabled and self.restricted_topics_detector:
            #     self._check_restricted_topics(span, prompt, response)
            #
            # if self.hallucination_config.enabled and self.hallucination_detector:
            #     self._check_hallucination(span, prompt, response, attributes)

        except Exception as e:
            logger.error("Error in evaluation span processor: %s", e, exc_info=True)

    def _extract_prompt(self, attributes: dict) -> Optional[str]:
        """Extract prompt text from span attributes.

        Args:
            attributes: Span attributes

        Returns:
            Optional[str]: Prompt text if found
        """
        # Try different attribute names used by various instrumentors
        prompt_keys = [
            "gen_ai.prompt",
            "gen_ai.prompt.0.content",
            "gen_ai.request.prompt",
            "llm.prompts",
            "gen_ai.content.prompt",
        ]

        for key in prompt_keys:
            if key in attributes:
                value = attributes[key]
                if isinstance(value, str):
                    return value
                elif isinstance(value, list) and value:
                    # Handle list of messages
                    if isinstance(value[0], dict) and "content" in value[0]:
                        return value[0]["content"]
                    elif isinstance(value[0], str):
                        return value[0]

        return None

    def _extract_response(self, attributes: dict) -> Optional[str]:
        """Extract response text from span attributes.

        Args:
            attributes: Span attributes

        Returns:
            Optional[str]: Response text if found
        """
        # Try different attribute names used by various instrumentors
        response_keys = [
            "gen_ai.response",
            "gen_ai.completion",
            "gen_ai.response.0.content",
            "llm.responses",
            "gen_ai.content.completion",
            "gen_ai.response.message.content",
        ]

        for key in response_keys:
            if key in attributes:
                value = attributes[key]
                if isinstance(value, str):
                    return value
                elif isinstance(value, list) and value:
                    # Handle list of messages
                    if isinstance(value[0], dict) and "content" in value[0]:
                        return value[0]["content"]
                    elif isinstance(value[0], str):
                        return value[0]

        return None

    def _check_pii(
        self, span: Span, prompt: Optional[str], response: Optional[str]
    ) -> None:
        """Check for PII in prompt and response.

        Args:
            span: The span to add PII attributes to
            prompt: Prompt text (optional)
            response: Response text (optional)
        """
        if not self.pii_detector:
            return

        try:
            # Check prompt for PII
            if prompt:
                result = self.pii_detector.detect(prompt)
                if result.has_pii:
                    span.set_attribute("evaluation.pii.prompt.detected", True)
                    span.set_attribute(
                        "evaluation.pii.prompt.entity_count", len(result.entities)
                    )
                    span.set_attribute(
                        "evaluation.pii.prompt.entity_types",
                        list(result.entity_counts.keys()),
                    )
                    span.set_attribute("evaluation.pii.prompt.score", result.score)

                    # Record metrics
                    self.pii_detection_counter.add(
                        1, {"location": "prompt", "mode": self.pii_config.mode.value}
                    )

                    # Add entity counts by type
                    for entity_type, count in result.entity_counts.items():
                        span.set_attribute(
                            f"evaluation.pii.prompt.{entity_type.lower()}_count", count
                        )
                        # Record entity metrics
                        self.pii_entity_counter.add(
                            count,
                            {
                                "entity_type": entity_type,
                                "location": "prompt",
                            },
                        )

                    # If blocking, set error status
                    if result.blocked:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                "Request blocked due to PII detection",
                            )
                        )
                        span.set_attribute("evaluation.pii.prompt.blocked", True)
                        # Record blocked metric
                        self.pii_blocked_counter.add(1, {"location": "prompt"})

                    # Add redacted text if available
                    if result.redacted_text:
                        span.set_attribute(
                            "evaluation.pii.prompt.redacted", result.redacted_text
                        )
                else:
                    span.set_attribute("evaluation.pii.prompt.detected", False)

            # Check response for PII
            if response:
                result = self.pii_detector.detect(response)
                if result.has_pii:
                    span.set_attribute("evaluation.pii.response.detected", True)
                    span.set_attribute(
                        "evaluation.pii.response.entity_count", len(result.entities)
                    )
                    span.set_attribute(
                        "evaluation.pii.response.entity_types",
                        list(result.entity_counts.keys()),
                    )
                    span.set_attribute("evaluation.pii.response.score", result.score)

                    # Record metrics
                    self.pii_detection_counter.add(
                        1, {"location": "response", "mode": self.pii_config.mode.value}
                    )

                    # Add entity counts by type
                    for entity_type, count in result.entity_counts.items():
                        span.set_attribute(
                            f"evaluation.pii.response.{entity_type.lower()}_count",
                            count,
                        )
                        # Record entity metrics
                        self.pii_entity_counter.add(
                            count,
                            {
                                "entity_type": entity_type,
                                "location": "response",
                            },
                        )

                    # If blocking, set error status
                    if result.blocked:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                "Response blocked due to PII detection",
                            )
                        )
                        span.set_attribute("evaluation.pii.response.blocked", True)
                        # Record blocked metric
                        self.pii_blocked_counter.add(1, {"location": "response"})

                    # Add redacted text if available
                    if result.redacted_text:
                        span.set_attribute(
                            "evaluation.pii.response.redacted", result.redacted_text
                        )
                else:
                    span.set_attribute("evaluation.pii.response.detected", False)

        except Exception as e:
            logger.error("Error checking PII: %s", e, exc_info=True)
            span.set_attribute("evaluation.pii.error", str(e))

    def shutdown(self) -> None:
        """Shutdown the span processor.

        Called when the tracer provider is shut down.
        """
        logger.info("EvaluationSpanProcessor shutting down")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            bool: True if successful
        """
        # No buffering in this processor
        return True
