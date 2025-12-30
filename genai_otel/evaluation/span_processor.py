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

from .bias_detector import BiasDetector
from .config import (
    BiasConfig,
    HallucinationConfig,
    PIIConfig,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
    ToxicityConfig,
)
from .hallucination_detector import HallucinationDetector
from .pii_detector import PIIDetector
from .prompt_injection_detector import PromptInjectionDetector
from .restricted_topics_detector import RestrictedTopicsDetector
from .toxicity_detector import ToxicityDetector

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
        self.restricted_topics_config = restricted_topics_config or RestrictedTopicsConfig()
        self.hallucination_config = hallucination_config or HallucinationConfig()

        # Initialize detectors
        self.pii_detector = None
        if self.pii_config.enabled:
            self.pii_detector = PIIDetector(self.pii_config)
            if not self.pii_detector.is_available():
                logger.warning(
                    "PII detector not available, PII detection will use fallback patterns"
                )

        self.toxicity_detector = None
        if self.toxicity_config.enabled:
            self.toxicity_detector = ToxicityDetector(self.toxicity_config)
            if not self.toxicity_detector.is_available():
                logger.warning(
                    "Toxicity detector not available, please install either:\n"
                    "  - Perspective API: pip install google-api-python-client\n"
                    "  - Detoxify: pip install detoxify"
                )

        self.bias_detector = None
        if self.bias_config.enabled:
            self.bias_detector = BiasDetector(self.bias_config)
            if not self.bias_detector.is_available():
                logger.warning(
                    "Bias detector not available (pattern-based detection always available)"
                )

        self.prompt_injection_detector = None
        if self.prompt_injection_config.enabled:
            self.prompt_injection_detector = PromptInjectionDetector(self.prompt_injection_config)

        self.restricted_topics_detector = None
        if self.restricted_topics_config.enabled:
            self.restricted_topics_detector = RestrictedTopicsDetector(
                self.restricted_topics_config
            )

        self.hallucination_detector = None
        if self.hallucination_config.enabled:
            self.hallucination_detector = HallucinationDetector(self.hallucination_config)

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

        # Toxicity Detection Metrics
        self.toxicity_detection_counter = meter.create_counter(
            name="genai.evaluation.toxicity.detections",
            description="Number of toxicity detections in prompts and responses",
            unit="1",
        )

        self.toxicity_category_counter = meter.create_counter(
            name="genai.evaluation.toxicity.categories",
            description="Toxicity detections by category",
            unit="1",
        )

        self.toxicity_blocked_counter = meter.create_counter(
            name="genai.evaluation.toxicity.blocked",
            description="Number of requests/responses blocked due to toxicity",
            unit="1",
        )

        self.toxicity_score_histogram = meter.create_histogram(
            name="genai.evaluation.toxicity.score",
            description="Toxicity score distribution",
            unit="1",
        )

        # Bias Detection Metrics
        self.bias_detection_counter = meter.create_counter(
            name="genai.evaluation.bias.detections",
            description="Number of bias detections in prompts and responses",
            unit="1",
        )

        self.bias_type_counter = meter.create_counter(
            name="genai.evaluation.bias.types",
            description="Bias detections by type",
            unit="1",
        )

        self.bias_blocked_counter = meter.create_counter(
            name="genai.evaluation.bias.blocked",
            description="Number of requests/responses blocked due to bias",
            unit="1",
        )

        self.bias_score_histogram = meter.create_histogram(
            name="genai.evaluation.bias.score",
            description="Bias score distribution",
            unit="1",
        )

        # Prompt Injection Detection Metrics
        self.prompt_injection_counter = meter.create_counter(
            name="genai.evaluation.prompt_injection.detections",
            description="Number of prompt injection attempts detected",
            unit="1",
        )

        self.prompt_injection_type_counter = meter.create_counter(
            name="genai.evaluation.prompt_injection.types",
            description="Prompt injection detections by type",
            unit="1",
        )

        self.prompt_injection_blocked_counter = meter.create_counter(
            name="genai.evaluation.prompt_injection.blocked",
            description="Number of requests blocked due to prompt injection",
            unit="1",
        )

        self.prompt_injection_score_histogram = meter.create_histogram(
            name="genai.evaluation.prompt_injection.score",
            description="Prompt injection score distribution",
            unit="1",
        )

        # Restricted Topics Metrics
        self.restricted_topics_counter = meter.create_counter(
            name="genai.evaluation.restricted_topics.detections",
            description="Number of restricted topics detected",
            unit="1",
        )

        self.restricted_topics_type_counter = meter.create_counter(
            name="genai.evaluation.restricted_topics.types",
            description="Restricted topics by type",
            unit="1",
        )

        self.restricted_topics_blocked_counter = meter.create_counter(
            name="genai.evaluation.restricted_topics.blocked",
            description="Number of requests/responses blocked due to restricted topics",
            unit="1",
        )

        self.restricted_topics_score_histogram = meter.create_histogram(
            name="genai.evaluation.restricted_topics.score",
            description="Restricted topics score distribution",
            unit="1",
        )

        # Hallucination Detection Metrics
        self.hallucination_counter = meter.create_counter(
            name="genai.evaluation.hallucination.detections",
            description="Number of potential hallucinations detected",
            unit="1",
        )

        self.hallucination_indicator_counter = meter.create_counter(
            name="genai.evaluation.hallucination.indicators",
            description="Hallucination detections by indicator type",
            unit="1",
        )

        self.hallucination_score_histogram = meter.create_histogram(
            name="genai.evaluation.hallucination.score",
            description="Hallucination score distribution",
            unit="1",
        )

        logger.info("EvaluationSpanProcessor initialized with features:")
        logger.info("  - PII Detection: %s", self.pii_config.enabled)
        logger.info("  - Toxicity Detection: %s", self.toxicity_config.enabled)
        logger.info("  - Bias Detection: %s", self.bias_config.enabled)
        logger.info("  - Prompt Injection Detection: %s", self.prompt_injection_config.enabled)
        logger.info("  - Restricted Topics: %s", self.restricted_topics_config.enabled)
        logger.info("  - Hallucination Detection: %s", self.hallucination_config.enabled)

        # Register detectors with BaseInstrumentor so they can run before span ends
        try:
            from genai_otel.instrumentors.base import BaseInstrumentor

            logger.info(
                f"Attempting to register detectors: pii_detector={self.pii_detector is not None}, toxicity_detector={self.toxicity_detector is not None}"
            )
            with BaseInstrumentor._evaluation_lock:
                if self.pii_detector:
                    BaseInstrumentor._pii_detector = self.pii_detector
                    logger.info("Registered PII detector with BaseInstrumentor")
                else:
                    logger.warning("PII detector is None, not registering")
                if self.toxicity_detector:
                    BaseInstrumentor._toxicity_detector = self.toxicity_detector
                    logger.info("Registered Toxicity detector with BaseInstrumentor")
                else:
                    logger.warning("Toxicity detector is None, not registering")
        except Exception as e:
            logger.warning(
                f"Failed to register detectors with BaseInstrumentor: {e}", exc_info=True
            )

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

        Note: This method runs AFTER span.end() is called, so the span is
        a ReadableSpan and cannot have attributes modified. Attributes must
        be set in BaseInstrumentor before span.end(). However, we still
        extract data and record metrics here.

        Args:
            span: The span that ended
        """
        # Don't check isinstance - span is ReadableSpan after end()

        try:
            # Helper to safely set attributes (will fail silently on ReadableSpan)
            def safe_set_attribute(key, value):
                try:
                    span.set_attribute(key, value)
                except AttributeError:
                    # ReadableSpan doesn't support set_attribute - attributes were already set by BaseInstrumentor
                    pass

            # Extract prompt and response from span attributes
            attributes = dict(span.attributes) if span.attributes else {}

            prompt = self._extract_prompt(attributes)
            response = self._extract_response(attributes)

            # Run PII detection
            if self.pii_config.enabled and self.pii_detector:
                self._check_pii(span, prompt, response, safe_set_attribute)

            # Run toxicity detection
            if self.toxicity_config.enabled and self.toxicity_detector:
                self._check_toxicity(span, prompt, response)

            # Run bias detection
            if self.bias_config.enabled and self.bias_detector:
                self._check_bias(span, prompt, response)

            # Run prompt injection detection (prompts only)
            if self.prompt_injection_config.enabled and self.prompt_injection_detector:
                self._check_prompt_injection(span, prompt)

            # Run restricted topics detection
            if self.restricted_topics_config.enabled and self.restricted_topics_detector:
                self._check_restricted_topics(span, prompt, response)

            # Run hallucination detection (responses only)
            if self.hallucination_config.enabled and self.hallucination_detector:
                self._check_hallucination(span, prompt, response, attributes)

        except Exception as e:
            logger.error("Error in evaluation span processor: %s", e, exc_info=True)

    def _extract_prompt(self, attributes: dict) -> Optional[str]:
        """Extract prompt text from span attributes.

        Args:
            attributes: Span attributes

        Returns:
            Optional[str]: Prompt text if found
        """
        import ast
        import json

        # Try different attribute names used by various instrumentors
        prompt_keys = [
            "gen_ai.prompt",
            "gen_ai.prompt.0.content",
            "gen_ai.request.prompt",
            "gen_ai.request.first_message",  # OpenAI instrumentor
            "llm.prompts",
            "gen_ai.content.prompt",
        ]

        for key in prompt_keys:
            if key in attributes:
                value = attributes[key]

                # Handle string values
                if isinstance(value, str):
                    # Check if it's a JSON/dict string (starts with '{' or '[')
                    if value.strip().startswith(("{", "[")):
                        try:
                            # Try JSON parsing first (handles proper JSON)
                            parsed = json.loads(value)
                            if isinstance(parsed, dict) and "content" in parsed:
                                return parsed["content"]
                            elif isinstance(parsed, list) and parsed:
                                if isinstance(parsed[0], dict) and "content" in parsed[0]:
                                    return parsed[0]["content"]
                                elif isinstance(parsed[0], str):
                                    return parsed[0]
                        except json.JSONDecodeError:
                            # Try ast.literal_eval for Python dict strings
                            try:
                                parsed = ast.literal_eval(value)
                                if isinstance(parsed, dict) and "content" in parsed:
                                    return parsed["content"]
                                elif isinstance(parsed, list) and parsed:
                                    if isinstance(parsed[0], dict) and "content" in parsed[0]:
                                        return parsed[0]["content"]
                                    elif isinstance(parsed[0], str):
                                        return parsed[0]
                            except (ValueError, SyntaxError):
                                # If parsing fails, treat as plain string
                                return value
                    else:
                        # Plain string prompt
                        return value

                # Handle actual dict/list objects (though rare with OTel attributes)
                elif isinstance(value, dict) and "content" in value:
                    return value["content"]
                elif isinstance(value, list) and value:
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
        self, span: Span, prompt: Optional[str], response: Optional[str], safe_set_attribute=None
    ) -> None:
        """Check for PII in prompt and response.

        Args:
            span: The span to add PII attributes to
            prompt: Prompt text (optional)
            response: Response text (optional)
            safe_set_attribute: Function to safely set attributes (optional, for ReadableSpan)
        """
        if not self.pii_detector:
            return

        # If no safe_set_attribute provided, use span.set_attribute directly
        if safe_set_attribute is None:
            safe_set_attribute = span.set_attribute

        try:
            # Check prompt for PII
            if prompt:
                result = self.pii_detector.detect(prompt)
                if result.has_pii:
                    safe_set_attribute("evaluation.pii.prompt.detected", True)
                    safe_set_attribute("evaluation.pii.prompt.entity_count", len(result.entities))
                    safe_set_attribute(
                        "evaluation.pii.prompt.entity_types",
                        list(result.entity_counts.keys()),
                    )
                    safe_set_attribute("evaluation.pii.prompt.score", result.score)

                    # Record metrics
                    self.pii_detection_counter.add(
                        1, {"location": "prompt", "mode": self.pii_config.mode.value}
                    )

                    # Add entity counts by type
                    for entity_type, count in result.entity_counts.items():
                        safe_set_attribute(
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
                        try:
                            span.set_status(
                                Status(
                                    StatusCode.ERROR,
                                    "Request blocked due to PII detection",
                                )
                            )
                        except AttributeError:
                            pass
                        safe_set_attribute("evaluation.pii.prompt.blocked", True)
                        # Record blocked metric
                        self.pii_blocked_counter.add(1, {"location": "prompt"})

                    # Add redacted text if available
                    if result.redacted_text:
                        safe_set_attribute("evaluation.pii.prompt.redacted", result.redacted_text)
                else:
                    safe_set_attribute("evaluation.pii.prompt.detected", False)

            # Check response for PII
            if response:
                result = self.pii_detector.detect(response)
                if result.has_pii:
                    safe_set_attribute("evaluation.pii.response.detected", True)
                    safe_set_attribute("evaluation.pii.response.entity_count", len(result.entities))
                    safe_set_attribute(
                        "evaluation.pii.response.entity_types",
                        list(result.entity_counts.keys()),
                    )
                    safe_set_attribute("evaluation.pii.response.score", result.score)

                    # Record metrics
                    self.pii_detection_counter.add(
                        1, {"location": "response", "mode": self.pii_config.mode.value}
                    )

                    # Add entity counts by type
                    for entity_type, count in result.entity_counts.items():
                        safe_set_attribute(
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
                        try:
                            span.set_status(
                                Status(
                                    StatusCode.ERROR,
                                    "Response blocked due to PII detection",
                                )
                            )
                        except AttributeError:
                            pass
                        safe_set_attribute("evaluation.pii.response.blocked", True)
                        # Record blocked metric
                        self.pii_blocked_counter.add(1, {"location": "response"})

                    # Add redacted text if available
                    if result.redacted_text:
                        safe_set_attribute("evaluation.pii.response.redacted", result.redacted_text)
                else:
                    safe_set_attribute("evaluation.pii.response.detected", False)

        except Exception as e:
            logger.error("Error checking PII: %s", e, exc_info=True)
            safe_set_attribute("evaluation.pii.error", str(e))

    def _check_toxicity(self, span: Span, prompt: Optional[str], response: Optional[str]) -> None:
        """Check for toxicity in prompt and response.

        Args:
            span: The span to add toxicity attributes to
            prompt: Prompt text (optional)
            response: Response text (optional)
        """
        if not self.toxicity_detector:
            return

        try:
            # Check prompt for toxicity
            if prompt:
                result = self.toxicity_detector.detect(prompt)
                if result.is_toxic:
                    span.set_attribute("evaluation.toxicity.prompt.detected", True)
                    span.set_attribute("evaluation.toxicity.prompt.max_score", result.max_score)
                    span.set_attribute(
                        "evaluation.toxicity.prompt.categories",
                        result.toxic_categories,
                    )

                    # Add individual category scores
                    for category, score in result.scores.items():
                        span.set_attribute(f"evaluation.toxicity.prompt.{category}_score", score)

                    # Record metrics
                    self.toxicity_detection_counter.add(1, {"location": "prompt"})
                    self.toxicity_score_histogram.record(result.max_score, {"location": "prompt"})

                    # Record category metrics
                    for category in result.toxic_categories:
                        self.toxicity_category_counter.add(
                            1, {"category": category, "location": "prompt"}
                        )

                    # If blocking, set error status
                    if result.blocked:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                "Request blocked due to toxic content",
                            )
                        )
                        span.set_attribute("evaluation.toxicity.prompt.blocked", True)
                        self.toxicity_blocked_counter.add(1, {"location": "prompt"})
                else:
                    span.set_attribute("evaluation.toxicity.prompt.detected", False)

            # Check response for toxicity
            if response:
                result = self.toxicity_detector.detect(response)
                if result.is_toxic:
                    span.set_attribute("evaluation.toxicity.response.detected", True)
                    span.set_attribute("evaluation.toxicity.response.max_score", result.max_score)
                    span.set_attribute(
                        "evaluation.toxicity.response.categories",
                        result.toxic_categories,
                    )

                    # Add individual category scores
                    for category, score in result.scores.items():
                        span.set_attribute(f"evaluation.toxicity.response.{category}_score", score)

                    # Record metrics
                    self.toxicity_detection_counter.add(1, {"location": "response"})
                    self.toxicity_score_histogram.record(result.max_score, {"location": "response"})

                    # Record category metrics
                    for category in result.toxic_categories:
                        self.toxicity_category_counter.add(
                            1, {"category": category, "location": "response"}
                        )

                    # If blocking, set error status
                    if result.blocked:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                "Response blocked due to toxic content",
                            )
                        )
                        span.set_attribute("evaluation.toxicity.response.blocked", True)
                        self.toxicity_blocked_counter.add(1, {"location": "response"})
                else:
                    span.set_attribute("evaluation.toxicity.response.detected", False)

        except Exception as e:
            logger.error("Error checking toxicity: %s", e, exc_info=True)
            span.set_attribute("evaluation.toxicity.error", str(e))

    def _check_bias(self, span: Span, prompt: Optional[str], response: Optional[str]) -> None:
        """Check for bias in prompt and response.

        Args:
            span: The span to add bias attributes to
            prompt: Prompt text (optional)
            response: Response text (optional)
        """
        if not self.bias_detector:
            return

        try:
            # Check prompt for bias
            if prompt:
                result = self.bias_detector.detect(prompt)
                if result.has_bias:
                    span.set_attribute("evaluation.bias.prompt.detected", True)
                    span.set_attribute("evaluation.bias.prompt.max_score", result.max_score)
                    span.set_attribute(
                        "evaluation.bias.prompt.detected_biases",
                        result.detected_biases,
                    )

                    # Add individual bias type scores
                    for bias_type, score in result.bias_scores.items():
                        if score > 0:
                            span.set_attribute(f"evaluation.bias.prompt.{bias_type}_score", score)

                    # Add patterns matched
                    for bias_type, patterns in result.patterns_matched.items():
                        span.set_attribute(
                            f"evaluation.bias.prompt.{bias_type}_patterns",
                            patterns[:5],  # Limit to first 5 patterns
                        )

                    # Record metrics
                    self.bias_detection_counter.add(1, {"location": "prompt"})
                    self.bias_score_histogram.record(result.max_score, {"location": "prompt"})

                    # Record bias type metrics
                    for bias_type in result.detected_biases:
                        self.bias_type_counter.add(
                            1, {"bias_type": bias_type, "location": "prompt"}
                        )

                    # If blocking mode and threshold exceeded, set error status
                    if self.bias_config.block_on_detection and result.has_bias:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                "Request blocked due to bias detection",
                            )
                        )
                        span.set_attribute("evaluation.bias.prompt.blocked", True)
                        self.bias_blocked_counter.add(1, {"location": "prompt"})
                else:
                    span.set_attribute("evaluation.bias.prompt.detected", False)

            # Check response for bias
            if response:
                result = self.bias_detector.detect(response)
                if result.has_bias:
                    span.set_attribute("evaluation.bias.response.detected", True)
                    span.set_attribute("evaluation.bias.response.max_score", result.max_score)
                    span.set_attribute(
                        "evaluation.bias.response.detected_biases",
                        result.detected_biases,
                    )

                    # Add individual bias type scores
                    for bias_type, score in result.bias_scores.items():
                        if score > 0:
                            span.set_attribute(f"evaluation.bias.response.{bias_type}_score", score)

                    # Add patterns matched
                    for bias_type, patterns in result.patterns_matched.items():
                        span.set_attribute(
                            f"evaluation.bias.response.{bias_type}_patterns",
                            patterns[:5],  # Limit to first 5 patterns
                        )

                    # Record metrics
                    self.bias_detection_counter.add(1, {"location": "response"})
                    self.bias_score_histogram.record(result.max_score, {"location": "response"})

                    # Record bias type metrics
                    for bias_type in result.detected_biases:
                        self.bias_type_counter.add(
                            1, {"bias_type": bias_type, "location": "response"}
                        )

                    # If blocking mode and threshold exceeded, set error status
                    if self.bias_config.block_on_detection and result.has_bias:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                "Response blocked due to bias detection",
                            )
                        )
                        span.set_attribute("evaluation.bias.response.blocked", True)
                        self.bias_blocked_counter.add(1, {"location": "response"})
                else:
                    span.set_attribute("evaluation.bias.response.detected", False)

        except Exception as e:
            logger.error("Error checking bias: %s", e, exc_info=True)
            span.set_attribute("evaluation.bias.error", str(e))

    def _check_prompt_injection(self, span: Span, prompt: Optional[str]) -> None:
        """Check for prompt injection attempts in prompt.

        Args:
            span: The span to add prompt injection attributes to
            prompt: Prompt text (optional)
        """
        if not self.prompt_injection_detector or not prompt:
            return

        try:
            result = self.prompt_injection_detector.detect(prompt)
            if result.is_injection:
                span.set_attribute("evaluation.prompt_injection.detected", True)
                span.set_attribute("evaluation.prompt_injection.score", result.injection_score)
                span.set_attribute("evaluation.prompt_injection.types", result.injection_types)

                # Add patterns matched
                for inj_type, patterns in result.patterns_matched.items():
                    span.set_attribute(
                        f"evaluation.prompt_injection.{inj_type}_patterns",
                        patterns[:5],  # Limit to first 5 patterns
                    )

                # Record metrics
                self.prompt_injection_counter.add(1, {"location": "prompt"})
                self.prompt_injection_score_histogram.record(
                    result.injection_score, {"location": "prompt"}
                )

                # Record injection type metrics
                for inj_type in result.injection_types:
                    self.prompt_injection_type_counter.add(1, {"injection_type": inj_type})

                # If blocking, set error status
                if result.blocked:
                    span.set_status(
                        Status(
                            StatusCode.ERROR,
                            "Request blocked due to prompt injection attempt",
                        )
                    )
                    span.set_attribute("evaluation.prompt_injection.blocked", True)
                    self.prompt_injection_blocked_counter.add(1, {})
            else:
                span.set_attribute("evaluation.prompt_injection.detected", False)

        except Exception as e:
            logger.error("Error checking prompt injection: %s", e, exc_info=True)
            span.set_attribute("evaluation.prompt_injection.error", str(e))

    def _check_restricted_topics(
        self, span: Span, prompt: Optional[str], response: Optional[str]
    ) -> None:
        """Check for restricted topics in prompt and response.

        Args:
            span: The span to add restricted topics attributes to
            prompt: Prompt text (optional)
            response: Response text (optional)
        """
        if not self.restricted_topics_detector:
            return

        try:
            # Check prompt for restricted topics
            if prompt:
                result = self.restricted_topics_detector.detect(prompt)
                if result.has_restricted_topic:
                    span.set_attribute("evaluation.restricted_topics.prompt.detected", True)
                    span.set_attribute(
                        "evaluation.restricted_topics.prompt.max_score", result.max_score
                    )
                    span.set_attribute(
                        "evaluation.restricted_topics.prompt.topics",
                        result.detected_topics,
                    )

                    # Add individual topic scores
                    for topic, score in result.topic_scores.items():
                        if score > 0:
                            span.set_attribute(
                                f"evaluation.restricted_topics.prompt.{topic}_score", score
                            )

                    # Record metrics
                    self.restricted_topics_counter.add(1, {"location": "prompt"})
                    self.restricted_topics_score_histogram.record(
                        result.max_score, {"location": "prompt"}
                    )

                    # Record topic metrics
                    for topic in result.detected_topics:
                        self.restricted_topics_type_counter.add(
                            1, {"topic": topic, "location": "prompt"}
                        )

                    # If blocking, set error status
                    if result.blocked:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                "Request blocked due to restricted topic",
                            )
                        )
                        span.set_attribute("evaluation.restricted_topics.prompt.blocked", True)
                        self.restricted_topics_blocked_counter.add(1, {"location": "prompt"})
                else:
                    span.set_attribute("evaluation.restricted_topics.prompt.detected", False)

            # Check response for restricted topics
            if response:
                result = self.restricted_topics_detector.detect(response)
                if result.has_restricted_topic:
                    span.set_attribute("evaluation.restricted_topics.response.detected", True)
                    span.set_attribute(
                        "evaluation.restricted_topics.response.max_score", result.max_score
                    )
                    span.set_attribute(
                        "evaluation.restricted_topics.response.topics",
                        result.detected_topics,
                    )

                    # Add individual topic scores
                    for topic, score in result.topic_scores.items():
                        if score > 0:
                            span.set_attribute(
                                f"evaluation.restricted_topics.response.{topic}_score", score
                            )

                    # Record metrics
                    self.restricted_topics_counter.add(1, {"location": "response"})
                    self.restricted_topics_score_histogram.record(
                        result.max_score, {"location": "response"}
                    )

                    # Record topic metrics
                    for topic in result.detected_topics:
                        self.restricted_topics_type_counter.add(
                            1, {"topic": topic, "location": "response"}
                        )

                    # If blocking, set error status
                    if result.blocked:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                "Response blocked due to restricted topic",
                            )
                        )
                        span.set_attribute("evaluation.restricted_topics.response.blocked", True)
                        self.restricted_topics_blocked_counter.add(1, {"location": "response"})
                else:
                    span.set_attribute("evaluation.restricted_topics.response.detected", False)

        except Exception as e:
            logger.error("Error checking restricted topics: %s", e, exc_info=True)
            span.set_attribute("evaluation.restricted_topics.error", str(e))

    def _check_hallucination(
        self,
        span: Span,
        prompt: Optional[str],
        response: Optional[str],
        attributes: dict,
    ) -> None:
        """Check for potential hallucinations in response.

        Args:
            span: The span to add hallucination attributes to
            prompt: Prompt text (optional, used as context)
            response: Response text (optional)
            attributes: Span attributes (for extracting context)
        """
        if not self.hallucination_detector or not response:
            return

        try:
            # Use prompt as context if available
            context = prompt

            # Try to extract additional context from attributes
            context_keys = [
                "gen_ai.context",
                "gen_ai.retrieval.documents",
                "gen_ai.rag.context",
            ]
            for key in context_keys:
                if key in attributes:
                    value = attributes[key]
                    if isinstance(value, str):
                        context = f"{context}\n{value}" if context else value
                        break

            result = self.hallucination_detector.detect(response, context)

            # Always set basic attributes
            span.set_attribute(
                "evaluation.hallucination.response.detected", result.has_hallucination
            )
            span.set_attribute(
                "evaluation.hallucination.response.score", result.hallucination_score
            )
            span.set_attribute("evaluation.hallucination.response.citations", result.citation_count)
            span.set_attribute(
                "evaluation.hallucination.response.hedge_words", result.hedge_words_count
            )
            span.set_attribute(
                "evaluation.hallucination.response.claims", result.factual_claim_count
            )

            if result.has_hallucination:
                span.set_attribute(
                    "evaluation.hallucination.response.indicators", result.hallucination_indicators
                )

                # Add unsupported claims
                if result.unsupported_claims:
                    span.set_attribute(
                        "evaluation.hallucination.response.unsupported_claims",
                        result.unsupported_claims[:3],  # Limit to first 3
                    )

                # Record metrics
                self.hallucination_counter.add(1, {"location": "response"})
                self.hallucination_score_histogram.record(
                    result.hallucination_score, {"location": "response"}
                )

                # Record indicator metrics
                for indicator in result.hallucination_indicators:
                    self.hallucination_indicator_counter.add(1, {"indicator": indicator})

        except Exception as e:
            logger.error("Error checking hallucination: %s", e, exc_info=True)
            span.set_attribute("evaluation.hallucination.error", str(e))

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
