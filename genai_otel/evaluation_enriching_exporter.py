"""Custom SpanExporter that enriches spans with evaluation attributes before export.

This exporter wraps another exporter (like OTLPSpanExporter or CostEnrichingSpanExporter)
and runs evaluation checks (PII, toxicity, bias, prompt injection, restricted topics,
hallucination) on spans before passing them to the wrapped exporter.

This is the primary evaluation path for OpenInference spans (LiteLLM, smolagents, MCP)
where the EvaluationSpanProcessor.on_end() cannot reliably write attributes to ReadableSpan.
"""

import logging
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


class EvaluationEnrichingSpanExporter(SpanExporter):
    """Wraps a SpanExporter and enriches spans with evaluation attributes before export.

    This exporter:
    1. Receives ReadableSpan objects from the SDK
    2. Extracts prompt and response from span attributes
    3. Runs enabled evaluation detectors (PII, toxicity, bias, etc.)
    4. Creates enriched span data with evaluation attributes
    5. Exports to the wrapped exporter (e.g., OTLP or CostEnrichingSpanExporter)
    """

    def __init__(self, wrapped_exporter: SpanExporter, evaluation_processor):
        """Initialize the evaluation enriching exporter.

        Args:
            wrapped_exporter: The underlying exporter to send enriched spans to.
            evaluation_processor: EvaluationSpanProcessor instance with configured detectors.
        """
        self.wrapped_exporter = wrapped_exporter
        self.evaluation_processor = evaluation_processor
        logger.info(
            "EvaluationEnrichingSpanExporter initialized, wrapping %s",
            type(wrapped_exporter).__name__,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans after enriching them with evaluation attributes.

        Args:
            spans: Sequence of ReadableSpan objects to export.

        Returns:
            SpanExportResult from the wrapped exporter.
        """
        try:
            enriched_spans = []
            for span in spans:
                enriched_span = self._enrich_span(span)
                enriched_spans.append(enriched_span)
            return self.wrapped_exporter.export(enriched_spans)
        except Exception as e:
            logger.error("Failed to export spans with evaluation enrichment: %s", e, exc_info=True)
            return SpanExportResult.FAILURE

    def _enrich_span(self, span: ReadableSpan) -> ReadableSpan:
        """Enrich a span with evaluation attributes if applicable.

        Args:
            span: The original ReadableSpan.

        Returns:
            A new ReadableSpan with evaluation attributes added, or the original.
        """
        try:
            if not span.attributes:
                return span

            attributes = dict(span.attributes)

            # Skip if evaluation attributes are already present
            if "evaluation.pii.prompt.detected" in attributes:
                return span

            # Extract prompt and response using the processor's existing methods
            prompt = self.evaluation_processor._extract_prompt(attributes)
            response = self.evaluation_processor._extract_response(attributes)

            if not prompt and not response:
                return span

            has_enrichment = False

            # Run PII detection
            if (
                self.evaluation_processor.pii_config.enabled
                and self.evaluation_processor.pii_detector
            ):
                has_enrichment |= self._check_pii(attributes, prompt, response)

            # Run toxicity detection
            if (
                self.evaluation_processor.toxicity_config.enabled
                and self.evaluation_processor.toxicity_detector
            ):
                has_enrichment |= self._check_toxicity(attributes, prompt, response)

            # Run bias detection
            if (
                self.evaluation_processor.bias_config.enabled
                and self.evaluation_processor.bias_detector
            ):
                has_enrichment |= self._check_bias(attributes, prompt, response)

            # Run prompt injection detection
            if (
                self.evaluation_processor.prompt_injection_config.enabled
                and self.evaluation_processor.prompt_injection_detector
            ):
                has_enrichment |= self._check_prompt_injection(attributes, prompt)

            # Run restricted topics detection
            if (
                self.evaluation_processor.restricted_topics_config.enabled
                and self.evaluation_processor.restricted_topics_detector
            ):
                has_enrichment |= self._check_restricted_topics(attributes, prompt, response)

            # Run hallucination detection
            if (
                self.evaluation_processor.hallucination_config.enabled
                and self.evaluation_processor.hallucination_detector
            ):
                has_enrichment |= self._check_hallucination(attributes, prompt, response)

            if has_enrichment:
                enriched_span = ReadableSpan(
                    name=span.name,
                    context=span.context,
                    kind=span.kind,
                    parent=span.parent,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    status=span.status,
                    attributes=attributes,
                    events=span.events,
                    links=span.links,
                    resource=span.resource,
                    instrumentation_scope=span.instrumentation_scope,
                )
                logger.debug("Enriched span '%s' with evaluation attributes", span.name)
                return enriched_span

        except Exception as e:
            logger.warning(
                "Failed to enrich span '%s' with evaluation: %s",
                getattr(span, "name", "unknown"),
                e,
                exc_info=True,
            )

        return span

    def _check_pii(self, attributes: dict, prompt: Optional[str], response: Optional[str]) -> bool:
        """Run PII detection and add results to attributes dict.

        Returns True if any attributes were added.
        """
        detector = self.evaluation_processor.pii_detector
        if not detector:
            return False

        added = False
        try:
            if prompt:
                result = detector.detect(prompt)
                if result.has_pii:
                    attributes["evaluation.pii.prompt.detected"] = True
                    attributes["evaluation.pii.prompt.entity_count"] = len(result.entities)
                    attributes["evaluation.pii.prompt.entity_types"] = list(
                        result.entity_counts.keys()
                    )
                    attributes["evaluation.pii.prompt.score"] = result.score
                    for entity_type, count in result.entity_counts.items():
                        attributes[f"evaluation.pii.prompt.{entity_type.lower()}_count"] = count
                    if result.blocked:
                        attributes["evaluation.pii.prompt.blocked"] = True
                    if result.redacted_text:
                        attributes["evaluation.pii.prompt.redacted"] = result.redacted_text
                    self.evaluation_processor.pii_detection_counter.add(
                        1,
                        {
                            "location": "prompt",
                            "mode": self.evaluation_processor.pii_config.mode.value,
                        },
                    )
                    for entity_type, count in result.entity_counts.items():
                        self.evaluation_processor.pii_entity_counter.add(
                            count, {"entity_type": entity_type, "location": "prompt"}
                        )
                    if result.blocked:
                        self.evaluation_processor.pii_blocked_counter.add(1, {"location": "prompt"})
                else:
                    attributes["evaluation.pii.prompt.detected"] = False
                added = True

            if response:
                result = detector.detect(response)
                if result.has_pii:
                    attributes["evaluation.pii.response.detected"] = True
                    attributes["evaluation.pii.response.entity_count"] = len(result.entities)
                    attributes["evaluation.pii.response.entity_types"] = list(
                        result.entity_counts.keys()
                    )
                    attributes["evaluation.pii.response.score"] = result.score
                    for entity_type, count in result.entity_counts.items():
                        attributes[f"evaluation.pii.response.{entity_type.lower()}_count"] = count
                    if result.blocked:
                        attributes["evaluation.pii.response.blocked"] = True
                    if result.redacted_text:
                        attributes["evaluation.pii.response.redacted"] = result.redacted_text
                    self.evaluation_processor.pii_detection_counter.add(
                        1,
                        {
                            "location": "response",
                            "mode": self.evaluation_processor.pii_config.mode.value,
                        },
                    )
                    for entity_type, count in result.entity_counts.items():
                        self.evaluation_processor.pii_entity_counter.add(
                            count, {"entity_type": entity_type, "location": "response"}
                        )
                    if result.blocked:
                        self.evaluation_processor.pii_blocked_counter.add(
                            1, {"location": "response"}
                        )
                else:
                    attributes["evaluation.pii.response.detected"] = False
                added = True

        except Exception as e:
            logger.error("Error checking PII in exporter: %s", e, exc_info=True)
            attributes["evaluation.pii.error"] = str(e)
            added = True

        return added

    def _check_toxicity(
        self, attributes: dict, prompt: Optional[str], response: Optional[str]
    ) -> bool:
        """Run toxicity detection and add results to attributes dict."""
        detector = self.evaluation_processor.toxicity_detector
        if not detector:
            return False

        added = False
        try:
            if prompt:
                result = detector.detect(prompt)
                if result.is_toxic:
                    attributes["evaluation.toxicity.prompt.detected"] = True
                    attributes["evaluation.toxicity.prompt.max_score"] = result.max_score
                    attributes["evaluation.toxicity.prompt.categories"] = result.toxic_categories
                    for category, score in result.scores.items():
                        attributes[f"evaluation.toxicity.prompt.{category}_score"] = score
                    self.evaluation_processor.toxicity_detection_counter.add(
                        1, {"location": "prompt"}
                    )
                    self.evaluation_processor.toxicity_score_histogram.record(
                        result.max_score, {"location": "prompt"}
                    )
                    for category in result.toxic_categories:
                        self.evaluation_processor.toxicity_category_counter.add(
                            1, {"category": category, "location": "prompt"}
                        )
                    if result.blocked:
                        attributes["evaluation.toxicity.prompt.blocked"] = True
                        self.evaluation_processor.toxicity_blocked_counter.add(
                            1, {"location": "prompt"}
                        )
                else:
                    attributes["evaluation.toxicity.prompt.detected"] = False
                added = True

            if response:
                result = detector.detect(response)
                if result.is_toxic:
                    attributes["evaluation.toxicity.response.detected"] = True
                    attributes["evaluation.toxicity.response.max_score"] = result.max_score
                    attributes["evaluation.toxicity.response.categories"] = result.toxic_categories
                    for category, score in result.scores.items():
                        attributes[f"evaluation.toxicity.response.{category}_score"] = score
                    self.evaluation_processor.toxicity_detection_counter.add(
                        1, {"location": "response"}
                    )
                    self.evaluation_processor.toxicity_score_histogram.record(
                        result.max_score, {"location": "response"}
                    )
                    for category in result.toxic_categories:
                        self.evaluation_processor.toxicity_category_counter.add(
                            1, {"category": category, "location": "response"}
                        )
                    if result.blocked:
                        attributes["evaluation.toxicity.response.blocked"] = True
                        self.evaluation_processor.toxicity_blocked_counter.add(
                            1, {"location": "response"}
                        )
                else:
                    attributes["evaluation.toxicity.response.detected"] = False
                added = True

        except Exception as e:
            logger.error("Error checking toxicity in exporter: %s", e, exc_info=True)
            attributes["evaluation.toxicity.error"] = str(e)
            added = True

        return added

    def _check_bias(self, attributes: dict, prompt: Optional[str], response: Optional[str]) -> bool:
        """Run bias detection and add results to attributes dict."""
        detector = self.evaluation_processor.bias_detector
        if not detector:
            return False

        added = False
        try:
            if prompt:
                result = detector.detect(prompt)
                if result.has_bias:
                    attributes["evaluation.bias.prompt.detected"] = True
                    attributes["evaluation.bias.prompt.max_score"] = result.max_score
                    attributes["evaluation.bias.prompt.detected_biases"] = result.detected_biases
                    for bias_type, score in result.bias_scores.items():
                        if score > 0:
                            attributes[f"evaluation.bias.prompt.{bias_type}_score"] = score
                    for bias_type, patterns in result.patterns_matched.items():
                        attributes[f"evaluation.bias.prompt.{bias_type}_patterns"] = patterns[:5]
                    self.evaluation_processor.bias_detection_counter.add(1, {"location": "prompt"})
                    self.evaluation_processor.bias_score_histogram.record(
                        result.max_score, {"location": "prompt"}
                    )
                    for bias_type in result.detected_biases:
                        self.evaluation_processor.bias_type_counter.add(
                            1, {"bias_type": bias_type, "location": "prompt"}
                        )
                    if self.evaluation_processor.bias_config.block_on_detection:
                        attributes["evaluation.bias.prompt.blocked"] = True
                        self.evaluation_processor.bias_blocked_counter.add(
                            1, {"location": "prompt"}
                        )
                else:
                    attributes["evaluation.bias.prompt.detected"] = False
                added = True

            if response:
                result = detector.detect(response)
                if result.has_bias:
                    attributes["evaluation.bias.response.detected"] = True
                    attributes["evaluation.bias.response.max_score"] = result.max_score
                    attributes["evaluation.bias.response.detected_biases"] = result.detected_biases
                    for bias_type, score in result.bias_scores.items():
                        if score > 0:
                            attributes[f"evaluation.bias.response.{bias_type}_score"] = score
                    for bias_type, patterns in result.patterns_matched.items():
                        attributes[f"evaluation.bias.response.{bias_type}_patterns"] = patterns[:5]
                    self.evaluation_processor.bias_detection_counter.add(
                        1, {"location": "response"}
                    )
                    self.evaluation_processor.bias_score_histogram.record(
                        result.max_score, {"location": "response"}
                    )
                    for bias_type in result.detected_biases:
                        self.evaluation_processor.bias_type_counter.add(
                            1, {"bias_type": bias_type, "location": "response"}
                        )
                    if self.evaluation_processor.bias_config.block_on_detection:
                        attributes["evaluation.bias.response.blocked"] = True
                        self.evaluation_processor.bias_blocked_counter.add(
                            1, {"location": "response"}
                        )
                else:
                    attributes["evaluation.bias.response.detected"] = False
                added = True

        except Exception as e:
            logger.error("Error checking bias in exporter: %s", e, exc_info=True)
            attributes["evaluation.bias.error"] = str(e)
            added = True

        return added

    def _check_prompt_injection(self, attributes: dict, prompt: Optional[str]) -> bool:
        """Run prompt injection detection and add results to attributes dict."""
        detector = self.evaluation_processor.prompt_injection_detector
        if not detector or not prompt:
            return False

        added = False
        try:
            result = detector.detect(prompt)
            if result.is_injection:
                attributes["evaluation.prompt_injection.detected"] = True
                attributes["evaluation.prompt_injection.score"] = result.injection_score
                attributes["evaluation.prompt_injection.types"] = result.injection_types
                for inj_type, patterns in result.patterns_matched.items():
                    attributes[f"evaluation.prompt_injection.{inj_type}_patterns"] = patterns[:5]
                self.evaluation_processor.prompt_injection_counter.add(1, {"location": "prompt"})
                self.evaluation_processor.prompt_injection_score_histogram.record(
                    result.injection_score, {"location": "prompt"}
                )
                for inj_type in result.injection_types:
                    self.evaluation_processor.prompt_injection_type_counter.add(
                        1, {"injection_type": inj_type}
                    )
                if result.blocked:
                    attributes["evaluation.prompt_injection.blocked"] = True
                    self.evaluation_processor.prompt_injection_blocked_counter.add(1, {})
            else:
                attributes["evaluation.prompt_injection.detected"] = False
            added = True

        except Exception as e:
            logger.error("Error checking prompt injection in exporter: %s", e, exc_info=True)
            attributes["evaluation.prompt_injection.error"] = str(e)
            added = True

        return added

    def _check_restricted_topics(
        self, attributes: dict, prompt: Optional[str], response: Optional[str]
    ) -> bool:
        """Run restricted topics detection and add results to attributes dict."""
        detector = self.evaluation_processor.restricted_topics_detector
        if not detector:
            return False

        added = False
        try:
            if prompt:
                result = detector.detect(prompt)
                if result.has_restricted_topic:
                    attributes["evaluation.restricted_topics.prompt.detected"] = True
                    attributes["evaluation.restricted_topics.prompt.max_score"] = result.max_score
                    attributes["evaluation.restricted_topics.prompt.topics"] = (
                        result.detected_topics
                    )
                    for topic, score in result.topic_scores.items():
                        if score > 0:
                            attributes[f"evaluation.restricted_topics.prompt.{topic}_score"] = score
                    self.evaluation_processor.restricted_topics_counter.add(
                        1, {"location": "prompt"}
                    )
                    self.evaluation_processor.restricted_topics_score_histogram.record(
                        result.max_score, {"location": "prompt"}
                    )
                    for topic in result.detected_topics:
                        self.evaluation_processor.restricted_topics_type_counter.add(
                            1, {"topic": topic, "location": "prompt"}
                        )
                    if result.blocked:
                        attributes["evaluation.restricted_topics.prompt.blocked"] = True
                        self.evaluation_processor.restricted_topics_blocked_counter.add(
                            1, {"location": "prompt"}
                        )
                else:
                    attributes["evaluation.restricted_topics.prompt.detected"] = False
                added = True

            if response:
                result = detector.detect(response)
                if result.has_restricted_topic:
                    attributes["evaluation.restricted_topics.response.detected"] = True
                    attributes["evaluation.restricted_topics.response.max_score"] = result.max_score
                    attributes["evaluation.restricted_topics.response.topics"] = (
                        result.detected_topics
                    )
                    for topic, score in result.topic_scores.items():
                        if score > 0:
                            attributes[f"evaluation.restricted_topics.response.{topic}_score"] = (
                                score
                            )
                    self.evaluation_processor.restricted_topics_counter.add(
                        1, {"location": "response"}
                    )
                    self.evaluation_processor.restricted_topics_score_histogram.record(
                        result.max_score, {"location": "response"}
                    )
                    for topic in result.detected_topics:
                        self.evaluation_processor.restricted_topics_type_counter.add(
                            1, {"topic": topic, "location": "response"}
                        )
                    if result.blocked:
                        attributes["evaluation.restricted_topics.response.blocked"] = True
                        self.evaluation_processor.restricted_topics_blocked_counter.add(
                            1, {"location": "response"}
                        )
                else:
                    attributes["evaluation.restricted_topics.response.detected"] = False
                added = True

        except Exception as e:
            logger.error("Error checking restricted topics in exporter: %s", e, exc_info=True)
            attributes["evaluation.restricted_topics.error"] = str(e)
            added = True

        return added

    def _check_hallucination(
        self, attributes: dict, prompt: Optional[str], response: Optional[str]
    ) -> bool:
        """Run hallucination detection and add results to attributes dict."""
        detector = self.evaluation_processor.hallucination_detector
        if not detector or not response:
            return False

        added = False
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

            result = detector.detect(response, context)

            attributes["evaluation.hallucination.response.detected"] = result.has_hallucination
            attributes["evaluation.hallucination.response.score"] = result.hallucination_score
            attributes["evaluation.hallucination.response.citations"] = result.citation_count
            attributes["evaluation.hallucination.response.hedge_words"] = result.hedge_words_count
            attributes["evaluation.hallucination.response.claims"] = result.factual_claim_count

            if result.has_hallucination:
                attributes["evaluation.hallucination.response.indicators"] = (
                    result.hallucination_indicators
                )
                if result.unsupported_claims:
                    attributes["evaluation.hallucination.response.unsupported_claims"] = (
                        result.unsupported_claims[:3]
                    )
                self.evaluation_processor.hallucination_counter.add(1, {"location": "response"})
                self.evaluation_processor.hallucination_score_histogram.record(
                    result.hallucination_score, {"location": "response"}
                )
                for indicator in result.hallucination_indicators:
                    self.evaluation_processor.hallucination_indicator_counter.add(
                        1, {"indicator": indicator}
                    )
            added = True

        except Exception as e:
            logger.error("Error checking hallucination in exporter: %s", e, exc_info=True)
            attributes["evaluation.hallucination.error"] = str(e)
            added = True

        return added

    def shutdown(self) -> None:
        """Shutdown the wrapped exporter."""
        logger.info("EvaluationEnrichingSpanExporter shutting down")
        self.wrapped_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the wrapped exporter.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            True if flush succeeded.
        """
        return self.wrapped_exporter.force_flush(timeout_millis)
