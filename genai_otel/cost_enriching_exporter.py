"""Custom SpanExporter that enriches spans with cost attributes before export.

This exporter wraps another exporter (like OTLPSpanExporter) and adds cost
attributes to spans before passing them to the wrapped exporter.
"""

import logging
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


class CostEnrichingSpanExporter(SpanExporter):
    """Wraps a SpanExporter and enriches spans with cost attributes before export.

    This exporter:
    1. Receives ReadableSpan objects from the SDK
    2. Extracts model name and token usage from span attributes
    3. Calculates cost using CostCalculator
    4. Creates enriched span data with cost attributes
    5. Exports to the wrapped exporter (e.g., OTLP)
    """

    def __init__(
        self, wrapped_exporter: SpanExporter, cost_calculator: Optional[CostCalculator] = None
    ):
        """Initialize the cost enriching exporter.

        Args:
            wrapped_exporter: The underlying exporter to send enriched spans to.
            cost_calculator: CostCalculator instance to use for cost calculations.
                           If None, creates a new instance.
        """
        self.wrapped_exporter = wrapped_exporter
        self.cost_calculator = cost_calculator or CostCalculator()
        logger.info(
            f"CostEnrichingSpanExporter initialized, wrapping {type(wrapped_exporter).__name__}"
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans after enriching them with cost attributes.

        Args:
            spans: Sequence of ReadableSpan objects to export.

        Returns:
            SpanExportResult from the wrapped exporter.
        """
        try:
            # Enrich spans with cost attributes
            enriched_spans = []
            for span in spans:
                enriched_span = self._enrich_span(span)
                enriched_spans.append(enriched_span)

            # Export to wrapped exporter
            return self.wrapped_exporter.export(enriched_spans)

        except Exception as e:
            logger.error(f"Failed to export spans: {e}", exc_info=True)
            return SpanExportResult.FAILURE

    def _enrich_span(self, span: ReadableSpan) -> ReadableSpan:
        """Enrich a span with cost attributes if applicable.

        Args:
            span: The original ReadableSpan.

        Returns:
            A new ReadableSpan with cost attributes added (or the original if not applicable).
        """
        try:
            # Check if span has LLM-related attributes
            if not span.attributes:
                return span

            attributes = dict(span.attributes)  # Make a mutable copy

            # Check for model name - support both GenAI and OpenInference conventions
            model = (
                attributes.get("gen_ai.request.model")
                or attributes.get("llm.model_name")
                or attributes.get("embedding.model_name")
            )
            if not model:
                return span

            # Skip if cost attributes are already present
            if "gen_ai.usage.cost.total" in attributes:
                logger.debug(f"Span '{span.name}' already has cost attributes, skipping enrichment")
                return span

            # Extract token usage - support GenAI, OpenInference, and legacy conventions
            prompt_tokens = (
                attributes.get("gen_ai.usage.prompt_tokens")
                or attributes.get("gen_ai.usage.input_tokens")
                or attributes.get("llm.token_count.prompt")  # OpenInference
                or 0
            )
            completion_tokens = (
                attributes.get("gen_ai.usage.completion_tokens")
                or attributes.get("gen_ai.usage.output_tokens")
                or attributes.get("llm.token_count.completion")  # OpenInference
                or 0
            )

            # Skip if no tokens recorded
            if prompt_tokens == 0 and completion_tokens == 0:
                return span

            # Get call type - support both GenAI and OpenInference conventions
            span_kind = attributes.get("openinference.span.kind", "").upper()
            call_type = attributes.get("gen_ai.operation.name") or span_kind.lower() or "chat"

            # Map operation names to call types
            call_type_mapping = {
                "chat": "chat",
                "completion": "chat",
                "embedding": "embedding",
                "embeddings": "embedding",
                "text_generation": "chat",
                "image_generation": "image",
                "audio": "audio",
                "llm": "chat",
                "chain": "chat",
                "retriever": "embedding",
                "reranker": "embedding",
                "tool": "chat",
                "agent": "chat",
            }
            normalized_call_type = call_type_mapping.get(str(call_type).lower(), "chat")

            # Calculate cost
            usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
            }

            cost_info = self.cost_calculator.calculate_granular_cost(
                model=str(model),
                usage=usage,
                call_type=normalized_call_type,
            )

            if cost_info and cost_info.get("total", 0.0) > 0:
                # Add cost attributes to the mutable copy
                attributes["gen_ai.usage.cost.total"] = cost_info["total"]

                if cost_info.get("prompt", 0.0) > 0:
                    attributes["gen_ai.usage.cost.prompt"] = cost_info["prompt"]
                if cost_info.get("completion", 0.0) > 0:
                    attributes["gen_ai.usage.cost.completion"] = cost_info["completion"]

                logger.info(
                    f"Enriched span '{span.name}' with cost: {cost_info['total']:.6f} USD "
                    f"for model {model} ({usage['total_tokens']} tokens)"
                )

                # Create a new ReadableSpan with enriched attributes
                # ReadableSpan is a NamedTuple, so we need to replace it
                from opentelemetry.sdk.trace import ReadableSpan as RS

                enriched_span = RS(
                    name=span.name,
                    context=span.context,
                    kind=span.kind,
                    parent=span.parent,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    status=span.status,
                    attributes=attributes,  # Use enriched attributes
                    events=span.events,
                    links=span.links,
                    resource=span.resource,
                    instrumentation_scope=span.instrumentation_scope,
                )
                return enriched_span

        except Exception as e:
            logger.warning(
                f"Failed to enrich span '{getattr(span, 'name', 'unknown')}' with cost: {e}",
                exc_info=True,
            )

        return span

    def shutdown(self) -> None:
        """Shutdown the wrapped exporter."""
        logger.info("CostEnrichingSpanExporter shutting down")
        self.wrapped_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the wrapped exporter.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            True if flush succeeded.
        """
        return self.wrapped_exporter.force_flush(timeout_millis)
