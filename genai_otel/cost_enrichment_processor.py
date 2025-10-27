"""Custom SpanProcessor to enrich OpenInference spans with cost tracking.

This processor adds cost attributes to spans created by OpenInference instrumentors
(smolagents, litellm, mcp) by extracting token usage and model information from
existing span attributes and calculating costs using our CostCalculator.

Supports both OpenTelemetry GenAI and OpenInference semantic conventions:
- GenAI: gen_ai.request.model, gen_ai.usage.{prompt_tokens,completion_tokens}
- OpenInference: llm.model_name, llm.token_count.{prompt,completion}
"""

import logging
from typing import Optional

from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import SpanContext

from .cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


class CostEnrichmentSpanProcessor(SpanProcessor):
    """Enriches spans with cost tracking attributes.

    This processor:
    1. Identifies spans from OpenInference instrumentors (smolagents, litellm, mcp)
    2. Extracts model name and token usage from span attributes
    3. Calculates cost using CostCalculator
    4. Adds cost attributes (gen_ai.usage.cost.total, etc.) to the span
    """

    def __init__(self, cost_calculator: Optional[CostCalculator] = None):
        """Initialize the cost enrichment processor.

        Args:
            cost_calculator: CostCalculator instance to use for cost calculations.
                           If None, creates a new instance.
        """
        self.cost_calculator = cost_calculator or CostCalculator()
        logger.info("CostEnrichmentSpanProcessor initialized")

    def on_start(self, span: Span, parent_context: Optional[SpanContext] = None) -> None:
        """Called when a span starts. No action needed."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends. Enriches with cost attributes if applicable.

        Args:
            span: The span that just ended.
        """
        try:
            # Only process spans that have LLM-related attributes
            if not span.attributes:
                return

            attributes = span.attributes

            # Check for model name - support both GenAI and OpenInference conventions
            model = (
                attributes.get("gen_ai.request.model")
                or attributes.get("llm.model_name")
                or attributes.get("embedding.model_name")
            )
            if not model:
                return

            # Skip if cost attributes are already present (added by instrumentor)
            if "gen_ai.usage.cost.total" in attributes:
                logger.debug(f"Span '{span.name}' already has cost attributes, skipping enrichment")
                return

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
                return

            # Get call type - support both GenAI and OpenInference conventions
            # OpenInference uses openinference.span.kind (values: LLM, EMBEDDING, etc.)
            span_kind = attributes.get("openinference.span.kind", "").upper()
            call_type = attributes.get("gen_ai.operation.name") or span_kind.lower() or "chat"

            # Map operation names to call types for cost calculator
            # Supports both GenAI and OpenInference conventions
            call_type_mapping = {
                # GenAI conventions
                "chat": "chat",
                "completion": "chat",
                "embedding": "embedding",
                "embeddings": "embedding",
                "text_generation": "chat",
                "image_generation": "image",
                "audio": "audio",
                # OpenInference conventions (span.kind values)
                "llm": "chat",
                "embedding": "embedding",
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

            # Use calculate_granular_cost to get detailed breakdown
            cost_info = self.cost_calculator.calculate_granular_cost(
                model=str(model),
                usage=usage,
                call_type=normalized_call_type,
            )

            if cost_info and cost_info.get("total", 0.0) > 0:
                # Add cost attributes to the span
                # Use duck typing to check if span supports set_attribute
                if hasattr(span, "set_attribute") and callable(getattr(span, "set_attribute")):
                    span.set_attribute("gen_ai.usage.cost.total", cost_info["total"])

                    if cost_info.get("prompt", 0.0) > 0:
                        span.set_attribute("gen_ai.usage.cost.prompt", cost_info["prompt"])
                    if cost_info.get("completion", 0.0) > 0:
                        span.set_attribute("gen_ai.usage.cost.completion", cost_info["completion"])

                    logger.info(
                        f"Enriched span '{span.name}' with cost: {cost_info['total']:.6f} USD "
                        f"for model {model} ({usage['total_tokens']} tokens)"
                    )
                else:
                    logger.warning(
                        f"Span '{span.name}' is not mutable (type: {type(span).__name__}), "
                        "cannot add cost attributes"
                    )

        except Exception as e:
            # Don't fail span processing due to cost enrichment errors
            logger.warning(
                f"Failed to enrich span '{getattr(span, 'name', 'unknown')}' with cost: {e}",
                exc_info=True,
            )

    def shutdown(self) -> None:
        """Called when the processor is shutdown."""
        logger.info("CostEnrichmentSpanProcessor shutdown")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            True if flush succeeded.
        """
        return True
