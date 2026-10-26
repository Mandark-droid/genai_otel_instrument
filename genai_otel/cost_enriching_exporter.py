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
        # Agent frameworks may report aggregate run usage on an AGENT span as
        # well as per-call usage on child model spans. Remember model usage by
        # trace so the agent aggregate doesn't get priced a second time when
        # parent and child batches are exported separately.
        self._traces_with_model_usage = set()
        logger.info(
            "CostEnrichingSpanExporter initialized, wrapping %s",
            type(wrapped_exporter).__name__,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans after enriching them with cost attributes.

        Args:
            spans: Sequence of ReadableSpan objects to export.

        Returns:
            SpanExportResult from the wrapped exporter.
        """
        try:
            # Discover LLM usage before enriching: an agent span can end after
            # its model children and may be exported in the same batch or a later
            # batch. Price the model calls and leave the aggregate agent span's
            # token counts visible without duplicating cost.
            batch_traces_with_model_usage = {
                span.context.trace_id
                for span in spans
                if span.context is not None
                and self._has_usage(span.attributes)
                and self._is_model_span(span.attributes)
                and self._has_model(span.attributes)
            }
            self._traces_with_model_usage.update(batch_traces_with_model_usage)

            enriched_spans = []
            for span in spans:
                attributes = span.attributes or {}
                trace_id = span.context.trace_id if span.context is not None else None
                skip_aggregate_cost = self._is_agent_span(attributes) and (
                    trace_id in batch_traces_with_model_usage
                    or trace_id in self._traces_with_model_usage
                )
                enriched_span = self._enrich_span(span, skip_cost=skip_aggregate_cost)
                enriched_spans.append(enriched_span)
                if span.parent is None and trace_id is not None:
                    self._traces_with_model_usage.discard(trace_id)

            # Export to wrapped exporter
            return self.wrapped_exporter.export(enriched_spans)

        except Exception as e:
            logger.error("Failed to export spans: %s", e, exc_info=True)
            return SpanExportResult.FAILURE

    @staticmethod
    def _has_usage(attributes) -> bool:
        if not attributes:
            return False
        return bool(
            attributes.get("gen_ai.usage.input_tokens")
            or attributes.get("gen_ai.usage.prompt_tokens")
            or attributes.get("llm.token_count.prompt")
            or attributes.get("decision.token_count.input")
            or attributes.get("gen_ai.usage.output_tokens")
            or attributes.get("gen_ai.usage.completion_tokens")
            or attributes.get("llm.token_count.completion")
            or attributes.get("decision.token_count.output")
        )

    @staticmethod
    def _is_model_span(attributes) -> bool:
        span_kind = str(attributes.get("openinference.span.kind", "")).upper()
        operation = str(attributes.get("gen_ai.operation.name", "")).lower()
        return span_kind in {"LLM", "EMBEDDING"} or operation in {
            "chat",
            "completion",
            "text_generation",
            "image_generation",
            "embedding",
        }

    @staticmethod
    def _has_model(attributes) -> bool:
        if not attributes:
            return False
        return bool(
            attributes.get("gen_ai.request.model")
            or attributes.get("llm.model_name")
            or attributes.get("embedding.model_name")
            or attributes.get("decision.model_name")
        )

    @staticmethod
    def _is_agent_span(attributes) -> bool:
        span_kind = str(attributes.get("openinference.span.kind", "")).upper()
        operation = str(attributes.get("gen_ai.operation.name", "")).lower()
        return span_kind == "AGENT" or operation == "invoke_agent"

    def _enrich_span(self, span: ReadableSpan, skip_cost: bool = False) -> ReadableSpan:
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

            if skip_cost:
                return span

            # Check for model name - support both GenAI and OpenInference conventions
            model = (
                attributes.get("gen_ai.request.model")
                or attributes.get("llm.model_name")
                or attributes.get("embedding.model_name")
                or attributes.get("decision.model_name")  # OpenInference DECISION spans
            )
            if not model:
                return span

            # Skip if cost attributes are already present
            if "gen_ai.usage.cost.total" in attributes:
                logger.debug(
                    "Span '%s' already has cost attributes, skipping enrichment", span.name
                )
                return span

            # Extract token usage - support GenAI, OpenInference, and legacy
            # conventions. The current GenAI names come first; the superseded
            # prompt_/completion_ names are only emitted under gen_ai/dup.
            prompt_tokens = (
                attributes.get("gen_ai.usage.input_tokens")
                or attributes.get("gen_ai.usage.prompt_tokens")  # superseded
                or attributes.get("llm.token_count.prompt")  # OpenInference
                or attributes.get("decision.token_count.input")  # OpenInference DECISION
                or 0
            )
            completion_tokens = (
                attributes.get("gen_ai.usage.output_tokens")
                or attributes.get("gen_ai.usage.completion_tokens")  # superseded
                or attributes.get("llm.token_count.completion")  # OpenInference
                or attributes.get("decision.token_count.output")  # OpenInference DECISION
                or 0
            )

            # Anthropic SDK v1.14+ reports input_tokens with cache reads and
            # writes included. The calculator prices those separately, so
            # remove them from the standard prompt bucket before adding their
            # dedicated cache rates.
            cache_read_tokens = (
                attributes.get("gen_ai.usage.cache_read.input_tokens")
                or attributes.get("cache_read_input_tokens")
                or 0
            )
            cache_write_tokens = (
                attributes.get("gen_ai.usage.cache_write.input_tokens")
                or attributes.get("cache_creation_input_tokens")
                or 0
            )
            prompt_tokens = max(
                int(prompt_tokens) - int(cache_read_tokens) - int(cache_write_tokens), 0
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
            # A decision model is priced from the decision section only. Read as a
            # chat call, a self-hosted one would pick up a parameter-count estimate
            # and gpt-6-luna would be billed for output it does not charge for.
            if span_kind == "DECISION" or attributes.get("gen_ai.request.type") == "decision":
                normalized_call_type = "decision"

            # Calculate cost
            usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": (
                    int(prompt_tokens)
                    + int(cache_read_tokens)
                    + int(cache_write_tokens)
                    + int(completion_tokens)
                ),
            }
            if cache_read_tokens:
                usage["cache_read_input_tokens"] = int(cache_read_tokens)
            if cache_write_tokens:
                usage["cache_creation_input_tokens"] = int(cache_write_tokens)

            cost_info = self.cost_calculator.calculate_granular_cost(
                model=str(model),
                usage=usage,
                call_type=normalized_call_type,
            )

            if cost_info and cost_info.get("total", 0.0) > 0:
                # Add cost attributes to the mutable copy
                attributes["gen_ai.usage.cost.total"] = cost_info["total"]
                if normalized_call_type == "decision":
                    attributes["gen_ai.usage.cost.decision"] = cost_info["total"]

                if cost_info.get("prompt", 0.0) > 0:
                    attributes["gen_ai.usage.cost.prompt"] = cost_info["prompt"]
                if cost_info.get("completion", 0.0) > 0:
                    attributes["gen_ai.usage.cost.completion"] = cost_info["completion"]

                logger.info(
                    "Enriched span '%s' with cost: %.6f USD for model %s (%s tokens)",
                    span.name,
                    cost_info["total"],
                    model,
                    usage["total_tokens"],
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
                "Failed to enrich span '%s' with cost: %s",
                getattr(span, "name", "unknown"),
                e,
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
