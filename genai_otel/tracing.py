"""Public tracing utilities for creating parent spans and trace hierarchy.

This module provides a simple API for users to create parent spans that group
nested LLM calls into a single trace, solving the single-span trace problem
where each LLM call appears as an isolated root span.

Example:
    from genai_otel.tracing import trace_operation

    async def process_loan(application):
        with trace_operation("loan_processing", {"loan.type": "auto_loan"}):
            # All litellm.acompletion() calls here become children of this span
            result = await litellm.acompletion(...)
            summary = await litellm.acompletion(...)
"""

import logging
from contextlib import contextmanager
from typing import Dict, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)

_TRACER_NAME = "genai_otel"


@contextmanager
def trace_operation(
    name: str,
    attributes: Optional[Dict[str, str]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """Context manager that creates a parent span for grouping nested LLM calls.

    All OpenTelemetry-instrumented calls (including LiteLLM, OpenAI, etc.) made
    within this context will automatically become child spans of the created span.

    Args:
        name: Name for the parent span (e.g., "loan_processing", "rag_pipeline").
        attributes: Optional dict of span attributes to set on the parent span.
        kind: SpanKind for the span (default: INTERNAL).

    Yields:
        The created span object, which can be used to add additional attributes
        or events.

    Example:
        with trace_operation("risk_assessment", {"customer.id": "123"}):
            response = await litellm.acompletion(model="gpt-4o", messages=[...])
    """
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as span:
        yield span
