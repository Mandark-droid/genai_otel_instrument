"""Base classes for OpenTelemetry instrumentors for GenAI libraries and tools.

This module defines the `BaseInstrumentor` abstract base class, which provides
common functionality and a standardized interface for instrumenting various
Generative AI (GenAI) libraries and Model Context Protocol (MCP) tools.
It includes methods for creating OpenTelemetry spans, recording metrics,
and handling configuration and cost calculation.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import wrapt
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from ..cost_calculator import CostCalculator

# Import semantic conventions
try:
    from openlit.semcov import SemanticConvention as SC
except ImportError:
    # Fallback if openlit not available
    class SC:
        GEN_AI_REQUESTS = "gen_ai.requests"
        GEN_AI_CLIENT_TOKEN_USAGE = "gen_ai.client.token.usage"
        GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"
        GEN_AI_USAGE_COST = "gen_ai.usage.cost"


# Import histogram bucket definitions
try:
    from genai_otel.metrics import _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
except ImportError:
    # Fallback buckets if import fails
    _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS = [
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    ]

logger = logging.getLogger(__name__)
# Global flag to track if shared metrics have been created
_SHARED_METRICS_CREATED = False
_SHARED_METRICS_LOCK = threading.Lock()


class BaseInstrumentor(ABC):  # pylint: disable=R0902
    """Abstract base class for all LLM library instrumentors.

    Provides common functionality for setting up OpenTelemetry spans, metrics,
    and handling common instrumentation patterns.
    """

    # Class-level shared metrics (created once, shared by all instances)
    _shared_request_counter = None
    _shared_token_counter = None
    _shared_latency_histogram = None
    _shared_cost_counter = None
    _shared_error_counter = None

    def __init__(self):
        """Initializes the instrumentor with OpenTelemetry tracers, meters, and common metrics."""
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        self.config: Optional[OTelConfig] = None
        self.cost_calculator = CostCalculator()
        self._instrumented = False

        # Use shared metrics to avoid duplicate warnings
        self._ensure_shared_metrics_created()

        # Reference the shared metrics
        self.request_counter = self._shared_request_counter
        self.token_counter = self._shared_token_counter
        self.latency_histogram = self._shared_latency_histogram
        self.cost_counter = self._shared_cost_counter
        self.error_counter = self._shared_error_counter

    @classmethod
    def _ensure_shared_metrics_created(cls):
        """Ensure shared metrics are created only once across all instrumentor instances."""
        global _SHARED_METRICS_CREATED

        with _SHARED_METRICS_LOCK:
            if _SHARED_METRICS_CREATED:
                return

            try:
                meter = metrics.get_meter(__name__)

                # Create shared metrics once using semantic conventions
                cls._shared_request_counter = meter.create_counter(
                    SC.GEN_AI_REQUESTS, description="Number of GenAI requests"
                )
                cls._shared_token_counter = meter.create_counter(
                    SC.GEN_AI_CLIENT_TOKEN_USAGE, description="Token usage for GenAI operations"
                )
                # Note: Histogram buckets should be configured via Views in MeterProvider
                # The advisory parameter is provided as a hint but Views take precedence
                cls._shared_latency_histogram = meter.create_histogram(
                    SC.GEN_AI_CLIENT_OPERATION_DURATION,
                    description="GenAI client operation duration",
                    unit="s",
                )
                cls._shared_cost_counter = meter.create_counter(
                    SC.GEN_AI_USAGE_COST, description="Cost of GenAI operations", unit="USD"
                )
                cls._shared_error_counter = meter.create_counter(
                    "gen_ai.client.errors", description="Number of GenAI client errors"
                )

                _SHARED_METRICS_CREATED = True
                logger.debug("Shared metrics created successfully")

            except Exception as e:
                logger.error("Failed to create shared metrics: %s", e, exc_info=True)
                # Create dummy metrics that do nothing to avoid crashes
                cls._shared_request_counter = None
                cls._shared_token_counter = None
                cls._shared_latency_histogram = None
                cls._shared_cost_counter = None
                cls._shared_error_counter = None

    @abstractmethod
    def instrument(self, config: OTelConfig):
        """Abstract method to implement library-specific instrumentation.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """

    def create_span_wrapper(
        self, span_name: str, extract_attributes: Optional[Callable[[Any, Any, Any], Dict]] = None
    ) -> Callable:
        """Create a decorator that instruments a function with an OpenTelemetry span."""

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):
            # If instrumentation failed during initialization, just call the original function.
            if not self._instrumented:
                logger.debug("Instrumentation not active, calling %s directly", span_name)
                return wrapped(*args, **kwargs)

            try:
                # Start a new span
                initial_attributes = {}
                if extract_attributes:
                    try:
                        extracted_attrs = extract_attributes(instance, args, kwargs)
                        for key, value in extracted_attrs.items():
                            if isinstance(value, (str, int, float, bool)):
                                initial_attributes[key] = value
                            else:
                                initial_attributes[key] = str(value)
                    except Exception as e:
                        logger.warning(
                            "Failed to extract attributes for span '%s': %s", span_name, e
                        )

                with self.tracer.start_as_current_span(
                    span_name, attributes=initial_attributes
                ) as span:
                    start_time = time.time()

                    try:

                        # Call the original function
                        result = wrapped(*args, **kwargs)

                        if self.request_counter:
                            self.request_counter.add(1, {"operation": span.name})

                        # Record metrics based on the result
                        try:
                            self._record_result_metrics(span, result, start_time, kwargs)
                        except Exception as e:
                            logger.warning(
                                "Failed to record metrics for span '%s': %s", span_name, e
                            )

                        # Set span status to OK on successful execution
                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        # Handle exceptions during the wrapped function execution
                        try:
                            if self.error_counter:
                                self.error_counter.add(
                                    1, {"operation": span_name, "error_type": type(e).__name__}
                                )
                        except Exception:
                            pass

                        # Set span status to ERROR and record the exception
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            except Exception as e:
                logger.error("Span creation failed for '%s': %s", span_name, e, exc_info=True)
                return wrapped(*args, **kwargs)

        return wrapper

    def _record_result_metrics(self, span, result, start_time: float, request_kwargs: dict = None):
        """Record metrics derived from the function result and execution time.

        Args:
            span: The OpenTelemetry span to record metrics on.
            result: The result from the wrapped function.
            start_time: The time when the function started executing.
            request_kwargs: The original request kwargs (for content capture).
        """
        # Record latency
        try:
            duration = time.time() - start_time
            if self.latency_histogram:
                self.latency_histogram.record(duration, {"operation": span.name})
        except Exception as e:
            logger.warning("Failed to record latency for span '%s': %s", span.name, e)

        # Extract and set response attributes if available
        try:
            if hasattr(self, "_extract_response_attributes"):
                response_attrs = self._extract_response_attributes(result)
                if response_attrs and isinstance(response_attrs, dict):
                    for key, value in response_attrs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(key, value)
                        elif isinstance(value, list):
                            # For arrays like finish_reasons
                            span.set_attribute(key, value)
                        else:
                            span.set_attribute(key, str(value))
        except Exception as e:
            logger.warning(
                "Failed to extract response attributes for span '%s': %s", span.name, e
            )

        # Add content events if content capture is enabled
        try:
            if (
                hasattr(self, "_add_content_events")
                and self.config
                and self.config.enable_content_capture
            ):
                self._add_content_events(span, result, request_kwargs or {})
        except Exception as e:
            logger.warning("Failed to add content events for span '%s': %s", span.name, e)

        # Extract and record token usage and cost
        try:
            usage = self._extract_usage(result)
            if usage and isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # Record token counts if available and positive
                # Support dual emission based on OTEL_SEMCONV_STABILITY_OPT_IN
                emit_old_attrs = (
                    self.config
                    and self.config.semconv_stability_opt_in
                    and "dup" in self.config.semconv_stability_opt_in
                )

                if (
                    self.token_counter
                    and isinstance(prompt_tokens, (int, float))
                    and prompt_tokens > 0
                ):
                    self.token_counter.add(
                        prompt_tokens, {"token_type": "prompt", "operation": span.name}
                    )
                    # New semantic convention
                    span.set_attribute("gen_ai.usage.prompt_tokens", int(prompt_tokens))
                    # Old semantic convention (if dual emission enabled)
                    if emit_old_attrs:
                        span.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))

                if (
                    self.token_counter
                    and isinstance(completion_tokens, (int, float))
                    and completion_tokens > 0
                ):
                    self.token_counter.add(
                        completion_tokens, {"token_type": "completion", "operation": span.name}
                    )
                    # New semantic convention
                    span.set_attribute("gen_ai.usage.completion_tokens", int(completion_tokens))
                    # Old semantic convention (if dual emission enabled)
                    if emit_old_attrs:
                        span.set_attribute("gen_ai.usage.output_tokens", int(completion_tokens))

                if isinstance(total_tokens, (int, float)) and total_tokens > 0:
                    span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))

                # Calculate and record cost if enabled and applicable
                if self.config and self.config.enable_cost_tracking and self._shared_cost_counter:
                    try:
                        model = span.attributes.get("gen_ai.request.model", "unknown")
                        # Assuming 'chat' as a default call_type for generic base instrumentor tests.
                        # Specific instrumentors will provide the actual call_type.
                        call_type = span.attributes.get("gen_ai.request.type", "chat")
                        cost = self.cost_calculator.calculate_cost(model, usage, call_type)
                        if cost and cost > 0:
                            self._shared_cost_counter.add(cost, {"model": str(model)})
                    except Exception as e:
                        logger.warning("Failed to calculate cost for span '%s': %s", span.name, e)

        except Exception as e:
            logger.warning(
                "Failed to extract or record usage metrics for span '%s': %s", span.name, e
            )

    @abstractmethod
    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Abstract method to extract token usage information from a function result.

        Subclasses must implement this to parse the specific library's response object
        and return a dictionary containing 'prompt_tokens', 'completion_tokens',
        and optionally 'total_tokens'.

        Args:
            result: The return value of the instrumented function.

        Returns:
            Optional[Dict[str, int]]: A dictionary with token counts, or None if usage cannot be extracted.
        """
