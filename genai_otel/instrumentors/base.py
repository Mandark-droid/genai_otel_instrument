"""Base classes for OpenTelemetry instrumentors for GenAI libraries and tools.

This module defines the `BaseInstrumentor` abstract base class, which provides
common functionality and a standardized interface for instrumenting various
Generative AI (GenAI) libraries and Model Context Protocol (MCP) tools.
It includes methods for creating OpenTelemetry spans, recording metrics,
and handling configuration and cost calculation.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import wrapt
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from ..cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


class BaseInstrumentor(ABC):  # pylint: disable=R0902
    """Abstract base class for all LLM library instrumentors.

    Provides common functionality for setting up OpenTelemetry spans, metrics,
    and handling common instrumentation patterns.
    """

    def __init__(self):
        """Initializes the instrumentor with OpenTelemetry tracers, meters, and common metrics."""
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        self.config: Optional[OTelConfig] = None
        self.cost_calculator = CostCalculator()
        self._instrumented = False

        try:
            # Create common metrics
            self.request_counter = self.meter.create_counter(
                "genai.requests", description="Number of LLM requests"
            )
            self.token_counter = self.meter.create_counter(
                "genai.tokens", description="Number of tokens processed"
            )
            self.latency_histogram = self.meter.create_histogram(
                "genai.latency", description="Request latency in seconds", unit="s"
            )
            self.cost_counter = self.meter.create_counter(
                "genai.cost", description="Estimated cost in USD", unit="USD"
            )
            self.error_counter = self.meter.create_counter(
                "genai.errors", description="Number of errors"
            )
        except Exception as e:
            logger.error(f"Failed to create metrics: {e}", exc_info=True)
            # Continue without metrics rather than failing

    @abstractmethod
    def instrument(self, config: OTelConfig):
        """Abstract method to implement library-specific instrumentation.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """

    def create_span_wrapper(
        self, span_name: str, extract_attributes: Optional[Callable[[Any, Any, Any], Dict]] = None
    ) -> Callable:  # pylint: disable=R1702
        """Create a decorator that instruments a function with an OpenTelemetry span.

        This method uses `wrapt.decorator` to wrap a target function. It handles:
        - Starting and ending spans.
        - Extracting attributes using a provided callable.
        - Recording metrics (latency, tokens, cost, errors).
        - Setting span status (OK or ERROR).
        - Handling exceptions during span creation or function execution.

        Args:
            span_name (str): The name for the OpenTelemetry span.
            extract_attributes (Optional[Callable[[Any, Any, Any], Dict]]): A callable that takes
                (instance, args, kwargs) and returns a dictionary of attributes to set on the span.

        Returns:
            Callable: A decorator function that can be applied to a method.
        """

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):
            # If instrumentation failed during initialization, just call the original function.
            if not self._instrumented:
                logger.debug(f"Instrumentation not active, calling {span_name} directly")
                return wrapped(*args, **kwargs)

            try:
                # Start a new span
                with self.tracer.start_as_current_span(span_name) as span:
                    start_time = time.time()

                    try:
                        # Extract and set attributes if a callable is provided
                        if extract_attributes:
                            try:
                                # Pass instance, args, and kwargs to the extractor
                                attrs = extract_attributes(instance, args, kwargs)
                                for key, value in attrs.items():
                                    try:
                                        # Ensure attribute values are valid types for OpenTelemetry
                                        if isinstance(value, (str, int, float, bool)):
                                            span.set_attribute(key, value)
                                        else:
                                            span.set_attribute(key, str(value))
                                    except Exception as e:
                                        logger.warning(f"Failed to set attribute {key}: {e}")
                            except Exception as e:
                                logger.warning(
                                    f"Failed to extract attributes for span '{span_name}': {e}"
                                )

                        # Call the original function
                        result = wrapped(*args, **kwargs)

                        # Record metrics based on the result
                        try:
                            self._record_result_metrics(span, result, start_time)
                        except Exception as e:
                            logger.warning(f"Failed to record metrics for span '{span_name}': {e}")

                        # Set span status to OK on successful execution
                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        # Handle exceptions during the wrapped function execution
                        # Record error metrics
                        try:
                            self.error_counter.add(
                                1, {"operation": span_name, "error_type": type(e).__name__}
                            )
                        except Exception:
                            # Avoid failing if error metric recording fails
                            pass

                        # Set span status to ERROR and record the exception
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise  # Re-raise the original exception

            except Exception as e:
                # Handle exceptions that occur during span creation itself
                logger.error(f"Span creation failed for '{span_name}': {e}", exc_info=True)
                # If span creation fails, still attempt to call the original function to maintain functionality
                return wrapped(*args, **kwargs)

        return wrapper

    def _record_result_metrics(self, span, result, start_time: float):
        """Record metrics derived from the function result and execution time.

        This method attempts to record latency, token usage, and estimated cost.
        It relies on `_extract_usage` to get token information.
        """
        # Record latency
        try:
            duration = time.time() - start_time
            self.latency_histogram.record(duration, {"operation": span.name})
        except Exception as e:
            logger.warning(f"Failed to record latency for span '{span.name}': {e}")

        # Extract and record token usage and cost
        try:
            usage = self._extract_usage(result)
            if usage and isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # Record token counts if available and positive
                if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
                    self.token_counter.add(
                        prompt_tokens, {"token_type": "prompt", "operation": span.name}
                    )
                    span.set_attribute("gen_ai.usage.prompt_tokens", int(prompt_tokens))

                if isinstance(completion_tokens, (int, float)) and completion_tokens > 0:
                    self.token_counter.add(
                        completion_tokens, {"token_type": "completion", "operation": span.name}
                    )
                    span.set_attribute("gen_ai.usage.completion_tokens", int(completion_tokens))

                if isinstance(total_tokens, (int, float)) and total_tokens > 0:
                    span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))

                # Calculate and record cost if enabled and applicable
                if self.config and self.config.enable_cost_tracking:
                    try:
                        model = span.attributes.get("gen_ai.request.model", "unknown")
                        cost = self.cost_calculator.calculate_cost(model, usage)
                        if cost > 0:
                            span.set_attribute("gen_ai.cost.amount", cost)
                            self.cost_counter.add(cost, {"model": str(model)})
                    except Exception as e:
                        logger.warning(f"Failed to calculate cost for span '{span.name}': {e}")

        except Exception as e:
            logger.warning(f"Failed to extract or record usage metrics for span '{span.name}': {e}")

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
