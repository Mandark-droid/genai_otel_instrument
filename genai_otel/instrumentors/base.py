"""Base classes for OpenTelemetry instrumentors for GenAI libraries and tools.

This module defines the `BaseInstrumentor` abstract base class, which provides
common functionality and a standardized interface for instrumenting various
Generative AI (GenAI) libraries and Model Context Protocol (MCP) tools.
It includes methods for creating OpenTelemetry spans, recording metrics,
and handling configuration and cost calculation.
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

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
from ..server_metrics import get_server_metrics

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
        GEN_AI_SERVER_TTFT = "gen_ai.server.ttft"
        GEN_AI_SERVER_TBT = "gen_ai.server.tbt"


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
    # Granular cost counters (Phase 3.2)
    _shared_prompt_cost_counter = None
    _shared_completion_cost_counter = None
    _shared_reasoning_cost_counter = None
    _shared_cache_read_cost_counter = None
    _shared_cache_write_cost_counter = None
    # Streaming metrics (Phase 3.4)
    _shared_ttft_histogram = None
    _shared_tbt_histogram = None
    # Token distribution histograms
    _shared_prompt_tokens_histogram = None
    _shared_completion_tokens_histogram = None
    # Finish reason tracking counters
    _shared_request_finish_counter = None
    _shared_request_success_counter = None
    _shared_request_failure_counter = None

    # Evaluation detectors (set by EvaluationSpanProcessor)
    _pii_detector = None
    _toxicity_detector = None
    _evaluation_lock = threading.Lock()

    def __init__(self):
        """Initializes the instrumentor with OpenTelemetry tracers, meters, and common metrics."""
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        self.config: Optional[OTelConfig] = None
        self.cost_calculator = CostCalculator()  # Will be updated when instrument() is called
        self._instrumented = False

        # Use shared metrics to avoid duplicate warnings
        self._ensure_shared_metrics_created()

        # Reference the shared metrics
        self.request_counter = self._shared_request_counter
        self.token_counter = self._shared_token_counter
        self.latency_histogram = self._shared_latency_histogram
        self.cost_counter = self._shared_cost_counter
        self.error_counter = self._shared_error_counter
        # Granular cost counters (Phase 3.2)
        self.prompt_cost_counter = self._shared_prompt_cost_counter
        self.completion_cost_counter = self._shared_completion_cost_counter
        self.reasoning_cost_counter = self._shared_reasoning_cost_counter
        self.cache_read_cost_counter = self._shared_cache_read_cost_counter
        self.cache_write_cost_counter = self._shared_cache_write_cost_counter
        # Streaming metrics
        self.ttft_histogram = self._shared_ttft_histogram
        self.tbt_histogram = self._shared_tbt_histogram
        # Token distribution histograms
        self.prompt_tokens_histogram = self._shared_prompt_tokens_histogram
        self.completion_tokens_histogram = self._shared_completion_tokens_histogram
        # Finish reason tracking counters
        self.request_finish_counter = self._shared_request_finish_counter
        self.request_success_counter = self._shared_request_success_counter
        self.request_failure_counter = self._shared_request_failure_counter

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
                # Granular cost counters (Phase 3.2)
                cls._shared_prompt_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.prompt", description="Prompt tokens cost", unit="USD"
                )
                cls._shared_completion_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.completion", description="Completion tokens cost", unit="USD"
                )
                cls._shared_reasoning_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.reasoning",
                    description="Reasoning tokens cost (o1 models)",
                    unit="USD",
                )
                cls._shared_cache_read_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.cache_read",
                    description="Cache read cost (Anthropic)",
                    unit="USD",
                )
                cls._shared_cache_write_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.cache_write",
                    description="Cache write cost (Anthropic)",
                    unit="USD",
                )
                cls._shared_error_counter = meter.create_counter(
                    "gen_ai.client.errors", description="Number of GenAI client errors"
                )
                # Streaming metrics (Phase 3.4)
                # Note: Buckets should be configured via Views in MeterProvider
                cls._shared_ttft_histogram = meter.create_histogram(
                    SC.GEN_AI_SERVER_TTFT,
                    description="Time to first token in seconds",
                    unit="s",
                )
                cls._shared_tbt_histogram = meter.create_histogram(
                    SC.GEN_AI_SERVER_TBT,
                    description="Time between tokens in seconds",
                    unit="s",
                )
                # Token distribution histograms
                cls._shared_prompt_tokens_histogram = meter.create_histogram(
                    "gen_ai.client.token.usage.prompt",
                    description="Distribution of prompt tokens per request",
                    unit="tokens",
                )
                cls._shared_completion_tokens_histogram = meter.create_histogram(
                    "gen_ai.client.token.usage.completion",
                    description="Distribution of completion tokens per request",
                    unit="tokens",
                )
                # Finish reason tracking counters
                cls._shared_request_finish_counter = meter.create_counter(
                    "gen_ai.server.request.finish",
                    description="Number of finished requests by finish reason",
                )
                cls._shared_request_success_counter = meter.create_counter(
                    "gen_ai.server.request.success",
                    description="Number of successfully completed requests",
                )
                cls._shared_request_failure_counter = meter.create_counter(
                    "gen_ai.server.request.failure",
                    description="Number of failed requests",
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
                cls._shared_prompt_cost_counter = None
                cls._shared_completion_cost_counter = None
                cls._shared_reasoning_cost_counter = None
                cls._shared_cache_read_cost_counter = None
                cls._shared_cache_write_cost_counter = None
                cls._shared_error_counter = None
                cls._shared_ttft_histogram = None
                cls._shared_tbt_histogram = None
                cls._shared_prompt_tokens_histogram = None
                cls._shared_completion_tokens_histogram = None
                cls._shared_request_finish_counter = None
                cls._shared_request_success_counter = None
                cls._shared_request_failure_counter = None

    def _setup_config(self, config: OTelConfig):
        """Set up configuration and reinitialize cost calculator with custom pricing if provided.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        self.config = config
        # Reinitialize cost calculator with custom pricing if provided
        if config.custom_pricing_json:
            self.cost_calculator = CostCalculator(custom_pricing_json=config.custom_pricing_json)
            logger.info("Cost calculator reinitialized with custom pricing")

    @abstractmethod
    def instrument(self, config: OTelConfig):
        """Abstract method to implement library-specific instrumentation.

        Implementers should call self._setup_config(config) at the beginning of this method
        to ensure custom pricing is loaded.

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

                # Check if this is a streaming request before creating the span
                is_streaming = kwargs.get("stream", False)

                # Start the span (but don't use context manager for streaming to keep it open)
                span = self.tracer.start_span(span_name, attributes=initial_attributes)
                start_time = time.time()

                # Increment server metrics: running requests counter
                server_metrics = get_server_metrics()
                if server_metrics:
                    server_metrics.increment_requests_running()
                    logger.debug(f"Incremented running requests for {span_name}")

                # Extract session and user context (Phase 4.1)
                if self.config:
                    if self.config.session_id_extractor:
                        try:
                            session_id = self.config.session_id_extractor(instance, args, kwargs)
                            if session_id:
                                span.set_attribute("session.id", session_id)
                                logger.debug("Set session.id: %s", session_id)
                        except Exception as e:
                            logger.debug("Failed to extract session ID: %s", e)

                    if self.config.user_id_extractor:
                        try:
                            user_id = self.config.user_id_extractor(instance, args, kwargs)
                            if user_id:
                                span.set_attribute("user.id", user_id)
                                logger.debug("Set user.id: %s", user_id)
                        except Exception as e:
                            logger.debug("Failed to extract user ID: %s", e)

                try:
                    # Call the original function
                    result = wrapped(*args, **kwargs)

                    if self.request_counter:
                        self.request_counter.add(1, {"operation": span.name})

                    # Handle streaming vs non-streaming responses (Phase 3.4)
                    if is_streaming:
                        # For streaming responses, wrap the iterator to capture TTFT/TBT
                        model = kwargs.get(
                            "model", initial_attributes.get("gen_ai.request.model", "unknown")
                        )
                        logger.debug(f"Detected streaming response for model: {model}")
                        # Wrap the streaming response - span will be finalized when iteration completes
                        return self._wrap_streaming_response(result, span, start_time, model)

                    # Non-streaming: record metrics and close span normally
                    try:
                        self._record_result_metrics(span, result, start_time, kwargs)
                    except Exception as e:
                        logger.warning("Failed to record metrics for span '%s': %s", span_name, e)

                    # Run evaluation checks BEFORE ending the span
                    try:
                        self._run_evaluation_checks(span, args, kwargs, result)
                    except Exception as e:
                        logger.warning(
                            "Failed to run evaluation checks for span '%s': %s", span_name, e
                        )

                    # Set span status to OK on successful execution
                    span.set_status(Status(StatusCode.OK))
                    span.end()

                    # Decrement server metrics: running requests counter
                    server_metrics = get_server_metrics()
                    if server_metrics:
                        server_metrics.decrement_requests_running()
                        logger.debug(f"Decremented running requests for {span_name}")

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
                    span.end()

                    # Decrement server metrics: running requests counter (error path)
                    server_metrics = get_server_metrics()
                    if server_metrics:
                        server_metrics.decrement_requests_running()
                        logger.debug(f"Decremented running requests for {span_name} (error)")

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
            logger.warning("Failed to extract response attributes for span '%s': %s", span.name, e)

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

                # Record prompt tokens
                if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
                    # Record counter metric if available
                    if self.token_counter:
                        self.token_counter.add(
                            prompt_tokens, {"token_type": "prompt", "operation": span.name}
                        )
                    # Record histogram for distribution analysis
                    if self.prompt_tokens_histogram:
                        model = span.attributes.get("gen_ai.request.model", "unknown")
                        self.prompt_tokens_histogram.record(
                            int(prompt_tokens), {"model": str(model), "operation": span.name}
                        )
                    # Always set span attributes (needed for cost calculation)
                    span.set_attribute("gen_ai.usage.prompt_tokens", int(prompt_tokens))
                    # Old semantic convention (if dual emission enabled)
                    if emit_old_attrs:
                        span.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))

                # Record completion tokens
                if isinstance(completion_tokens, (int, float)) and completion_tokens > 0:
                    # Record counter metric if available
                    if self.token_counter:
                        self.token_counter.add(
                            completion_tokens, {"token_type": "completion", "operation": span.name}
                        )
                    # Record histogram for distribution analysis
                    if self.completion_tokens_histogram:
                        model = span.attributes.get("gen_ai.request.model", "unknown")
                        self.completion_tokens_histogram.record(
                            int(completion_tokens), {"model": str(model), "operation": span.name}
                        )
                    # Always set span attributes (needed for cost calculation)
                    span.set_attribute("gen_ai.usage.completion_tokens", int(completion_tokens))
                    # Old semantic convention (if dual emission enabled)
                    if emit_old_attrs:
                        span.set_attribute("gen_ai.usage.output_tokens", int(completion_tokens))

                # Record total tokens
                if isinstance(total_tokens, (int, float)) and total_tokens > 0:
                    span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))

                # Calculate and record cost if enabled and applicable
                logger.debug(
                    f"Cost tracking check: config={self.config is not None}, "
                    f"enable_cost_tracking={self.config.enable_cost_tracking if self.config else 'N/A'}"
                )
                if self.config and self.config.enable_cost_tracking:
                    try:
                        model = span.attributes.get("gen_ai.request.model", "unknown")
                        # Assuming 'chat' as a default call_type for generic base instrumentor tests.
                        # Specific instrumentors will provide the actual call_type.
                        call_type = span.attributes.get("gen_ai.request.type", "chat")

                        logger.debug(
                            f"Calculating cost for model={model}, call_type={call_type}, "
                            f"prompt_tokens={usage.get('prompt_tokens')}, "
                            f"completion_tokens={usage.get('completion_tokens')}"
                        )

                        # Use granular cost calculation for chat requests
                        if call_type == "chat":
                            costs = self.cost_calculator.calculate_granular_cost(
                                model, usage, call_type
                            )
                            total_cost = costs["total"]

                            # Record total cost
                            if total_cost > 0:
                                if self.cost_counter:
                                    self.cost_counter.add(total_cost, {"model": str(model)})
                                # Always set span attributes (needed for cost tracking)
                                span.set_attribute("gen_ai.usage.cost.total", total_cost)
                                logger.debug(
                                    f"Set cost attribute: gen_ai.usage.cost.total={total_cost}"
                                )
                            else:
                                logger.debug(
                                    f"Cost is zero, not setting attributes. Costs: {costs}"
                                )

                            # Record and set attributes for granular costs
                            # Note: Metrics recording is optional, span attributes are always set
                            if costs["prompt"] > 0:
                                if self.prompt_cost_counter:
                                    self.prompt_cost_counter.add(
                                        costs["prompt"], {"model": str(model)}
                                    )
                                span.set_attribute("gen_ai.usage.cost.prompt", costs["prompt"])

                            if costs["completion"] > 0:
                                if self.completion_cost_counter:
                                    self.completion_cost_counter.add(
                                        costs["completion"], {"model": str(model)}
                                    )
                                span.set_attribute(
                                    "gen_ai.usage.cost.completion", costs["completion"]
                                )

                            if costs["reasoning"] > 0:
                                if self.reasoning_cost_counter:
                                    self.reasoning_cost_counter.add(
                                        costs["reasoning"], {"model": str(model)}
                                    )
                                span.set_attribute(
                                    "gen_ai.usage.cost.reasoning", costs["reasoning"]
                                )

                            if costs["cache_read"] > 0:
                                if self.cache_read_cost_counter:
                                    self.cache_read_cost_counter.add(
                                        costs["cache_read"], {"model": str(model)}
                                    )
                                span.set_attribute(
                                    "gen_ai.usage.cost.cache_read", costs["cache_read"]
                                )

                            if costs["cache_write"] > 0:
                                if self.cache_write_cost_counter:
                                    self.cache_write_cost_counter.add(
                                        costs["cache_write"], {"model": str(model)}
                                    )
                                span.set_attribute(
                                    "gen_ai.usage.cost.cache_write", costs["cache_write"]
                                )
                        else:
                            # For non-chat requests, use simple cost calculation
                            cost = self.cost_calculator.calculate_cost(model, usage, call_type)
                            if cost and cost > 0:
                                if self.cost_counter:
                                    self.cost_counter.add(cost, {"model": str(model)})
                    except Exception as e:
                        logger.warning("Failed to calculate cost for span '%s': %s", span.name, e)

        except Exception as e:
            logger.warning(
                "Failed to extract or record usage metrics for span '%s': %s", span.name, e
            )

        # Extract and record finish reason if available (for request outcome tracking)
        try:
            if hasattr(self, "_extract_finish_reason"):
                finish_reason = self._extract_finish_reason(result)
                if finish_reason:
                    model = span.attributes.get("gen_ai.request.model", "unknown")

                    # Record finish reason counter
                    if self.request_finish_counter:
                        self.request_finish_counter.add(
                            1, {"finish_reason": finish_reason, "model": str(model)}
                        )

                    # Set span attribute
                    span.set_attribute("gen_ai.response.finish_reason", finish_reason)

                    # Track success vs failure based on finish reason
                    # Success: stop, length, end_turn, etc.
                    # Failure: error, content_filter, timeout, etc.
                    success_reasons = {"stop", "length", "end_turn", "max_tokens"}
                    failure_reasons = {"error", "content_filter", "timeout", "rate_limit"}

                    if finish_reason in success_reasons:
                        if self.request_success_counter:
                            self.request_success_counter.add(1, {"model": str(model)})
                    elif finish_reason in failure_reasons:
                        if self.request_failure_counter:
                            self.request_failure_counter.add(
                                1, {"finish_reason": finish_reason, "model": str(model)}
                            )
        except Exception as e:
            logger.debug(
                "Failed to extract or record finish reason for span '%s': %s", span.name, e
            )

    def _run_evaluation_checks(self, span, args, kwargs, result):
        """Run PII and toxicity evaluation checks before ending the span.

        This method extracts prompt and response from span attributes and runs
        evaluation detectors if they are configured. Attributes are added to the
        span BEFORE it ends, ensuring they appear in exported traces.

        Args:
            span: The active span (mutable, not yet ended)
            args: Original function arguments
            kwargs: Original function keyword arguments
            result: The LLM response object
        """
        if not BaseInstrumentor._pii_detector and not BaseInstrumentor._toxicity_detector:
            return  # No detectors configured

        try:
            # Extract prompt and response from span attributes
            import ast
            import json

            attrs = dict(span.attributes)

            # Extract prompt
            prompt = None
            if "gen_ai.request.first_message" in attrs:
                value = attrs["gen_ai.request.first_message"]
                logger.debug(f"Found gen_ai.request.first_message: {value[:100]}")
                if isinstance(value, str) and value.strip().startswith("{"):
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, dict) and "content" in parsed:
                            prompt = parsed["content"]
                            logger.info(f"Extracted prompt for evaluation: {prompt[:100]}...")
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"Failed to parse first_message: {e}")

            # Extract response - try to get from result object
            response = None
            try:
                if hasattr(result, "choices") and result.choices:
                    if hasattr(result.choices[0], "message"):
                        response = result.choices[0].message.content
                    elif hasattr(result.choices[0], "text"):
                        response = result.choices[0].text
            except Exception:
                pass

            # Run PII detection
            if BaseInstrumentor._pii_detector and prompt:
                try:
                    logger.info(f"Running PII detection on prompt: {prompt[:50]}...")
                    pii_result = BaseInstrumentor._pii_detector.detect(prompt)
                    logger.info(
                        f"PII detection result: has_pii={pii_result.has_pii}, entities={list(pii_result.entity_counts.keys()) if pii_result.has_pii else []}"
                    )
                    if pii_result.has_pii:
                        span.set_attribute("evaluation.pii.prompt.detected", True)
                        span.set_attribute(
                            "evaluation.pii.prompt.entity_count", len(pii_result.entities)
                        )
                        span.set_attribute(
                            "evaluation.pii.prompt.entity_types",
                            list(pii_result.entity_counts.keys()),
                        )
                        span.set_attribute("evaluation.pii.prompt.score", pii_result.score)

                        # Add entity counts by type
                        for entity_type, count in pii_result.entity_counts.items():
                            span.set_attribute(
                                f"evaluation.pii.prompt.{entity_type.lower()}_count", count
                            )

                        if pii_result.redacted_text:
                            span.set_attribute(
                                "evaluation.pii.prompt.redacted", pii_result.redacted_text
                            )
                        if pii_result.blocked:
                            span.set_attribute("evaluation.pii.prompt.blocked", True)
                    else:
                        span.set_attribute("evaluation.pii.prompt.detected", False)
                except Exception as e:
                    logger.warning(f"PII detection failed: {e}", exc_info=True)

            # Run toxicity detection
            if BaseInstrumentor._toxicity_detector and prompt:
                try:
                    toxicity_result = BaseInstrumentor._toxicity_detector.detect(prompt)
                    if toxicity_result.is_toxic:
                        span.set_attribute("evaluation.toxicity.prompt.detected", True)
                        span.set_attribute(
                            "evaluation.toxicity.prompt.max_score", toxicity_result.max_score
                        )
                        span.set_attribute(
                            "evaluation.toxicity.prompt.categories",
                            toxicity_result.toxic_categories,
                        )

                        # Add individual category scores
                        for category, score in toxicity_result.scores.items():
                            span.set_attribute(
                                f"evaluation.toxicity.prompt.{category}_score", score
                            )
                    else:
                        span.set_attribute("evaluation.toxicity.prompt.detected", False)
                except Exception as e:
                    logger.warning(f"Toxicity detection failed: {e}", exc_info=True)

        except Exception as e:
            logger.warning(f"Evaluation checks failed: {e}", exc_info=True)

    def _wrap_streaming_response(self, stream, span, start_time: float, model: str):
        """Wrap a streaming response to capture TTFT and TBT metrics.

        This generator wrapper yields chunks from the streaming response while
        measuring time to first token (TTFT) and time between tokens (TBT).
        The span is finalized when the stream completes or errors.

        Args:
            stream: The streaming response iterator
            span: The OpenTelemetry span for this request
            start_time: Request start time (for TTFT calculation)
            model: Model name/identifier for metric attributes

        Yields:
            Chunks from the original stream
        """
        from opentelemetry.trace import Status, StatusCode

        first_token = True
        last_token_time = start_time
        token_count = 0
        last_chunk = None  # Store last chunk to extract usage

        try:
            for chunk in stream:
                current_time = time.time()
                token_count += 1

                if first_token:
                    # Record Time to First Token
                    ttft = current_time - start_time
                    span.set_attribute("gen_ai.server.ttft", ttft)
                    if self.ttft_histogram:
                        self.ttft_histogram.record(ttft, {"model": model, "operation": span.name})
                    logger.debug(f"TTFT for {model}: {ttft:.3f}s")
                    first_token = False
                else:
                    # Record Time Between Tokens
                    tbt = current_time - last_token_time
                    if self.tbt_histogram:
                        self.tbt_histogram.record(tbt, {"model": model, "operation": span.name})

                last_token_time = current_time
                last_chunk = chunk  # Keep track of last chunk for usage extraction
                yield chunk

            # Stream completed successfully
            duration = time.time() - start_time
            if self.latency_histogram:
                self.latency_histogram.record(duration, {"operation": span.name})
            span.set_attribute("gen_ai.streaming.token_count", token_count)

            # Extract usage from last chunk and calculate cost
            # Many providers (OpenAI, Anthropic, etc.) include usage in the final chunk
            try:
                if last_chunk is not None:
                    usage = self._extract_usage(last_chunk)
                    if usage and isinstance(usage, dict):
                        # Record token usage metrics and calculate cost
                        # This will set span attributes and record cost metrics
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)

                        # Record token counts
                        if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
                            if self.token_counter:
                                self.token_counter.add(
                                    prompt_tokens, {"token_type": "prompt", "operation": span.name}
                                )
                            # Record histogram for distribution analysis
                            if self.prompt_tokens_histogram:
                                self.prompt_tokens_histogram.record(
                                    int(prompt_tokens), {"model": model, "operation": span.name}
                                )
                            span.set_attribute("gen_ai.usage.prompt_tokens", int(prompt_tokens))

                        if isinstance(completion_tokens, (int, float)) and completion_tokens > 0:
                            if self.token_counter:
                                self.token_counter.add(
                                    completion_tokens,
                                    {"token_type": "completion", "operation": span.name},
                                )
                            # Record histogram for distribution analysis
                            if self.completion_tokens_histogram:
                                self.completion_tokens_histogram.record(
                                    int(completion_tokens), {"model": model, "operation": span.name}
                                )
                            span.set_attribute(
                                "gen_ai.usage.completion_tokens", int(completion_tokens)
                            )

                        if isinstance(total_tokens, (int, float)) and total_tokens > 0:
                            span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))

                        # Calculate and record cost if enabled
                        if self.config and self.config.enable_cost_tracking:
                            try:
                                # Get call_type from span attributes or default to "chat"
                                call_type = span.attributes.get("gen_ai.request.type", "chat")

                                # Use granular cost calculation for chat requests
                                if call_type == "chat":
                                    costs = self.cost_calculator.calculate_granular_cost(
                                        model, usage, call_type
                                    )
                                    total_cost = costs["total"]

                                    # Record total cost
                                    if total_cost > 0:
                                        if self.cost_counter:
                                            self.cost_counter.add(total_cost, {"model": str(model)})
                                        span.set_attribute("gen_ai.usage.cost.total", total_cost)
                                        logger.debug(f"Streaming cost: {total_cost} USD")

                                    # Record granular costs
                                    if costs["prompt"] > 0:
                                        if self.prompt_cost_counter:
                                            self.prompt_cost_counter.add(
                                                costs["prompt"], {"model": str(model)}
                                            )
                                        span.set_attribute(
                                            "gen_ai.usage.cost.prompt", costs["prompt"]
                                        )

                                    if costs["completion"] > 0:
                                        if self.completion_cost_counter:
                                            self.completion_cost_counter.add(
                                                costs["completion"], {"model": str(model)}
                                            )
                                        span.set_attribute(
                                            "gen_ai.usage.cost.completion", costs["completion"]
                                        )

                                    if costs["reasoning"] > 0:
                                        if self.reasoning_cost_counter:
                                            self.reasoning_cost_counter.add(
                                                costs["reasoning"], {"model": str(model)}
                                            )
                                        span.set_attribute(
                                            "gen_ai.usage.cost.reasoning", costs["reasoning"]
                                        )

                                    if costs["cache_read"] > 0:
                                        if self.cache_read_cost_counter:
                                            self.cache_read_cost_counter.add(
                                                costs["cache_read"], {"model": str(model)}
                                            )
                                        span.set_attribute(
                                            "gen_ai.usage.cost.cache_read", costs["cache_read"]
                                        )

                                    if costs["cache_write"] > 0:
                                        if self.cache_write_cost_counter:
                                            self.cache_write_cost_counter.add(
                                                costs["cache_write"], {"model": str(model)}
                                            )
                                        span.set_attribute(
                                            "gen_ai.usage.cost.cache_write", costs["cache_write"]
                                        )
                                else:
                                    # For non-chat requests, use simple cost calculation
                                    cost = self.cost_calculator.calculate_cost(
                                        model, usage, call_type
                                    )
                                    if cost and cost > 0:
                                        if self.cost_counter:
                                            self.cost_counter.add(cost, {"model": str(model)})
                                        span.set_attribute("gen_ai.usage.cost.total", cost)
                            except Exception as e:
                                logger.warning(
                                    "Failed to calculate cost for streaming response: %s", e
                                )
                    else:
                        logger.debug("No usage information found in streaming response")
            except Exception as e:
                logger.warning("Failed to extract usage from streaming response: %s", e)

            span.set_status(Status(StatusCode.OK))
            span.end()  # Close the span when streaming completes

            # Decrement server metrics: running requests counter (streaming success)
            server_metrics = get_server_metrics()
            if server_metrics:
                server_metrics.decrement_requests_running()
                logger.debug("Decremented running requests (streaming success)")

            logger.debug(f"Streaming completed: {token_count} chunks in {duration:.3f}s")

        except Exception as e:
            # Stream failed
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()  # Close the span even on error

            # Decrement server metrics: running requests counter (streaming error)
            server_metrics = get_server_metrics()
            if server_metrics:
                server_metrics.decrement_requests_running()
                logger.debug("Decremented running requests (streaming error)")

            if self.error_counter:
                self.error_counter.add(1, {"operation": span.name, "error_type": type(e).__name__})
            logger.warning(f"Error in streaming wrapper: {e}")
            raise

    # Phase 4.2: RAG/Embedding Helper Methods
    def add_embedding_attributes(
        self, span, model: str, input_text: str, vector: Optional[List[float]] = None
    ):
        """Add embedding-specific attributes to a span.

        Args:
            span: The OpenTelemetry span
            model: The embedding model name
            input_text: The text being embedded (will be truncated to 500 chars)
            vector: Optional embedding vector (use with caution - can be large!)
        """
        span.set_attribute("embedding.model_name", model)
        span.set_attribute("embedding.text", input_text[:500])  # Truncate to avoid large spans

        if vector and self.config and hasattr(self.config, "capture_embedding_vectors"):
            # Only capture vectors if explicitly enabled (they can be very large)
            span.set_attribute("embedding.vector", json.dumps(vector))
            span.set_attribute("embedding.vector.dimension", len(vector))

    def add_retrieval_attributes(
        self,
        span,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        max_docs: int = 5,
    ):
        """Add retrieval/RAG-specific attributes to a span.

        Args:
            span: The OpenTelemetry span
            documents: List of retrieved documents. Each dict should have:
                - id: Document identifier
                - score: Relevance score
                - content: Document content
                - metadata: Optional metadata dict
            query: Optional query string
            max_docs: Maximum number of documents to include in attributes (default: 5)
        """
        if query:
            span.set_attribute("retrieval.query", query[:500])  # Truncate

        # Limit to first N documents to avoid attribute explosion
        for i, doc in enumerate(documents[:max_docs]):
            prefix = f"retrieval.documents.{i}.document"

            if "id" in doc:
                span.set_attribute(f"{prefix}.id", str(doc["id"]))
            if "score" in doc:
                span.set_attribute(f"{prefix}.score", float(doc["score"]))
            if "content" in doc:
                # Truncate content to avoid large attributes
                content = str(doc["content"])[:500]
                span.set_attribute(f"{prefix}.content", content)

            # Add metadata if present
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                for key, value in doc["metadata"].items():
                    # Flatten metadata, limit key names to avoid explosion
                    safe_key = str(key)[:50]  # Limit key length
                    safe_value = str(value)[:200]  # Limit value length
                    span.set_attribute(f"{prefix}.metadata.{safe_key}", safe_value)

        span.set_attribute("retrieval.document_count", len(documents))

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
