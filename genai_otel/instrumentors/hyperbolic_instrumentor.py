"""OpenTelemetry instrumentor for Hyperbolic API calls.

This instrumentor automatically traces HTTP requests to Hyperbolic's API,
capturing relevant LLM attributes such as model name and token usage from
the raw HTTP response.
"""

import json
import logging
from typing import Any, Dict, Optional

import wrapt

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class HyperbolicInstrumentor(BaseInstrumentor):
    """Instrumentor for Hyperbolic API (raw HTTP requests)"""

    HYPERBOLIC_API_BASE = "https://api.hyperbolic.xyz"

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._requests_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if requests library is available."""
        try:
            import requests

            self._requests_available = True
            logger.debug("Requests library detected, Hyperbolic instrumentation available")
        except ImportError:
            logger.debug("Requests library not installed, Hyperbolic instrumentation skipped")
            self._requests_available = False

    def instrument(self, config: OTelConfig):
        """Instrument requests library for Hyperbolic API calls.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._requests_available:
            logger.debug("Skipping Hyperbolic instrumentation - requests library not available")
            return

        self.config = config

        try:
            import requests

            # Wrap requests.post to intercept Hyperbolic API calls
            original_post = requests.post

            @wrapt.decorator
            def hyperbolic_post_wrapper(wrapped, instance, args, kwargs):
                # Check if this is a Hyperbolic API call
                url = args[0] if args else kwargs.get("url", "")
                if not url.startswith(self.HYPERBOLIC_API_BASE):
                    # Not a Hyperbolic call, pass through
                    return wrapped(*args, **kwargs)

                # Extract attributes before the call
                request_data = kwargs.get("json", {})
                attrs = self._extract_request_attributes(request_data)

                # Create span wrapper
                with self.tracer.start_as_current_span("hyperbolic.chat.completion") as span:
                    # Set request attributes
                    for key, value in attrs.items():
                        span.set_attribute(key, value)

                    # Record request metric
                    model = attrs.get("gen_ai.request.model", "unknown")
                    if self.request_counter:
                        self.request_counter.add(1, {"model": model, "provider": "hyperbolic"})

                    try:
                        # Make the actual API call
                        response = wrapped(*args, **kwargs)

                        # Extract response attributes
                        if response.status_code == 200:
                            response_data = response.json()
                            self._extract_and_record_response(span, response_data)
                        else:
                            span.set_attribute("error", True)
                            span.set_attribute("http.status_code", response.status_code)

                        return response

                    except Exception as e:
                        span.set_attribute("error", True)
                        span.record_exception(e)
                        if self.error_counter:
                            self.error_counter.add(
                                1,
                                {
                                    "operation": "chat.completion",
                                    "error.type": type(e).__name__,
                                    "provider": "hyperbolic",
                                },
                            )
                        raise

            # Apply the wrapper
            requests.post = hyperbolic_post_wrapper(original_post)
            self._instrumented = True
            logger.info("Hyperbolic instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Hyperbolic: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_request_attributes(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from Hyperbolic API request.

        Args:
            request_data: The JSON request payload.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "hyperbolic"
        attrs["gen_ai.request.model"] = request_data.get("model", "unknown")
        attrs["gen_ai.operation.name"] = "chat"

        messages = request_data.get("messages", [])
        attrs["gen_ai.request.message_count"] = len(messages)

        # Request parameters
        if "temperature" in request_data:
            attrs["gen_ai.request.temperature"] = request_data["temperature"]
        if "top_p" in request_data:
            attrs["gen_ai.request.top_p"] = request_data["top_p"]
        if "max_tokens" in request_data:
            attrs["gen_ai.request.max_tokens"] = request_data["max_tokens"]

        # First message preview
        if messages:
            first_message = str(messages[0])[:200]
            attrs["gen_ai.request.first_message"] = first_message

        return attrs

    def _extract_and_record_response(self, span, response_data: Dict[str, Any]):
        """Extract response attributes and record metrics.

        Args:
            span: The OpenTelemetry span.
            response_data: The JSON response from Hyperbolic API.
        """
        # Response ID
        if "id" in response_data:
            span.set_attribute("gen_ai.response.id", response_data["id"])

        # Response model
        if "model" in response_data:
            span.set_attribute("gen_ai.response.model", response_data["model"])

        # Finish reasons
        choices = response_data.get("choices", [])
        if choices:
            finish_reasons = [
                choice.get("finish_reason") for choice in choices if "finish_reason" in choice
            ]
            if finish_reasons:
                span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)

        # Extract token usage
        usage_data = response_data.get("usage", {})
        if usage_data:
            usage_dict = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

            # Record token usage as span attributes
            span.set_attribute("gen_ai.usage.prompt_tokens", usage_dict["prompt_tokens"])
            span.set_attribute("gen_ai.usage.completion_tokens", usage_dict["completion_tokens"])
            span.set_attribute("gen_ai.usage.total_tokens", usage_dict["total_tokens"])

            # Record token metrics
            if self.token_counter:
                model = span.attributes.get("gen_ai.request.model", "unknown")
                self.token_counter.add(
                    usage_dict["prompt_tokens"],
                    {"token_type": "prompt", "model": model, "provider": "hyperbolic"},
                )
                self.token_counter.add(
                    usage_dict["completion_tokens"],
                    {"token_type": "completion", "model": model, "provider": "hyperbolic"},
                )

            # Calculate and record cost
            if self.config.enable_cost_tracking:
                from ..cost_calculator import CostCalculator

                cost_calc = CostCalculator(custom_pricing_json=self.config.custom_pricing_json)
                model = span.attributes.get("gen_ai.request.model", "unknown")
                cost = cost_calc.calculate_cost(
                    model_name=model,
                    prompt_tokens=usage_dict["prompt_tokens"],
                    completion_tokens=usage_dict["completion_tokens"],
                    call_type="chat",
                )

                if cost > 0 and self.cost_counter:
                    span.set_attribute("gen_ai.cost.amount", cost)
                    self.cost_counter.add(
                        cost, {"model": model, "provider": "hyperbolic", "call_type": "chat"}
                    )

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from response.

        Note: This method is required by BaseInstrumentor but not used for HTTP-based
        instrumentation. Token extraction is handled in _extract_and_record_response.

        Args:
            result: The API response (unused for HTTP instrumentation).

        Returns:
            None
        """
        return None
