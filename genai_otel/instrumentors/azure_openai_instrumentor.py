"""OpenTelemetry instrumentor for Azure OpenAI SDK.

This instrumentor automatically traces calls to Azure OpenAI models, capturing
relevant attributes such as model name and token usage.
"""

import logging
import time
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


def _cap_content(config, text):
    """Bound captured content to config.content_max_length (0/None/unset = unlimited)."""
    if text is None:
        return text
    text = str(text)
    max_len = getattr(config, "content_max_length", 0) if config else 0
    if isinstance(max_len, int) and max_len > 0:
        return text[:max_len]
    return text


class AzureOpenAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Azure OpenAI"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._azure_openai_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Azure AI OpenAI library is available."""
        try:
            import azure.ai.openai  # Moved to top

            self._azure_openai_available = True
            logger.debug("Azure AI OpenAI library detected and available for instrumentation")
        except ImportError:
            logger.debug("Azure AI OpenAI library not installed, instrumentation will be skipped")
            self._azure_openai_available = False

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from azure.ai.openai import OpenAIClient

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(OpenAIClient, "_genai_otel_azure_instrumented", False) is True:
                logger.debug("Azure OpenAI already instrumented, skipping")
                return

            original_complete = OpenAIClient.complete

            def wrapped_complete(instance, *args, **kwargs):
                # Started explicitly rather than with `with`: a streamed call
                # outlives this function, and the context manager would close
                # the span before the first token arrived.
                span = self.tracer.start_span("azure.openai.complete")
                token = otel_context.attach(trace.set_span_in_context(span))
                start_time = time.time()
                handed_to_stream = False
                try:
                    model = kwargs.get("model", "unknown")

                    span.set_attribute("gen_ai.system", "azure_openai")
                    span.set_attribute("gen_ai.request.model", model)

                    # Capture request content for evaluation support
                    # Azure OpenAI supports both messages and prompt
                    messages = kwargs.get("messages", [])
                    if messages:
                        try:
                            fm = self._build_first_message(messages)
                            if fm:
                                span.set_attribute("gen_ai.request.first_message", fm)
                        except (IndexError, AttributeError) as e:
                            logger.debug("Failed to extract request content: %s", e)
                    elif "prompt" in kwargs:
                        # Fallback to prompt if messages not present
                        try:
                            fm = self._build_first_message(
                                [{"role": "user", "content": str(kwargs.get("prompt", ""))}]
                            )
                            if fm:
                                span.set_attribute("gen_ai.request.first_message", fm)
                        except Exception as e:
                            logger.debug("Failed to extract prompt content: %s", e)

                    if self.request_counter:
                        self.request_counter.add(1, {"model": model, "provider": "azure_openai"})

                    result = original_complete(instance, *args, **kwargs)

                    handled, value = self._install_stream_measurement(
                        span, result, start_time, model, kwargs
                    )
                    if handled:
                        handed_to_stream = True
                        return value

                    self._record_result_metrics(span, result, 0)

                    # Capture response content for evaluation support
                    response_attrs = self._extract_response_attributes(result)
                    for key, value in response_attrs.items():
                        span.set_attribute(key, value)

                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    otel_context.detach(token)
                    # The stream wrapper owns the span's end once it takes over.
                    if not handed_to_stream:
                        span.end()

            OpenAIClient.complete = wrapped_complete
            try:
                OpenAIClient._genai_otel_azure_instrumented = True
            except Exception:  # noqa: BLE001
                pass

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        if hasattr(result, "usage") and result.usage:
            return {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from Azure OpenAI response for evaluation support.

        Args:
            result: The API response object.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        # Extract response content for evaluation support
        try:
            # Azure OpenAI uses OpenAI-compatible format: choices[0].message.content
            if hasattr(result, "choices") and result.choices:
                first_choice = result.choices[0]
                if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                    response_content = first_choice.message.content
                    if response_content:
                        attrs["gen_ai.response"] = response_content
                # Fallback to text attribute for completions
                elif hasattr(first_choice, "text"):
                    response_content = first_choice.text
                    if response_content:
                        attrs["gen_ai.response"] = response_content
        except (IndexError, AttributeError) as e:
            logger.debug("Failed to extract response content: %s", e)

        # Bound captured completion text (audit content stays, but capped).
        if "gen_ai.response" in attrs:
            attrs["gen_ai.response"] = _cap_content(
                getattr(self, "config", None), attrs["gen_ai.response"]
            )

        return attrs
