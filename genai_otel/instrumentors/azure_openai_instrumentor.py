"""OpenTelemetry instrumentor for Azure OpenAI SDK.

This instrumentor automatically traces calls to Azure OpenAI models, capturing
relevant attributes such as model name and token usage.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


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

            original_complete = OpenAIClient.complete

            def wrapped_complete(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("azure.openai.complete") as span:
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
                    self._record_result_metrics(span, result, 0)

                    # Capture response content for evaluation support
                    response_attrs = self._extract_response_attributes(result)
                    for key, value in response_attrs.items():
                        span.set_attribute(key, value)

                    return result

            OpenAIClient.complete = wrapped_complete

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

        return attrs
