"""OpenTelemetry instrumentor for the Groq SDK.

This instrumentor automatically traces chat completion calls to Groq models,
capturing relevant attributes such as the model name and token usage.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class GroqInstrumentor(BaseInstrumentor):
    """Instrumentor for Groq"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._groq_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Groq library is available."""
        try:
            import groq

            self._groq_available = True
            logger.debug("Groq library detected and available for instrumentation")
        except ImportError:
            logger.debug("Groq library not installed, instrumentation will be skipped")
            self._groq_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Groq SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._groq_available:
            logger.debug("Skipping Groq instrumentation - library not available")
            return

        self.config = config

        try:
            import groq

            original_init = groq.Groq.__init__

            def wrapped_init(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                self._instrument_client(instance)
                return instance

            groq.Groq.__init__ = wrapped_init
            self._instrumented = True
            logger.info("Groq instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Groq: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument Groq client methods.

        Args:
            client: The Groq client instance to instrument.
        """
        original_create = client.chat.completions.create

        def wrapped_create(*args, **kwargs):
            with self.tracer.start_as_current_span("groq.chat.completions") as span:
                model = kwargs.get("model", "unknown")

                span.set_attribute("gen_ai.system", "groq")
                span.set_attribute("gen_ai.request.model", model)

                # Capture request content for evaluation support
                messages = kwargs.get("messages", [])
                if messages:
                    try:
                        fm = self._build_first_message(messages)
                        if fm:
                            span.set_attribute("gen_ai.request.first_message", fm)
                    except (IndexError, AttributeError) as e:
                        logger.debug("Failed to extract request content: %s", e)

                if self.request_counter:
                    self.request_counter.add(1, {"model": model, "provider": "groq"})

                result = original_create(*args, **kwargs)
                self._record_result_metrics(span, result, 0)

                # Capture response content for evaluation support
                response_attrs = self._extract_response_attributes(result)
                for key, value in response_attrs.items():
                    span.set_attribute(key, value)

                return result

        client.chat.completions.create = wrapped_create

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Groq response.

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        if hasattr(result, "usage"):
            return {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from Groq response for evaluation support.

        Args:
            result: The API response object.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        # Extract response content for evaluation support
        try:
            # Groq responses use OpenAI-compatible format: choices[0].message.content
            if hasattr(result, "choices") and result.choices:
                first_choice = result.choices[0]
                if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                    response_content = first_choice.message.content
                    if response_content:
                        attrs["gen_ai.response"] = response_content
        except (IndexError, AttributeError) as e:
            logger.debug("Failed to extract response content: %s", e)

        return attrs
