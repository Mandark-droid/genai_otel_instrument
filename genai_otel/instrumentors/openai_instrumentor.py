import wrapt
from typing import Dict, Any, Optional
from .base import BaseInstrumentor
from ..config import OTelConfig


class OpenAIInstrumentor(BaseInstrumentor):
    """Instrumentor for OpenAI SDK"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import openai

            # Instrument ChatCompletion.create using create_span_wrapper
            if hasattr(openai, "ChatCompletion") and hasattr(openai.ChatCompletion, "create"):
                original_create_method = openai.ChatCompletion.create
                instrumented_create_method = self.create_span_wrapper(
                    span_name="openai.chat.completion",
                    extract_attributes=self._extract_openai_attributes
                )
                openai.ChatCompletion.create = instrumented_create_method

            # Instrument OpenAI client initialization
            if hasattr(openai, "OpenAI"):
                original_init = openai.OpenAI.__init__

                def wrapped_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_client(instance)
                    return result

                openai.OpenAI.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)

        except ImportError:
            pass

    def _instrument_client(self, client):
        if hasattr(client, "chat") and hasattr(client.chat, "completions") and hasattr(client.chat.completions, "create"):
            original_create_method = client.chat.completions.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="openai.chat.completion",
                extract_attributes=self._extract_openai_attributes
            )
            client.chat.completions.create = instrumented_create_method

    # New method to extract attributes
    def _extract_openai_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        attrs = {}
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        attrs["gen_ai.system"] = "openai"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.request.message_count"] = len(messages)
        if messages:
            # Consider truncating if messages can be very long
            attrs["gen_ai.request.first_message"] = str(messages[0])
        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        # Add safety checks for result and result.usage
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            return {
                "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(usage, 'completion_tokens', 0),
                "total_tokens": getattr(usage, 'total_tokens', 0)
            }
        return None
