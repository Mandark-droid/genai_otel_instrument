from typing import Dict, Optional, Any, Callable
import wrapt
from .base import BaseInstrumentor
from ..config import OTelConfig


class AnthropicInstrumentor(BaseInstrumentor):
    """Instrumentor for Anthropic Claude SDK"""

    def instrument(self, config: OTelConfig):
        self.config = config

        try:
            import anthropic

            if hasattr(anthropic, "Anthropic"):
                original_init = anthropic.Anthropic.__init__

                def wrapped_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_client(instance)
                    return result

                anthropic.Anthropic.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)

        except ImportError:
            pass

    def _instrument_client(self, client):
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            original_create_method = client.messages.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="anthropic.messages.create",
                extract_attributes=self._extract_anthropic_attributes
            )
            client.messages.create = instrumented_create_method

    # New method to extract attributes
    def _extract_anthropic_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        attrs = {}
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        attrs["gen_ai.system"] = "anthropic"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.request.message_count"] = len(messages)
        # Anthropic messages format might be different, check if first message is useful to log
        # if messages:
        #     attrs["gen_ai.request.first_message"] = str(messages[0])
        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        # Add safety checks for result and result.usage
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            return {
                "prompt_tokens": getattr(usage, 'input_tokens', 0),
                "completion_tokens": getattr(usage, 'output_tokens', 0),
                "total_tokens": getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)
            }
        return None