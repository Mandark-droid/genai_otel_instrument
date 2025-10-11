import wrapt
from typing import Dict, Optional, Any, Callable
import json
from .base import BaseInstrumentor
from ..config import OTelConfig

logger = logging.getLogger(__name__)

class AWSBedrockInstrumentor(BaseInstrumentor):
    """Instrumentor for AWS Bedrock"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import boto3

            # Instrument bedrock-runtime client
            original_client = boto3.client

            def wrapped_client(*args, **kwargs):
                client = original_client(*args, **kwargs)
                if args and args[0] == "bedrock-runtime":
                    self._instrument_bedrock_client(client)
                return client

            boto3.client = wrapped_client

        except ImportError:
            pass

    def _instrument_bedrock_client(self, client):
        # Use create_span_wrapper for invoke_model
        if hasattr(client, "invoke_model"):
            original_invoke_method = client.invoke_model
            instrumented_invoke_method = self.create_span_wrapper(
                span_name="aws.bedrock.invoke_model",
                extract_attributes=self._extract_aws_bedrock_attributes
            )
            client.invoke_model = instrumented_invoke_method

    # New method to extract attributes, accepting instance
    def _extract_aws_bedrock_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        attrs = {}
        model_id = kwargs.get("modelId", "unknown")
        # Body might contain prompt, but it's complex to parse here and might not be needed for span attributes.

        attrs["gen_ai.system"] = "aws_bedrock"
        attrs["gen_ai.request.model"] = model_id
        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        # Attempt to parse the response body for usage metadata
        if hasattr(result, "get"):
            content_type = result.get("contentType", "").lower()
            body_str = result.get("body", "")

            if "application/json" in content_type and body_str:
                try:
                    body = json.loads(body_str)
                    # Check for common usage patterns from different models via Bedrock
                    # Anthropic Claude usage:
                    if "usage" in body and isinstance(body["usage"], dict):
                        usage = body["usage"]
                        return {
                            "prompt_tokens": getattr(usage, 'inputTokens', 0),
                            "completion_tokens": getattr(usage, 'outputTokens', 0),
                            "total_tokens": getattr(usage, 'inputTokens', 0) + getattr(usage, 'outputTokens', 0)
                        }
                    # Gemini usage (if applicable via Bedrock):
                    elif "usageMetadata" in body and isinstance(body["usageMetadata"], dict):
                        usage = body["usageMetadata"]
                        return {
                            "prompt_tokens": getattr(usage, 'promptTokenCount', 0),
                            "completion_tokens": getattr(usage, 'candidatesTokenCount', 0),
                            "total_tokens": getattr(usage, 'totalTokenCount', 0)
                        }
                    # Add other model specific usage patterns if known
                except json.JSONDecodeError:
                    logger.debug("Failed to parse Bedrock response body as JSON.")
                except Exception as e:
                    logger.debug(f"Error extracting usage from Bedrock response: {e}")
        return None