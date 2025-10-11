import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class AWSBedrockInstrumentor(BaseInstrumentor):
    """Instrumentor for AWS Bedrock"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._boto3_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if boto3 library is available."""
        try:
            import boto3  # Moved to top

            self._boto3_available = True
            logger.debug("boto3 library detected and available for instrumentation")
        except ImportError:
            logger.debug("boto3 library not installed, instrumentation will be skipped")
            self._boto3_available = False

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import boto3  # Moved to top

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
        if hasattr(client, "invoke_model"):
            instrumented_invoke_method = self.create_span_wrapper(
                span_name="aws.bedrock.invoke_model",
                extract_attributes=self._extract_aws_bedrock_attributes,
            )
            client.invoke_model = instrumented_invoke_method

    def _extract_aws_bedrock_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:  # pylint: disable=W0613
        attrs = {}
        model_id = kwargs.get("modelId", "unknown")

        attrs["gen_ai.system"] = "aws_bedrock"
        attrs["gen_ai.request.model"] = model_id
        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:  # pylint: disable=R1705
        if hasattr(result, "get"):
            content_type = result.get("contentType", "").lower()
            body_str = result.get("body", "")

            if "application/json" in content_type and body_str:
                try:
                    body = json.loads(body_str)
                    if "usage" in body and isinstance(body["usage"], dict):
                        usage = body["usage"]
                        return {
                            "prompt_tokens": getattr(usage, "inputTokens", 0),
                            "completion_tokens": getattr(usage, "outputTokens", 0),
                            "total_tokens": getattr(usage, "inputTokens", 0)
                            + getattr(usage, "outputTokens", 0),
                        }
                    elif "usageMetadata" in body and isinstance(body["usageMetadata"], dict):
                        usage = body["usageMetadata"]
                        return {
                            "prompt_tokens": getattr(usage, "promptTokenCount", 0),
                            "completion_tokens": getattr(usage, "candidatesTokenCount", 0),
                            "total_tokens": getattr(usage, "totalTokenCount", 0),
                        }
                except json.JSONDecodeError:
                    logger.debug("Failed to parse Bedrock response body as JSON.")
                except Exception as e:
                    logger.debug("Error extracting usage from Bedrock response: %s", e)
        return None
