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

        # Capture request content for evaluation support
        body = kwargs.get("body", "")
        if body:
            try:
                # Body is usually a JSON string
                if isinstance(body, (str, bytes)):
                    body_dict = (
                        json.loads(body)
                        if isinstance(body, str)
                        else json.loads(body.decode("utf-8"))
                    )
                else:
                    body_dict = body

                # Extract content based on model family
                # Claude format: messages array
                if "messages" in body_dict and body_dict["messages"]:
                    fm = self._build_first_message(body_dict["messages"])
                    if fm:
                        attrs["gen_ai.request.first_message"] = fm
                # Llama/Titan format: prompt field
                elif "prompt" in body_dict:
                    fm = self._build_first_message(
                        [{"role": "user", "content": str(body_dict["prompt"])}]
                    )
                    if fm:
                        attrs["gen_ai.request.first_message"] = fm
                # Generic input field
                elif "inputText" in body_dict:
                    fm = self._build_first_message(
                        [{"role": "user", "content": str(body_dict["inputText"])}]
                    )
                    if fm:
                        attrs["gen_ai.request.first_message"] = fm
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                logger.debug("Failed to extract request content: %s", e)

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

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from AWS Bedrock response for evaluation support.

        Args:
            result: The API response object.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        # Extract response content for evaluation support
        try:
            if hasattr(result, "get"):
                body_str = result.get("body", "")

                if body_str:
                    # Parse response body
                    if isinstance(body_str, bytes):
                        body_str = body_str.decode("utf-8")

                    body = json.loads(body_str) if isinstance(body_str, str) else body_str

                    # Extract content based on model family response format
                    # Claude format: content array
                    if "content" in body:
                        content = body["content"]
                        if isinstance(content, list) and len(content) > 0:
                            # Claude returns list of content blocks
                            first_content = content[0]
                            if isinstance(first_content, dict) and "text" in first_content:
                                attrs["gen_ai.response"] = first_content["text"]
                            else:
                                attrs["gen_ai.response"] = str(content[0])
                        elif isinstance(content, str):
                            attrs["gen_ai.response"] = content
                    # Llama/Titan format: completion or generation field
                    elif "completion" in body:
                        attrs["gen_ai.response"] = body["completion"]
                    elif "generation" in body:
                        attrs["gen_ai.response"] = body["generation"]
                    # Generic output field
                    elif "outputText" in body:
                        attrs["gen_ai.response"] = body["outputText"]
                    # Results array format
                    elif (
                        "results" in body
                        and isinstance(body["results"], list)
                        and len(body["results"]) > 0
                    ):
                        first_result = body["results"][0]
                        if isinstance(first_result, dict) and "outputText" in first_result:
                            attrs["gen_ai.response"] = first_result["outputText"]
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError) as e:
            logger.debug("Failed to extract response content: %s", e)

        return attrs
