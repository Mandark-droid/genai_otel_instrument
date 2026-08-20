import json
import logging
from typing import Any, Dict, Optional

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

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(boto3, "_genai_otel_bedrock_instrumented", False) is True:
                logger.debug("AWS Bedrock already instrumented, skipping")
                self._instrumented = True
                return

            original_client = boto3.client

            def wrapped_client(*args, **kwargs):
                client = original_client(*args, **kwargs)
                if args and args[0] == "bedrock-runtime":
                    self._instrument_bedrock_client(client)
                return client

            boto3.client = wrapped_client
            try:
                boto3._genai_otel_bedrock_instrumented = True
            except Exception:  # noqa: BLE001
                pass
            # Required so create_span_wrapper actually records spans instead of
            # short-circuiting to the unwrapped call.
            self._instrumented = True

        except ImportError:
            pass

    def _instrument_bedrock_client(self, client):
        if hasattr(client, "invoke_model"):
            # Capture the bound method and apply the span-wrapper factory to it;
            # assigning the factory itself (without calling it on the original)
            # makes invoke_model raise TypeError on first use.
            original_invoke_model = client.invoke_model
            client.invoke_model = self.create_span_wrapper(
                span_name="aws.bedrock.invoke_model",
                extract_attributes=self._extract_aws_bedrock_attributes,
            )(original_invoke_model)

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

                # "inputText" alone is not a reliable embedding signal: Titan Text
                # generation requests use the same {"inputText": ...} body shape, and
                # textGenerationConfig is optional there (defaults apply if omitted), so
                # its absence can't be used to rule out a Titan Text call either. Titan
                # Text and Titan Embed model IDs are unambiguous ("amazon.titan-text-*"
                # vs "amazon.titan-embed-*"), so use the model family as the deciding
                # signal for a bare inputText body instead.
                model_id_lower = str(model_id).lower()
                is_embedding = (
                    "embed" in model_id_lower
                    or any(key in body_dict for key in ("texts", "inputs"))
                    or (
                        "inputText" in body_dict
                        and "titan-text" not in model_id_lower
                        and "textGenerationConfig" not in body_dict
                        and "messages" not in body_dict
                    )
                )
                attrs["gen_ai.operation.name"] = "embeddings" if is_embedding else "chat"
                attrs["gen_ai.request.type"] = "embedding" if is_embedding else "chat"
                if is_embedding:
                    values = body_dict.get(
                        "texts", body_dict.get("inputs", body_dict.get("inputText"))
                    )
                    if isinstance(values, (list, tuple)):
                        count = len(values)
                    else:
                        count = 1 if values is not None else 0
                    attrs["gen_ai.request.input_count"] = count

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
                        input_tokens = usage.get("inputTokens", 0)
                        output_tokens = usage.get("outputTokens", 0)
                        return {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        }
                    elif "usageMetadata" in body and isinstance(body["usageMetadata"], dict):
                        usage = body["usageMetadata"]
                        input_tokens = usage.get("promptTokenCount", 0)
                        output_tokens = usage.get("candidatesTokenCount", 0)
                        return {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": usage.get(
                                "totalTokenCount", input_tokens + output_tokens
                            ),
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

                    vectors = body.get("embeddings")
                    if vectors is None and body.get("embedding") is not None:
                        vectors = [body["embedding"]]
                    if vectors is not None:
                        vectors = list(vectors)
                        attrs["gen_ai.response.embedding_count"] = len(vectors)
                        if vectors and isinstance(vectors[0], (list, tuple)):
                            attrs["gen_ai.response.vector_size"] = len(vectors[0])
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError) as e:
            logger.debug("Failed to extract response content: %s", e)

        # Bound captured completion text (audit content stays, but capped).
        if "gen_ai.response" in attrs:
            attrs["gen_ai.response"] = _cap_content(
                getattr(self, "config", None), attrs["gen_ai.response"]
            )

        return attrs
