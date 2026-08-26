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


def _converse_blocks_text(blocks: Any) -> str:
    """Flatten Converse content blocks to plain text.

    Converse content is a list of typed blocks -- ``text``, ``image``,
    ``toolUse``, ``toolResult`` -- rather than a plain string. Only the text
    blocks carry anything worth putting on a span.
    """
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, (list, tuple)):
        parts = []
        for block in blocks:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


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

        # The streaming sibling of the call above: same request shape, same
        # extractor. Without it the covered path went dark the moment a caller
        # streamed.
        if hasattr(client, "invoke_model_with_response_stream"):
            original_ims = client.invoke_model_with_response_stream
            client.invoke_model_with_response_stream = self.create_span_wrapper(
                span_name="aws.bedrock.invoke_model_with_response_stream",
                extract_attributes=self._extract_aws_bedrock_attributes,
            )(original_ims)

        # Converse is the unified API AWS points callers at, and the practical
        # path for every non-Anthropic model. Being model-agnostic, it needs no
        # per-model body shim -- unlike invoke_model.
        if hasattr(client, "converse"):
            original_converse = client.converse
            client.converse = self.create_span_wrapper(
                span_name="aws.bedrock.converse",
                extract_attributes=self._extract_converse_attributes,
            )(original_converse)

        if hasattr(client, "converse_stream"):
            client.converse_stream = self._wrap_converse_stream(client.converse_stream)

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

    def _extract_converse_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:  # pylint: disable=W0613
        """Extract span attributes from a Converse request.

        Converse is model-agnostic, so unlike invoke_model the shape does not
        have to be driven off modelId.
        """
        attrs: Dict[str, Any] = {}
        messages = kwargs.get("messages") or []

        attrs["gen_ai.system"] = "aws_bedrock"
        attrs["gen_ai.request.model"] = kwargs.get("modelId", "unknown")
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.message_count"] = len(messages)

        config = getattr(self, "config", None)

        # `system` is a top-level parameter carrying its own content blocks, not
        # a role inside messages[], so it must not inflate the message count.
        system = kwargs.get("system")
        if system:
            instructions = _converse_blocks_text(system)
            if instructions:
                attrs["gen_ai.request.instructions"] = _cap_content(config, instructions)

        inference = kwargs.get("inferenceConfig") or {}
        if isinstance(inference, dict):
            if "maxTokens" in inference:
                attrs["gen_ai.request.max_tokens"] = inference["maxTokens"]
            if "temperature" in inference:
                attrs["gen_ai.request.temperature"] = inference["temperature"]
            if "topP" in inference:
                attrs["gen_ai.request.top_p"] = inference["topP"]
            if "stopSequences" in inference:
                attrs["gen_ai.request.stop_sequences"] = inference["stopSequences"]

        tool_config = kwargs.get("toolConfig")
        if tool_config:
            try:
                attrs["llm.tools"] = json.dumps(tool_config.get("tools", tool_config))
            except (TypeError, ValueError) as e:
                logger.debug("Failed to serialize Bedrock toolConfig: %s", e)

        # Normalise onto the chat message shape so the shared first-message
        # machinery applies unchanged.
        normalised = [
            {
                "role": m.get("role", "user"),
                "content": _converse_blocks_text(m.get("content", "")),
            }
            for m in messages
            if isinstance(m, dict)
        ]
        first_message = self._build_first_message(normalised)
        if first_message:
            attrs["gen_ai.request.first_message"] = first_message

        return attrs

    def _wrap_converse_stream(self, original):
        """Span wrapper for converse_stream.

        Bedrock hands back ``{"stream": EventStream}`` immediately -- the model
        generates while the caller iterates -- and there is no ``stream=True``
        kwarg for the generic wrapper to key on. So the span is opened here and
        given to the shared stream wrapper, which closes it once the event
        stream is exhausted and reads usage from the trailing ``metadata``
        event. Closing the span on return instead would report near-zero
        latency and no tokens.
        """
        import time

        from opentelemetry.trace import Status, StatusCode

        instrumentor = self

        def wrapper(*args, **kwargs):
            if not instrumentor._instrumented:
                return original(*args, **kwargs)

            attributes: Dict[str, Any] = {}
            try:
                attributes = instrumentor._extract_converse_attributes(None, args, kwargs)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to extract converse_stream attributes: %s", e)

            span = instrumentor.tracer.start_span(
                "aws.bedrock.converse_stream", attributes=attributes
            )
            start_time = time.time()
            model = kwargs.get("modelId", "unknown")

            try:
                result = original(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise

            # Sampled-out spans come back as NonRecordingSpan, which has no
            # `.name` -- the measurement path below reads it, so measuring here
            # would raise on every call once sampling drops the span. Hand the
            # caller the untouched stream instead; the measurements would have
            # been discarded anyway.
            if not span.is_recording():
                span.end()
                return result

            stream = result.get("stream") if hasattr(result, "get") else None
            if stream is None:
                # Nothing to measure -- close the span rather than leak it.
                span.set_status(Status(StatusCode.OK))
                span.end()
                return result

            if instrumentor.request_counter:
                instrumentor.request_counter.add(1, {"operation": "aws.bedrock.converse_stream"})

            result["stream"] = instrumentor._wrap_streaming_response(
                stream, span, start_time, model
            )
            return result

        return wrapper

    @staticmethod
    def _converse_usage(usage: Any) -> Optional[Dict[str, int]]:
        """Map Converse's camelCase token counts onto the canonical keys."""
        if not isinstance(usage, dict):
            return None
        if "inputTokens" not in usage and "outputTokens" not in usage:
            return None
        input_tokens = usage.get("inputTokens", 0) or 0
        output_tokens = usage.get("outputTokens", 0) or 0
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": usage.get("totalTokens", input_tokens + output_tokens),
        }

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Return Converse's stopReason, from the response or a messageStop event."""
        try:
            if hasattr(result, "get"):
                reason = result.get("stopReason")
                if isinstance(reason, str) and reason:
                    return reason
                message_stop = result.get("messageStop")
                if isinstance(message_stop, dict):
                    reason = message_stop.get("stopReason")
                    if isinstance(reason, str) and reason:
                        return reason
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to extract Bedrock stopReason: %s", e)
        return None

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:  # pylint: disable=R1705
        if hasattr(result, "get"):
            # Converse reports usage at the top level in camelCase, with no
            # `body` and no `contentType` -- the contentType gate below would
            # return None and leave the span priced at zero.
            converse = self._converse_usage(result.get("usage"))
            if converse is not None:
                return converse

            # converse_stream reports tokens only in the trailing metadata event.
            metadata = result.get("metadata")
            if isinstance(metadata, dict):
                converse = self._converse_usage(metadata.get("usage"))
                if converse is not None:
                    return converse

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

        # Converse returns a plain dict: the completion, the stop reason and any
        # tool calls live under output.message.content[], not in a JSON body.
        try:
            if hasattr(result, "get") and isinstance(result.get("output"), dict):
                message = result["output"].get("message") or {}
                blocks = message.get("content") or []

                text = _converse_blocks_text(blocks)
                if text:
                    attrs["gen_ai.response"] = _cap_content(getattr(self, "config", None), text)

                finish_reason = self._extract_finish_reason(result)
                if finish_reason:
                    attrs["gen_ai.response.finish_reasons"] = [finish_reason]

                tc_idx = 0
                for block in blocks:
                    if not isinstance(block, dict) or "toolUse" not in block:
                        continue
                    tool_use = block["toolUse"] or {}
                    prefix = f"llm.output_messages.0.message.tool_calls.{tc_idx}"
                    if tool_use.get("toolUseId"):
                        attrs[f"{prefix}.tool_call.id"] = tool_use["toolUseId"]
                    if tool_use.get("name"):
                        attrs[f"{prefix}.tool_call.function.name"] = tool_use["name"]
                    if "input" in tool_use:
                        try:
                            attrs[f"{prefix}.tool_call.function.arguments"] = json.dumps(
                                tool_use["input"]
                            )
                        except (TypeError, ValueError) as e:
                            logger.debug("Failed to serialize toolUse input: %s", e)
                    tc_idx += 1
                return attrs
        except (AttributeError, KeyError, TypeError) as e:
            logger.debug("Failed to extract Converse response content: %s", e)

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
