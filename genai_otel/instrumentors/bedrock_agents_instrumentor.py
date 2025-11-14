"""OpenTelemetry instrumentor for AWS Bedrock Agents.

This instrumentor automatically traces agent invocations, action groups,
knowledge base queries, and agent orchestration using AWS Bedrock Agents.

AWS Bedrock Agents is a managed service that helps you build and deploy
generative AI applications with agents that can reason, take actions, and
access knowledge bases.

Requirements:
    pip install boto3 botocore
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class BedrockAgentsInstrumentor(BaseInstrumentor):
    """Instrumentor for AWS Bedrock Agents"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._bedrock_agents_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Bedrock Agents runtime is available."""
        try:
            import boto3

            # Check if bedrock-agent-runtime service is available
            session = boto3.session.Session()
            if "bedrock-agent-runtime" in session.get_available_services():
                self._bedrock_agents_available = True
                logger.debug(
                    "AWS Bedrock Agents runtime detected and available for instrumentation"
                )
            else:
                logger.debug("AWS Bedrock Agents runtime service not available")
                self._bedrock_agents_available = False
        except ImportError:
            logger.debug("boto3 not installed, Bedrock Agents instrumentation will be skipped")
            self._bedrock_agents_available = False

    def instrument(self, config: OTelConfig):
        """Instrument AWS Bedrock Agents if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._bedrock_agents_available:
            logger.debug("Skipping Bedrock Agents instrumentation - library not available")
            return

        self.config = config

        try:
            import wrapt
            from botocore.client import BaseClient

            # Store original make_request method
            original_make_request = BaseClient._make_request

            # Wrap the _make_request method
            def wrapped_make_request(self, operation_model, request_dict, *args, **kwargs):
                # Only instrument bedrock-agent-runtime operations
                if (
                    hasattr(self, "_service_model")
                    and self._service_model.service_name == "bedrock-agent-runtime"
                ):
                    operation_name = operation_model.name

                    # Instrument invoke_agent operation
                    if operation_name == "InvokeAgent":
                        return self._instrumentor.create_span_wrapper(
                            span_name="bedrock.agents.invoke_agent",
                            extract_attributes=lambda inst, args, kwargs: self._instrumentor._extract_invoke_agent_attributes(
                                request_dict
                            ),
                        )(original_make_request)(
                            self, operation_model, request_dict, *args, **kwargs
                        )

                    # Instrument retrieve operation
                    elif operation_name == "Retrieve":
                        return self._instrumentor.create_span_wrapper(
                            span_name="bedrock.agents.retrieve",
                            extract_attributes=lambda inst, args, kwargs: self._instrumentor._extract_retrieve_attributes(
                                request_dict
                            ),
                        )(original_make_request)(
                            self, operation_model, request_dict, *args, **kwargs
                        )

                    # Instrument retrieve_and_generate operation
                    elif operation_name == "RetrieveAndGenerate":
                        return self._instrumentor.create_span_wrapper(
                            span_name="bedrock.agents.retrieve_and_generate",
                            extract_attributes=lambda inst, args, kwargs: self._instrumentor._extract_retrieve_and_generate_attributes(
                                request_dict
                            ),
                        )(original_make_request)(
                            self, operation_model, request_dict, *args, **kwargs
                        )

                # Call original for non-bedrock-agent-runtime operations
                return original_make_request(self, operation_model, request_dict, *args, **kwargs)

            # Attach instrumentor reference for access in wrapper
            BaseClient._instrumentor = self

            # Replace the method
            BaseClient._make_request = wrapped_make_request

            self._instrumented = True
            logger.info("AWS Bedrock Agents instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument AWS Bedrock Agents: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_invoke_agent_attributes(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from InvokeAgent request.

        Args:
            request_dict: The request dictionary from boto3.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "bedrock_agents"
        attrs["gen_ai.operation.name"] = "invoke_agent"

        try:
            # Extract from body (already serialized JSON)
            body = request_dict.get("body", {})
            if isinstance(body, str):
                body = json.loads(body)

            # Extract agent identifiers
            if "agentId" in body:
                attrs["bedrock.agent.id"] = body["agentId"]
            if "agentAliasId" in body:
                attrs["bedrock.agent.alias_id"] = body["agentAliasId"]

            # Extract session information
            if "sessionId" in body:
                attrs["bedrock.agent.session_id"] = body["sessionId"]

            # Extract input text
            if "inputText" in body:
                attrs["bedrock.agent.input_text"] = str(body["inputText"])[:500]

            # Extract session state if present
            if "sessionState" in body:
                session_state = body["sessionState"]
                if isinstance(session_state, dict):
                    # Extract prompt session attributes
                    if "promptSessionAttributes" in session_state:
                        attrs["bedrock.agent.prompt_attributes"] = str(
                            session_state["promptSessionAttributes"]
                        )[:200]

                    # Extract session attributes
                    if "sessionAttributes" in session_state:
                        attrs["bedrock.agent.session_attributes"] = str(
                            session_state["sessionAttributes"]
                        )[:200]

            # Extract enable trace flag
            if "enableTrace" in body:
                attrs["bedrock.agent.enable_trace"] = body["enableTrace"]

        except Exception as e:
            logger.debug("Failed to extract invoke_agent attributes: %s", e)

        return attrs

    def _extract_retrieve_attributes(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from Retrieve request.

        Args:
            request_dict: The request dictionary from boto3.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "bedrock_agents"
        attrs["gen_ai.operation.name"] = "retrieve"

        try:
            body = request_dict.get("body", {})
            if isinstance(body, str):
                body = json.loads(body)

            # Extract knowledge base ID
            if "knowledgeBaseId" in body:
                attrs["bedrock.knowledge_base.id"] = body["knowledgeBaseId"]

            # Extract retrieval query
            if "retrievalQuery" in body:
                query = body["retrievalQuery"]
                if isinstance(query, dict) and "text" in query:
                    attrs["bedrock.retrieval.query"] = str(query["text"])[:500]

            # Extract retrieval configuration
            if "retrievalConfiguration" in body:
                config = body["retrievalConfiguration"]
                if isinstance(config, dict):
                    # Extract vector search config
                    if "vectorSearchConfiguration" in config:
                        vector_config = config["vectorSearchConfiguration"]
                        if "numberOfResults" in vector_config:
                            attrs["bedrock.retrieval.number_of_results"] = vector_config[
                                "numberOfResults"
                            ]
                        if "overrideSearchType" in vector_config:
                            attrs["bedrock.retrieval.search_type"] = vector_config[
                                "overrideSearchType"
                            ]

        except Exception as e:
            logger.debug("Failed to extract retrieve attributes: %s", e)

        return attrs

    def _extract_retrieve_and_generate_attributes(
        self, request_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract attributes from RetrieveAndGenerate request.

        Args:
            request_dict: The request dictionary from boto3.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "bedrock_agents"
        attrs["gen_ai.operation.name"] = "retrieve_and_generate"

        try:
            body = request_dict.get("body", {})
            if isinstance(body, str):
                body = json.loads(body)

            # Extract input
            if "input" in body:
                input_data = body["input"]
                if isinstance(input_data, dict) and "text" in input_data:
                    attrs["bedrock.rag.input_text"] = str(input_data["text"])[:500]

            # Extract session ID
            if "sessionId" in body:
                attrs["bedrock.rag.session_id"] = body["sessionId"]

            # Extract retrieve and generate configuration
            if "retrieveAndGenerateConfiguration" in body:
                config = body["retrieveAndGenerateConfiguration"]
                if isinstance(config, dict):
                    # Extract type
                    if "type" in config:
                        attrs["bedrock.rag.type"] = config["type"]

                    # Extract knowledge base configuration
                    if "knowledgeBaseConfiguration" in config:
                        kb_config = config["knowledgeBaseConfiguration"]
                        if "knowledgeBaseId" in kb_config:
                            attrs["bedrock.knowledge_base.id"] = kb_config["knowledgeBaseId"]
                        if "modelArn" in kb_config:
                            attrs["gen_ai.request.model"] = kb_config["modelArn"]

                        # Extract generation configuration
                        if "generationConfiguration" in kb_config:
                            gen_config = kb_config["generationConfiguration"]
                            if "inferenceConfig" in gen_config:
                                inference = gen_config["inferenceConfig"]
                                if "temperature" in inference:
                                    attrs["gen_ai.request.temperature"] = inference["temperature"]
                                if "maxTokens" in inference:
                                    attrs["gen_ai.request.max_tokens"] = inference["maxTokens"]
                                if "topP" in inference:
                                    attrs["gen_ai.request.top_p"] = inference["topP"]

        except Exception as e:
            logger.debug("Failed to extract retrieve_and_generate attributes: %s", e)

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from agent response.

        Note: Bedrock Agents may not expose token usage directly.
        Token usage is captured by underlying Bedrock model calls.

        Args:
            result: The agent invocation result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # Bedrock Agents responses don't typically include token usage
        # Token usage is tracked by underlying Bedrock model instrumentor
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from agent result.

        Args:
            result: The agent invocation result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # For invoke_agent responses
            if isinstance(result, dict):
                # Extract session ID
                if "sessionId" in result:
                    attrs["bedrock.agent.response.session_id"] = result["sessionId"]

                # Extract content type
                if "contentType" in result:
                    attrs["bedrock.agent.response.content_type"] = result["contentType"]

                # For streaming responses, we get an event stream
                # The actual response parsing would happen in the application code
                # We can log that a response was received
                if "completion" in result:
                    attrs["bedrock.agent.response.has_completion"] = True

                # For retrieve responses
                if "retrievalResults" in result:
                    retrieval_results = result["retrievalResults"]
                    if isinstance(retrieval_results, list):
                        attrs["bedrock.retrieval.results_count"] = len(retrieval_results)

                # For retrieve_and_generate responses
                if "output" in result:
                    output = result["output"]
                    if isinstance(output, dict):
                        if "text" in output:
                            attrs["bedrock.rag.output_text"] = str(output["text"])[:500]

                # Extract citations if present
                if "citations" in result:
                    citations = result["citations"]
                    if isinstance(citations, list):
                        attrs["bedrock.rag.citations_count"] = len(citations)

        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from agent result.

        Args:
            result: The agent invocation result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            # For streaming responses, finish reason might be in the final event
            # For non-streaming, completion indicates success
            if isinstance(result, dict):
                # Check for stop reason in streaming events
                if "stopReason" in result:
                    return result["stopReason"]

                # If we have output/completion, assume successful completion
                if "output" in result or "completion" in result:
                    return "completed"

        except Exception as e:
            logger.debug("Failed to extract finish reason: %s", e)

        return None
