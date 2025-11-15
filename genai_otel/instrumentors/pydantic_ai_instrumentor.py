"""OpenTelemetry instrumentor for Pydantic AI framework.

This instrumentor automatically traces agent execution, tool calls, and model
interactions using the Pydantic AI type-safe agent framework.

Pydantic AI is a new framework (Dec 2024) by the Pydantic team that provides
type-safe agent development with full Pydantic validation and multi-provider support.

Requirements:
    pip install pydantic-ai
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class PydanticAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Pydantic AI agent framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._pydantic_ai_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Pydantic AI library is available."""
        try:
            import pydantic_ai

            self._pydantic_ai_available = True
            logger.debug("Pydantic AI library detected and available for instrumentation")
        except ImportError:
            logger.debug("Pydantic AI library not installed, instrumentation will be skipped")
            self._pydantic_ai_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Pydantic AI framework if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._pydantic_ai_available:
            logger.debug("Skipping Pydantic AI instrumentation - library not available")
            return

        self.config = config

        try:
            import wrapt
            from pydantic_ai import Agent

            # Instrument Agent.run (synchronous execution)
            if hasattr(Agent, "run"):
                original_run = Agent.run
                Agent.run = wrapt.FunctionWrapper(original_run, self._wrap_agent_run)

            # Instrument Agent.run_sync (explicit synchronous execution)
            if hasattr(Agent, "run_sync"):
                original_run_sync = Agent.run_sync
                Agent.run_sync = wrapt.FunctionWrapper(original_run_sync, self._wrap_agent_run_sync)

            # Instrument Agent.run_stream (streaming execution)
            if hasattr(Agent, "run_stream"):
                original_run_stream = Agent.run_stream
                Agent.run_stream = wrapt.FunctionWrapper(
                    original_run_stream, self._wrap_agent_run_stream
                )

            self._instrumented = True
            logger.info("Pydantic AI instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Pydantic AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_agent_run(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run method with span.

        Args:
            wrapped: The original method.
            instance: The Agent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="pydantic_ai.agent.run",
            extract_attributes=self._extract_agent_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_agent_run_sync(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run_sync method with span.

        Args:
            wrapped: The original method.
            instance: The Agent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="pydantic_ai.agent.run_sync",
            extract_attributes=self._extract_agent_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_agent_run_stream(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run_stream method with span.

        Args:
            wrapped: The original method.
            instance: The Agent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="pydantic_ai.agent.run_stream",
            extract_attributes=self._extract_agent_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _extract_agent_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Agent.run call.

        Args:
            instance: The Agent instance.
            args: Positional arguments (user prompt, etc.).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "pydantic_ai"
        attrs["gen_ai.operation.name"] = "agent.run"

        # Extract agent name if available
        if hasattr(instance, "name") and instance.name:
            attrs["pydantic_ai.agent.name"] = instance.name

        # Extract model information
        if hasattr(instance, "model") and instance.model:
            model = instance.model
            # Get model name/provider
            if hasattr(model, "name"):
                attrs["gen_ai.request.model"] = model.name
                attrs["pydantic_ai.model.name"] = model.name
            elif hasattr(model, "model_name"):
                attrs["gen_ai.request.model"] = model.model_name
                attrs["pydantic_ai.model.name"] = model.model_name

            # Extract provider if available
            if hasattr(model, "__class__"):
                provider = model.__class__.__name__
                attrs["pydantic_ai.model.provider"] = provider

        # Extract system prompts
        if hasattr(instance, "_system_prompts") and instance._system_prompts:
            try:
                # System prompts can be list of strings or functions
                prompts = []
                for prompt in instance._system_prompts:
                    if callable(prompt):
                        prompts.append("<function>")
                    else:
                        prompts.append(str(prompt)[:200])  # Truncate
                attrs["pydantic_ai.system_prompts"] = prompts[:5]  # Limit to 5
            except Exception as e:
                logger.debug("Failed to extract system prompts: %s", e)

        # Extract tools/functions if available
        if hasattr(instance, "_function_tools") and instance._function_tools:
            try:
                tool_names = []
                for tool_name in instance._function_tools.keys():
                    tool_names.append(tool_name)
                attrs["pydantic_ai.tools"] = tool_names[:10]  # Limit to 10
                attrs["pydantic_ai.tools.count"] = len(tool_names)
            except Exception as e:
                logger.debug("Failed to extract tools: %s", e)

        # Extract result type if specified
        if hasattr(instance, "_result_type"):
            try:
                result_type = instance._result_type
                if result_type:
                    attrs["pydantic_ai.result_type"] = str(result_type)
            except Exception as e:
                logger.debug("Failed to extract result type: %s", e)

        # Extract user prompt (first positional argument)
        user_prompt = None
        if len(args) > 0:
            user_prompt = args[0]
        elif "user_prompt" in kwargs:
            user_prompt = kwargs["user_prompt"]
        elif "prompt" in kwargs:
            user_prompt = kwargs["prompt"]

        if user_prompt:
            if isinstance(user_prompt, str):
                attrs["pydantic_ai.user_prompt"] = user_prompt[:500]  # Truncate
            else:
                attrs["pydantic_ai.user_prompt"] = str(user_prompt)[:500]

        # Extract message history if provided
        if "message_history" in kwargs:
            try:
                history = kwargs["message_history"]
                if history and hasattr(history, "__len__"):
                    attrs["pydantic_ai.message_history.count"] = len(history)
            except Exception as e:
                logger.debug("Failed to extract message history: %s", e)

        # Extract model settings if provided
        if "model_settings" in kwargs:
            try:
                settings = kwargs["model_settings"]
                if isinstance(settings, dict):
                    if "temperature" in settings:
                        attrs["gen_ai.request.temperature"] = settings["temperature"]
                    if "max_tokens" in settings:
                        attrs["gen_ai.request.max_tokens"] = settings["max_tokens"]
                    if "top_p" in settings:
                        attrs["gen_ai.request.top_p"] = settings["top_p"]
            except Exception as e:
                logger.debug("Failed to extract model settings: %s", e)

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from agent result.

        Args:
            result: The agent execution result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # Try to extract usage from result
        try:
            # Pydantic AI results typically have usage information
            if hasattr(result, "usage") and result.usage:
                usage = result.usage
                return {
                    "prompt_tokens": getattr(usage, "request_tokens", 0),
                    "completion_tokens": getattr(usage, "response_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

            # Try alternative attribute names
            if hasattr(result, "_usage"):
                usage = result._usage
                return {
                    "prompt_tokens": getattr(usage, "request_tokens", 0),
                    "completion_tokens": getattr(usage, "response_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

        except Exception as e:
            logger.debug("Failed to extract usage: %s", e)

        # Token usage is also captured by underlying provider instrumentors
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from agent result.

        Args:
            result: The agent execution result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Extract result data
            if hasattr(result, "data"):
                try:
                    data = result.data
                    # If data is a Pydantic model, convert to dict
                    if hasattr(data, "model_dump"):
                        data_dict = data.model_dump()
                        attrs["pydantic_ai.result.data"] = str(data_dict)[:500]
                    else:
                        attrs["pydantic_ai.result.data"] = str(data)[:500]
                except Exception as e:
                    logger.debug("Failed to extract result data: %s", e)

            # Extract messages from result
            if hasattr(result, "messages") and result.messages:
                try:
                    attrs["pydantic_ai.result.messages.count"] = len(result.messages)

                    # Extract last message content
                    if result.messages:
                        last_msg = result.messages[-1]
                        if hasattr(last_msg, "content"):
                            attrs["pydantic_ai.result.last_message"] = str(last_msg.content)[:500]
                        if hasattr(last_msg, "role"):
                            attrs["pydantic_ai.result.last_role"] = last_msg.role
                except Exception as e:
                    logger.debug("Failed to extract messages: %s", e)

            # Extract timestamp if available
            if hasattr(result, "timestamp"):
                try:
                    attrs["pydantic_ai.result.timestamp"] = str(result.timestamp)
                except Exception as e:
                    logger.debug("Failed to extract timestamp: %s", e)

            # Extract cost if available
            if hasattr(result, "cost") and result.cost:
                try:
                    attrs["pydantic_ai.result.cost"] = float(result.cost)
                except Exception as e:
                    logger.debug("Failed to extract cost: %s", e)

            # Extract model name from result if available
            if hasattr(result, "model"):
                attrs["gen_ai.response.model"] = str(result.model)

        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from agent result.

        Args:
            result: The agent execution result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            # Check if result has finish_reason
            if hasattr(result, "finish_reason"):
                return str(result.finish_reason)

            # Check messages for finish reason
            if hasattr(result, "messages") and result.messages:
                last_msg = result.messages[-1]
                if hasattr(last_msg, "finish_reason"):
                    return str(last_msg.finish_reason)

            # If we have data, assume completion
            if hasattr(result, "data"):
                return "completed"

        except Exception as e:
            logger.debug("Failed to extract finish reason: %s", e)

        return None
