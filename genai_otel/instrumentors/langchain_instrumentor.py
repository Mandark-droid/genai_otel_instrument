"""OpenTelemetry instrumentor for the LangChain framework.

This instrumentor automatically traces various components within LangChain,
including chains, agents, and chat models, capturing relevant attributes for observability.
"""

import asyncio
import functools
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class LangChainInstrumentor(BaseInstrumentor):
    """Instrumentor for LangChain"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._langchain_available = False
        self._langchain_core_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if langchain library is available."""
        try:
            import langchain

            self._langchain_available = True
            logger.debug("langchain library detected and available for instrumentation")
        except ImportError:
            logger.debug("langchain library not installed, instrumentation will be skipped")
            self._langchain_available = False

        # Check for langchain_core (required for chat model instrumentation)
        try:
            import langchain_core

            self._langchain_core_available = True
            logger.debug("langchain_core library detected and available for instrumentation")
        except ImportError:
            logger.debug(
                "langchain_core library not installed, chat model instrumentation will be skipped"
            )
            self._langchain_core_available = False

    def instrument(self, config: OTelConfig):
        """Instrument langchain components if available."""
        if not self._langchain_available:
            logger.debug("Skipping instrumentation - library not available")
            return

        self.config = config

        # Instrument chains and agents
        self._instrument_chains_and_agents()

        # Instrument chat models if langchain_core is available
        if self._langchain_core_available:
            self._instrument_chat_models()

    def _instrument_chains_and_agents(self):
        """Instrument LangChain chains and agents."""
        try:
            from langchain.agents.agent import AgentExecutor
            from langchain.chains.base import Chain

            # Instrument Chains
            original_call = Chain.__call__

            def wrapped_call(instance, *args, **kwargs):
                chain_type = instance.__class__.__name__
                with self.tracer.start_as_current_span(f"langchain.chain.{chain_type}") as span:
                    span.set_attribute("langchain.chain.type", chain_type)
                    result = original_call(instance, *args, **kwargs)
                    return result

            Chain.__call__ = wrapped_call

            # Instrument Agents
            original_agent_call = AgentExecutor.__call__

            def wrapped_agent_call(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("langchain.agent.execute") as span:
                    agent_name = getattr(instance, "agent", {}).get("name", "unknown")
                    span.set_attribute("langchain.agent.name", agent_name)
                    result = original_agent_call(instance, *args, **kwargs)
                    return result

            AgentExecutor.__call__ = wrapped_agent_call
            logger.debug("Chains and agents instrumentation completed")

        except ImportError:
            logger.debug("Could not import chains or agents, skipping instrumentation")

    def _instrument_chat_models(self):
        """Instrument LangChain chat models."""
        try:
            from langchain_core.language_models.chat_models import BaseChatModel

            # Instrument invoke method
            original_invoke = BaseChatModel.invoke

            @functools.wraps(original_invoke)
            def wrapped_invoke(instance, *args, **kwargs):
                import time

                model_name = self._get_model_name(instance)
                with self.tracer.start_as_current_span("langchain.chat_model.invoke") as span:
                    start_time = time.time()
                    self._set_chat_attributes(span, instance, args, kwargs, model_name)

                    result = original_invoke(instance, *args, **kwargs)

                    # Use standard metrics recording from BaseInstrumentor
                    # This will extract usage, calculate costs, and record metrics
                    self._record_result_metrics(span, result, start_time, kwargs)

                    return result

            BaseChatModel.invoke = wrapped_invoke

            # Instrument ainvoke (async invoke) method
            original_ainvoke = BaseChatModel.ainvoke

            @functools.wraps(original_ainvoke)
            async def wrapped_ainvoke(instance, *args, **kwargs):
                import time

                model_name = self._get_model_name(instance)
                with self.tracer.start_as_current_span("langchain.chat_model.ainvoke") as span:
                    start_time = time.time()
                    self._set_chat_attributes(span, instance, args, kwargs, model_name)

                    result = await original_ainvoke(instance, *args, **kwargs)

                    # Use standard metrics recording from BaseInstrumentor
                    # This will extract usage, calculate costs, and record metrics
                    self._record_result_metrics(span, result, start_time, kwargs)

                    return result

            BaseChatModel.ainvoke = wrapped_ainvoke

            # Instrument batch method
            original_batch = BaseChatModel.batch

            @functools.wraps(original_batch)
            def wrapped_batch(instance, *args, **kwargs):
                import time

                model_name = self._get_model_name(instance)
                with self.tracer.start_as_current_span("langchain.chat_model.batch") as span:
                    start_time = time.time()

                    # Set standard GenAI attributes
                    provider = self._extract_provider(instance)
                    if provider:
                        span.set_attribute("gen_ai.system", provider)
                    else:
                        span.set_attribute("gen_ai.system", "langchain")

                    span.set_attribute("gen_ai.request.model", model_name)
                    span.set_attribute("gen_ai.operation.name", "batch")

                    # Also set LangChain-specific attributes
                    span.set_attribute("langchain.chat_model.name", model_name)
                    span.set_attribute("langchain.chat_model.operation", "batch")

                    # Get batch size
                    if args and len(args) > 0:
                        batch_size = len(args[0]) if hasattr(args[0], "__len__") else 1
                        span.set_attribute("langchain.chat_model.batch_size", batch_size)

                    result = original_batch(instance, *args, **kwargs)

                    # Record metrics (though batch results may not have usage info)
                    self._record_result_metrics(span, result, start_time, kwargs)

                    return result

            BaseChatModel.batch = wrapped_batch

            # Instrument abatch (async batch) method
            original_abatch = BaseChatModel.abatch

            @functools.wraps(original_abatch)
            async def wrapped_abatch(instance, *args, **kwargs):
                import time

                model_name = self._get_model_name(instance)
                with self.tracer.start_as_current_span("langchain.chat_model.abatch") as span:
                    start_time = time.time()

                    # Set standard GenAI attributes
                    provider = self._extract_provider(instance)
                    if provider:
                        span.set_attribute("gen_ai.system", provider)
                    else:
                        span.set_attribute("gen_ai.system", "langchain")

                    span.set_attribute("gen_ai.request.model", model_name)
                    span.set_attribute("gen_ai.operation.name", "batch")

                    # Also set LangChain-specific attributes
                    span.set_attribute("langchain.chat_model.name", model_name)
                    span.set_attribute("langchain.chat_model.operation", "abatch")

                    # Get batch size
                    if args and len(args) > 0:
                        batch_size = len(args[0]) if hasattr(args[0], "__len__") else 1
                        span.set_attribute("langchain.chat_model.batch_size", batch_size)

                    result = await original_abatch(instance, *args, **kwargs)

                    # Record metrics (though batch results may not have usage info)
                    self._record_result_metrics(span, result, start_time, kwargs)

                    return result

            BaseChatModel.abatch = wrapped_abatch

            logger.info("LangChain chat models instrumentation completed")

        except ImportError as e:
            logger.debug(f"Could not import langchain_core chat models: {e}")
        except Exception as e:
            logger.error(f"Error instrumenting chat models: {e}", exc_info=True)

    def _get_model_name(self, instance: Any) -> str:
        """Extract model name from chat model instance."""
        # Try common attribute names for model name
        for attr in ["model_name", "model", "model_id"]:
            if hasattr(instance, attr):
                value = getattr(instance, attr)
                if value:
                    return str(value)

        # Fallback to class name
        return instance.__class__.__name__

    def _set_chat_attributes(self, span, instance: Any, args: tuple, kwargs: dict, model_name: str):
        """Set span attributes for chat model invocations."""
        # Set standard GenAI semantic convention attributes
        provider = self._extract_provider(instance)
        if provider:
            span.set_attribute("gen_ai.system", provider)
        else:
            span.set_attribute("gen_ai.system", "langchain")

        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("gen_ai.operation.name", "chat")

        # Also set LangChain-specific attributes for backward compatibility
        span.set_attribute("langchain.chat_model.name", model_name)
        span.set_attribute("langchain.chat_model.operation", "invoke")
        if provider:
            span.set_attribute("langchain.chat_model.provider", provider)

        # Count messages if available
        if args and len(args) > 0:
            messages = args[0]
            if hasattr(messages, "__len__"):
                message_count = len(messages)
                span.set_attribute("gen_ai.request.message_count", message_count)
                span.set_attribute("langchain.chat_model.message_count", message_count)

    def _extract_provider(self, instance: Any) -> Optional[str]:
        """Extract provider name from chat model instance."""
        class_name = instance.__class__.__name__.lower()
        module_name = instance.__class__.__module__.lower()

        # Map class names to providers
        provider_mapping = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google": "google",
            "ollama": "ollama",
            "bedrock": "bedrock",
            "cohere": "cohere",
            "groq": "groq",
            "mistral": "mistral",
        }

        # Check class name
        for key, value in provider_mapping.items():
            if key in class_name:
                return value

        # Check module name
        for key, value in provider_mapping.items():
            if key in module_name:
                return value

        return None

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract usage information for BaseInstrumentor compatibility."""
        try:
            usage_data = None

            if hasattr(result, "usage_metadata") and result.usage_metadata:
                usage_data = result.usage_metadata
            elif hasattr(result, "response_metadata") and result.response_metadata:
                metadata = result.response_metadata
                if "token_usage" in metadata:
                    usage_data = metadata["token_usage"]
                elif "usage" in metadata:
                    usage_data = metadata["usage"]

            if usage_data:
                if isinstance(usage_data, dict):
                    prompt_tokens = usage_data.get("input_tokens") or usage_data.get(
                        "prompt_tokens"
                    )
                    completion_tokens = usage_data.get("output_tokens") or usage_data.get(
                        "completion_tokens"
                    )
                else:
                    prompt_tokens = getattr(usage_data, "input_tokens", None) or getattr(
                        usage_data, "prompt_tokens", None
                    )
                    completion_tokens = getattr(usage_data, "output_tokens", None) or getattr(
                        usage_data, "completion_tokens", None
                    )

                if prompt_tokens or completion_tokens:
                    return {
                        "prompt_tokens": int(prompt_tokens) if prompt_tokens else 0,
                        "completion_tokens": int(completion_tokens) if completion_tokens else 0,
                        "total_tokens": int(prompt_tokens or 0) + int(completion_tokens or 0),
                    }
        except Exception:
            pass

        return None
