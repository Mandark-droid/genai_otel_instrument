"""OpenTelemetry instrumentor for Haystack NLP framework.

This instrumentor automatically traces pipeline execution, component operations,
and document processing using the Haystack framework.

Haystack is a modular NLP framework for building search and question-answering
systems with support for various LLMs, retrievers, and document stores.

Requirements:
    pip install haystack-ai
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class HaystackInstrumentor(BaseInstrumentor):
    """Instrumentor for Haystack NLP framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._haystack_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Haystack library is available."""
        try:
            import haystack

            self._haystack_available = True
            logger.debug("Haystack library detected and available for instrumentation")
        except ImportError:
            logger.debug("Haystack library not installed, instrumentation will be skipped")
            self._haystack_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Haystack framework if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._haystack_available:
            logger.debug("Skipping Haystack instrumentation - library not available")
            return

        self.config = config

        try:
            import wrapt

            # Try to import Pipeline (v2.x)
            try:
                from haystack import Pipeline

                # Instrument Pipeline.run (main execution method)
                if hasattr(Pipeline, "run"):
                    original_run = Pipeline.run
                    Pipeline.run = wrapt.FunctionWrapper(original_run, self._wrap_pipeline_run)

                # Instrument Pipeline.run_async (async execution)
                if hasattr(Pipeline, "run_async"):
                    original_run_async = Pipeline.run_async
                    Pipeline.run_async = wrapt.FunctionWrapper(
                        original_run_async, self._wrap_pipeline_run_async
                    )

            except ImportError:
                logger.debug("Haystack Pipeline not available (v2.x API)")

            # Try to instrument individual components
            try:
                # Instrument Generator components (LLM interaction)
                from haystack.components.generators import OpenAIChatGenerator, OpenAIGenerator

                if hasattr(OpenAIGenerator, "run"):
                    original_gen_run = OpenAIGenerator.run
                    OpenAIGenerator.run = wrapt.FunctionWrapper(
                        original_gen_run, self._wrap_generator_run
                    )

                if hasattr(OpenAIChatGenerator, "run"):
                    original_chat_run = OpenAIChatGenerator.run
                    OpenAIChatGenerator.run = wrapt.FunctionWrapper(
                        original_chat_run, self._wrap_chat_generator_run
                    )

            except ImportError:
                logger.debug("Haystack generator components not available")

            # Try to instrument Retriever components
            try:
                from haystack.components.retrievers import InMemoryBM25Retriever

                if hasattr(InMemoryBM25Retriever, "run"):
                    original_retriever_run = InMemoryBM25Retriever.run
                    InMemoryBM25Retriever.run = wrapt.FunctionWrapper(
                        original_retriever_run, self._wrap_retriever_run
                    )

            except ImportError:
                logger.debug("Haystack retriever components not available")

            self._instrumented = True
            logger.info("Haystack instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Haystack: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_pipeline_run(self, wrapped, instance, args, kwargs):
        """Wrap Pipeline.run method with span.

        Args:
            wrapped: The original method.
            instance: The Pipeline instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="haystack.pipeline.run",
            extract_attributes=self._extract_pipeline_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_pipeline_run_async(self, wrapped, instance, args, kwargs):
        """Wrap Pipeline.run_async method with span.

        Args:
            wrapped: The original method.
            instance: The Pipeline instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="haystack.pipeline.run_async",
            extract_attributes=self._extract_pipeline_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_generator_run(self, wrapped, instance, args, kwargs):
        """Wrap Generator.run method with span.

        Args:
            wrapped: The original method.
            instance: The Generator instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="haystack.generator.run",
            extract_attributes=self._extract_generator_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_chat_generator_run(self, wrapped, instance, args, kwargs):
        """Wrap ChatGenerator.run method with span.

        Args:
            wrapped: The original method.
            instance: The ChatGenerator instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="haystack.chat_generator.run",
            extract_attributes=self._extract_chat_generator_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_retriever_run(self, wrapped, instance, args, kwargs):
        """Wrap Retriever.run method with span.

        Args:
            wrapped: The original method.
            instance: The Retriever instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="haystack.retriever.run",
            extract_attributes=self._extract_retriever_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _extract_pipeline_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Pipeline.run call.

        Args:
            instance: The Pipeline instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "haystack"
        attrs["gen_ai.operation.name"] = "pipeline.run"

        # Extract pipeline metadata
        if hasattr(instance, "metadata") and instance.metadata:
            try:
                metadata = instance.metadata
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        attrs[f"haystack.pipeline.metadata.{key}"] = str(value)[:200]
            except Exception as e:
                logger.debug("Failed to extract pipeline metadata: %s", e)

        # Extract graph information
        if hasattr(instance, "graph"):
            try:
                graph = instance.graph
                # Get nodes (components)
                if hasattr(graph, "nodes"):
                    nodes = list(graph.nodes())
                    attrs["haystack.pipeline.components.count"] = len(nodes)
                    attrs["haystack.pipeline.components"] = [str(n) for n in nodes[:10]]

                # Get edges (connections)
                if hasattr(graph, "edges"):
                    edges = list(graph.edges())
                    attrs["haystack.pipeline.connections.count"] = len(edges)

            except Exception as e:
                logger.debug("Failed to extract pipeline graph: %s", e)

        # Extract input data
        if "data" in kwargs:
            try:
                data = kwargs["data"]
                if isinstance(data, dict):
                    attrs["haystack.pipeline.input.keys"] = list(data.keys())[:10]
                    # Extract query if present
                    if "query" in data:
                        attrs["haystack.pipeline.input.query"] = str(data["query"])[:500]
            except Exception as e:
                logger.debug("Failed to extract pipeline input: %s", e)

        # Extract include/exclude components
        if "include_outputs_from" in kwargs:
            try:
                include = kwargs["include_outputs_from"]
                if isinstance(include, (list, set)):
                    attrs["haystack.pipeline.include_outputs"] = list(include)[:10]
            except Exception as e:
                logger.debug("Failed to extract include_outputs: %s", e)

        return attrs

    def _extract_generator_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from Generator.run call.

        Args:
            instance: The Generator instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "haystack"
        attrs["gen_ai.operation.name"] = "generator.run"
        attrs["haystack.component.type"] = "generator"

        # Extract model information
        if hasattr(instance, "model") and instance.model:
            attrs["gen_ai.request.model"] = instance.model
            attrs["haystack.generator.model"] = instance.model

        # Extract generation parameters
        if hasattr(instance, "generation_kwargs"):
            try:
                gen_kwargs = instance.generation_kwargs
                if isinstance(gen_kwargs, dict):
                    if "max_tokens" in gen_kwargs:
                        attrs["gen_ai.request.max_tokens"] = gen_kwargs["max_tokens"]
                    if "temperature" in gen_kwargs:
                        attrs["gen_ai.request.temperature"] = gen_kwargs["temperature"]
                    if "top_p" in gen_kwargs:
                        attrs["gen_ai.request.top_p"] = gen_kwargs["top_p"]
            except Exception as e:
                logger.debug("Failed to extract generation kwargs: %s", e)

        # Extract prompt
        if "prompt" in kwargs:
            try:
                prompt = kwargs["prompt"]
                attrs["haystack.generator.prompt"] = str(prompt)[:500]
            except Exception as e:
                logger.debug("Failed to extract prompt: %s", e)

        return attrs

    def _extract_chat_generator_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from ChatGenerator.run call.

        Args:
            instance: The ChatGenerator instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "haystack"
        attrs["gen_ai.operation.name"] = "chat_generator.run"
        attrs["haystack.component.type"] = "chat_generator"

        # Extract model information
        if hasattr(instance, "model") and instance.model:
            attrs["gen_ai.request.model"] = instance.model
            attrs["haystack.chat_generator.model"] = instance.model

        # Extract generation parameters
        if hasattr(instance, "generation_kwargs"):
            try:
                gen_kwargs = instance.generation_kwargs
                if isinstance(gen_kwargs, dict):
                    if "max_tokens" in gen_kwargs:
                        attrs["gen_ai.request.max_tokens"] = gen_kwargs["max_tokens"]
                    if "temperature" in gen_kwargs:
                        attrs["gen_ai.request.temperature"] = gen_kwargs["temperature"]
            except Exception as e:
                logger.debug("Failed to extract generation kwargs: %s", e)

        # Extract messages
        if "messages" in kwargs:
            try:
                messages = kwargs["messages"]
                if isinstance(messages, list):
                    attrs["haystack.chat_generator.messages.count"] = len(messages)
                    # Extract last message
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "content"):
                            attrs["haystack.chat_generator.last_message"] = str(last_msg.content)[
                                :500
                            ]
                        if hasattr(last_msg, "role"):
                            attrs["haystack.chat_generator.last_role"] = last_msg.role
            except Exception as e:
                logger.debug("Failed to extract messages: %s", e)

        return attrs

    def _extract_retriever_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from Retriever.run call.

        Args:
            instance: The Retriever instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "haystack"
        attrs["gen_ai.operation.name"] = "retriever.run"
        attrs["haystack.component.type"] = "retriever"

        # Extract query
        if "query" in kwargs:
            try:
                query = kwargs["query"]
                attrs["haystack.retriever.query"] = str(query)[:500]
            except Exception as e:
                logger.debug("Failed to extract query: %s", e)

        # Extract top_k
        if "top_k" in kwargs:
            try:
                attrs["haystack.retriever.top_k"] = kwargs["top_k"]
            except Exception as e:
                logger.debug("Failed to extract top_k: %s", e)

        # Extract filters if present
        if "filters" in kwargs:
            try:
                filters = kwargs["filters"]
                if filters:
                    attrs["haystack.retriever.filters"] = str(filters)[:200]
            except Exception as e:
                logger.debug("Failed to extract filters: %s", e)

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from pipeline result.

        Args:
            result: The pipeline execution result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # Haystack results are typically dicts with component outputs
        # Token usage is captured by underlying LLM provider instrumentors
        # Try to extract if available in result metadata
        try:
            if isinstance(result, dict):
                # Check for generator outputs
                for key, value in result.items():
                    if isinstance(value, dict) and "meta" in value:
                        meta = value["meta"]
                        if isinstance(meta, list) and meta:
                            usage_info = meta[0].get("usage", {})
                            if usage_info:
                                return {
                                    "prompt_tokens": usage_info.get("prompt_tokens", 0),
                                    "completion_tokens": usage_info.get("completion_tokens", 0),
                                    "total_tokens": usage_info.get("total_tokens", 0),
                                }
        except Exception as e:
            logger.debug("Failed to extract usage: %s", e)

        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from pipeline result.

        Args:
            result: The pipeline execution result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            if isinstance(result, dict):
                # Extract output keys
                attrs["haystack.pipeline.output.keys"] = list(result.keys())[:10]

                # Try to extract replies from generator outputs
                for key, value in result.items():
                    if isinstance(value, dict):
                        # Check for replies (generator output)
                        if "replies" in value:
                            replies = value["replies"]
                            if isinstance(replies, list) and replies:
                                attrs[f"haystack.output.{key}.replies.count"] = len(replies)
                                attrs[f"haystack.output.{key}.first_reply"] = str(replies[0])[:500]

                        # Check for documents (retriever output)
                        if "documents" in value:
                            documents = value["documents"]
                            if isinstance(documents, list):
                                attrs[f"haystack.output.{key}.documents.count"] = len(documents)

        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from pipeline result.

        Args:
            result: The pipeline execution result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            if isinstance(result, dict):
                # Check generator outputs for finish reason
                for key, value in result.items():
                    if isinstance(value, dict) and "meta" in value:
                        meta = value["meta"]
                        if isinstance(meta, list) and meta:
                            finish_reason = meta[0].get("finish_reason")
                            if finish_reason:
                                return str(finish_reason)

            # If we have result, assume completion
            if result:
                return "completed"

        except Exception as e:
            logger.debug("Failed to extract finish reason: %s", e)

        return None
