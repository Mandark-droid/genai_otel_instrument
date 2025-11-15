"""OpenTelemetry instrumentor for the LangGraph framework.

This instrumentor automatically traces graph execution, nodes, edges, state updates,
and checkpoints using the LangGraph stateful workflow framework.
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class LangGraphInstrumentor(BaseInstrumentor):
    """Instrumentor for LangGraph stateful workflow framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._langgraph_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if LangGraph library is available."""
        try:
            import langgraph

            self._langgraph_available = True
            logger.debug("LangGraph library detected and available for instrumentation")
        except ImportError:
            logger.debug("LangGraph library not installed, instrumentation will be skipped")
            self._langgraph_available = False

    def instrument(self, config: OTelConfig):
        """Instrument LangGraph framework if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._langgraph_available:
            logger.debug("Skipping LangGraph instrumentation - library not available")
            return

        self.config = config

        try:
            import wrapt
            from langgraph.graph import StateGraph

            # Instrument StateGraph.compile() to wrap the resulting CompiledGraph
            if hasattr(StateGraph, "compile"):
                original_compile = StateGraph.compile

                def wrapped_compile(wrapped, instance, args, kwargs):
                    # Get the compiled graph
                    compiled_graph = wrapped(*args, **kwargs)

                    # Instrument the compiled graph's execution methods
                    self._instrument_compiled_graph(compiled_graph, instance)

                    return compiled_graph

                StateGraph.compile = wrapt.FunctionWrapper(original_compile, wrapped_compile)

                self._instrumented = True
                logger.info("LangGraph instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument LangGraph: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_compiled_graph(self, compiled_graph, state_graph):
        """Instrument a compiled graph's execution methods.

        Args:
            compiled_graph: The compiled graph object.
            state_graph: The original StateGraph instance.
        """
        import wrapt

        # Instrument invoke method (synchronous execution)
        if hasattr(compiled_graph, "invoke"):
            original_invoke = compiled_graph.invoke
            compiled_graph.invoke = wrapt.FunctionWrapper(
                original_invoke,
                lambda w, i, a, kw: self._wrap_graph_invoke(
                    w, i, a, kw, state_graph, is_async=False
                ),
            )

        # Instrument stream method (synchronous streaming)
        if hasattr(compiled_graph, "stream"):
            original_stream = compiled_graph.stream
            compiled_graph.stream = wrapt.FunctionWrapper(
                original_stream,
                lambda w, i, a, kw: self._wrap_graph_stream(
                    w, i, a, kw, state_graph, is_async=False
                ),
            )

        # Instrument ainvoke method (asynchronous execution)
        if hasattr(compiled_graph, "ainvoke"):
            original_ainvoke = compiled_graph.ainvoke
            compiled_graph.ainvoke = wrapt.FunctionWrapper(
                original_ainvoke,
                lambda w, i, a, kw: self._wrap_graph_invoke(
                    w, i, a, kw, state_graph, is_async=True
                ),
            )

        # Instrument astream method (asynchronous streaming)
        if hasattr(compiled_graph, "astream"):
            original_astream = compiled_graph.astream
            compiled_graph.astream = wrapt.FunctionWrapper(
                original_astream,
                lambda w, i, a, kw: self._wrap_graph_stream(
                    w, i, a, kw, state_graph, is_async=True
                ),
            )

    def _wrap_graph_invoke(self, wrapped, instance, args, kwargs, state_graph, is_async):
        """Wrap graph invoke/ainvoke method with span.

        Args:
            wrapped: The original method.
            instance: The compiled graph instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
            state_graph: The original StateGraph instance.
            is_async: Whether this is async invocation.
        """
        operation_name = "graph.ainvoke" if is_async else "graph.invoke"
        return self.create_span_wrapper(
            span_name=f"langgraph.{operation_name}",
            extract_attributes=lambda i, a, kw: self._extract_graph_attributes(
                i, a, kw, state_graph
            ),
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_graph_stream(self, wrapped, instance, args, kwargs, state_graph, is_async):
        """Wrap graph stream/astream method with span.

        Args:
            wrapped: The original method.
            instance: The compiled graph instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
            state_graph: The original StateGraph instance.
            is_async: Whether this is async streaming.
        """
        operation_name = "graph.astream" if is_async else "graph.stream"
        return self.create_span_wrapper(
            span_name=f"langgraph.{operation_name}",
            extract_attributes=lambda i, a, kw: self._extract_graph_attributes(
                i, a, kw, state_graph
            ),
        )(wrapped)(instance, *args, **kwargs)

    def _extract_graph_attributes(
        self, instance: Any, args: Any, kwargs: Any, state_graph: Any
    ) -> Dict[str, Any]:
        """Extract attributes from graph execution.

        Args:
            instance: The compiled graph instance.
            args: Positional arguments (input state).
            kwargs: Keyword arguments.
            state_graph: The original StateGraph instance.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "langgraph"
        attrs["gen_ai.operation.name"] = "graph.execution"

        # Extract graph structure information from StateGraph
        try:
            # Get nodes from the graph
            if hasattr(state_graph, "nodes"):
                nodes = state_graph.nodes
                if nodes:
                    node_names = list(nodes.keys())
                    attrs["langgraph.node_count"] = len(node_names)
                    attrs["langgraph.nodes"] = node_names[:10]  # Limit to 10 nodes

            # Get edges from the graph
            if hasattr(state_graph, "edges"):
                edges = state_graph.edges
                if edges:
                    attrs["langgraph.edge_count"] = len(edges)

            # Get channels (state schema) if available
            if hasattr(state_graph, "channels"):
                channels = state_graph.channels
                if channels:
                    channel_names = list(channels.keys())
                    attrs["langgraph.channels"] = channel_names[:10]
                    attrs["langgraph.channel_count"] = len(channel_names)

        except Exception as e:
            logger.debug("Failed to extract graph structure: %s", e)

        # Extract input state (first positional argument)
        input_state = None
        if len(args) > 0:
            input_state = args[0]
        elif "input" in kwargs:
            input_state = kwargs["input"]

        if input_state:
            try:
                if isinstance(input_state, dict):
                    # Store input state keys
                    state_keys = list(input_state.keys())
                    attrs["langgraph.input.keys"] = state_keys[:10]

                    # Store truncated values for important keys
                    for key in ["messages", "query", "question", "input"][:3]:
                        if key in input_state:
                            value = str(input_state[key])[:200]
                            attrs[f"langgraph.input.{key}"] = value
                else:
                    # Non-dict input
                    attrs["langgraph.input"] = str(input_state)[:200]
            except Exception as e:
                logger.debug("Failed to extract input state: %s", e)

        # Extract config if provided
        config = kwargs.get("config")
        if config:
            try:
                # Extract configurable values
                if isinstance(config, dict):
                    if "configurable" in config:
                        configurable = config["configurable"]
                        if "thread_id" in configurable:
                            attrs["langgraph.thread_id"] = configurable["thread_id"]
                        if "checkpoint_id" in configurable:
                            attrs["langgraph.checkpoint_id"] = configurable["checkpoint_id"]

                    # Extract recursion limit
                    if "recursion_limit" in config:
                        attrs["langgraph.recursion_limit"] = config["recursion_limit"]

            except Exception as e:
                logger.debug("Failed to extract config: %s", e)

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from graph execution result.

        Note: LangGraph doesn't directly expose token usage in the result.
        Token usage is captured by underlying LLM provider instrumentors.

        Args:
            result: The graph execution result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # LangGraph doesn't directly expose usage
        # Token usage is captured by LLM provider instrumentors (OpenAI, Anthropic, etc.)
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from graph execution result.

        Args:
            result: The graph execution result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # LangGraph result is typically a dict (the final state)
            if isinstance(result, dict):
                # Store output state keys
                state_keys = list(result.keys())
                attrs["langgraph.output.keys"] = state_keys[:10]

                # Store truncated values for important keys
                for key in ["messages", "answer", "output", "result"][:3]:
                    if key in result:
                        value_str = str(result[key])[:500]
                        attrs[f"langgraph.output.{key}"] = value_str

                # Count the number of state updates/steps
                if "messages" in result and isinstance(result["messages"], list):
                    attrs["langgraph.message_count"] = len(result["messages"])

            # Try to extract metadata if available
            if hasattr(result, "__metadata__"):
                try:
                    metadata = result.__metadata__
                    if "step" in metadata:
                        attrs["langgraph.steps"] = metadata["step"]
                except Exception as e:
                    logger.debug("Failed to extract metadata: %s", e)

        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from graph execution result.

        Args:
            result: The graph execution result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        # LangGraph doesn't typically provide a finish_reason
        # We could infer completion status
        if result:
            return "completed"
        return None
