"""OpenTelemetry instrumentor for the LangGraph framework.

Every runnable LangGraph graph is a ``langgraph.pregel.Pregel``: a compiled
``StateGraph``, a prebuilt agent, the copy ``with_config()`` returns, a functional-API
entrypoint. This instrumentor wraps the four run methods of that class (``invoke``,
``stream``, ``ainvoke``, ``astream``), so each graph run is one span, and whatever its
nodes call (LLM clients, tools, other graphs) is traced beneath it.

Each node execution also gets a span, by wrapping the function LangGraph runs every
node task through (``run_with_retry`` / ``arun_with_retry``), so the node's LLM and
tool calls nest beneath it. Node spans carry ``metadata.langgraph_node`` /
``langgraph_step`` and ``graph.node.id``, which Arize AX reads for its agent graph,
with the run span as the graph's root node. Edges and checkpoints get no spans of
their own; the graph's structure is recorded as attributes on the run span.
"""

import contextvars
import importlib
import importlib.util
import json
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Tuple

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..agent_budget import extract_budget_attributes
from ..config import OTelConfig
from ..server_metrics import get_server_metrics
from ..sessions import current_session_id, inbound_session_id
from .base import BaseInstrumentor, strip_content_attributes

logger = logging.getLogger(__name__)

# Pregel.invoke() is a loop over self.stream(), and ainvoke() over self.astream(). All
# four are wrapped, so one invoke used to produce two graph spans, the inner one with a
# session id of its own. A run method leaves a note naming its graph here; the stream
# call it then makes on that graph takes the note and adds no span.
_INNER_STREAM: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_langgraph_inner_stream", default=None
)

# Set on the Pregel class once its run methods are wrapped, so a second instrumentor
# instance cannot stack another layer of wrappers, and spans, on top.
_WRAPPED_FLAG = "_genai_otel_langgraph_wrapped"

# Where PregelRunner looks up the per-node run functions: ``_runner`` from 0.6 / 1.x,
# ``runner`` before. The runner imports them by name, so they are replaced there.
_NODE_RUNNER_MODULES = ("langgraph.pregel._runner", "langgraph.pregel.runner")


def _module_available(name: str) -> bool:
    """Whether ``name`` can be imported, decided without importing it."""
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _is_inner_stream(instance: Any) -> bool:
    """True, once, for the stream call that ``instance``'s own invoke/ainvoke makes."""
    note = _INNER_STREAM.get()
    if note is not None and note[0] is instance:
        note[0] = None
        return True
    return False


# The instrumentor the class-level wrappers trace with. The wrappers are installed
# once and stay on Pregel, while uninstrument() followed by instrument() builds a new
# instrumentor on a new tracer provider; bound to the first one, they would keep
# writing to a provider that has been shut down.
_ACTIVE: Optional["LangGraphInstrumentor"] = None


def _dispatch_to_active(method_name: str):
    """A wrapt wrapper that hands the call to the current instrumentor's method."""

    def wrapper(wrapped, instance, args, kwargs):
        active = _ACTIVE
        if active is None:
            return wrapped(*args, **kwargs)
        return getattr(active, method_name)(wrapped, instance, args, kwargs)

    return wrapper


def _forget_inner_stream(note) -> None:
    """Drop the one-shot note. Never raises into the caller.

    A pending run that is closed from another context (a cancelled task finalised
    at loop shutdown) cannot reset a token made in its own.
    """
    try:
        _INNER_STREAM.reset(note)
    except (ValueError, LookupError):
        pass


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

            # ``langgraph`` is a namespace package that langgraph-checkpoint,
            # langgraph-prebuilt and others install into as well, so the import above
            # succeeds with the graph runtime itself absent. Look for the runtime.
            self._langgraph_available = _module_available("langgraph.pregel")
        except ImportError:
            self._langgraph_available = False

        if self._langgraph_available:
            logger.debug("LangGraph library detected and available for instrumentation")
        else:
            logger.debug("LangGraph library not installed, instrumentation will be skipped")

    def instrument(self, config: OTelConfig):
        """Instrument LangGraph framework if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._langgraph_available:
            logger.debug("Skipping LangGraph instrumentation - library not available")
            return

        # Idempotency guard: repeated instrument() calls must not stack wrappers
        # on the shared Pregel run methods.
        if self._instrumented:
            logger.debug("LangGraph already instrumented, skipping repeat instrument()")
            return

        self.config = config

        try:
            import wrapt
            from langgraph.pregel import Pregel

            # Wrap the run methods on the class, not on each compiled graph. Wrappers
            # stored on the instance did not survive Pregel.copy(), which with_config()
            # uses and which rebuilds the graph from its instance attributes: the copy
            # (what langchain.agents.create_agent returns) was not traced at all, and on
            # LangGraph 0.2.x the copy itself raised TypeError.
            if vars(Pregel).get(_WRAPPED_FLAG):
                logger.debug("LangGraph run methods already wrapped, not wrapping again")
            else:
                wrappers = {
                    "invoke": "_wrap_graph_invoke",
                    "ainvoke": "_wrap_graph_ainvoke",
                    "stream": "_wrap_graph_stream",
                    "astream": "_wrap_graph_astream",
                }
                for name, method_name in wrappers.items():
                    original = getattr(Pregel, name, None)
                    if original is not None:
                        setattr(
                            Pregel,
                            name,
                            wrapt.FunctionWrapper(original, _dispatch_to_active(method_name)),
                        )
                setattr(Pregel, _WRAPPED_FLAG, True)

            self._wrap_node_runner(wrapt)

            global _ACTIVE
            _ACTIVE = self
            self._instrumented = True
            logger.info("LangGraph instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument LangGraph: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    @staticmethod
    def _wrap_node_runner(wrapt) -> None:
        """Wrap the functions PregelRunner runs each node task through.

        These are LangGraph internals, so a version that moved them loses node spans
        and nothing else.
        """
        for module_name in _NODE_RUNNER_MODULES:
            try:
                if not _module_available(module_name):
                    continue
                module = importlib.import_module(module_name)
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not import %s: %s", module_name, e)
                continue
            if not hasattr(module, "run_with_retry"):
                continue
            if vars(module).get(_WRAPPED_FLAG):
                return
            for name, method_name in (
                ("run_with_retry", "_run_node"),
                ("arun_with_retry", "_arun_node"),
            ):
                original = getattr(module, name, None)
                if original is not None:
                    setattr(
                        module,
                        name,
                        wrapt.FunctionWrapper(original, _dispatch_to_active(method_name)),
                    )
            setattr(module, _WRAPPED_FLAG, True)
            return
        logger.debug("LangGraph node runner not found, nodes get no spans of their own")

    def _graph_node_enabled(self) -> bool:
        """Whether the Arize agent graph keys are written (GENAI_ENABLE_AGENT_GRAPH_ATTRIBUTES)."""
        return bool(getattr(self.config, "enable_agent_graph_attributes", True))

    def _node_span(self, task: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Span name and start attributes for a node task, or None to leave it untraced."""
        name = getattr(task, "name", None)
        # __start__ writes the graph input into its channels; it is not a node.
        if not isinstance(name, str) or not name or name.startswith("__"):
            return None
        metadata = {}
        task_config = getattr(task, "config", None)
        if isinstance(task_config, dict) and isinstance(task_config.get("metadata"), dict):
            metadata = task_config["metadata"]
        step = metadata.get("langgraph_step")

        attrs: Dict[str, Any] = {"langgraph.node": name}
        if isinstance(step, int):
            attrs["langgraph.step"] = step
        triggers = getattr(task, "triggers", None)
        if isinstance(triggers, (list, tuple)) and triggers:
            attrs["langgraph.triggers"] = [str(t) for t in triggers[:10]]
        if self._graph_node_enabled():
            attrs["graph.node.id"] = name
            attrs["openinference.span.kind"] = "CHAIN"
            # OpenInference's metadata attribute, which Arize reads LangGraph
            # nodes from. Only the node keys: the run metadata can hold user data.
            node_metadata: Dict[str, Any] = {"langgraph_node": name}
            if isinstance(step, int):
                node_metadata["langgraph_step"] = step
            attrs["metadata"] = json.dumps(node_metadata)
        return f"langgraph.node.{name}", attrs

    def _prepare_node_span(self, args: Any, kwargs: Any):
        if not self._instrumented:
            return None
        try:
            task = args[0] if args else kwargs.get("task")
            return self._node_span(task)
        except Exception as e:  # noqa: BLE001 - tracing must never cost the node its run
            logger.debug("Failed to prepare LangGraph node span: %s", e)
            return None

    @staticmethod
    def _record_node_error(span, error: Exception) -> None:
        try:
            from langgraph.errors import GraphBubbleUp

            # interrupt() and Command(graph=PARENT) travel as exceptions; they are
            # control flow, not failures.
            if isinstance(error, GraphBubbleUp):
                span.set_attribute("langgraph.node.interrupted", True)
                return
        except ImportError:
            pass
        try:
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            span.record_exception(error)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to record LangGraph node error: %s", e)

    def _run_node(self, wrapped, instance, args, kwargs):
        """Run one node task inside a span of its own."""
        prepared = self._prepare_node_span(args, kwargs)
        if prepared is None:
            return wrapped(*args, **kwargs)
        span_name, attrs = prepared
        with self.tracer.start_as_current_span(
            span_name, attributes=attrs, record_exception=False, set_status_on_exception=False
        ) as span:
            try:
                return wrapped(*args, **kwargs)
            except Exception as e:
                self._record_node_error(span, e)
                raise

    def _arun_node(self, wrapped, instance, args, kwargs):
        """Async counterpart of ``_run_node``."""
        prepared = self._prepare_node_span(args, kwargs)
        if prepared is None:
            return wrapped(*args, **kwargs)
        span_name, attrs = prepared

        async def _traced():
            with self.tracer.start_as_current_span(
                span_name, attributes=attrs, record_exception=False, set_status_on_exception=False
            ) as span:
                try:
                    return await wrapped(*args, **kwargs)
                except Exception as e:
                    self._record_node_error(span, e)
                    raise

        return _traced()

    def _graph_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Attributes for a run of ``instance``, whose StateGraph is its ``builder``."""
        return self._extract_graph_attributes(
            instance, args, kwargs, getattr(instance, "builder", None)
        )

    def _wrap_graph_invoke(self, wrapped, instance, args, kwargs):
        """Wrap graph invoke method with span.

        Args:
            wrapped: The original method.
            instance: The compiled graph instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        traced = self.create_span_wrapper(
            span_name="langgraph.graph.invoke",
            extract_attributes=lambda i, a, kw: self._graph_attributes(instance, a, kw),
        )(wrapped)
        note = _INNER_STREAM.set([instance])
        try:
            return traced(*args, **kwargs)
        finally:
            _forget_inner_stream(note)

    def _wrap_graph_ainvoke(self, wrapped, instance, args, kwargs):
        """Wrap graph ainvoke method with span.

        Args:
            wrapped: The original method.
            instance: The compiled graph instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        traced = self.create_span_wrapper(
            span_name="langgraph.graph.ainvoke",
            extract_attributes=lambda i, a, kw: self._graph_attributes(instance, a, kw),
        )(wrapped)

        async def _ainvoke():
            # The span is opened here, when the coroutine runs, not when ainvoke() is
            # called. The caller may hand the coroutine to another task (asyncio.run,
            # gather, create_task). A span made current at the call stayed current in
            # the caller's context after it had ended, and adopted every later run.
            note = _INNER_STREAM.set([instance])
            try:
                return await traced(*args, **kwargs)
            finally:
                _forget_inner_stream(note)

        return _ainvoke()

    def _wrap_graph_stream(self, wrapped, instance, args, kwargs):
        """Wrap graph stream method with span.

        Args:
            wrapped: The original method.
            instance: The compiled graph instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        chunks = wrapped(*args, **kwargs)
        if not self._instrumented or _is_inner_stream(instance):
            return chunks
        return self._trace_stream(
            chunks, "langgraph.graph.stream", instance, args, kwargs, otel_context.get_current()
        )

    def _wrap_graph_astream(self, wrapped, instance, args, kwargs):
        """Wrap graph astream method with span.

        Args:
            wrapped: The original method.
            instance: The compiled graph instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        chunks = wrapped(*args, **kwargs)
        if not self._instrumented or _is_inner_stream(instance):
            return chunks
        return self._trace_astream(
            chunks, "langgraph.graph.astream", instance, args, kwargs, otel_context.get_current()
        )

    # A streamed run is traced here rather than through the base span wrapper. That
    # wrapper starts the span, and makes it current, when stream() is called and leaves
    # it current until the generator is exhausted. The graph does nothing until it is
    # iterated, so a stream that was never iterated never ended its span, and one the
    # consumer stopped reading left its span current in the consumer's context.

    def _trace_stream(self, chunks, span_name, instance, args, kwargs, parent):
        """Yield ``chunks`` inside a span that lasts as long as the iteration does.

        ``parent`` is the context in which the stream was requested. The span starts
        at the first iteration, which may be after the caller's own span has ended,
        and it still belongs under it.
        """
        iterator = iter(chunks)
        started = self._start_stream_span(span_name, instance, args, kwargs, parent)
        if started is None:
            yield from iterator
            return

        span, context = started
        exhausted = False
        try:
            while True:
                # Current only while the graph produces the next chunk, which is when
                # its nodes run. Nothing stays attached across the yield.
                token = otel_context.attach(context)
                try:
                    chunk = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                finally:
                    otel_context.detach(token)
                yield chunk
        except Exception as e:
            self._record_stream_error(span, span_name, e)
            raise
        finally:
            try:
                close = getattr(iterator, "close", None)
                if not exhausted and close is not None:
                    # The consumer stopped early: let the graph unwind inside its span.
                    token = otel_context.attach(context)
                    try:
                        close()
                    finally:
                        otel_context.detach(token)
            finally:
                self._end_stream_span(span, span_name, exhausted)

    async def _trace_astream(self, chunks, span_name, instance, args, kwargs, parent):
        """Async counterpart of :meth:`_trace_stream`."""
        iterator = chunks.__aiter__()
        started = self._start_stream_span(span_name, instance, args, kwargs, parent)
        if started is None:
            async for chunk in iterator:
                yield chunk
            return

        span, context = started
        exhausted = False
        try:
            while True:
                token = otel_context.attach(context)
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    exhausted = True
                    break
                finally:
                    otel_context.detach(token)
                yield chunk
        except Exception as e:
            self._record_stream_error(span, span_name, e)
            raise
        finally:
            try:
                aclose = getattr(iterator, "aclose", None)
                if not exhausted and aclose is not None:
                    token = otel_context.attach(context)
                    try:
                        await aclose()
                    finally:
                        otel_context.detach(token)
            finally:
                self._end_stream_span(span, span_name, exhausted)

    def _start_stream_span(
        self, span_name, instance, args, kwargs, parent=None
    ) -> Optional[Tuple[Any, Any]]:
        """Start the span for a streamed run, with what the base wrapper puts on a span.

        Returns the span and the context that makes it current, or None when the span
        could not be set up: tracing must never cost the caller the stream.
        """
        try:
            attributes: Dict[str, Any] = {}
            try:
                extracted = self._graph_attributes(instance, args, kwargs)
                extracted = self._with_provider_aliases(extracted)
                extracted = strip_content_attributes(extracted, self.config)
                for key, value in extracted.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[key] = value
                    else:
                        attributes[key] = str(value)
            except Exception as e:
                logger.warning("Failed to extract attributes for span '%s': %s", span_name, e)

            span = self.tracer.start_span(span_name, context=parent, attributes=attributes)

            if self.config:
                if self.config.session_id_extractor:
                    try:
                        session_id = self.config.session_id_extractor(instance, args, kwargs)
                        if session_id:
                            span.set_attribute("session.id", session_id)
                            span.set_attribute("gen_ai.conversation.id", session_id)
                    except Exception as e:
                        logger.debug("Failed to extract session ID: %s", e)
                if self.config.user_id_extractor:
                    try:
                        user_id = self.config.user_id_extractor(instance, args, kwargs)
                        if user_id:
                            span.set_attribute("user.id", user_id)
                    except Exception as e:
                        logger.debug("Failed to extract user ID: %s", e)

            server_metrics = get_server_metrics()
            if server_metrics:
                server_metrics.increment_requests_running()

            return span, trace.set_span_in_context(span)
        except Exception as e:
            logger.error("Span creation failed for '%s': %s", span_name, e, exc_info=True)
            return None

    def _record_stream_error(self, span, span_name, error) -> None:
        """Mark a streamed run's span as failed."""
        try:
            if self.error_counter:
                self.error_counter.add(
                    1, {"operation": span_name, "error_type": type(error).__name__}
                )
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        except Exception as e:
            logger.debug("Failed to record error on span '%s': %s", span_name, e)

    def _end_stream_span(self, span, span_name, exhausted) -> None:
        """End a streamed run's span. Only a stream read to its end counts as a run."""
        try:
            if exhausted:
                if self.request_counter:
                    self.request_counter.add(1, {"operation": span_name})
                span.set_status(Status(StatusCode.OK))
            span.end()
            server_metrics = get_server_metrics()
            if server_metrics:
                server_metrics.decrement_requests_running()
        except Exception as e:
            logger.debug("Failed to end span '%s': %s", span_name, e)

    def _cap_content(self, text: Any, default: int = 200) -> str:
        """Cap captured content per ``config.content_max_length``.

        Content capture is a required audit feature, so content keeps flowing;
        this only bounds its size. ``content_max_length == 0`` (or missing) means
        unlimited. When no config is present (unit tests) the historical default
        cap is used.
        """
        s = text if isinstance(text, str) else str(text)
        cfg = getattr(self, "config", None)
        max_len = default
        cfg_len = getattr(cfg, "content_max_length", None) if cfg is not None else None
        if isinstance(cfg_len, int):
            max_len = cfg_len
        if max_len and max_len > 0:
            return s[:max_len]
        return s

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

        # The run config is the second positional parameter of every run method, and
        # graph.invoke(input, config) is the usual way to pass it (it is also how
        # invoke() itself calls stream()). Reading only the keyword lost the thread id.
        config = kwargs.get("config")
        if config is None and len(args) > 1:
            config = args[1]

        # Core attributes
        attrs["gen_ai.system"] = "langgraph"
        attrs["gen_ai.operation.name"] = "graph.execution"
        if self._graph_node_enabled():
            # The run is the root node of the agent graph; its nodes hang beneath it.
            graph_name = getattr(instance, "name", None)
            attrs["graph.node.id"] = (
                graph_name if isinstance(graph_name, str) and graph_name else "LangGraph"
            )
            attrs["openinference.span.kind"] = "CHAIN"
        # LangGraph carries recursion_limit in the run config dict.
        attrs.update(extract_budget_attributes(kwargs, config, instance))

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

                    # Store bounded values for important keys
                    for key in ["messages", "query", "question", "input"][:3]:
                        if key in input_state:
                            attrs[f"langgraph.input.{key}"] = self._cap_content(input_state[key])
                else:
                    # Non-dict input
                    attrs["langgraph.input"] = self._cap_content(input_state)
            except Exception as e:
                logger.debug("Failed to extract input state: %s", e)

        # Extract config if provided
        if config:
            try:
                # Extract configurable values
                if isinstance(config, dict):
                    if "configurable" in config:
                        configurable = config["configurable"]
                        # LangGraph runs a subgraph with a config of its own, in which
                        # these keys can be present and None.
                        if configurable.get("thread_id") is not None:
                            attrs["langgraph.thread_id"] = configurable["thread_id"]
                        if configurable.get("checkpoint_id") is not None:
                            attrs["langgraph.checkpoint_id"] = configurable["checkpoint_id"]

                    # Extract recursion limit
                    if "recursion_limit" in config:
                        attrs["langgraph.recursion_limit"] = config["recursion_limit"]

            except Exception as e:
                logger.debug("Failed to extract config: %s", e)

        # --- Session ID extraction (required for TraceVerse session aggregation) ---
        session_id = None

        # Priority 1: App-provided session_id in input state
        if isinstance(input_state, dict):
            session_id = input_state.get("session_id") or input_state.get("session.id")

        # Priority 2: an enclosing genai_otel.session() block
        if not session_id:
            session_id = current_session_id()

        # Priority 3: LangGraph thread_id from config (native session concept)
        if not session_id and config and isinstance(config, dict):
            configurable = config.get("configurable", {})
            if isinstance(configurable, dict):
                session_id = configurable.get("thread_id")

        # Priority 4: OTelConfig.session_id_extractor callable
        if not session_id and self.config and self.config.session_id_extractor:
            try:
                session_id = self.config.session_id_extractor(instance, args, kwargs)
            except Exception as e:
                logger.debug("Failed to extract session ID via extractor: %s", e)

        # Priority 5: a session an upstream service sent, when accepted
        if not session_id:
            session_id = inbound_session_id()

        # Priority 6: Auto-generate a new UUID for this invocation
        if not session_id:
            session_id = str(uuid.uuid4())

        attrs["session.id"] = session_id
        attrs["langgraph.session.id"] = session_id
        # Cross-framework conversation correlation per upstream proposal
        # semantic-conventions-genai#145. For LangGraph this is the
        # innermost conversation primitive (thread_id when the app
        # supplied one; falls through to session_id otherwise).
        attrs["gen_ai.conversation.id"] = session_id

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

                # Store bounded values for important keys
                for key in ["messages", "answer", "output", "result"][:3]:
                    if key in result:
                        attrs[f"langgraph.output.{key}"] = self._cap_content(result[key])

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
