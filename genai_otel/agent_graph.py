"""Agent graph attributes for Arize AX / Phoenix agent-trajectory views.

Arize builds its agent graph from ``graph.node.id`` (required) and
``graph.node.parent_id`` (optional; inferred from the span tree when absent), and
labels nodes with ``graph.node.display_name``. Our agent instrumentors report the
agent as ``gen_ai.agent.name``, so this processor adds those keys, plus the
OpenInference ``agent.name`` and span kind, to every span that names an agent.

The parent link is the nearest enclosing agent, which can sit several spans up
(an LLM or tool span usually separates two agents), so each span inside an agent
records the graph node it belongs to. Everything happens in ``on_start``: the SDK
freezes a span's attributes before any end-of-span hook runs. An instrumentor
must therefore pass the agent name in the span's start attributes.
"""

import logging
import threading
from typing import Dict, NamedTuple, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import Span, SpanProcessor

logger = logging.getLogger(__name__)

AGENT_NAME = "agent.name"
GRAPH_NODE_ID = "graph.node.id"
GRAPH_NODE_NAME = "graph.node.name"
# The Arize AX agent page names the label graph.node.display_name; the
# OpenInference spec calls it graph.node.name. Both are written.
GRAPH_NODE_DISPLAY_NAME = "graph.node.display_name"
GRAPH_NODE_PARENT_ID = "graph.node.parent_id"
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"

# Spans still open inside an agent. Entries leave on span end; the cap only
# guards against spans that are never ended.
_MAX_TRACKED_SPANS = 10000


class _GraphNode(NamedTuple):
    node_id: str
    parent_id: Optional[str]


def _span_id(span) -> Optional[int]:
    try:
        span_context = span.get_span_context()
    except Exception:  # noqa: BLE001
        return None
    if span_context is None or not span_context.is_valid:
        return None
    return span_context.span_id


def _node_id(attributes) -> Optional[str]:
    """The span's graph node, or None when the span is not an agent.

    An explicit ``graph.node.id`` always makes a node. Otherwise an agent name does,
    unless the span declares a kind other than AGENT: a CrewAI task names the agent
    that runs it, but it is a step, and the agent span inside it is the node.
    """
    explicit = attributes.get(GRAPH_NODE_ID)
    if explicit:
        return str(explicit)
    kind = attributes.get(OPENINFERENCE_SPAN_KIND)
    if kind and str(kind).upper() != "AGENT":
        return None
    for key in (AGENT_NAME, "gen_ai.agent.name"):
        value = attributes.get(key)
        if value:
            return str(value)
    return None


class AgentGraphSpanProcessor(SpanProcessor):
    """Add ``agent.name`` and ``graph.node.*`` to spans that name an agent.

    Keys a span already carries are left alone, so OpenInference instrumentors
    that set their own graph attributes are not overwritten.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, _GraphNode] = {}
        self._lock = threading.Lock()

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        try:
            span_id = _span_id(span)
            if span_id is None:
                return
            parent_span_id = _span_id(trace.get_current_span(parent_context))
            with self._lock:
                enclosing = self._nodes.get(parent_span_id) if parent_span_id is not None else None

            attributes = span.attributes or {}
            node_id = _node_id(attributes)
            if node_id is None:
                node = enclosing
            else:
                node = self._annotate(span, attributes, node_id, enclosing)

            if node is not None:
                with self._lock:
                    if len(self._nodes) >= _MAX_TRACKED_SPANS:
                        self._nodes.clear()
                    self._nodes[span_id] = node
        except Exception as e:  # noqa: BLE001 - graph attributes must never break a span
            logger.debug("Failed to add agent graph attributes: %s", e)

    @staticmethod
    def _annotate(
        span: Span, attributes, node_id: str, enclosing: Optional[_GraphNode]
    ) -> _GraphNode:
        if GRAPH_NODE_PARENT_ID in attributes:
            parent_id = str(attributes[GRAPH_NODE_PARENT_ID]) or None
        elif enclosing is None:
            parent_id = None
        elif enclosing.node_id == node_id:
            # The same agent one level down (a CrewAI task and the agent running
            # it, an AutoGen run and its on_messages): one node, not a self-loop.
            parent_id = enclosing.parent_id
        else:
            parent_id = enclosing.node_id

        agent_name = attributes.get(AGENT_NAME) or attributes.get("gen_ai.agent.name")
        if agent_name and AGENT_NAME not in attributes:
            span.set_attribute(AGENT_NAME, str(agent_name))
        if GRAPH_NODE_ID not in attributes:
            span.set_attribute(GRAPH_NODE_ID, node_id)
        if GRAPH_NODE_NAME not in attributes:
            span.set_attribute(GRAPH_NODE_NAME, node_id)
        if GRAPH_NODE_DISPLAY_NAME not in attributes:
            span.set_attribute(
                GRAPH_NODE_DISPLAY_NAME, str(attributes.get(GRAPH_NODE_NAME) or node_id)
            )
        # An unset parent marks the root node, so only a real parent is written.
        if parent_id and GRAPH_NODE_PARENT_ID not in attributes:
            span.set_attribute(GRAPH_NODE_PARENT_ID, parent_id)
        if OPENINFERENCE_SPAN_KIND not in attributes:
            span.set_attribute(OPENINFERENCE_SPAN_KIND, "AGENT")
        return _GraphNode(node_id, parent_id)

    def on_end(self, span) -> None:
        try:
            span_id = _span_id(span)
            if span_id is not None:
                with self._lock:
                    self._nodes.pop(span_id, None)
        except Exception:  # noqa: BLE001
            pass

    def shutdown(self) -> None:
        with self._lock:
            self._nodes.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
