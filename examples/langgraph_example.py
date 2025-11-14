"""LangGraph Stateful Workflow Framework Instrumentation Example.

This example demonstrates automatic OpenTelemetry instrumentation for
LangGraph, a framework for building stateful, graph-based AI workflows
with checkpoints and persistence.

LangGraph powers production AI agents at Uber and LinkedIn, with 11K+
GitHub stars and 4.2M monthly downloads.

Requirements:
    pip install genai-otel-instrument
    pip install langgraph langchain langchain-openai
    export OPENAI_API_KEY=your_api_key
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os
from typing import TypedDict

import genai_otel

# Initialize instrumentation - LangGraph is enabled automatically
genai_otel.instrument(
    service_name="langgraph-example",
    endpoint="http://localhost:4318",
)

print("\n" + "=" * 80)
print("LangGraph Stateful Workflow Framework OpenTelemetry Instrumentation Example")
print("=" * 80 + "\n")

# Import LangGraph
try:
    from langgraph.graph import END, StateGraph
except ImportError:
    print("ERROR: LangGraph not installed. Install with:")
    print("  pip install langgraph")
    exit(1)

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Get your API key from: https://platform.openai.com/api-keys")
    exit(1)

print("1. Simple Linear Graph...")
print("-" * 80)


# Define state structure
class SimpleState(TypedDict):
    """State for a simple workflow."""

    messages: list[str]
    count: int


# Define nodes (functions that process state)
def node_a(state: SimpleState) -> SimpleState:
    """First processing node."""
    print("  Executing Node A...")
    state["messages"].append("Processed by Node A")
    state["count"] += 1
    return state


def node_b(state: SimpleState) -> SimpleState:
    """Second processing node."""
    print("  Executing Node B...")
    state["messages"].append("Processed by Node B")
    state["count"] += 1
    return state


def node_c(state: SimpleState) -> SimpleState:
    """Final processing node."""
    print("  Executing Node C...")
    state["messages"].append("Processed by Node C")
    state["count"] += 1
    return state


# Create the graph
builder = StateGraph(SimpleState)

# Add nodes
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)

# Add edges (define flow)
builder.set_entry_point("node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", END)

# Compile the graph (this is what gets instrumented)
graph = builder.compile()

# Run the graph
initial_state = {"messages": [], "count": 0}
print("Running simple linear graph...")
result = graph.invoke(initial_state)

print("\nResult:")
print(f"  Messages: {result['messages']}")
print(f"  Total nodes executed: {result['count']}")
print()

print("2. Conditional Graph with Branching...")
print("-" * 80)


# Define state for conditional workflow
class ConditionalState(TypedDict):
    """State for conditional workflow."""

    input: str
    sentiment: str
    response: str


# Define nodes
def analyze_sentiment(state: ConditionalState) -> ConditionalState:
    """Analyze the sentiment of input text."""
    print("  Analyzing sentiment...")
    # Simple sentiment analysis (in practice, use LLM)
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "sad"]

    text = state["input"].lower()
    if any(word in text for word in positive_words):
        state["sentiment"] = "positive"
    elif any(word in text for word in negative_words):
        state["sentiment"] = "negative"
    else:
        state["sentiment"] = "neutral"

    return state


def handle_positive(state: ConditionalState) -> ConditionalState:
    """Handle positive sentiment."""
    print("  Handling positive sentiment...")
    state["response"] = "Great to hear positive feedback!"
    return state


def handle_negative(state: ConditionalState) -> ConditionalState:
    """Handle negative sentiment."""
    print("  Handling negative sentiment...")
    state["response"] = "We'll work on improving that."
    return state


def handle_neutral(state: ConditionalState) -> ConditionalState:
    """Handle neutral sentiment."""
    print("  Handling neutral sentiment...")
    state["response"] = "Thank you for your input."
    return state


# Conditional edge function
def route_by_sentiment(state: ConditionalState) -> str:
    """Route to appropriate handler based on sentiment."""
    return state["sentiment"]


# Build conditional graph
conditional_builder = StateGraph(ConditionalState)

# Add nodes
conditional_builder.add_node("analyze", analyze_sentiment)
conditional_builder.add_node("positive", handle_positive)
conditional_builder.add_node("negative", handle_negative)
conditional_builder.add_node("neutral", handle_neutral)

# Set entry point
conditional_builder.set_entry_point("analyze")

# Add conditional edges
conditional_builder.add_conditional_edges(
    "analyze",
    route_by_sentiment,
    {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    },
)

# Add edges to END
conditional_builder.add_edge("positive", END)
conditional_builder.add_edge("negative", END)
conditional_builder.add_edge("neutral", END)

# Compile
conditional_graph = conditional_builder.compile()

# Test with different inputs
test_inputs = [
    "This is great!",
    "This is terrible.",
    "This is okay.",
]

for test_input in test_inputs:
    print(f"\nTesting: '{test_input}'")
    result = conditional_graph.invoke({"input": test_input, "sentiment": "", "response": ""})
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Response: {result['response']}")

print()

print("3. Graph with Streaming...")
print("-" * 80)


# Define state for streaming
class StreamingState(TypedDict):
    """State for streaming workflow."""

    steps: list[str]


def step_1(state: StreamingState) -> StreamingState:
    """First step."""
    state["steps"].append("Step 1 completed")
    return state


def step_2(state: StreamingState) -> StreamingState:
    """Second step."""
    state["steps"].append("Step 2 completed")
    return state


def step_3(state: StreamingState) -> StreamingState:
    """Third step."""
    state["steps"].append("Step 3 completed")
    return state


# Build streaming graph
streaming_builder = StateGraph(StreamingState)
streaming_builder.add_node("step1", step_1)
streaming_builder.add_node("step2", step_2)
streaming_builder.add_node("step3", step_3)
streaming_builder.set_entry_point("step1")
streaming_builder.add_edge("step1", "step2")
streaming_builder.add_edge("step2", "step3")
streaming_builder.add_edge("step3", END)

streaming_graph = streaming_builder.compile()

# Stream the execution
print("Streaming graph execution...")
for i, chunk in enumerate(streaming_graph.stream({"steps": []})):
    print(f"  Chunk {i + 1}: {chunk}")

print()

print("=" * 80)
print("Telemetry Data Collected:")
print("=" * 80)
print(
    """
For each graph execution, the following data is automatically collected:

TRACES (Spans):
- Span name: langgraph.graph.invoke (or .stream, .ainvoke, .astream)
- Attributes:
  - gen_ai.system: "langgraph"
  - gen_ai.operation.name: "graph.execution"
  - langgraph.node_count: Number of nodes in the graph
  - langgraph.nodes: List of node names
  - langgraph.edge_count: Number of edges
  - langgraph.channels: State schema (keys)
  - langgraph.input.keys: Input state keys
  - langgraph.input.*: Input state values (truncated)
  - langgraph.output.keys: Output state keys
  - langgraph.output.*: Output state values (truncated)
  - langgraph.thread_id: Thread ID for persistence (if using checkpoints)
  - langgraph.checkpoint_id: Checkpoint ID for resumability
  - langgraph.message_count: Message count (for conversational workflows)
  - langgraph.steps: Number of execution steps

METRICS:
- genai.requests: Graph execution count
- genai.tokens: Token usage from LLM calls within nodes
- genai.latency: Graph execution duration
- genai.cost: Total cost for LLM calls (if nodes use LLMs)

View these metrics in your observability platform (Grafana, Jaeger, etc.)
"""
)

print("=" * 80)
print("LangGraph Framework Features:")
print("=" * 80)
print(
    """
Key Features Instrumented:
- Graph Execution: Tracks invoke(), stream(), ainvoke(), astream()
- Node Structure: Monitors graph topology (nodes and edges)
- State Management: Captures state updates across execution
- Conditional Logic: Tracks branching and routing decisions
- Streaming: Observability for streaming graph execution
- Checkpoints: Monitors persistence and resumability
- Async Support: Full support for async graph execution

Graph Types:
- StateGraph: Stateful workflows with shared state
- Linear: Sequential node execution
- Branching: Conditional routing based on state
- Cyclic: Loops and iterative processing

Benefits:
- Production-proven (powers Uber & LinkedIn agents)
- 11K+ stars, 4.2M monthly downloads
- State machines for complex workflows
- Checkpointing for fault tolerance
- Time-travel debugging with state history
- Complete observability with zero code changes
"""
)

print("=" * 80)
print("Example complete! Check your OTLP collector/Grafana for traces and metrics.")
print("=" * 80 + "\n")
