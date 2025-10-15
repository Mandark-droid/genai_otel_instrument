# Comparison: genai_otel_instrument vs OpenInference vs OpenTelemetry GenAI Conventions

## ✅ What You're Doing Well
Your implementation covers these standard attributes correctly:

- `gen_ai.system` - Provider name ✓
- `gen_ai.request.model` - Model identifier ✓
- `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens` - Token counts ✓
- `gen_ai.cost.amount` - Cost tracking ✓

## 🔴 Major Gaps in OpenTelemetry Semantic Conventions

### 1. Missing Core GenAI Span Attributes
**OpenTelemetry Standard Attributes You're Missing:**
```python
# Required/Recommended by OTel GenAI spec:
gen_ai.operation.name          # e.g., "chat", "text_completion", "embedding"
gen_ai.request.temperature     # Model temperature setting
gen_ai.request.top_p           # Top-p sampling parameter
gen_ai.request.max_tokens      # Maximum tokens requested
gen_ai.request.frequency_penalty
gen_ai.request.presence_penalty
gen_ai.response.id             # Response ID from provider
gen_ai.response.model          # Actual model used (may differ from request)
gen_ai.response.finish_reasons # Why generation stopped
gen_ai.usage.input_tokens      # NEW: Replaces prompt_tokens
gen_ai.usage.output_tokens     # NEW: Replaces completion_tokens
```

### 2. Missing Event-Based Tracing
OpenTelemetry GenAI conventions specify events for prompt/completion content (not just attributes):
```python
# You should emit these as span events:
span.add_event(
    name="gen_ai.content.prompt",
    attributes={
        "gen_ai.prompt": "[{\"role\": \"user\", \"content\": \"...\"}]"
    }
)

span.add_event(
    name="gen_ai.content.completion", 
    attributes={
        "gen_ai.completion": "[{\"role\": \"assistant\", \"content\": \"...\"}]"
    }
)
```
**Why this matters:** Events allow you to capture full prompt/completion content without bloating span attributes, and they're timestamped.

### 3. Missing OpenInference Span Kind
OpenInference adds a critical attribute your implementation lacks:
```python
openinference.span.kind = "LLM"  # or "CHAIN", "RETRIEVER", "RERANKER", "AGENT"
```
This helps categorize different types of GenAI operations in your trace visualization.

### 4. Incomplete Message Structure
- **Your approach (likely):** Storing messages as simple strings or JSON
- **OpenInference standard:** Structured message objects with proper flattening

```python
# OpenInference way (better):
span.set_attribute("llm.input_messages.0.message.role", "user")
span.set_attribute("llm.input_messages.0.message.content", "Hello")
span.set_attribute("llm.input_messages.1.message.role", "assistant")
span.set_attribute("llm.input_messages.1.message.content", "Hi there")

# Also supports:
message.tool_calls              # For function calling
message.function_call_name      # Legacy function calls
message.contents                # Multi-modal content (text + images)
```

### 5. Missing Tool/Function Call Instrumentation
OpenInference has rich support for tool calling that you're missing:

```python
llm.tools                       # List of tools advertised to LLM
tool_call.function.name         # Name of called function
tool_call.function.arguments    # Arguments passed
tool_call.id                    # Tool call ID
message.tool_calls              # Array of tool calls in message
```

### 6. Incomplete Cost Tracking
Your implementation has `gen_ai.cost.amount`, but OpenInference specifies granular cost attributes:

```python
llm.cost.prompt                 # Input cost
llm.cost.completion             # Output cost
llm.cost.total                  # Total cost
llm.cost.completion_details.reasoning  # Reasoning token costs (GPT-4o)
llm.cost.prompt_details.cache_read     # Cache hit costs (Anthropic)
llm.cost.prompt_details.cache_write    # Cache miss costs (Anthropic)
```

### 7. Missing Retrieval & Embedding Attributes
If you're instrumenting RAG pipelines, you need:

```python
# For vector search operations:
retrieval.documents.{i}.document.id
retrieval.documents.{i}.document.content
retrieval.documents.{i}.document.score
retrieval.documents.{i}.document.metadata

# For embeddings:
embedding.model_name
embedding.vector                # The actual embedding
embedding.text                  # Text that was embedded

# For reranking:
reranker.model_name
reranker.query
reranker.top_k
reranker.input_documents
reranker.output_documents
```

### 8. Missing Session & User Tracking

```python
session.id                      # Group related requests
user.id                         # Track per-user usage
```

### 9. Missing Graph/Agent Attributes
For agentic workflows (which you mention supporting):

```python
agent.name                      # Name of the agent
graph.node.id                   # Node ID in execution graph
graph.node.name                 # Human-readable node name
graph.node.parent_id            # Parent node for graph visualization
```

## 📊 Metric Naming Issues
Your metrics use custom prefixes:

```python
genai.requests
genai.tokens
genai.latency
genai.cost
genai.gpu.*
```

OpenTelemetry standard uses different naming:

```python
gen_ai.client.operation.duration       # Not genai.latency
gen_ai.client.token.usage              # Not genai.tokens
gen_ai.server.request.duration
```

## 🔧 Recommendations
### Immediate Actions:

1.  **Add missing core attributes:**
    ```python
    # In your LLM instrumentor:
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.request.temperature", temperature)
    span.set_attribute("gen_ai.request.max_tokens", max_tokens)
    span.set_attribute("gen_ai.response.id", response.id)
    span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
    ```

2.  **Implement event-based content capture:**
    ```python
    # Instead of storing as attributes, use events:
    span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": json.dumps(messages)})
    span.add_event("gen_ai.content.completion", {"gen_ai.completion": json.dumps(response)})
    ```

3.  **Add OpenInference span kind:**
    ```python
    span.set_attribute("openinference.span.kind", "LLM")  # or appropriate type
    ```

4.  **Flatten message structures properly:**
    ```python
    for i, msg in enumerate(messages):
        span.set_attribute(f"llm.input_messages.{i}.message.role", msg["role"])
        span.set_attribute(f"llm.input_messages.{i}.message.content", msg["content"])
    ```

5.  **Add granular cost tracking:**
    ```python
    span.set_attribute("llm.cost.prompt", prompt_cost)
    span.set_attribute("llm.cost.completion", completion_cost)
    span.set_attribute("llm.cost.total", total_cost)
    ```

6.  **Support `OTEL_SEMCONV_STABILITY_OPT_IN`:**
    ```python
    # Allow users to opt into latest conventions:
    if os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN") == "gen_ai_latest_experimental":
        # Use latest attribute names
        use_new_conventions = True
    ```

7.  **Add tool/function call support:**
    ```python
    if hasattr(message, "tool_calls"):
        for i, tool_call in enumerate(message.tool_calls):
            span.set_attribute(f"message.tool_calls.{i}.tool_call.function.name", 
                              tool_call.function.name)
            span.set_attribute(f"message.tool_calls.{i}.tool_call.function.arguments",
                              tool_call.function.arguments)
    ```

- Added new OpenInference instrumentation packages: `openinference-instrumentation`, `openinference-instrumentation-litellm`, `openinference-instrumentation-mcp`, `openinference-instrumentation-smolagents`.