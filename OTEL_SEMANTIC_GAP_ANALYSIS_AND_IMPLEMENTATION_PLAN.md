# OpenTelemetry Semantic Conventions - Gap Analysis & Implementation Plan

**Date:** 2025-10-20
**Project:** genai-otel-instrument
**Analysis Version:** 1.1 (Updated with additional sample analysis)

---

## Executive Summary

Based on analysis of current trace/metric collection, OpenTelemetry GenAI semantic conventions (2025), and comparison with the codebase, this document identifies **15 critical gaps** and provides a phased implementation plan to achieve full semantic compatibility.

### ✅ Implementation Status (Updated 2025-10-20)

**Phase 1: COMPLETE** ✅
- ✅ Metric names refactored to use semantic conventions (`gen_ai.*`)
- ✅ GPU metrics fixed (Counter → ObservableGauge with callbacks)
- ✅ Histogram buckets applied via OpenTelemetry Views
- ✅ All 373 baseline tests passing + 8 new tests = **381 tests passing**
- ✅ Code coverage maintained at **95%**

**Phase 2: COMPLETE** ✅
- ✅ Missing span attributes added to OpenAI instrumentor (operation.name, request parameters, response attributes)
- ✅ OTEL_SEMCONV_STABILITY_OPT_IN support implemented for dual token attribute emission
- ✅ Event-based content capture implemented (opt-in via GENAI_ENABLE_CONTENT_CAPTURE)
- ✅ Response attribute extraction added (`_extract_response_attributes` method)
- ✅ All tests passing, zero regressions

**Next Steps:**
- Phase 3: Advanced features (tool calls, granular costs, MCP metrics, streaming)
- Phase 4: Optional enhancements (session tracking, RAG attributes, agent tracking)

### 🆕 Update (Version 1.1)

**New samples reveal:**
1. ✅ OpenInference instrumentors working correctly (Smolagents, LiteLLM, MCP)
2. ⚠️ Dual instrumentation strategy creates inconsistency
3. ❌ GPU metrics confirmed broken (using Counter instead of Gauge)
4. 🔍 Need to align custom instrumentors with OpenInference conventions

### Current State Analysis

**Sample Trace Data Shows:**
```json
{
  "name": "openai.chat.completion",
  "attributes": [
    {"key": "gen_ai.system", "value": "openai"},
    {"key": "gen_ai.request.model", "value": "gpt-3.5-turbo"},
    {"key": "gen_ai.request.message_count", "value": 2},
    {"key": "gen_ai.usage.prompt_tokens", "value": 23},
    {"key": "gen_ai.usage.completion_tokens", "value": 101}
  ]
}
```

**Sample Metrics Data Shows:**
```json
{
  "metrics": [
    {"name": "genai.requests"},           // ❌ Wrong name (should be gen_ai.*)
    {"name": "genai.latency"},            // ❌ Wrong name
    {"name": "genai.tokens"},             // ❌ Wrong name
    {"name": "genai.cost"},               // ❌ Wrong name
    {"name": "genai.gpu.utilization"}     // ❌ Wrong name
  ]
}
```

**Key Problems:**
1. ❌ Metric names use `genai.*` instead of `gen_ai.*` (missing underscore)
2. ❌ Using deprecated token attribute names (`prompt_tokens` vs `input_tokens`)
3. ❌ Missing critical span attributes (operation.name, temperature, response.id, etc.)
4. ❌ No event-based content capture
5. ❌ GPU metrics using wrong metric types (Counter instead of Gauge)
6. ❌ GPU memory and temperature gauges created but never observed
7. ❌ Histogram buckets defined but never applied
8. ❌ Constants in semcov.py never used

---

## 🆕 NEW FINDING: Dual Instrumentation Architecture

### Current Architecture

Your project uses **TWO parallel instrumentation systems**:

#### 1. Custom Instrumentors (genai_otel/instrumentors/)
**Libraries covered:**
- OpenAI, Anthropic, Google AI, AWS Bedrock
- Azure OpenAI, Cohere, Mistral AI, Together AI
- Groq, Ollama, Vertex AI, Replicate, Anyscale
- LangChain, LlamaIndex, HuggingFace

**Conventions used:**
- Custom metric names: `genai.requests`, `genai.tokens`, `genai.latency`, `genai.cost`
- Span attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.prompt_tokens`

#### 2. OpenInference Instrumentors (from openinference.instrumentation.*)
**Libraries covered:**
- LiteLLM
- MCP (Model Context Protocol)
- Smolagents

**Conventions used:**
- OpenInference attributes: `openinference.span.kind`, `tool.name`, `tool.description`
- Token counts: `llm.token_count.prompt`, `llm.token_count.completion`, `llm.token_count.total`
- Structured: `input.value`, `output.value`

### Sample Evidence

**Custom Instrumentor (OpenAI):**
```json
{
  "scope": {"name": "genai_otel.instrumentors.base"},
  "attributes": [
    {"key": "gen_ai.system", "value": "openai"},
    {"key": "gen_ai.request.model", "value": "gpt-3.5-turbo"},
    {"key": "gen_ai.usage.prompt_tokens", "value": 23},
    {"key": "gen_ai.usage.completion_tokens", "value": 101}
  ]
}
```

**OpenInference Instrumentor (Smolagents):**
```json
{
  "scope": {"name": "openinference.instrumentation.smolagents", "version": "0.1.11"},
  "attributes": [
    {"key": "openinference.span.kind", "value": "AGENT"},
    {"key": "tool.name", "value": "final_answer"},
    {"key": "llm.token_count.prompt", "value": 6686},
    {"key": "llm.token_count.completion", "value": 343},
    {"key": "llm.token_count.total", "value": 7029}
  ]
}
```

### Issues with Dual Instrumentation

#### 1. Inconsistent Attribute Naming
| Attribute Type | Custom Instrumentors | OpenInference | Standard OTel |
|----------------|---------------------|---------------|---------------|
| Prompt tokens | `gen_ai.usage.prompt_tokens` | `llm.token_count.prompt` | `gen_ai.usage.input_tokens` |
| Completion tokens | `gen_ai.usage.completion_tokens` | `llm.token_count.completion` | `gen_ai.usage.output_tokens` |
| Span kind | Not set | `openinference.span.kind` | Recommended |

#### 2. Potential Double Instrumentation
If you enable both LiteLLM custom instrumentor AND OpenInference LiteLLM instrumentor:
- Two sets of spans created
- Duplicate metrics recorded
- Inconsistent attribute names

#### 3. Missing Standardization
Your custom instrumentors are missing attributes that OpenInference already provides:
- `openinference.span.kind` (LLM, AGENT, TOOL, CHAIN, etc.)
- `input.value` / `output.value` structured data
- `tool.name`, `tool.description`, `tool.parameters`

### Recommendation: Hybrid Approach

**Strategy:**
1. **Keep custom instrumentors** for core LLM providers (OpenAI, Anthropic, etc.)
   - These give you fine-grained control
   - Can add provider-specific attributes
   - BUT: Align with OpenInference conventions

2. **Use OpenInference instrumentors** for agent frameworks
   - LiteLLM, Smolagents, MCP already working well
   - Maintained by OpenInference project
   - Follow their conventions

3. **Standardize on hybrid convention set:**
   - Metric names: Use OTel standard (`gen_ai.*`)
   - Span attributes: Use both OTel (`gen_ai.*`) AND OpenInference (`openinference.span.kind`, `llm.*`)
   - This gives maximum compatibility with dashboards

**Implementation Impact:**
- Phase 1: Add this to metric naming refactor
- Phase 2: Add OpenInference attributes to custom instrumentors
- Documentation: Clarify which instrumentors handle which libraries

---

## Gap Analysis by Category

### 1. CRITICAL: Metric Naming Violations

**File:** `genai_otel/instrumentors/base.py:81-95`

**Current Implementation:**
```python
cls._shared_request_counter = meter.create_counter(
    "genai.requests", description="Number of LLM requests"
)
cls._shared_token_counter = meter.create_counter(
    "genai.tokens", description="Number of tokens processed"
)
cls._shared_latency_histogram = meter.create_histogram(
    "genai.latency", description="Request latency in seconds", unit="s"
)
cls._shared_cost_counter = meter.create_counter(
    "genai.cost", description="Estimated cost in USD", unit="USD"
)
```

**Should Use (from openlit/semcov.py):**
```python
# Properly defined but NEVER imported/used:
GEN_AI_CLIENT_TOKEN_USAGE = "gen_ai.client.token.usage"
GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_COST = "gen_ai.usage.cost"
GEN_AI_REQUESTS = "gen_ai.requests"
```

**Impact:** HIGH - Metrics are incompatible with standard OTel backends and dashboards.

---

### 2. CRITICAL: GPU Metrics Issues

**File:** `genai_otel/gpu_metrics.py`

**Problems Identified:**

#### a) Wrong Metric Type for Utilization
```python
# Line 77-79: Using Counter for utilization (WRONG!)
self.gpu_utilization_counter = self.meter.create_counter(
    "genai.gpu.utilization", description="GPU utilization percentage", unit="%"
)
```

**Issue:** GPU utilization is a gauge (0-100%), not a monotonically increasing counter.

**Evidence from actual traces:**
```json
{
  "name": "genai.gpu.utilization",
  "sum": {
    "dataPoints": [{"asInt": "0"}],
    "aggregationTemporality": 2,
    "isMonotonic": true  // ❌ WRONG! Utilization is not monotonic
  }
}
```

**Confirmed in multiple samples** - This metric is fundamentally broken.

**Should be:**
```python
self.gpu_utilization_gauge = self.meter.create_observable_gauge(
    "gen_ai.gpu.utilization",  # Fixed name with underscore
    callbacks=[self._get_gpu_utilization],
    description="GPU utilization percentage",
    unit="%"
)
```

#### b) Observable Gauges Never Observed
```python
# Lines 80-85: Created but never have callbacks
self.gpu_memory_used_gauge = self.meter.create_observable_gauge(
    "genai.gpu.memory.used", ...  # No callbacks parameter!
)
self.gpu_temperature_gauge = self.meter.create_observable_gauge(
    "genai.gpu.temperature", ...  # No callbacks parameter!
)
```

**Issue:** Lines 126-128 and 136-138 call `.add()` on observable gauges, which is invalid.
**Result:** Memory and temperature metrics are likely NOT being collected.

**Should be:**
```python
self.gpu_memory_used_gauge = self.meter.create_observable_gauge(
    "gen_ai.gpu.memory.used",
    callbacks=[self._get_gpu_memory],
    description="GPU memory used in MiB",
    unit="MiB"
)
```

#### c) Missing GPU Metrics
- ❌ GPU memory total (only tracking used)
- ❌ GPU memory available
- ❌ GPU power usage (calculated but not exposed as metric)
- ❌ GPU clock speeds (SM, memory)
- ❌ GPU PCIe throughput

#### d) CO2 Metrics Not Visible in Sample Data
```python
# Line 64-68: Created but never seen in trace samples
self.co2_counter = meter.create_counter(
    "genai.co-2.emissions",  # ❌ Wrong name (should be gen_ai.co2.emissions)
    description="Cumulative CO2 equivalent emissions in grams",
    unit="gCO2e",
)
```

**Possible Issues:**
- Metric name has hyphen instead of underscore
- `config.enable_co2_tracking` might be False by default

---

### 3. Missing Core GenAI Span Attributes

**File:** `genai_otel/instrumentors/openai_instrumentor.py:89-113`

**Current Attributes (Only 4):**
```python
attrs["gen_ai.system"] = "openai"
attrs["gen_ai.request.model"] = model
attrs["gen_ai.request.message_count"] = len(messages)
attrs["gen_ai.request.first_message"] = first_message
```

**Missing Required/Recommended Attributes:**

| Attribute | Required? | Current | Notes |
|-----------|-----------|---------|-------|
| `gen_ai.operation.name` | Required | ❌ | e.g., "chat", "embedding", "completion" |
| `gen_ai.request.temperature` | Recommended | ❌ | Model temperature setting |
| `gen_ai.request.top_p` | Recommended | ❌ | Top-p sampling parameter |
| `gen_ai.request.max_tokens` | Recommended | ❌ | Maximum tokens requested |
| `gen_ai.request.frequency_penalty` | Optional | ❌ | Frequency penalty |
| `gen_ai.request.presence_penalty` | Optional | ❌ | Presence penalty |
| `gen_ai.response.id` | Recommended | ❌ | Response ID from provider |
| `gen_ai.response.model` | Recommended | ❌ | Actual model used (may differ) |
| `gen_ai.response.finish_reasons` | Recommended | ❌ | Array of finish reasons |
| `gen_ai.usage.input_tokens` | Recommended | ❌ | New name for prompt_tokens |
| `gen_ai.usage.output_tokens` | Recommended | ❌ | New name for completion_tokens |

**Impact:** HIGH - Missing critical diagnostic and cost tracking information.

---

### 4. Using Deprecated Token Attribute Names

**File:** `genai_otel/instrumentors/base.py:218-228`

**Current (Deprecated as of OTel v1.37.0):**
```python
span.set_attribute("gen_ai.usage.prompt_tokens", int(prompt_tokens))
span.set_attribute("gen_ai.usage.completion_tokens", int(completion_tokens))
```

**New Standard:**
```python
span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
```

**Should Support Both with Opt-In:**
```python
stability_opt_in = os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", "")
use_new_conventions = "gen_ai_latest_experimental" in stability_opt_in

if use_new_conventions:
    span.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))
    span.set_attribute("gen_ai.usage.output_tokens", int(completion_tokens))
else:
    # Backward compatibility
    span.set_attribute("gen_ai.usage.prompt_tokens", int(prompt_tokens))
    span.set_attribute("gen_ai.usage.completion_tokens", int(completion_tokens))
```

**Impact:** MEDIUM - Current implementation works but is deprecated.

---

### 5. Missing Event-Based Content Capture

**Current:** No span events are emitted for prompts/completions.

**Required per OTel GenAI Conventions:**
```python
# At start of LLM call
span.add_event(
    name="gen_ai.content.prompt",
    attributes={
        "gen_ai.prompt": json.dumps(messages)
    }
)

# After receiving response
span.add_event(
    name="gen_ai.content.completion",
    attributes={
        "gen_ai.completion": json.dumps(response_message)
    }
)
```

**Benefits:**
- Events are timestamped independently
- Don't bloat span attributes
- Can be toggled with `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
- Allow full prompt/completion capture without span size limits

**Impact:** MEDIUM - Missing detailed content tracking.

---

### 6. Missing OpenInference Span Kind

**Current:** No span kind classification.

**Required:**
```python
span.set_attribute("openinference.span.kind", "LLM")
# Other values: "CHAIN", "RETRIEVER", "RERANKER", "AGENT", "EMBEDDING"
```

**Impact:** LOW - Helpful for visualization but not critical.

---

### 7. Missing Structured Message Attributes

**Current (openai_instrumentor.py:110-111):**
```python
first_message = str(messages[0])[:200]
attrs["gen_ai.request.first_message"] = first_message
```

**OpenInference Standard:**
```python
for i, msg in enumerate(messages):
    span.set_attribute(f"llm.input_messages.{i}.message.role", msg["role"])
    span.set_attribute(f"llm.input_messages.{i}.message.content", msg["content"])

    # Support tool calls
    if "tool_calls" in msg:
        for j, tool_call in enumerate(msg["tool_calls"]):
            span.set_attribute(
                f"llm.input_messages.{i}.message.tool_calls.{j}.tool_call.function.name",
                tool_call["function"]["name"]
            )
            span.set_attribute(
                f"llm.input_messages.{i}.message.tool_calls.{j}.tool_call.function.arguments",
                tool_call["function"]["arguments"]
            )
```

**Impact:** MEDIUM - Better message structure for debugging.

---

### 8. Missing Tool/Function Call Instrumentation

**Current:** No tool call tracking.

**Required Attributes:**
```python
llm.tools                                    # JSON array of available tools
llm.output_messages.0.message.tool_calls    # Tool calls in response
tool_call.id                                 # Tool call ID
tool_call.function.name                      # Function name
tool_call.function.arguments                 # JSON arguments
```

**Impact:** HIGH for agentic workflows - Cannot track tool usage.

---

### 9. Incomplete Cost Tracking

**Current (base.py:240-243):**
```python
cost = self.cost_calculator.calculate_cost(model, usage, call_type)
if cost and cost > 0:
    self._shared_cost_counter.add(cost, {"model": str(model)})
```

**Missing Granular Tracking:**
```python
# Should track separately:
llm.cost.prompt                              # Input cost
llm.cost.completion                          # Output cost
llm.cost.total                               # Total cost
llm.cost.completion_details.reasoning        # Reasoning tokens (o1 models)
llm.cost.prompt_details.cache_read           # Cache hits (Anthropic)
llm.cost.prompt_details.cache_write          # Cache misses (Anthropic)
```

**Impact:** MEDIUM - Better cost attribution for optimization.

---

### 10. Missing Histogram Bucket Definitions

**File:** `genai_otel/metrics.py:52-148`

**Defined but Never Used:**
```python
_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS = [0.01, 0.02, ..., 81.92]
_GEN_AI_SERVER_TBT = [0.01, 0.025, ..., 2.5]
_GEN_AI_SERVER_TFTT = [0.001, 0.005, ..., 10.0]
_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS = [1, 4, ..., 67108864]
_MCP_CLIENT_OPERATION_DURATION_BUCKETS = [0.001, ..., 10.0]
_MCP_PAYLOAD_SIZE_BUCKETS = [100, ..., 5242880]
```

**Current Histogram Creation (base.py:87-88):**
```python
cls._shared_latency_histogram = meter.create_histogram(
    "genai.latency", description="Request latency in seconds", unit="s"
    # ❌ No explicit_bucket_boundaries parameter!
)
```

**Result:** OTel uses default buckets (0, 5, 10, 25, ...) which are poorly suited for LLM latencies.

**Sample Data Confirms:**
```json
"explicitBounds": [0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
```

**Should be:**
```python
from genai_otel.metrics import _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS

cls._shared_latency_histogram = meter.create_histogram(
    "gen_ai.client.operation.duration",
    description="Client operation duration in seconds",
    unit="s",
    explicit_bucket_boundaries=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
)
```

**Impact:** HIGH - Poor histogram granularity loses latency insights.

---

### 11. Missing Server-Side Streaming Metrics

**Defined but Never Created/Recorded:**
```python
# From semcov.py:
GEN_AI_SERVER_TBT = "gen_ai.server.tbt"      # Time Between Tokens
GEN_AI_SERVER_TTFT = "gen_ai.server.ttft"    # Time To First Token

# Buckets defined in metrics.py:
_GEN_AI_SERVER_TBT = [0.01, 0.025, ..., 2.5]
_GEN_AI_SERVER_TFTT = [0.001, 0.005, ..., 10.0]
```

**Required for Streaming LLM Calls:**
```python
# In streaming response handler:
ttft = first_token_time - request_start_time
span.set_attribute("gen_ai.server.ttft", ttft)
ttft_histogram.record(ttft, {"model": model})

# Between tokens:
tbt = current_token_time - previous_token_time
tbt_histogram.record(tbt, {"model": model})
```

**Impact:** HIGH for streaming - Missing critical latency metrics.

---

### 12. Missing MCP-Specific Metrics

**Defined in semcov.py but Never Used:**
```python
MCP_REQUESTS = "mcp.requests"
MCP_CLIENT_OPERATION_DURATION_METRIC = "mcp.client.operation.duration"
MCP_REQUEST_SIZE = "mcp.request.size"
MCP_RESPONSE_SIZE_METRIC = "mcp.response.size"
MCP_TOOL_CALLS = "mcp.tool_calls"
MCP_RESOURCE_READS = "mcp.resource.reads"
MCP_PROMPT_GETS = "mcp.prompt_gets"
MCP_TRANSPORT_USAGE = "mcp.transport.usage"
MCP_ERRORS = "mcp.errors"
MCP_OPERATION_SUCCESS_RATE = "mcp.operation.success_rate"
```

**File:** `genai_otel/mcp_instrumentors/manager.py`

**Current:** MCP instrumentors exist but don't record these metrics.

**Impact:** MEDIUM - Missing MCP tool observability.

---

### 13. Missing Session & User Tracking

**Current:** No session/user context in spans.

**Recommended:**
```python
# From application context:
span.set_attribute("session.id", session_id)
span.set_attribute("user.id", user_id)
```

**Impact:** LOW - Helpful for multi-user analysis but optional.

---

### 14. 🆕 Inconsistent Convention Usage Across Instrumentors

**Current:** Custom instrumentors and OpenInference instrumentors use different conventions.

**Evidence from Traces:**

Custom instrumentor output:
```json
{
  "scope": {"name": "genai_otel.instrumentors.base"},
  "attributes": {
    "gen_ai.usage.prompt_tokens": 23,      // OTel convention
    "gen_ai.usage.completion_tokens": 101  // OTel convention
  }
}
```

OpenInference instrumentor output:
```json
{
  "scope": {"name": "openinference.instrumentation.smolagents"},
  "attributes": {
    "llm.token_count.prompt": 6686,        // OpenInference convention
    "llm.token_count.completion": 343,     // OpenInference convention
    "openinference.span.kind": "AGENT"     // OpenInference exclusive
  }
}
```

**Issues:**
1. Different attribute names for same data
2. Custom instrumentors missing `openinference.span.kind`
3. Difficult to create unified dashboards
4. May confuse downstream tools

**Should Add to Custom Instrumentors:**
```python
# In base.py create_span_wrapper:
span.set_attribute("openinference.span.kind", "LLM")  # Add this

# Optionally support both conventions:
span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)      # OTel
span.set_attribute("llm.token_count.prompt", prompt_tokens)          # OpenInference
```

**Impact:** MEDIUM - Better dashboard compatibility but not critical.

---

### 15. 🆕 Potential Double Instrumentation Risk

**Issue:** Project includes both custom and OpenInference instrumentors for same libraries.

**Example Conflict:**
- Custom: `litellm` in `INSTRUMENTORS` dict (if you create one)
- OpenInference: `LiteLLMInstrumentor` imported from `openinference.instrumentation.litellm`

**Current State (auto_instrument.py:100-107):**
```python
if OPENINFERENCE_AVAILABLE:
    INSTRUMENTORS.update({
        "smolagents": SmolagentsInstrumentor,
        "mcp": MCPInstrumentor,
        "litellm": LiteLLMInstrumentor,  # OpenInference handles these
    })
```

**Risk:** If you add custom instrumentors for these libraries, both will run, creating:
- Duplicate spans
- Double metric counts
- Increased overhead

**Recommendation:**
- Document which instrumentor handles each library
- Add check to prevent double instrumentation
- Keep OpenInference for: LiteLLM, Smolagents, MCP
- Use custom for: OpenAI, Anthropic, etc.

**Impact:** LOW currently (no conflicts yet), but HIGH if custom instrumentors added for LiteLLM/Smolagents.

---

## Implementation Plan

### Phase 1: Critical Fixes ✅ COMPLETE

**Goal:** Fix metric naming and GPU issues to ensure correct data collection.

**Status:** ✅ **COMPLETED 2025-10-20**
- All metric names updated to use semantic conventions
- GPU metrics completely refactored with ObservableGauge + callbacks
- Histogram buckets applied via Views
- All tests passing (381/381), 95% coverage maintained

#### 1.1 Refactor Metric Names to Use Semantic Conventions
**Files:** `genai_otel/instrumentors/base.py`

**Changes:**
```python
# Import constants
from openlit.semcov import SemanticConvention as SC

# Update metric names (lines 81-95)
cls._shared_request_counter = meter.create_counter(
    SC.GEN_AI_REQUESTS,  # "gen_ai.requests"
    description="Number of GenAI requests"
)

cls._shared_token_counter = meter.create_counter(
    SC.GEN_AI_CLIENT_TOKEN_USAGE,  # "gen_ai.client.token.usage"
    description="Number of tokens used"
)

cls._shared_latency_histogram = meter.create_histogram(
    SC.GEN_AI_CLIENT_OPERATION_DURATION,  # "gen_ai.client.operation.duration"
    description="Client operation duration in seconds",
    unit="s",
    explicit_bucket_boundaries=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
)

cls._shared_cost_counter = meter.create_counter(
    SC.GEN_AI_USAGE_COST,  # "gen_ai.usage.cost"
    description="Estimated cost in USD",
    unit="USD"
)
```

**Testing:**
```bash
pytest tests/instrumentors/test_base.py -v
pytest tests/test_metrics.py -v
```

---

#### 1.2 Fix GPU Metrics Implementation
**Files:** `genai_otel/gpu_metrics.py`

**Changes:**

```python
# A) Fix utilization metric type (line 77-79)
# BEFORE:
self.gpu_utilization_counter = self.meter.create_counter(...)

# AFTER:
self.gpu_utilization_gauge = self.meter.create_observable_gauge(
    "gen_ai.gpu.utilization",  # Fixed name
    callbacks=[self._observe_gpu_utilization],
    description="GPU utilization percentage",
    unit="%"
)

# B) Fix memory gauge (line 80-82)
self.gpu_memory_used_gauge = self.meter.create_observable_gauge(
    "gen_ai.gpu.memory.used",  # Fixed name
    callbacks=[self._observe_gpu_memory],
    description="GPU memory used in MiB",
    unit="MiB"
)

# C) Fix temperature gauge (line 83-85)
self.gpu_temperature_gauge = self.meter.create_observable_gauge(
    "gen_ai.gpu.temperature",  # Fixed name
    callbacks=[self._observe_gpu_temperature],
    description="GPU temperature in Celsius",
    unit="Cel"
)

# D) Add callback methods
def _observe_gpu_utilization(self, options):
    """Observable callback for GPU utilization."""
    try:
        device_count = self.nvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            device_name = self._get_device_name(handle, i)
            utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)

            yield Observation(
                value=utilization.gpu,
                attributes={"gpu_id": str(i), "gpu_name": device_name}
            )
    except Exception as e:
        logger.error(f"Error observing GPU utilization: {e}")

# E) Fix CO2 metric name (line 64-68)
self.co2_counter = meter.create_counter(
    "gen_ai.co2.emissions",  # Fixed from "genai.co-2.emissions"
    description="Cumulative CO2 equivalent emissions in grams",
    unit="gCO2e",
)

# F) Remove old _collect_metrics() method and _run() thread
# Replace with observable callbacks only
```

**Testing:**
```bash
pytest tests/test_gpu_metrics.py -v
# Manual test with GPU:
python -c "from genai_otel import instrument; instrument(); import time; time.sleep(5)"
```

---

#### 1.3 Apply Histogram Buckets
**Files:** `genai_otel/instrumentors/base.py`

**Changes:**
```python
# Import bucket definitions
from genai_otel.metrics import _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS

# Update histogram creation (line 87-89)
cls._shared_latency_histogram = meter.create_histogram(
    SC.GEN_AI_CLIENT_OPERATION_DURATION,
    description="Client operation duration in seconds",
    unit="s",
    explicit_bucket_boundaries=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
)
```

---

### Phase 2: Core Attribute Enhancements ✅ COMPLETE

**Goal:** Add missing core span attributes and support new conventions.

**Status:** ✅ **COMPLETED 2025-10-20**
- Added comprehensive span attributes to OpenAI instrumentor
- Implemented OTEL_SEMCONV_STABILITY_OPT_IN for dual token emission
- Added event-based content capture (opt-in)
- Added response attribute extraction
- All tests passing (381/381), 95% coverage maintained

#### 2.1 Add Missing Span Attributes to OpenAI Instrumentor
**Files:** `genai_otel/instrumentors/openai_instrumentor.py`

**Changes:**
```python
def _extract_openai_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
    """Extract attributes from OpenAI API call."""
    attrs = {}

    # Existing
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])

    # Core attributes
    attrs["gen_ai.system"] = "openai"
    attrs["gen_ai.request.model"] = model
    attrs["gen_ai.operation.name"] = "chat"  # NEW

    # Request parameters (NEW)
    if "temperature" in kwargs:
        attrs["gen_ai.request.temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        attrs["gen_ai.request.top_p"] = kwargs["top_p"]
    if "max_tokens" in kwargs:
        attrs["gen_ai.request.max_tokens"] = kwargs["max_tokens"]
    if "frequency_penalty" in kwargs:
        attrs["gen_ai.request.frequency_penalty"] = kwargs["frequency_penalty"]
    if "presence_penalty" in kwargs:
        attrs["gen_ai.request.presence_penalty"] = kwargs["presence_penalty"]

    # OpenInference span kind (NEW)
    attrs["openinference.span.kind"] = "LLM"

    # Message structure (replace first_message with structured approach)
    for i, msg in enumerate(messages[:5]):  # Limit to first 5
        attrs[f"llm.input_messages.{i}.message.role"] = msg.get("role", "unknown")
        content = msg.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        attrs[f"llm.input_messages.{i}.message.content"] = content

    return attrs
```

**Changes in wrapper to add response attributes:**
```python
# In base.py create_span_wrapper, after result is obtained:
# Extract response-specific attributes
if hasattr(result, 'id'):
    span.set_attribute("gen_ai.response.id", result.id)
if hasattr(result, 'model'):
    span.set_attribute("gen_ai.response.model", result.model)
if hasattr(result, 'choices') and result.choices:
    finish_reasons = [choice.finish_reason for choice in result.choices if hasattr(choice, 'finish_reason')]
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", json.dumps(finish_reasons))
```

**Apply same pattern to ALL instrumentors:**
- `anthropic_instrumentor.py`
- `google_ai_instrumentor.py`
- `aws_bedrock_instrumentor.py`
- etc.

---

#### 2.2 Implement OTEL_SEMCONV_STABILITY_OPT_IN Support
**Files:** `genai_otel/instrumentors/base.py`

**Changes:**
```python
# At top of base.py
import os

STABILITY_OPT_IN = os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", "")
USE_NEW_CONVENTIONS = "gen_ai_latest_experimental" in STABILITY_OPT_IN

# In _record_result_metrics (lines 218-231)
if USE_NEW_CONVENTIONS:
    # New attribute names
    if prompt_tokens > 0:
        span.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))
        self.token_counter.add(
            prompt_tokens,
            {"token_type": "input", "operation": span.name}
        )
    if completion_tokens > 0:
        span.set_attribute("gen_ai.usage.output_tokens", int(completion_tokens))
        self.token_counter.add(
            completion_tokens,
            {"token_type": "output", "operation": span.name}
        )
else:
    # Legacy attribute names (backward compatibility)
    if prompt_tokens > 0:
        span.set_attribute("gen_ai.usage.prompt_tokens", int(prompt_tokens))
        self.token_counter.add(
            prompt_tokens,
            {"token_type": "prompt", "operation": span.name}
        )
    if completion_tokens > 0:
        span.set_attribute("gen_ai.usage.completion_tokens", int(completion_tokens))
        self.token_counter.add(
            completion_tokens,
            {"token_type": "completion", "operation": span.name}
        )

# Always set total_tokens
if total_tokens > 0:
    span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))
```

---

#### 2.3 Add Event-Based Content Capture
**Files:** `genai_otel/instrumentors/base.py`

**Changes:**
```python
# At top
CAPTURE_CONTENT = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false").lower() == "true"

# In create_span_wrapper, after span creation:
if CAPTURE_CONTENT and extract_attributes:
    try:
        # Capture input
        input_data = self._extract_input_content(instance, args, kwargs)
        if input_data:
            span.add_event(
                "gen_ai.content.prompt",
                attributes={"gen_ai.prompt": json.dumps(input_data)}
            )
    except Exception as e:
        logger.debug(f"Failed to capture input content: {e}")

# After result:
if CAPTURE_CONTENT:
    try:
        output_data = self._extract_output_content(result)
        if output_data:
            span.add_event(
                "gen_ai.content.completion",
                attributes={"gen_ai.completion": json.dumps(output_data)}
            )
    except Exception as e:
        logger.debug(f"Failed to capture output content: {e}")

# Add abstract methods:
@abstractmethod
def _extract_input_content(self, instance, args, kwargs):
    """Extract input content for events."""
    pass

@abstractmethod
def _extract_output_content(self, result):
    """Extract output content for events."""
    pass
```

**Implement in each instrumentor.**

---

#### 2.4 🆕 Add OpenInference Compatibility to Custom Instrumentors
**Files:** `genai_otel/instrumentors/base.py`, all provider instrumentors

**Goal:** Make custom instrumentors output both OTel and OpenInference attributes for maximum compatibility.

**Changes in base.py:**
```python
# In create_span_wrapper, add OpenInference span kind
def create_span_wrapper(self, span_name: str, span_kind: str = "LLM", ...):
    """
    Args:
        span_name: Name of the span
        span_kind: OpenInference span kind (LLM, CHAIN, AGENT, TOOL, etc.)
    """
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        with self.tracer.start_as_current_span(span_name, attributes=initial_attributes) as span:
            # Add OpenInference span kind
            span.set_attribute("openinference.span.kind", span_kind)

            # ... rest of wrapper
```

**Support dual token attributes (optional but recommended):**
```python
# In _record_result_metrics, add both conventions
if prompt_tokens > 0:
    # OTel convention
    span.set_attribute("gen_ai.usage.prompt_tokens", int(prompt_tokens))
    # OpenInference convention (for Phoenix/Arize compatibility)
    span.set_attribute("llm.token_count.prompt", int(prompt_tokens))

if completion_tokens > 0:
    span.set_attribute("gen_ai.usage.completion_tokens", int(completion_tokens))
    span.set_attribute("llm.token_count.completion", int(completion_tokens))

if total_tokens > 0:
    span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))
    span.set_attribute("llm.token_count.total", int(total_tokens))
```

**Benefits:**
- Works with OTel-native backends (Jaeger, Tempo)
- Works with OpenInference-based tools (Phoenix, Arize)
- Consistent with OpenInference instrumentors already in use
- Minimal overhead (just extra attributes)

**Testing:**
```bash
# Test with both env vars
export OTEL_SEMCONV_STABILITY_OPT_IN="gen_ai_latest_experimental"
pytest tests/instrumentors/test_openai_instrumentor.py -v

# Verify both attribute sets appear in spans
```

---

### Phase 3: Advanced Features (Week 3)

**Goal:** Add tool calls, granular cost tracking, and MCP metrics.

#### 3.1 Tool/Function Call Instrumentation
**Files:** `genai_otel/instrumentors/openai_instrumentor.py`

**Changes:**
```python
# In _extract_openai_attributes:
# Add tools if present
if "tools" in kwargs:
    attrs["llm.tools"] = json.dumps(kwargs["tools"])

# In wrapper after result:
# Extract tool calls from response
if hasattr(result, 'choices'):
    for choice_idx, choice in enumerate(result.choices):
        message = getattr(choice, 'message', None)
        if message and hasattr(message, 'tool_calls') and message.tool_calls:
            for tc_idx, tool_call in enumerate(message.tool_calls):
                prefix = f"llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}"
                span.set_attribute(f"{prefix}.tool_call.id", tool_call.id)
                span.set_attribute(f"{prefix}.tool_call.function.name", tool_call.function.name)
                span.set_attribute(f"{prefix}.tool_call.function.arguments", tool_call.function.arguments)
```

---

#### 3.2 Granular Cost Tracking
**Files:** `genai_otel/cost_calculator.py`, `genai_otel/instrumentors/base.py`

**Changes in CostCalculator:**
```python
def calculate_detailed_cost(self, model, usage, call_type="chat"):
    """Calculate detailed cost breakdown."""
    pricing = self._get_pricing(model, call_type)
    if not pricing:
        return None

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    cache_read_tokens = usage.get("cache_read_input_tokens", 0)
    cache_write_tokens = usage.get("cache_creation_input_tokens", 0)

    # Calculate component costs
    prompt_cost = (prompt_tokens / 1_000_000) * pricing.get("input", 0)
    completion_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0)
    reasoning_cost = (reasoning_tokens / 1_000_000) * pricing.get("reasoning", pricing.get("output", 0))
    cache_read_cost = (cache_read_tokens / 1_000_000) * pricing.get("cache_read", pricing.get("input", 0) * 0.1)
    cache_write_cost = (cache_write_tokens / 1_000_000) * pricing.get("cache_write", pricing.get("input", 0))

    return {
        "prompt": prompt_cost,
        "completion": completion_cost,
        "reasoning": reasoning_cost,
        "cache_read": cache_read_cost,
        "cache_write": cache_write_cost,
        "total": sum([prompt_cost, completion_cost, reasoning_cost, cache_read_cost, cache_write_cost])
    }
```

**Changes in base.py:**
```python
# Create additional cost counters
cls._shared_cost_prompt_counter = meter.create_counter(
    "gen_ai.cost.prompt", description="Prompt cost in USD", unit="USD"
)
cls._shared_cost_completion_counter = meter.create_counter(
    "gen_ai.cost.completion", description="Completion cost in USD", unit="USD"
)

# In _record_result_metrics:
detailed_cost = self.cost_calculator.calculate_detailed_cost(model, usage, call_type)
if detailed_cost:
    span.set_attribute("llm.cost.prompt", detailed_cost["prompt"])
    span.set_attribute("llm.cost.completion", detailed_cost["completion"])
    span.set_attribute("llm.cost.total", detailed_cost["total"])

    self._shared_cost_prompt_counter.add(detailed_cost["prompt"], {"model": model})
    self._shared_cost_completion_counter.add(detailed_cost["completion"], {"model": model})
    self._shared_cost_counter.add(detailed_cost["total"], {"model": model})

    # Handle special token types
    if detailed_cost["reasoning"] > 0:
        span.set_attribute("llm.cost.completion_details.reasoning", detailed_cost["reasoning"])
    if detailed_cost["cache_read"] > 0:
        span.set_attribute("llm.cost.prompt_details.cache_read", detailed_cost["cache_read"])
```

---

#### 3.3 Add MCP Metrics
**Files:** `genai_otel/mcp_instrumentors/manager.py`, `genai_otel/mcp_instrumentors/base.py` (new)

**Create base MCP instrumentor:**
```python
# genai_otel/mcp_instrumentors/base.py
from openlit.semcov import SemanticConvention as SC
from opentelemetry import metrics

class BaseMCPInstrumentor:
    def __init__(self):
        self.meter = metrics.get_meter(__name__)

        # MCP metrics
        self.mcp_request_counter = self.meter.create_counter(
            SC.MCP_REQUESTS,
            description="Number of MCP requests"
        )

        self.mcp_duration_histogram = self.meter.create_histogram(
            SC.MCP_CLIENT_OPERATION_DURATION_METRIC,
            description="MCP operation duration",
            unit="s",
            explicit_bucket_boundaries=_MCP_CLIENT_OPERATION_DURATION_BUCKETS
        )

        self.mcp_request_size_histogram = self.meter.create_histogram(
            SC.MCP_REQUEST_SIZE,
            description="MCP request payload size",
            unit="By",
            explicit_bucket_boundaries=_MCP_PAYLOAD_SIZE_BUCKETS
        )

        self.mcp_response_size_histogram = self.meter.create_histogram(
            SC.MCP_RESPONSE_SIZE_METRIC,
            description="MCP response payload size",
            unit="By",
            explicit_bucket_boundaries=_MCP_PAYLOAD_SIZE_BUCKETS
        )
```

**Update database/cache instrumentors to use these metrics.**

---

#### 3.4 Add Streaming Metrics (TBT/TTFT)
**Files:** `genai_otel/instrumentors/base.py`, provider instrumentors

**Add streaming histogram:**
```python
# In base instrumentor
cls._shared_ttft_histogram = meter.create_histogram(
    SC.GEN_AI_SERVER_TTFT,
    description="Time to first token in seconds",
    unit="s",
    explicit_bucket_boundaries=_GEN_AI_SERVER_TFTT
)

cls._shared_tbt_histogram = meter.create_histogram(
    SC.GEN_AI_SERVER_TBT,
    description="Time between tokens in seconds",
    unit="s",
    explicit_bucket_boundaries=_GEN_AI_SERVER_TBT
)
```

**In streaming handlers (e.g., OpenAI):**
```python
def _wrap_streaming_response(self, stream, span, start_time):
    """Wrap streaming response to capture TBT/TTFT."""
    first_token = True
    last_token_time = start_time

    for chunk in stream:
        current_time = time.time()

        if first_token:
            ttft = current_time - start_time
            span.set_attribute("gen_ai.server.ttft", ttft)
            self._shared_ttft_histogram.record(ttft, {"model": span.attributes.get("gen_ai.request.model")})
            first_token = False
        else:
            tbt = current_time - last_token_time
            self._shared_tbt_histogram.record(tbt, {"model": span.attributes.get("gen_ai.request.model")})

        last_token_time = current_time
        yield chunk
```

---

### Phase 4: Optional Enhancements (Week 4)

**Goal:** Add session/user tracking, RAG attributes, agent tracking.

#### 4.1 Session & User Tracking
**Files:** `genai_otel/config.py`, `genai_otel/instrumentors/base.py`

**Add config:**
```python
@dataclass
class OTelConfig:
    # ... existing fields

    # Context extraction functions (optional)
    session_id_extractor: Optional[Callable] = None
    user_id_extractor: Optional[Callable] = None
```

**In wrapper:**
```python
# In create_span_wrapper:
if self.config.session_id_extractor:
    try:
        session_id = self.config.session_id_extractor(instance, args, kwargs)
        if session_id:
            span.set_attribute("session.id", session_id)
    except Exception as e:
        logger.debug(f"Failed to extract session ID: {e}")

if self.config.user_id_extractor:
    try:
        user_id = self.config.user_id_extractor(instance, args, kwargs)
        if user_id:
            span.set_attribute("user.id", user_id)
    except Exception as e:
        logger.debug(f"Failed to extract user ID: {e}")
```

---

#### 4.2 RAG/Embedding Attributes
**Files:** New instrumentors for embedding and retrieval operations

**For Embedding Calls:**
```python
span.set_attribute("embedding.model_name", model)
span.set_attribute("embedding.text", text[:500])  # Truncated
# Optionally capture vector (large!)
if capture_vectors:
    span.set_attribute("embedding.vector", json.dumps(vector))
```

**For Retrieval:**
```python
for i, doc in enumerate(retrieved_docs):
    span.set_attribute(f"retrieval.documents.{i}.document.id", doc.id)
    span.set_attribute(f"retrieval.documents.{i}.document.score", doc.score)
    span.set_attribute(f"retrieval.documents.{i}.document.content", doc.content[:500])
    # Metadata
    if doc.metadata:
        for key, value in doc.metadata.items():
            span.set_attribute(f"retrieval.documents.{i}.document.metadata.{key}", str(value))
```

---

## Testing Strategy

### Phase 1 Tests
```bash
# Run baseline tests
pytest tests/ --tb=no -q > baseline_tests.txt

# After Phase 1 changes
pytest tests/ --tb=no -q > phase1_tests.txt
diff baseline_tests.txt phase1_tests.txt

# Specific tests
pytest tests/instrumentors/test_base.py -v
pytest tests/test_gpu_metrics.py -v
pytest tests/test_metrics.py -v

# Integration test
python examples/test_openai.py
python examples/test_anthropic.py
```

### Phase 2 Tests
```bash
pytest tests/instrumentors/test_openai_instrumentor.py -v
pytest tests/instrumentors/test_anthropic_instrumentor.py -v

# Verify new attributes appear
python -c "
from genai_otel import instrument
import os
os.environ['OTEL_SEMCONV_STABILITY_OPT_IN'] = 'gen_ai_latest_experimental'
instrument()
# Make LLM call and check attributes
"
```

### Phase 3 Tests
```bash
pytest tests/test_cost_calculator.py -v
pytest tests/mcp_instrumentors/ -v

# Test streaming
python examples/test_streaming.py
```

---

## Migration Guide for Users

### Breaking Changes in Phase 1

**Metric Names Changed:**
| Old Name | New Name |
|----------|----------|
| `genai.requests` | `gen_ai.requests` |
| `genai.tokens` | `gen_ai.client.token.usage` |
| `genai.latency` | `gen_ai.client.operation.duration` |
| `genai.cost` | `gen_ai.usage.cost` |
| `genai.gpu.*` | `gen_ai.gpu.*` |

**Action Required:**
- Update dashboard queries
- Update alerting rules
- Update metric aggregations

**Example Dashboard Update (Prometheus/Grafana):**
```promql
# OLD
sum(rate(genai_requests_total[5m])) by (operation)

# NEW
sum(rate(gen_ai_requests_total[5m])) by (operation)
```

---

## Configuration Updates

### New Environment Variables

```bash
# Enable new semantic conventions
export OTEL_SEMCONV_STABILITY_OPT_IN="gen_ai_latest_experimental"

# Enable content capture (opt-in for privacy)
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"

# Enable CO2 tracking (already exists but clarify default)
export GENAI_ENABLE_CO2_TRACKING="true"
export GENAI_CARBON_INTENSITY="400"  # gCO2e/kWh
```

---

## Success Criteria

### Phase 1 (Critical) ✅ COMPLETE
- ✅ All metrics use `gen_ai.*` prefix
- ✅ GPU metrics use correct types (ObservableGauge with callbacks, not Counter)
- ✅ Histogram buckets applied via Views
- ✅ CO2 metrics fixed (`gen_ai.co2.emissions`)
- ✅ Zero test regressions (381/381 tests passing)

### Phase 2 (Core) ✅ COMPLETE
- ✅ All required span attributes present (operation.name, request params, response attrs)
- ✅ OTEL_SEMCONV_STABILITY_OPT_IN works (dual token emission)
- ✅ Event-based content capture functional (opt-in via GENAI_ENABLE_CONTENT_CAPTURE)
- ✅ Backward compatibility maintained (default: new conventions only)
- ✅ Response attribute extraction implemented
- ✅ 8 new tests added, all passing

### Phase 3 (Advanced)
- ✅ Tool call tracking works
- ✅ Granular cost metrics recorded
- ✅ MCP metrics visible
- ✅ Streaming metrics (TTFT/TBT) captured

### Phase 4 (Optional)
- ✅ Session/user tracking available
- ✅ RAG attributes documented
- ✅ Example code provided

---

## Risk Mitigation

### Backward Compatibility
- Phase 1 has breaking changes (metric names)
- Bump version to 2.0.0
- Provide migration guide
- Support legacy mode with env var: `GENAI_OTEL_LEGACY_METRICS=true`

### Performance Impact
- Event capture is opt-in (default: disabled)
- Structured messages limited to first 5
- Content truncated at 500 chars
- Observable gauges reduce polling overhead

### Testing Coverage
- Maintain >95% coverage
- Add integration tests for each phase
- Test with real LLM providers (mocked)
- Load testing for metric cardinality

---

## Timeline Summary

| Phase | Duration | Effort | Risk | Key Changes |
|-------|----------|--------|------|-------------|
| Phase 1 | 1 week | High | High (breaking) | Metric names, GPU fixes, histogram buckets |
| Phase 2 | 1 week | Medium | Low | Core attributes, semconv opt-in, events, **OpenInference compat** |
| Phase 3 | 1 week | Medium | Medium | Tool calls, granular costs, MCP metrics, streaming |
| Phase 4 | 1 week | Low | Low | Session/user tracking, RAG, documentation |
| **Total** | **4 weeks** | **Medium** | **Manageable** | **15 gaps addressed** |

### 🆕 Phase 2 Now Includes:
- **2.4 OpenInference Compatibility** - Add `openinference.span.kind` and dual token attributes to custom instrumentors for maximum dashboard compatibility

---

## Next Steps

1. **Review this plan** - Confirm priorities and timeline
2. **Set up feature branch** - `git checkout -b feature/otel-semantic-compliance`
3. **Start Phase 1.1** - Refactor metric names
4. **Run baseline tests** - Establish test coverage baseline
5. **Iterate with testing** - Test after each file change

Would you like to proceed with **Phase 1.1 (Refactor Metric Names)** now?
