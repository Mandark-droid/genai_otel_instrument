# Comprehensive Field Analysis - All Missing Fields

This document lists ALL fields found in actual spans that are NOT currently extracted by the OpenSearch setup.

## Analysis Sources
- **crewai-example service**: 18 spans analyzed (36 unique tag keys)
- **genai-test-app service**: 17 spans analyzed (35 unique tag keys)
- **autogen-example service**: 17 spans analyzed (23 unique tag keys)

## Missing Fields by Category

### 1. OpenTelemetry Core Fields
- **`otel.scope.name`** (string) - Instrumentation scope name
  - Examples: "crewai.telemetry", "genai_otel.instrumentors.base"
  - MUST ADD - Critical for understanding which instrumentation library created the span

- **`otel.scope.version`** (string) - Version of the instrumentation scope
  - SHOULD ADD - Useful for debugging instrumentation issues

### 2. CrewAI-Specific Fields

**Crew Metadata:**
- **`crewai_version`** (string) - CrewAI library version (e.g., "1.8.0")
- **`python_version`** (string) - Python runtime version (e.g., "3.12.6")
- **`crew_key`** (string) - Unique crew identifier
- **`crew_id`** (string) - Crew instance ID
- **`crew_fingerprint`** (string) - Crew configuration fingerprint
- **`crew_process`** (string) - Execution process type (e.g., "sequential", "hierarchical")
- **`crew_memory`** (boolean) - Whether crew memory is enabled
- **`crew_number_of_tasks`** (integer) - Number of tasks in crew
- **`crew_number_of_agents`** (integer) - Number of agents in crew
- **`crew_fingerprint_created_at`** (string) - Timestamp of crew fingerprint creation
- **`crew_agents`** (text) - JSON array of agent configurations
- **`crew_tasks`** (text) - JSON array of task configurations

**Task Metadata:**
- **`task_key`** (string) - Unique task identifier
- **`task_id`** (string) - Task instance ID
- **`task_fingerprint`** (string) - Task configuration fingerprint
- **`task_fingerprint_created_at`** (string) - Timestamp of task fingerprint creation

**Agent Metadata:**
- **`agent_fingerprint`** (string) - Agent configuration fingerprint
- **`agent_role`** (string) - Agent role/name (e.g., "Research Specialist")

**Recommendation:** MUST ADD - Essential for CrewAI observability and debugging

### 3. OpenInference Semantic Conventions

- **`openinference.span.kind`** (string) - OpenInference span type
  - Examples: "TOOL", "AGENT", "CHAIN", "LLM"
  - MUST ADD - Required for proper OpenInference integration

- **`input.value`** (text) - Input to the operation
  - SHOULD ADD - Critical for debugging

- **`output.value`** (text) - Output from the operation
  - SHOULD ADD - Critical for debugging

- **`output.mime_type`** (string) - MIME type of output
  - OPTIONAL - Useful for understanding output format

### 4. Tool/MCP Fields

- **`tool.name`** (string) - Name of the tool being invoked
- **`tool.description`** (text) - Description of the tool
- **`tool.parameters`** (text) - JSON parameters passed to tool

**Recommendation:** MUST ADD - Essential for MCP and tool observability

### 5. Smolagents-Specific Fields

- **`smolagents.max_steps`** (integer) - Maximum steps for agent execution
- **`smolagents.tools_names`** (string) - Comma-separated list of available tools

**Recommendation:** SHOULD ADD - Useful for Smolagents debugging

### 6. Alternative Token Count Fields

Some instrumentors use different field names for token counts:

- **`llm.token_count.prompt`** (integer) - Alternative to gen_ai.usage.prompt_tokens
- **`llm.token_count.completion`** (integer) - Alternative to gen_ai.usage.completion_tokens
- **`llm.token_count.total`** (integer) - Alternative to gen_ai.usage.total_tokens

**Recommendation:** SHOULD ADD - Provides compatibility with different instrumentation libraries

### 7. LLM Function/Tool Calling (CRITICAL for AutoGen)

- **`llm.tools`** (text) - JSON array of tool/function definitions available to the LLM
  - Contains function names, descriptions, and parameter schemas
  - Example: `[{"type": "function", "function": {"description": "Calculate ROI", "name": "calculate_roi", ...}}]`
  - MUST ADD - Essential for tracking what tools/functions are available

- **`llm.output_messages.0.message.tool_calls.0.tool_call.id`** (keyword) - Tool call ID
  - Unique identifier for each tool invocation
  - Example: "call_nqnIMLNxQZrLyleweLxCJBfn"
  - MUST ADD - Required for tracking tool execution

- **`llm.output_messages.0.message.tool_calls.0.tool_call.function.name`** (keyword) - Function name
  - Name of the function being called by the LLM
  - Example: "calculate_roi"
  - MUST ADD - Critical for function call observability

- **`llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments`** (text) - Function arguments
  - JSON-encoded arguments passed to the function
  - Example: `{"investment": 10000, "return_value": 12500}`
  - MUST ADD - Essential for debugging tool calls

**Note:** The tool call fields use array indices (`.0.`) which indicates they're from the first message/tool call. For comprehensive coverage, we should extract the first tool call's details. Multiple tool calls in a single response would require special handling.

**Recommendation:** MUST ADD - Critical for AutoGen and function-calling LLM observability

## Fields Already Added in v2.2

The following fields were identified as missing but have been added in v2.2:
- ✅ `gen_ai.response` - Full response text
- ✅ `gen_ai.streaming.token_count` - Streaming token count
- ✅ `gen_ai.server.ttft` - Time to first token
- ✅ `gen_ai.request.max_tokens` - Maximum tokens parameter
- ✅ `gen_ai.request.temperature` - Temperature parameter
- ✅ `gen_ai.request.top_p` - Top-p parameter
- ✅ `session.id` - Session identifier
- ✅ `user.id` - User identifier
- ✅ `gen_ai.usage.input_tokens` - Token alias
- ✅ `gen_ai.usage.output_tokens` - Token alias

## Priority Summary

### CRITICAL (Must Add in v2.3):
1. `otel.scope.name` - Core OTel field
2. CrewAI fields (all 18 fields) - Complete CrewAI observability
3. `openinference.span.kind` - OpenInference compatibility
4. Tool fields (`tool.name`, `tool.description`, `tool.parameters`) - MCP observability
5. `input.value`, `output.value` - Critical for debugging

### HIGH Priority (Should Add in v2.3):
6. `otel.scope.version` - Instrumentation debugging
7. Smolagents fields - Framework-specific observability
8. Alternative token count fields (`llm.token_count.*`) - Library compatibility
9. `output.mime_type` - Output format understanding

### Summary Statistics:
- **Total unique tag keys found**: 94 (36 from crewai-example + 35 from genai-test-app + 23 from autogen-example)
- **Currently extracted in v2.5**: ~126 fields
- **ALL discovered fields are now extracted**: Complete coverage achieved!

## Version History

**Version 2.5 (COMPLETED):**
- **CRITICAL BUG FIX**: span_status now checks otel_status_code first
  - Spans with otel_status_code="ERROR" now correctly get span_status="ERROR" (not "OK")
- LangGraph stateful workflow fields (12)
  - `langgraph.node_count`, `langgraph.nodes`, `langgraph.edge_count`
  - `langgraph.channels`, `langgraph.channel_count`
  - `langgraph.input.keys`, `langgraph.output.keys`
  - `langgraph.thread_id`, `langgraph.checkpoint_id`, `langgraph.recursion_limit`
  - `langgraph.message_count`, `langgraph.steps`
- Pydantic AI agent framework fields (15)
  - `pydantic_ai.agent.name`, `pydantic_ai.model.name`, `pydantic_ai.model.provider`
  - `pydantic_ai.system_prompts`, `pydantic_ai.tools`, `pydantic_ai.tools.count`
  - `pydantic_ai.result_type`, `pydantic_ai.user_prompt`
  - `pydantic_ai.message_history.count`, `pydantic_ai.result.data`
  - `pydantic_ai.result.messages.count`, `pydantic_ai.result.last_message`
  - `pydantic_ai.result.last_role`, `pydantic_ai.result.timestamp`, `pydantic_ai.result.cost`

Total new fields in v2.5: **27 fields + 1 critical bug fix** (bringing total to ~126 fields)

**Version 2.4 (COMPLETED):**
- LLM function/tool calling fields (4)
  - `llm.tools` - JSON array of tool/function definitions
  - `llm.output_messages.0.message.tool_calls.0.tool_call.id` - Tool call ID
  - `llm.output_messages.0.message.tool_calls.0.tool_call.function.name` - Function name
  - `llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments` - Function arguments

Total new fields in v2.4: **4 fields** (bringing total to ~99 fields)

**Version 2.3 (COMPLETED):**
- All CrewAI fields (18)
- All OpenInference fields (4)
- All Tool/MCP fields (3)
- otel.scope.name and otel.scope.version (2)
- Smolagents fields (2)
- Alternative token count fields (3)

Total fields in v2.3: **32 fields** (bringing total to ~95 fields)

## Implementation Status

✅ **ALL FIELDS IMPLEMENTED** - As of v2.5, all discovered fields from comprehensive analysis are now extracted and indexed in OpenSearch.

**Coverage by Framework:**
- ✅ GenAI Semantic Conventions - Complete
- ✅ CrewAI - Complete (all 18 fields)
- ✅ OpenInference - Complete (all 4 fields)
- ✅ Tool/MCP - Complete (all 3 fields)
- ✅ Smolagents - Complete (all 2 fields)
- ✅ AutoGen Function Calling - Complete (all 4 fields)
- ✅ LangGraph Stateful Workflows - Complete (all 12 fields)
- ✅ Pydantic AI Agents - Complete (all 15 fields)
- ✅ OTel Core - Complete (scope name/version)
- ✅ Evaluation Metrics - Complete (PII, Toxicity, Bias, Hallucination, Prompt Injection)

**Total Field Count:** ~126 fields extracted from spans

**Script Location:** `examples/demo/opensearch-setup.sh` (v2.5)

**Critical Bug Fix in v2.5:** span_status calculation now prioritizes otel_status_code
