# AI Framework Research for OpenTelemetry Instrumentation

**Report Date:** November 13, 2025
**Project:** TraceVerde - GenAI OpenTelemetry Instrumentation
**Objective:** Research popular AI frameworks and SDKs for potential instrumentation support

---

## Executive Summary

This report analyzes popular AI agent frameworks and SDKs that could be added to TraceVerde's instrumentation coverage. We identified **8 high-priority frameworks** across three categories:

1. **Multi-Agent Frameworks** (4): CrewAI, AutoGen, LangGraph, Pydantic AI
2. **Enhanced SDK Support** (3): OpenAI Agents SDK, Google GenAI SDK, AWS Bedrock AgentCore SDK
3. **Additional Frameworks** (2): Haystack, Semantic Kernel

Our research shows that while TraceVerde already supports **20+ LLM providers** and **2 frameworks** (LangChain, LlamaIndex), there's significant opportunity to expand into the rapidly growing multi-agent orchestration space.

**Key Finding:** Multi-agent frameworks (CrewAI, AutoGen, LangGraph) have collectively garnered **90,000+ GitHub stars** and millions of monthly downloads, indicating strong market demand.

---

## Current State: What TraceVerde Already Supports

### ✅ Currently Instrumented (20+ Providers)

**LLM Providers with Full Cost Tracking:**
- **OpenAI** (GPT-3.5, GPT-4, o1, etc.) - **Already supported**
- **Anthropic** (Claude with caching support)
- **Google AI** (Gemini models) - **Partial support via google-ai-generativelanguage**
- **AWS Bedrock** (Multiple models) - **Basic support via boto3**
- **Azure OpenAI** (GPT models via Azure)
- **Mistral AI, Cohere, Together AI, Groq, Ollama, Vertex AI**
- **SambaNova, Hyperbolic** (recently added in v0.1.23)

**Hardware/Local:**
- Replicate, HuggingFace Transformers, Anyscale

**Frameworks:**
- **LangChain** (chains, agents, tools)
- **LlamaIndex** (query engines, indices)

**OpenInference (Python 3.10+):**
- Smolagents, LiteLLM, MCP

### 🔧 MCP Tool Instrumentation (Already Supported)
- Databases: PostgreSQL, MySQL, MongoDB, SQLAlchemy
- Caching: Redis
- Message Queues: Apache Kafka
- Vector DBs: Pinecone, Weaviate, Qdrant, ChromaDB, Milvus, FAISS
- APIs: HTTP/REST (requests, httpx)

---

## Framework Analysis

### 1. CrewAI - Multi-Agent Collaboration Framework

**Overview:**
- **GitHub Stars:** 30,000+
- **Monthly Downloads:** ~1 million
- **Python Version:** 3.10+
- **License:** MIT
- **Status:** Active development, 100,000+ certified developers

**Architecture:**
```
CrewAI Components:
├── Agents (role-based AI entities with goals, tools, backstory)
├── Tasks (specific assignments for agents)
├── Crews (collections of agents working together)
├── Processes (sequential, hierarchical, consensual)
└── Tools (functions agents use to interact with systems)
```

**Key Features:**
- **Role-Based Agents:** Each agent has specific expertise (researcher, writer, analyst)
- **Collaborative Intelligence:** Agents work together with output chaining
- **Two Orchestration Modes:**
  - **Crews:** Autonomous multi-agent teams with shared goals
  - **Flows:** Event-driven control for single LLM calls, supports Crews natively
- **Process Types:**
  - Sequential: Tasks execute in order, output feeds next task
  - Hierarchical: Manager agent delegates and validates
  - Consensual: (planned) Agents vote on decisions

**What to Instrument:**
```python
# Key instrumentation points:
1. Crew execution (crew.kickoff())
2. Individual agent task execution
3. Inter-agent communication (handoffs)
4. Tool calls made by agents
5. Process orchestration (sequential/hierarchical flows)
6. LLM calls within each agent (may already be captured)
```

**Instrumentation Complexity:** 🟡 **Medium**
- Already built from scratch (not based on LangChain)
- Well-defined abstraction layers
- Clear execution lifecycle with kickoff() entry point

**Cost Tracking:** ✅ Can leverage existing LLM provider instrumentors
- CrewAI agents use underlying LLM APIs (OpenAI, Anthropic, etc.)
- Need to aggregate costs across multi-agent workflows
- Track per-agent costs for analysis

**Recommended Attributes:**
```yaml
Span Attributes:
  - crewai.crew.id: Unique crew identifier
  - crewai.crew.name: Crew name
  - crewai.agent.role: Agent's role (researcher, writer, etc.)
  - crewai.agent.goal: Agent's assigned goal
  - crewai.task.id: Task identifier
  - crewai.task.description: Task description
  - crewai.process.type: sequential | hierarchical | consensual
  - crewai.handoff.from_agent: Source agent in handoff
  - crewai.handoff.to_agent: Target agent in handoff
  - crewai.tool.name: Tool used by agent

Metrics:
  - crewai.crew.duration: Crew execution time
  - crewai.task.duration: Individual task duration
  - crewai.agent.tasks_completed: Tasks completed per agent
  - crewai.crew.cost: Aggregated cost for entire crew execution
```

**Priority:** 🔴 **HIGH** - Very popular, clear use case, growing community

---

### 2. Microsoft AutoGen - Multi-Agent Conversation Framework

**Overview:**
- **GitHub Stars:** 51,600+ (highest in category)
- **Python Version:** 3.10+
- **License:** MIT (maintained by Microsoft)
- **Status:** ⚠️ **Entering maintenance mode** - Merging with Semantic Kernel into "Microsoft Agent Framework" (October 2025)

**Architecture:**
```
AutoGen Components:
├── ConversableAgent (base class for all agents)
│   ├── AssistantAgent (AI assistant using LLMs)
│   └── UserProxyAgent (proxy for humans, can execute code)
├── Group Chat (multi-agent orchestration)
│   └── GroupChatManager (orchestrates agent selection)
└── Conversation Patterns
    ├── Two-Agent Chat
    ├── Sequential Chat (chained conversations)
    ├── Group Chat (multi-agent collaboration)
    └── Nested Chat (complex workflow packaging)
```

**Key Features:**
- **Conversable Agents:** Generic agent abstraction for LLM + tool + human interaction
- **Dynamic Agent Selection:** GroupChatManager selects next speaker using:
  - `round_robin`: Fixed rotation
  - `random`: Random selection
  - `manual`: Human selection
  - `auto`: LLM-based selection (default)
- **Event-Driven Programming:** Scalable multi-agent systems
- **Cross-Language Support:** Python and .NET

**Migration Note:**
AutoGen is being unified with Semantic Kernel into **Microsoft Agent Framework** (public preview Oct 2025). Future development will focus on the unified platform.

**What to Instrument:**
```python
# Key instrumentation points:
1. Agent initialization and registration
2. Conversable agent message exchanges
3. Group chat orchestration (agent selection logic)
4. Sequential chat chains
5. Nested chat execution
6. Code execution by UserProxyAgent
7. LLM calls within agents (may already be captured)
```

**Instrumentation Complexity:** 🟡 **Medium-High**
- Well-documented message exchange protocol
- Multiple conversation patterns to handle
- GroupChatManager logic adds complexity
- Migration to Agent Framework may simplify future maintenance

**Cost Tracking:** ✅ Can leverage existing LLM provider instrumentors
- Aggregate costs across multi-agent conversations
- Track per-agent conversation costs
- Important for group chats with dynamic agent selection

**Recommended Attributes:**
```yaml
Span Attributes:
  - autogen.agent.name: Agent name
  - autogen.agent.type: AssistantAgent | UserProxyAgent | Custom
  - autogen.conversation.id: Conversation identifier
  - autogen.conversation.pattern: two_agent | sequential | group_chat | nested
  - autogen.group_chat.selection_mode: round_robin | random | manual | auto
  - autogen.message.sender: Sender agent name
  - autogen.message.recipient: Recipient agent name
  - autogen.message.index: Message index in conversation
  - autogen.code_execution.result: Code execution result (if applicable)

Metrics:
  - autogen.conversation.duration: Conversation duration
  - autogen.conversation.messages: Message count
  - autogen.agent.messages_sent: Messages sent per agent
  - autogen.group_chat.agent_selections: Agent selection counts
  - autogen.conversation.cost: Total conversation cost
```

**Priority:** 🟠 **MEDIUM-HIGH** - Popular but entering maintenance mode. Monitor Microsoft Agent Framework migration.

**Note:** Consider instrumenting the new **Microsoft Agent Framework** instead once it reaches GA (expected Q1 2025).

---

### 3. LangGraph - Stateful Multi-Agent Workflows

**Overview:**
- **GitHub Stars:** 11,700+ (13,900+ in some sources)
- **Monthly Downloads:** 4.2 million
- **License:** MIT (open-source)
- **Managed Platform:** LangGraph Platform available
- **Company:** LangChain Inc.

**Architecture:**
```
LangGraph Components:
├── State (shared data structure, current application snapshot)
├── Nodes (agents in the graph)
├── Edges (connections between agents, control flow)
├── Checkpoints (persistence layer for state)
└── Workflows (directed graphs modeling agent behavior)
```

**Key Features:**
- **Graph-Based Orchestration:** Agents are nodes, communication is edges
- **State Machines:** Core CS concept applied to AI agents
- **Persistence Layer:**
  - **Memory:** Persists arbitrary application state
  - **Human-in-the-Loop:** Interrupt and resume execution
  - **Time-Travel Debugging:** Step through execution history
- **Flexible Control Flows:** Single agent, multi-agent, hierarchical, sequential
- **Production-Ready:** Powers Uber & LinkedIn's AI agents
- **Fault Tolerance:** Robust error handling and recovery

**What to Instrument:**
```python
# Key instrumentation points:
1. Graph compilation and initialization
2. Node execution (individual agent operations)
3. Edge transitions (control flow between agents)
4. State updates (mutations to shared state)
5. Checkpoint creation and restoration
6. Human-in-the-loop interruptions
7. Error handling and retries
8. LLM calls within nodes (may already be captured)
```

**Instrumentation Complexity:** 🔴 **High**
- Graph-based architecture requires careful span modeling
- State management tracking adds complexity
- Checkpoint persistence needs special handling
- Human-in-the-loop introduces async complexity

**Cost Tracking:** ✅ Can leverage existing LLM provider instrumentors
- Track costs across graph execution
- Per-node cost attribution
- Important for long-running workflows with retries

**Recommended Attributes:**
```yaml
Span Attributes:
  - langgraph.graph.id: Graph identifier
  - langgraph.graph.name: Graph name
  - langgraph.node.id: Node identifier
  - langgraph.node.name: Node name (agent name)
  - langgraph.edge.from: Source node
  - langgraph.edge.to: Target node
  - langgraph.state.key: State key being updated
  - langgraph.checkpoint.id: Checkpoint identifier
  - langgraph.human_in_loop: Whether execution was interrupted
  - langgraph.execution.status: running | interrupted | completed | failed

Metrics:
  - langgraph.graph.duration: Total graph execution time
  - langgraph.node.duration: Individual node execution time
  - langgraph.state.updates: State update count
  - langgraph.checkpoints.created: Checkpoint creation count
  - langgraph.graph.cost: Total graph execution cost
  - langgraph.node.cost: Per-node cost
```

**Priority:** 🔴 **HIGH** - Production-proven, high download count, unique architecture

**Note:** LangGraph is part of the LangChain ecosystem. Since LangChain is already instrumented, some tracing may already exist via LangChain instrumentation.

---

### 4. OpenAI Agents SDK - Production-Ready Agent Framework

**Overview:**
- **Release Date:** March 11, 2025
- **Status:** Production-ready upgrade of Swarm (experimental framework)
- **Languages:** Python, JavaScript/TypeScript
- **Maintainer:** OpenAI

**Architecture:**
```
OpenAI Agents SDK Primitives:
├── Agents (LLMs with instructions and tools)
├── Handoffs (task delegation between agents)
├── Guardrails (input/output validation)
├── Sessions (automatic conversation history management)
└── Tracing (built-in visualization and debugging)
```

**Key Features:**
- **Agents:** LLMs equipped with instructions and tools for specific tasks
- **Handoffs:** Seamless task delegation based on agent capabilities
  - Agents transfer responsibility when query is outside scope
  - Better-suited agents handle specialized tasks
- **Guardrails:** Configurable safety checks
  - Input validation
  - Output validation
  - Real-time safety enforcement
  - Critical for enterprise security/compliance
- **Sessions:** Automatic conversation history across agent runs
  - Eliminates manual state handling
  - Transparent context management
- **Built-in Tracing:** Visualize and debug agentic flows
- **Evaluation:** Built-in eval tools
- **Fine-tuning:** Model fine-tuning support

**Gap Analysis:**
- **OpenAI SDK Already Supported:** TraceVerde instruments OpenAI's base SDK
- **Agents SDK is NEW Layer:** Higher-level abstractions on top of base SDK
- **What's Missing:** Agent-specific orchestration, handoffs, sessions

**What to Instrument:**
```python
# Key instrumentation points:
1. Agent initialization and configuration
2. Agent execution and tool calls
3. Handoff events (agent → agent transfers)
4. Session creation and management
5. Guardrail validation (input/output checks)
6. Multi-agent workflow orchestration
7. Base OpenAI calls already captured by existing instrumentor
```

**Instrumentation Complexity:** 🟢 **Low-Medium**
- Built on top of OpenAI SDK (already instrumented)
- Clean abstraction with 4 core primitives
- Lightweight framework design
- Well-documented API

**Cost Tracking:** ✅ Already captured via existing OpenAI instrumentor
- Need to aggregate costs across agent sessions
- Track per-agent costs in multi-agent workflows
- Session-level cost attribution

**Recommended Attributes:**
```yaml
Span Attributes:
  - openai.agent.id: Agent identifier
  - openai.agent.name: Agent name
  - openai.agent.instructions: Agent system instructions (truncated)
  - openai.agent.tools: Available tools (array)
  - openai.handoff.from_agent: Source agent
  - openai.handoff.to_agent: Target agent
  - openai.handoff.reason: Handoff reason/context
  - openai.session.id: Session identifier
  - openai.session.message_count: Messages in session
  - openai.guardrail.input_validated: Input validation result
  - openai.guardrail.output_validated: Output validation result
  - openai.guardrail.violation: Guardrail violation details

Metrics:
  - openai.agent.executions: Agent execution count
  - openai.agent.duration: Agent execution time
  - openai.handoffs: Handoff count
  - openai.session.duration: Session duration
  - openai.guardrail.violations: Guardrail violation count
  - openai.agent.cost: Per-agent cost in session
```

**Priority:** 🔴 **HIGH** - Official OpenAI framework, production-ready, aligns with roadmap's guardrail features

**Synergy with Roadmap:** TraceVerde v0.2.0 roadmap includes guardrails and safety features. OpenAI Agents SDK's built-in guardrails provide a reference implementation.

---

### 5. Google GenAI SDK - Unified Google AI Platform

**Overview:**
- **Package:** `google-genai` (PyPI)
- **Status:** General Availability (GA) as of May 2025
- **Latest Release:** November 12, 2025
- **Python Version:** 3.9+
- **Deprecated:** Old `generative-ai-python` SDK (support ends Nov 30, 2025)

**Architecture:**
```
Google GenAI SDK:
├── Gemini Developer API (API key auth)
├── Vertex AI API (project/location auth)
├── Models (Gemini 2.0, Gemini 1.5 Pro/Flash)
└── Features (multi-modal, streaming, embeddings)
```

**Key Features:**
- **Unified SDK:** Single interface for Gemini Developer API + Vertex AI
- **Stable & Production-Ready:** GA status, fully supported
- **Latest Features:** Access to newest Gemini capabilities
- **Best Performance:** Optimized for Gemini models
- **Multi-Modal:** Text, image, video, audio support

**Gap Analysis:**
- **TraceVerde Currently Has:** `google_ai_instrumentor.py` (87 lines)
- **Likely Targets:** Old `google-generativelanguage` SDK
- **Migration Needed:** Support new `google-genai` SDK

**What to Update:**
```python
# Current instrumentation (google_ai_instrumentor.py):
- Likely instruments: google.generativeai (old SDK)
- Need to verify: Does it work with google-genai (new SDK)?

# If migration needed:
1. Update imports: from google import genai
2. Instrument genai.Client initialization
3. Instrument genai.models.generate_content()
4. Instrument genai.models.generate_content_stream() (streaming)
5. Instrument embeddings API
6. Handle both Gemini Developer API and Vertex AI modes
```

**Instrumentation Complexity:** 🟢 **Low**
- Similar API patterns to existing Google AI instrumentor
- Well-documented SDK with stable API
- May be mostly a migration/update task

**Cost Tracking:** ✅ Already in llm_pricing.json
- Gemini 1.5/2.0 Pro/Flash pricing exists
- Verify coverage of all new Gemini 2.0 models

**Recommended Action:**
- **Audit existing `google_ai_instrumentor.py`**
- Verify if it works with new `google-genai` SDK
- Update if targeting deprecated SDK
- Ensure Gemini 2.0 models are covered

**Priority:** 🟠 **MEDIUM-HIGH** - Important migration to avoid deprecated SDK, but may be quick update

---

### 6. AWS Bedrock AgentCore SDK - Production-Ready Agent Infrastructure

**Overview:**
- **Repository:** `aws/bedrock-agentcore-sdk-python`
- **Purpose:** Transform any AI agent into production-ready application
- **Architecture:** Framework-agnostic primitives (runtime, memory, auth, tools)
- **Infrastructure:** AWS-managed

**Architecture:**
```
AWS Bedrock Components:
├── bedrock-runtime (model invocation)
├── bedrock-agent (agent management)
├── bedrock-agent-runtime (agent execution)
└── bedrock-agentcore-sdk (production primitives)
```

**Key Features:**
- **Agent Runtime:** Invoke pre-configured Bedrock agents
- **Multi-Turn Conversations:** Session management
- **Knowledge Bases:** RAG with managed vector stores
- **Action Groups:** Custom tool/API integrations
- **Guardrails:** AWS Bedrock Guardrails integration
- **Framework-Agnostic:** Works with any agent framework

**Gap Analysis:**
- **TraceVerde Currently Has:** `aws_bedrock_instrumentor.py` (94 lines)
- **Likely Coverage:** Basic bedrock-runtime API (model invocation)
- **Potentially Missing:**
  - bedrock-agent-runtime (agent execution)
  - AgentCore SDK primitives
  - Bedrock Guardrails instrumentation

**What to Add:**
```python
# Extend existing aws_bedrock_instrumentor.py:

1. Instrument bedrock-agent-runtime client:
   - invoke_agent() method
   - retrieve() for knowledge bases
   - retrieve_and_generate() for RAG

2. Instrument AgentCore SDK (if applicable):
   - Session management
   - Memory operations
   - Tool/action group execution

3. Instrument Guardrails:
   - Input validation spans
   - Output validation spans
   - Policy violation tracking
```

**Instrumentation Complexity:** 🟡 **Medium**
- Boto3-based (consistent with existing AWS instrumentors)
- Multiple client types (runtime, agent, agent-runtime)
- Guardrails add validation logic

**Cost Tracking:** ✅ Already in llm_pricing.json
- AWS Bedrock models covered (Titan, Claude, Llama, Mistral)
- May need pricing for:
  - Knowledge base storage
  - Guardrail API calls

**Recommended Action:**
- **Audit existing `aws_bedrock_instrumentor.py`**
- Verify coverage of bedrock-agent-runtime
- Add agent-specific attributes (agent ID, session ID, action groups)
- Integrate Bedrock Guardrails instrumentation

**Priority:** 🟠 **MEDIUM** - AWS ecosystem important, but basic Bedrock already covered

**Synergy with Roadmap:** Bedrock Guardrails align with TraceVerde v0.2.0 guardrail features

---

### 7. Pydantic AI - Type-Safe Agent Framework

**Overview:**
- **Repository:** `pydantic/pydantic-ai`
- **PyPI:** `pydantic-ai`
- **Latest Release:** November 12, 2025
- **Tagline:** "GenAI Agent Framework, the Pydantic way"
- **Goal:** "Bring that FastAPI feeling to GenAI app development"

**Architecture:**
```
Pydantic AI Components:
├── Type-Safe Agents (Pydantic validation)
├── Multi-Model Support (20+ providers)
├── MCP Integration (Model Context Protocol)
├── Agent2Agent Communication
├── Durable Agents (preserve progress across failures)
└── Logfire Integration (observability platform)
```

**Key Features:**
- **Type Safety:** IDE auto-completion and compile-time error checking
  - Move errors from runtime to write-time
  - Pydantic validation for inputs/outputs
- **Model Support:** Virtually every provider
  - OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, Perplexity
  - Azure AI Foundry, Amazon Bedrock, Google Vertex AI
  - Ollama, LiteLLM, Groq, OpenRouter, Together AI
- **MCP Integration:** Access to external tools and data
- **Agent2Agent:** Agent interoperability
- **Durable Agents:** Preserve progress across:
  - Transient API failures
  - Application errors
  - Application restarts
- **Pydantic Logfire Integration:**
  - **OpenTelemetry-based** observability platform
  - Real-time debugging
  - Evals-based performance monitoring
  - Behavior, tracing, and **cost tracking**

**Instrumentation Consideration:**
⚠️ **Pydantic AI already integrates with Pydantic Logfire (OpenTelemetry-based)**

- Logfire is Pydantic's general-purpose observability platform
- Built on OpenTelemetry standards
- Includes cost tracking out-of-the-box
- Question: Can TraceVerde coexist or enhance Logfire instrumentation?

**Gap Analysis:**
- **TraceVerde Coverage:** May already capture underlying LLM calls (OpenAI, Anthropic, etc.)
- **Missing:** Pydantic AI-specific abstractions
  - Agent configuration and lifecycle
  - Agent2Agent communication
  - Durable agent checkpointing
  - MCP tool calls (TraceVerde has MCP instrumentation, but need integration)

**What to Instrument:**
```python
# Key instrumentation points:
1. Agent initialization and configuration
2. Agent execution (pydantic_ai.Agent.run())
3. Agent2Agent communication
4. Durable agent state persistence/recovery
5. MCP tool invocations (may already be covered)
6. Underlying LLM calls (already covered by provider instrumentors)
7. Pydantic validation errors
```

**Instrumentation Complexity:** 🟡 **Medium**
- Clean API design (FastAPI-inspired)
- Type-safe abstractions well-defined
- Durable agent state tracking adds complexity
- Integration with existing Logfire instrumentation may be tricky

**Cost Tracking:** ✅ Can leverage existing provider instrumentors
- Logfire already includes cost tracking
- TraceVerde can provide complementary cost attribution
- Aggregate costs across durable agent runs

**Recommended Attributes:**
```yaml
Span Attributes:
  - pydantic_ai.agent.id: Agent identifier
  - pydantic_ai.agent.name: Agent name
  - pydantic_ai.agent.model: Model configuration
  - pydantic_ai.agent.system_prompt: System prompt (truncated)
  - pydantic_ai.agent.durable: Whether agent is durable
  - pydantic_ai.agent.checkpoint_id: Checkpoint identifier (if restored)
  - pydantic_ai.a2a.source_agent: Source agent in A2A communication
  - pydantic_ai.a2a.target_agent: Target agent in A2A communication
  - pydantic_ai.validation.error: Pydantic validation error

Metrics:
  - pydantic_ai.agent.runs: Agent execution count
  - pydantic_ai.agent.duration: Agent run duration
  - pydantic_ai.agent.checkpoints: Checkpoint count
  - pydantic_ai.validation.errors: Validation error count
  - pydantic_ai.a2a.messages: Agent2Agent message count
```

**Priority:** 🟠 **MEDIUM** - Growing adoption, type-safety focus, but Logfire integration may reduce need

**Note:** Consider partnership/integration with Pydantic Logfire rather than competing instrumentation

---

### 8. Haystack - Production RAG and Agent Framework

**Overview:**
- **Repository:** `deepset-ai/haystack`
- **PyPI:** `haystack-ai` (farm-haystack is legacy)
- **Purpose:** AI orchestration framework for production LLM applications
- **Focus:** RAG, question answering, semantic search, conversational agents

**Architecture:**
```
Haystack Components:
├── Components (modular building blocks)
│   ├── Retrievers (document retrieval)
│   ├── Generators (LLM text generation)
│   ├── Embedders (embedding generation)
│   └── Custom Components
├── Pipelines (directed multigraphs)
│   ├── Loops & Branches (complex logic)
│   └── AsyncPipeline (parallel execution)
└── Agents (tool-driven multi-step reasoning)
    └── Tools (individual modules for specific tasks)
```

**Key Features:**
- **Component-Based:** Modular, reusable building blocks
  - Models, vector DBs, file converters
  - Easy to create custom components
- **Flexible Pipelines:** Directed multigraphs
  - Loops and branches for complex workflows
  - AsyncPipeline for parallel execution
- **Advanced Retrieval:** Best suited for RAG use cases
  - Hybrid search (keyword + vector)
  - Re-ranking and filtering
- **Agent Framework:** Tool-driven structure
  - Dynamic tool selection by agent
  - Order of tool usage determined at runtime
  - Combine tool outputs intelligently

**What to Instrument:**
```python
# Key instrumentation points:
1. Pipeline execution (Pipeline.run())
2. AsyncPipeline parallel execution
3. Component execution (individual component runs)
4. Retriever operations (document retrieval)
5. Generator operations (LLM calls) - may already be captured
6. Agent execution and tool selection
7. Tool invocations by agents
8. Loop and branch decision points
```

**Instrumentation Complexity:** 🟡 **Medium**
- Well-defined component abstraction
- Pipeline graph structure needs careful span modeling
- AsyncPipeline adds concurrency complexity
- Agent tool selection logic to track

**Cost Tracking:** ✅ Can leverage existing LLM provider instrumentors
- Track costs across pipeline execution
- Per-component cost attribution
- Important for complex RAG pipelines with multiple LLM calls

**Recommended Attributes:**
```yaml
Span Attributes:
  - haystack.pipeline.id: Pipeline identifier
  - haystack.pipeline.name: Pipeline name
  - haystack.pipeline.async: Whether AsyncPipeline
  - haystack.component.type: Component type (Retriever, Generator, etc.)
  - haystack.component.name: Component name
  - haystack.retriever.query: Retrieval query
  - haystack.retriever.top_k: Number of documents retrieved
  - haystack.retriever.document_count: Actual documents returned
  - haystack.agent.id: Agent identifier
  - haystack.agent.tool.name: Tool name
  - haystack.agent.tool.selected: Whether tool was selected
  - haystack.loop.iteration: Loop iteration number
  - haystack.branch.taken: Branch taken in decision point

Metrics:
  - haystack.pipeline.duration: Pipeline execution time
  - haystack.pipeline.executions: Pipeline execution count
  - haystack.component.duration: Component execution time
  - haystack.retriever.documents_retrieved: Documents retrieved count
  - haystack.agent.tool_calls: Tool call count
  - haystack.pipeline.cost: Total pipeline cost
```

**Priority:** 🟠 **MEDIUM** - Strong RAG focus, production-proven, complements existing LangChain/LlamaIndex support

**Note:** Haystack's focus on RAG pipelines complements TraceVerde's existing RAG observability features (Phase 4 - Session and RAG tracking)

---

### 9. Semantic Kernel - Microsoft's Multi-Agent Framework

**Overview:**
- **Repository:** `microsoft/semantic-kernel`
- **Languages:** Python, .NET
- **Status:** GA expected Q1 2025
- **Future:** Merging with AutoGen into "Microsoft Agent Framework"

**Architecture:**
```
Semantic Kernel Components:
├── Agents (AI entities with specific capabilities)
├── Plugins (functions agents can use)
├── Orchestration Patterns
│   ├── Sequential (pipeline)
│   ├── Concurrent (parallel processing)
│   ├── GroupChat (multi-agent collaboration)
│   ├── Handoff (agent delegation)
│   └── Magnetic (TBD)
├── Planners (automated task planning)
└── Memory (state management)
```

**Key Features:**
- **Multi-Agent Orchestration:** Introduced May 2025
- **Unified Interface:** Same patterns across all orchestrations
- **Azure AI Foundry Integration:** Deploy to Azure AI
- **Production-Ready:** Transitioning to GA (Q1 2025)
- **Enterprise Focus:** Designed for production applications

**Orchestration Patterns:**
1. **Sequential:** Agents in pipeline, output feeds next agent
2. **Concurrent:** Agents work in parallel, results aggregated
3. **GroupChat:** Multi-agent collaboration with shared context
4. **Handoff:** Dynamic agent delegation based on capabilities
5. **Magnetic:** (Future pattern)

**Migration Note:**
Semantic Kernel + AutoGen → **Microsoft Agent Framework** (public preview Oct 2025)

**Gap Analysis:**
- Not currently instrumented by TraceVerde
- Significant overlap with AutoGen (both merging)
- May be better to wait for unified Microsoft Agent Framework

**What to Instrument:**
```python
# Key instrumentation points:
1. Agent initialization and configuration
2. Orchestration pattern execution (Sequential, Concurrent, etc.)
3. Plugin/function calls
4. Planning operations
5. Memory operations
6. GroupChat orchestration
7. Handoff events
8. Azure AI Foundry integration (if applicable)
9. LLM calls (may already be captured)
```

**Instrumentation Complexity:** 🟡 **Medium-High**
- Multiple orchestration patterns to handle
- Plugin system adds abstraction layer
- Planner logic tracking
- Azure integration complexity

**Cost Tracking:** ✅ Can leverage existing provider instrumentors
- Track costs across orchestration patterns
- Per-agent cost attribution
- Important for complex multi-agent workflows

**Recommended Attributes:**
```yaml
Span Attributes:
  - semantic_kernel.agent.id: Agent identifier
  - semantic_kernel.agent.name: Agent name
  - semantic_kernel.orchestration.pattern: sequential | concurrent | group_chat | handoff
  - semantic_kernel.plugin.name: Plugin name
  - semantic_kernel.plugin.function: Function name
  - semantic_kernel.planner.type: Planner type
  - semantic_kernel.planner.steps: Planned steps
  - semantic_kernel.memory.key: Memory key accessed
  - semantic_kernel.handoff.from_agent: Source agent
  - semantic_kernel.handoff.to_agent: Target agent

Metrics:
  - semantic_kernel.orchestration.duration: Orchestration duration
  - semantic_kernel.agent.executions: Agent execution count
  - semantic_kernel.plugin.calls: Plugin call count
  - semantic_kernel.planner.plans_created: Plan creation count
  - semantic_kernel.orchestration.cost: Total orchestration cost
```

**Priority:** 🟡 **LOW-MEDIUM** - Wait for Microsoft Agent Framework GA (Q1 2025) rather than instrumenting both AutoGen and Semantic Kernel separately

**Recommendation:** **Defer until Microsoft Agent Framework GA**, then instrument unified platform

---

## Implementation Recommendations

### Priority Ranking (High → Low)

#### 🔴 Tier 1: High Priority (Immediate Implementation)

1. **CrewAI** - 30K+ stars, 1M monthly downloads, unique role-based architecture
2. **LangGraph** - 11K+ stars, 4.2M monthly downloads, powers Uber/LinkedIn
3. **OpenAI Agents SDK** - Official OpenAI framework, production-ready, aligns with roadmap

**Rationale:**
- Highest market demand and adoption
- Clear, well-documented APIs
- Unique value propositions (CrewAI's roles, LangGraph's graphs, OpenAI's sessions)
- Complement existing LangChain/LlamaIndex instrumentation

#### 🟠 Tier 2: Medium-High Priority (Next 3-6 Months)

4. **Google GenAI SDK** - Audit/update existing instrumentor for new SDK
5. **AutoGen** - 51K+ stars but entering maintenance mode
6. **Pydantic AI** - Growing adoption, type-safety focus, consider Logfire integration

**Rationale:**
- Google SDK is likely a quick update/migration
- AutoGen still popular despite maintenance mode (users won't migrate immediately)
- Pydantic AI has unique positioning but Logfire integration may reduce need

#### 🟡 Tier 3: Medium Priority (Future Consideration)

7. **Haystack** - Strong RAG focus, complements existing frameworks
8. **AWS Bedrock AgentCore** - Audit/extend existing Bedrock instrumentor
9. **Microsoft Agent Framework** - Wait for GA (Q1 2025), then evaluate

**Rationale:**
- Haystack is solid but overlaps with LangChain/LlamaIndex
- Bedrock basic coverage exists, agents are incremental
- Agent Framework not yet GA, avoid instrumenting transitional frameworks

---

## Technical Implementation Considerations

### 1. Instrumentation Pattern Consistency

All new instrumentors should follow TraceVerde's established pattern:

```python
class CrewAIInstrumentor(BaseInstrumentor):
    """Instruments CrewAI multi-agent framework."""

    def __init__(self):
        super().__init__()
        self._original_kickoff = None

    def instrument(self, config: OTelConfig):
        """Instrument CrewAI's Crew execution."""
        if not self.is_available():
            return

        # Wrap Crew.kickoff() method
        from crewai import Crew
        Crew.kickoff = wrapt.FunctionWrapper(
            Crew.kickoff,
            self._instrument_kickoff
        )

    def _instrument_kickoff(self, wrapped, instance, args, kwargs):
        """Wrap crew execution with span."""
        return self.create_span_wrapper(
            span_name="crewai.crew.execution",
            extract_attributes=self._extract_crew_attributes
        )(wrapped)(instance, *args, **kwargs)

    def _extract_crew_attributes(self, instance, args, kwargs):
        """Extract crew-specific attributes."""
        return {
            "crewai.crew.id": getattr(instance, "id", None),
            "crewai.process.type": getattr(instance, "process", None),
            "crewai.agent_count": len(getattr(instance, "agents", [])),
            # ... more attributes
        }
```

### 2. Cost Aggregation for Multi-Agent Workflows

Multi-agent frameworks require **cost aggregation** across multiple LLM calls:

```python
class MultiAgentCostAggregator:
    """Aggregate costs across multi-agent workflows."""

    def __init__(self):
        self._workflow_costs = {}  # workflow_id -> cost_breakdown

    def record_agent_cost(self, workflow_id, agent_id, costs):
        """Record cost for individual agent."""
        if workflow_id not in self._workflow_costs:
            self._workflow_costs[workflow_id] = {"agents": {}, "total": 0}

        self._workflow_costs[workflow_id]["agents"][agent_id] = costs
        self._workflow_costs[workflow_id]["total"] += costs["total"]

    def get_workflow_cost(self, workflow_id):
        """Get aggregated cost for workflow."""
        return self._workflow_costs.get(workflow_id, {"total": 0})
```

**Implementation:**
- Use context propagation to track workflow IDs
- Aggregate costs in parent span
- Emit both per-agent and total costs as attributes

### 3. Span Hierarchy for Nested Orchestrations

Multi-agent frameworks often have nested execution:

```
Crew Execution (Root Span)
├── Agent 1: Researcher (Child Span)
│   ├── Task: Research Topic
│   └── LLM Call: OpenAI GPT-4 (Grandchild Span - already instrumented)
├── Agent 2: Writer (Child Span)
│   ├── Task: Write Article
│   └── LLM Call: Anthropic Claude (Grandchild Span - already instrumented)
└── Agent 3: Editor (Child Span)
    ├── Task: Edit Article
    └── LLM Call: OpenAI GPT-4 (Grandchild Span - already instrumented)
```

**Best Practice:**
- Framework-level span (crew, graph, conversation)
- Agent-level spans (individual agents)
- LLM calls automatically captured by existing instrumentors
- Use OpenTelemetry context propagation for hierarchy

### 4. Handling Framework-Specific Features

| Framework | Special Feature | Implementation Approach |
|-----------|----------------|------------------------|
| **CrewAI** | Sequential/Hierarchical processes | Track process type, manager agent |
| **AutoGen** | Dynamic agent selection | Log selection mode and selected agent |
| **LangGraph** | Checkpoints & time-travel | Record checkpoint IDs, state snapshots |
| **OpenAI Agents** | Guardrails | Validation results, violation events |
| **Pydantic AI** | Durable agents | Checkpoint IDs, recovery events |
| **Haystack** | AsyncPipeline | Parallel span execution, join points |

### 5. Testing Strategy

Each new instrumentor requires:

1. **Availability Tests:** Verify library detection
2. **Instrumentation Tests:** Confirm span creation
3. **Attribute Tests:** Validate attribute extraction
4. **Cost Tests:** Verify cost aggregation
5. **Integration Tests:** Test with real frameworks
6. **Streaming Tests:** If applicable (agent outputs)

**Example Test Structure:**
```python
def test_crewai_instrumentation(self):
    """Test CrewAI instrumentation."""
    from crewai import Agent, Task, Crew

    # Create test crew
    researcher = Agent(role="Researcher", goal="Research topics")
    task = Task(description="Research AI agents", agent=researcher)
    crew = Crew(agents=[researcher], tasks=[task])

    # Execute with instrumentation
    with self.trace_exporter() as exporter:
        crew.kickoff()

    # Verify spans
    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    crew_span = spans[0]
    assert crew_span.name == "crewai.crew.execution"
    assert "crewai.crew.id" in crew_span.attributes
```

---

## Estimated Development Effort

| Framework | Complexity | Estimated LOC | Dev Time | Priority |
|-----------|-----------|---------------|----------|----------|
| **CrewAI** | 🟡 Medium | 200-300 | 2-3 weeks | 🔴 High |
| **LangGraph** | 🔴 High | 300-400 | 3-4 weeks | 🔴 High |
| **OpenAI Agents SDK** | 🟢 Low-Medium | 150-250 | 1-2 weeks | 🔴 High |
| **Google GenAI SDK** | 🟢 Low | 50-100 (update) | 3-5 days | 🟠 Medium-High |
| **AutoGen** | 🟡 Medium-High | 250-350 | 2-3 weeks | 🟠 Medium-High |
| **Pydantic AI** | 🟡 Medium | 200-300 | 2-3 weeks | 🟠 Medium |
| **Haystack** | 🟡 Medium | 200-300 | 2-3 weeks | 🟡 Medium |
| **AWS Bedrock AgentCore** | 🟡 Medium | 100-150 (extend) | 1 week | 🟡 Medium |
| **Microsoft Agent Framework** | 🟡 Medium-High | 250-350 | 2-3 weeks | 🟡 Low-Medium (defer) |

**Total Estimated Effort (Tier 1 + Tier 2):** ~12-18 weeks for 6 frameworks

---

## Pricing Data Gaps

Review `llm_pricing.json` for coverage of models used by new frameworks:

### ✅ Already Covered
- OpenAI (GPT-3.5, GPT-4, o1, embeddings)
- Anthropic (Claude 3.5, Claude 3)
- Google (Gemini 1.5/2.0 Pro/Flash)
- AWS Bedrock (Titan, Claude, Llama, Mistral)
- Azure OpenAI (same as OpenAI)

### ❓ Verify Coverage
- **Gemini 2.0 Models:** Ensure all variants covered (Flash, Pro)
- **DeepSeek-R1:** Check if available (used in Together AI, Pydantic AI)
- **Grok Models:** Check if pricing available (X.AI models)
- **Perplexity Models:** Check if pricing available

### 🔧 Action Items
1. Audit llm_pricing.json against model lists from Pydantic AI documentation
2. Add missing models from popular providers (Grok, Perplexity, DeepSeek-R1)
3. Update Google Gemini pricing for 2.0 release

---

## Synergies with TraceVerde Roadmap (v0.2.0)

TraceVerde's planned features align well with several frameworks:

### 1. Guardrails & Safety (v0.2.0 Roadmap)

**Aligned Frameworks:**
- **OpenAI Agents SDK:** Built-in guardrails for input/output validation
- **AWS Bedrock:** Bedrock Guardrails for safety checks
- **Pydantic AI:** Type-safe validation (Pydantic models)

**Opportunity:** Study these implementations as reference for TraceVerde's guardrail features

### 2. Bias, Toxicity, Hallucination Detection (v0.2.0 Roadmap)

**Aligned Frameworks:**
- **Pydantic AI + Logfire:** Evals-based performance monitoring
- **OpenAI Agents SDK:** Evaluation features built-in

**Opportunity:** Integrate eval metrics from agent frameworks into TraceVerde telemetry

### 3. Session Tracking (Already Implemented - Phase 4)

**Aligned Frameworks:**
- **OpenAI Agents SDK:** Sessions with automatic history management
- **AutoGen:** Multi-turn conversations with session context
- **LangGraph:** State persistence across executions

**Opportunity:** TraceVerde's session tracking can capture agent session IDs and costs

### 4. RAG Observability (Already Implemented - Phase 4)

**Aligned Frameworks:**
- **Haystack:** Advanced RAG pipelines
- **LangGraph:** Retrieval nodes in graphs
- **AWS Bedrock:** Knowledge Bases for RAG

**Opportunity:** TraceVerde's RAG attributes can enrich agent retrieval operations

---

## Competitive Analysis

### Existing Instrumentation Solutions

| Solution | Coverage | Approach |
|----------|----------|----------|
| **Pydantic Logfire** | Pydantic AI, general Python | OpenTelemetry-based, Pydantic-first |
| **LangSmith** | LangChain, LangGraph | Proprietary platform, LangChain-first |
| **Helicone** | OpenAI, Anthropic, general LLMs | Proxy-based instrumentation |
| **Weights & Biases** | General ML/LLM tracking | Experiment tracking focus |
| **Arize Phoenix** | OpenInference, general LLM | Open-source, OpenInference focus |

**TraceVerde's Differentiation:**
- ✅ **Broadest Provider Coverage:** 20+ LLM providers
- ✅ **Zero-Code Setup:** Environment variable configuration
- ✅ **Comprehensive Cost Tracking:** 340+ models, granular pricing
- ✅ **MCP Tool Support:** Databases, APIs, vector DBs
- ✅ **Open-Source + Self-Hosted:** AGPL-3.0, no vendor lock-in
- ⚠️ **Agent Framework Gap:** LangChain/LlamaIndex only (this research addresses gap)

**Strategic Opportunity:**
By adding multi-agent framework support (CrewAI, AutoGen, LangGraph, OpenAI Agents), TraceVerde can become the **most comprehensive open-source GenAI observability solution**.

---

## Conclusion & Next Steps

### Summary

This research identified **8 high-value frameworks** for TraceVerde instrumentation:

**Tier 1 (Immediate):**
1. CrewAI - Role-based multi-agent collaboration
2. LangGraph - Graph-based stateful workflows
3. OpenAI Agents SDK - Production-ready agent primitives

**Tier 2 (3-6 Months):**
4. Google GenAI SDK - Audit/update existing instrumentor
5. AutoGen - Multi-agent conversations (maintenance mode)
6. Pydantic AI - Type-safe agents with Logfire integration

**Tier 3 (Future):**
7. Haystack - RAG-focused pipelines
8. AWS Bedrock AgentCore - Extend existing Bedrock instrumentor
9. Microsoft Agent Framework - Wait for GA (Q1 2025)

### Recommended Roadmap

**Phase 1 (Q1 2025):** Tier 1 Frameworks
- Week 1-2: OpenAI Agents SDK instrumentor
- Week 3-5: CrewAI instrumentor
- Week 6-9: LangGraph instrumentor
- Week 10: Integration testing, documentation

**Phase 2 (Q2 2025):** Tier 2 Frameworks
- Week 1: Audit Google GenAI SDK instrumentor
- Week 2-4: AutoGen instrumentor
- Week 5-7: Pydantic AI instrumentor (coordinate with Logfire)
- Week 8: Integration testing, documentation

**Phase 3 (Q3 2025):** Tier 3 Frameworks
- Evaluate market demand for Haystack, Bedrock AgentCore
- Monitor Microsoft Agent Framework GA release
- Prioritize based on community feedback

### Action Items

1. **Create GitHub Issues:**
   - [ ] CrewAI instrumentation (#XX)
   - [ ] LangGraph instrumentation (#XX)
   - [ ] OpenAI Agents SDK instrumentation (#XX)
   - [ ] Google GenAI SDK audit (#XX)
   - [ ] AutoGen instrumentation (#XX)
   - [ ] Pydantic AI instrumentation (#XX)

2. **Update Documentation:**
   - [ ] Add agent framework support to roadmap
   - [ ] Document multi-agent cost aggregation approach
   - [ ] Update README with planned framework support

3. **Pricing Database:**
   - [ ] Audit llm_pricing.json for Gemini 2.0, DeepSeek-R1, Grok, Perplexity
   - [ ] Add missing model pricing

4. **Community Engagement:**
   - [ ] Create RFC/discussion for agent framework priorities
   - [ ] Gather user feedback on desired frameworks
   - [ ] Consider partnerships (Pydantic, LangChain, etc.)

5. **Technical Preparation:**
   - [ ] Design multi-agent cost aggregation mechanism
   - [ ] Create agent framework instrumentor base class (if needed)
   - [ ] Set up test environments for each framework

---

## References

### Documentation Links

**CrewAI:**
- Docs: https://docs.crewai.com/en/introduction
- GitHub: https://github.com/crewAIInc/crewAI
- PyPI: https://pypi.org/project/crewai/

**AutoGen:**
- Docs: https://microsoft.github.io/autogen/stable/
- GitHub: https://github.com/microsoft/autogen
- Migration Guide: https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/

**LangGraph:**
- Docs: https://www.langchain.com/langgraph
- GitHub: (part of LangChain monorepo)

**OpenAI Agents SDK:**
- Docs: https://openai.github.io/openai-agents-python/
- GitHub: https://github.com/openai/openai-agents-python

**Google GenAI SDK:**
- Docs: https://googleapis.github.io/python-genai/
- GitHub: https://github.com/googleapis/python-genai
- PyPI: https://pypi.org/project/google-genai/

**AWS Bedrock:**
- Docs: https://docs.aws.amazon.com/bedrock/
- GitHub: https://github.com/aws/bedrock-agentcore-sdk-python
- Boto3 Docs: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime.html

**Pydantic AI:**
- Docs: https://ai.pydantic.dev/
- GitHub: https://github.com/pydantic/pydantic-ai
- PyPI: https://pypi.org/project/pydantic-ai/

**Haystack:**
- Docs: https://docs.haystack.deepset.ai/
- GitHub: https://github.com/deepset-ai/haystack
- PyPI: https://pypi.org/project/haystack-ai/

**Semantic Kernel:**
- Docs: https://learn.microsoft.com/en-us/semantic-kernel/
- GitHub: https://github.com/microsoft/semantic-kernel

---

**Report Compiled By:** Claude (Anthropic AI)
**Research Date:** November 13, 2025
**Framework Count Analyzed:** 9 frameworks
**Total Web Searches:** 10 queries
**Document Version:** 1.0
