# Multi-Agent Frameworks

TraceVerde instruments 8 multi-agent and LLM frameworks with complete trace hierarchy.

## CrewAI

Complete span hierarchy: Crew -> Task -> Agent -> LLM calls.

```bash
pip install genai-otel-instrument[crewai]
```

**Instrumented methods:**
- `Crew.kickoff()`, `kickoff_async()`, `akickoff()`, `kickoff_for_each()`
- `Task.execute_sync()`, `Task.execute_async()`
- `Agent.execute_task()`

**Span types:**
- `crewai.crew.execution` - crew-level attributes (process type, agent/task counts)
- `crewai.task.execution` - task attributes (description, expected output)
- `crewai.agent.execution` - agent attributes (role, goal, backstory, LLM model)

Includes automatic `ThreadPoolExecutor` context propagation for worker threads.

## LangGraph

Stateful workflows with graph-based orchestration.

```bash
pip install genai-otel-instrument[langgraph]
```

Instruments `invoke()`, `stream()`, `ainvoke()`, `astream()` with automatic session ID propagation.

## Google ADK

Google Agent Development Kit instrumentation.

```bash
pip install genai-otel-instrument[google-adk]
```

Instruments `Runner.run_async()` and `InMemoryRunner.run_debug()`. Captures agent name, model, tools, sub-agents, and session info.

## AutoGen AgentChat (v0.4+)

```bash
pip install genai-otel-instrument[autogen-agentchat]
```

Instruments agent `run()`/`run_stream()` and team execution (RoundRobinGroupChat, SelectorGroupChat, Swarm). Captures participants, task content, and termination conditions.

## OpenAI Agents SDK

Agent orchestration with handoffs, sessions, and guardrails.

Automatically instrumented when OpenAI SDK is installed.

## Pydantic AI

Type-safe agents with Pydantic validation and multi-provider support.

```bash
pip install genai-otel-instrument[pydantic-ai]
```

## LangChain

Chains, agents, and tools.

```bash
pip install genai-otel-instrument[langchain]
```

## Other Frameworks

- **LlamaIndex** - Query engines and indices (`[llamaindex]`)
- **Haystack** - Modular NLP pipelines with RAG support (`[haystack]`)
- **DSPy** - Declarative LM programming (`[dspy]`)
- **Instructor** - Structured output extraction (`[instructor]`)
- **Guardrails AI** - Input/output validation (`[guardrails]`)
