# Multi-Agent Frameworks

TraceVerde instruments 8 multi-agent and LLM frameworks with complete trace hierarchy.

## CrewAI

Complete span hierarchy: Crew -> Task -> Agent -> LLM calls.

```bash
pip install genai-otel-instrument[crewai]
```

```python
import genai_otel

genai_otel.instrument(
    service_name="my-crewai-app",
    enabled_instrumentors=["crewai", "openai"],
    enable_cost_tracking=True,
)

from crewai import Crew, Agent, Task

researcher = Agent(
    role="Senior Researcher",
    goal="Research topics thoroughly",
    backstory="Expert researcher with 10 years experience",
    llm="gpt-4"
)

task = Task(
    description="Research OpenTelemetry best practices",
    expected_output="Comprehensive report",
    agent=researcher
)

crew = Crew(agents=[researcher], tasks=[task], process="sequential")
result = crew.kickoff()
# Complete traces: Crew -> Agent -> Task -> LLM calls
```

**Instrumented methods:**

- `Crew.kickoff()`, `kickoff_async()`, `akickoff()`, `kickoff_for_each()` and async batch variants
- `Task.execute_sync()`, `Task.execute_async()`
- `Agent.execute_task()`

**Span types:**

- `crewai.crew.execution` - crew-level attributes (process type, agent/task counts)
- `crewai.task.execution` - task attributes (description, expected output)
- `crewai.agent.execution` - agent attributes (role, goal, backstory, LLM model)

Includes automatic `ThreadPoolExecutor` context propagation for worker threads.

See [CrewAI example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/crewai_example.py).

## LangGraph

Stateful workflows with graph-based orchestration.

```bash
pip install genai-otel-instrument[langgraph]
```

Instruments `invoke()`, `stream()`, `ainvoke()`, `astream()` with automatic session ID propagation.

See [LangGraph example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/langgraph_example.py).

## Google ADK

Google Agent Development Kit instrumentation.

```bash
pip install genai-otel-instrument[google-adk]
```

Instruments `Runner.run_async()` and `InMemoryRunner.run_debug()`. Captures agent name, model, tools, sub-agents, and session info.

See [Google ADK example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/google_adk_example.py).

## AutoGen (Legacy)

Microsoft multi-agent conversations with group chats.

```bash
pip install genai-otel-instrument[autogen]
```

```python
import genai_otel
genai_otel.instrument(service_name="autogen-example")

import autogen

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": "..."}]},
)

user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config=False,
)

user_proxy.initiate_chat(
    assistant,
    message="What is the capital of France?",
    max_turns=2,
)
# Traces: initiate_chat -> OpenAI calls with cost tracking
```

**Instrumented methods:**

- `ConversableAgent.initiate_chat()` - agent-to-agent conversations
- `GroupChat.select_speaker()` - speaker selection in group chats
- `GroupChatManager.run()` - group chat orchestration

See [AutoGen example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/autogen_example.py).

## AutoGen AgentChat (v0.4+)

```bash
pip install genai-otel-instrument[autogen-agentchat]
```

Instruments agent `run()`/`run_stream()` and team execution (RoundRobinGroupChat, SelectorGroupChat, Swarm). Captures participants, task content, and termination conditions.

## OpenAI Agents SDK

Agent orchestration with handoffs, sessions, and guardrails.

Automatically instrumented when OpenAI SDK is installed.

See [OpenAI Agents example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/openai_agents_example.py).

## AWS Bedrock Agents

Managed agent runtime with knowledge bases and RAG.

```bash
pip install genai-otel-instrument[aws]
```

See [Bedrock Agents example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bedrock_agents_example.py).

## Pydantic AI

Type-safe agents with Pydantic validation and multi-provider support.

```bash
pip install genai-otel-instrument[pydantic-ai]
```

See [Pydantic AI example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pydantic_ai_example.py).

## Other Frameworks

| Framework | Description | Install Extra | Example |
|-----------|-------------|---------------|---------|
| LangChain | Chains, agents, and tools | `[langchain]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/langchain/example.py) |
| LlamaIndex | Query engines and indices | `[llamaindex]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/llamaindex/example.py) |
| Haystack | Modular NLP pipelines with RAG | `[haystack]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/haystack_example.py) |
| DSPy | Declarative LM programming | `[dspy]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/dspy_example.py) |
| Instructor | Structured output extraction | `[instructor]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/instructor_example.py) |

## All Examples

Browse all framework examples in the [examples/ directory](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples).
