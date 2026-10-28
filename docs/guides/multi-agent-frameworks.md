# Multi-Agent Frameworks

TraceVerde instruments 13 multi-agent and LLM frameworks, including native OpenTelemetry
frameworks and OpenInference integrations. Framework spans are connected to provider
spans when the provider is also instrumented.

## CrewAI

Span hierarchy: Crew -> Task -> Agent -> LLM calls, in one trace per kickoff. For the
Arize AX agent graph the crew span is the root node and each agent execution a node
beneath it; see [Arize AX](arize-ax.md#agent-graph).
Verified against CrewAI 1.15.27 and 1.14.4.

```bash
pip install genai-otel-instrument[crewai]
```

```python
import genai_otel

genai_otel.instrument(
    service_name="my-crewai-app",
    enabled_instrumentors=["crewai", "openai"],
)

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool


@tool("word_count")
def word_count(text: str) -> str:
    """Count the words in the given text and return the number."""
    return str(len(text.split()))


llm = LLM(model="gpt-4o-mini", max_tokens=80, temperature=0)

counter = Agent(
    role="Word Counter",
    goal="Count words exactly using the word_count tool",
    backstory="A careful clerk who never guesses a count.",
    llm=llm,
    tools=[word_count],
)
reporter = Agent(
    role="Reporter",
    goal="State results in one short sentence",
    backstory="A terse reporter.",
    llm=llm,
)

count_task = Task(
    description="Use the word_count tool on 'OpenTelemetry traces every agent step'.",
    expected_output="The word count as a number.",
    agent=counter,
)
report_task = Task(
    description="Write one sentence stating the word count found.",
    expected_output="One short sentence.",
    agent=reporter,
)

crew = Crew(
    agents=[counter, reporter],
    tasks=[count_task, report_task],
    process=Process.sequential,
)
result = crew.kickoff()
# crewai.crew.execution -> crewai.task.execution -> crewai.agent.execution -> openai.chat.completion
```

Call `genai_otel.instrument()` before importing `crewai` and before creating any `LLM`
or `Agent`. An OpenAI client built before instrumentation is not traced, and on CrewAI
older than 1.15.15 importing `crewai` first lets CrewAI's own telemetry take the global
tracer provider, in which case no TraceVerde span is exported (a warning is logged).

**Instrumented methods:**

- `Crew.kickoff()`, `kickoff_async()`, `akickoff()`, `kickoff_for_each()` and async batch variants
- `Task.execute_sync()`, `Task.execute_async()`, `Task.aexecute_sync()`
- `Agent.execute_task()`, `Agent.aexecute_task()`

**Span types:**

- `crewai.crew.execution` - crew-level attributes (process type, agent/task counts, output)
- `crewai.task.execution` - task attributes (description, expected output, `gen_ai.agent.name`)
- `crewai.agent.execution` - agent attributes (role, goal, backstory, LLM model)

**What to expect:**

- Tokens and cost are on the LLM spans. The crew span reports them itself only when no
  LLM span beneath it did, and then for that kickoff alone. Batch runs that overlap in
  time (`kickoff_for_each_async`) share their LLM's counters, so in that case each crew
  span can include usage from the others.
- `session.id` is `inputs={"session_id": ...}` when you pass one, else the enclosing
  `genai_otel.session()` block, else the crew's id. Every task, agent, LLM and tool span
  of the kickoff carries it.
- Tool calls have no span of their own; they appear as `llm.tools` and tool-call
  attributes on the LLM span.
- A task with `output_pydantic` goes through the OpenAI SDK's `parse` call, which is not
  traced yet, so it has no LLM span.
- `kickoff_for_each` produces one outer `crewai.crew.execution` span around one per input.
- TraceVerde sets `CREWAI_DISABLE_TELEMETRY=true` so CrewAI's built-in telemetry does not
  run alongside it. Set the variable yourself to keep your own choice.
  `CREWAI_TRACING_ENABLED=false` removes CrewAI's first-run prompt to share a trace.

Includes automatic `ThreadPoolExecutor` context propagation for worker threads.

See [CrewAI example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/crewai_example.py).

## LangGraph

Stateful workflows with graph-based orchestration. Verified against LangGraph 0.1.19,
0.2.76, 0.6.11 and 1.2.14.

```bash
pip install genai-otel-instrument[langgraph]
```

```python
import genai_otel

genai_otel.instrument(
    service_name="my-langgraph-app",
    enabled_instrumentors=["langgraph", "openai"],
)

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    question: str
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=12)


def answer(state: State):
    return {"answer": llm.invoke(state["question"]).content}


builder = StateGraph(State)
builder.add_node("answer", answer)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
graph = builder.compile()

result = graph.invoke(
    {"question": "Capital of France? One word."},
    {"configurable": {"thread_id": "user-42"}},
)
# langgraph.graph.invoke -> langgraph.node.answer -> openai.chat.completion
```

**How it hooks in.** TraceVerde wraps four methods on the `langgraph.pregel.Pregel` class
with `wrapt`: `invoke`, `ainvoke`, `stream` and `astream`. It is not a LangChain callback
handler and registers no callbacks. Every runnable graph is a `Pregel`, so each run
produces one span however the graph was built: a compiled `StateGraph`,
`create_react_agent`, `langchain.agents.create_agent`, a `with_config()` copy, a
functional-API `@entrypoint`, a subgraph (nested under its parent's span), or a graph
compiled before `instrument()` was called. Each node execution gets a span too: the
instrumentor wraps `run_with_retry` / `arun_with_retry`, the functions LangGraph runs
every node task through. Those are LangGraph internals; on a version that moves them,
node spans are skipped and graph spans are unaffected.

**Span types:** `langgraph.graph.invoke`, `langgraph.graph.ainvoke`,
`langgraph.graph.stream`, `langgraph.graph.astream`, and `langgraph.node.<name>` per
node execution.

**What to expect:**

- `session.id`, `langgraph.session.id` and `gen_ai.conversation.id` are a `session_id`
  in the input state, else the enclosing `genai_otel.session()` block, else the config
  `thread_id`, otherwise a random id per run. Node, LLM and tool spans inherit it. `langgraph.thread_id`
  and `langgraph.recursion_limit` are recorded when present in the config.
- Graph spans carry the graph's shape (`langgraph.node_count`, `langgraph.nodes`,
  `langgraph.edge_count`, `langgraph.channels`) and the input and output keys.
- Node spans carry `langgraph.node`, `langgraph.step` and `langgraph.triggers`, and the
  node's LLM and tool calls nest beneath them. A node that raises is marked as an error;
  one that pauses with `interrupt()` is not, and carries `langgraph.node.interrupted`.
  `__start__` gets no span.
- For the Arize AX agent graph, node spans also carry `graph.node.id` (the node name),
  `metadata` (`{"langgraph_node", "langgraph_step"}`) and `openinference.span.kind=CHAIN`,
  and the graph span is the root node, named after the graph (`compile(name=...)`,
  default `LangGraph`). See [Arize AX](arize-ax.md#agent-graph).
- Tools have no span of their own. LLM spans, tokens and cost come from the provider
  instrumentor enabled alongside (`openai` above); graph and node spans carry no token
  usage, so nothing is counted twice.
- A stream span starts at the first chunk and ends when the stream is exhausted, closed
  or abandoned. A stream that is never iterated produces no span.
- Enabling `langchain` as well adds a `langchain.chat_model.*` span around each model
  call. Usage stays on the provider span.

See [LangGraph example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/langgraph_example.py).

## Google ADK

Google Agent Development Kit instrumentation.

```bash
pip install genai-otel-instrument[google-adk]
```

Instruments `Runner.run_async()` and `InMemoryRunner.run_debug()`. Captures agent name, model, tools, sub-agents, and session info.

See [Google ADK example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/google_adk_example.py).

## Google Antigravity SDK

Native instrumentation for the Google Antigravity Python agent SDK. The
`antigravity.agent.chat` span covers the full lazy response stream, including
token usage reported after the stream is consumed. Prompts and generated text
are recorded only when content capture is enabled.

```bash
pip install 'genai-otel-instrument[antigravity]'
```

```python
import genai_otel

genai_otel.instrument(
    service_name="my-antigravity-app",
    enabled_instrumentors=["antigravity"],
)

from google.antigravity import Agent, LocalAgentConfig

async with Agent(LocalAgentConfig()) as agent:
    response = await agent.chat("Explain OpenTelemetry in one sentence.")
    print(await response.text())
```

The `google-antigravity` extra requires Python 3.10+. The SDK's response is
lazy, so consume it (for example with `await response.text()`) to finish the
span and record usage.

See the [Antigravity example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/antigravity/example.py).

## AutoGen (Legacy)

Multi-agent conversations with group chats, for the `import autogen` API. Verified
against pyautogen 0.2.35 and 0.9, and autogen 0.14.1.

```bash
pip install genai-otel-instrument[autogen]
```

The extra installs `pyautogen` below 0.10. From 0.10 on, `pyautogen` is an empty proxy
for `autogen-agentchat` and has no `autogen` module; use
[AutoGen AgentChat](#autogen-agentchat-v04) for that API.

```python
import genai_otel

genai_otel.instrument(
    service_name="autogen-example",
    enabled_instrumentors=["autogen", "openai"],
)

import autogen

llm_config = {"config_list": [{"model": "gpt-4o-mini"}], "max_tokens": 30, "cache_seed": None}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="Answer in at most 6 words.",
)

user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config=False,
)

user_proxy.initiate_chat(
    assistant,
    message="What is the capital of France?",
    max_turns=1,
)
# autogen.initiate_chat -> openai.chat.completion
```

**Instrumented methods:**

- `ConversableAgent.initiate_chat()` - agent-to-agent conversations (`autogen.initiate_chat`)
- `GroupChat.select_speaker()` - speaker selection (`autogen.group_chat.select_speaker`)
- `GroupChatManager.run_chat()` - group chat orchestration (`autogen.group_chat.run`)

Tokens and cost are on the LLM spans. `a_initiate_chat()` is not traced yet.

See [AutoGen example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/autogen_example.py).

## AutoGen AgentChat (v0.4+)

Verified against autogen-agentchat 0.7.5.

```bash
pip install genai-otel-instrument[autogen-agentchat] "autogen-ext[openai]"
```

```python
import asyncio

import genai_otel

genai_otel.instrument(
    service_name="agentchat-example",
    enabled_instrumentors=["autogen_agentchat", "openai"],
)

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return "Sunny, 21 C in {0}".format(city)


async def main():
    client = OpenAIChatCompletionClient(model="gpt-4o-mini", max_tokens=40)
    agent = AssistantAgent(
        name="weather_agent",
        model_client=client,
        tools=[get_weather],
        system_message="Use the get_weather tool, then answer in under 10 words.",
        reflect_on_tool_use=True,
    )
    await agent.run(task="What is the weather in Paris?")
    await client.close()


asyncio.run(main())
# autogen_agentchat.agent.run -> invoke_agent weather_agent (AutoGen's own span)
#   -> autogen_agentchat.agent.on_messages -> openai.chat.completion
```

**Span types:**

- `autogen_agentchat.agent.run`, `autogen_agentchat.agent.run_stream` - agent name and
  type, model, task
- `autogen_agentchat.agent.on_messages` - under `run`, or for a direct call
- `autogen_agentchat.team.run`, `autogen_agentchat.team.run_stream` - team type,
  participants, termination condition, stop reason

**What to expect:**

- Tokens and cost are on the LLM spans; agent and team spans carry none.
- AutoGen 0.7 emits its own OpenTelemetry spans through the same tracer provider, so
  they appear in the same trace: `create_agent`, `invoke_agent` and `execute_tool`
  (scope `autogen-core`), plus one span per runtime message (scope
  `autogen SingleThreadedAgentRuntime`). A two-agent team run produced 32 of the
  runtime spans; `AUTOGEN_DISABLE_RUNTIME_TRACING=true`, set before the team is
  created, removes them.
- If you stop reading a `run_stream()` early, call `await stream.aclose()` so its span
  ends there.

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

Type-safe agents with Pydantic validation and multi-provider support. Verified against
Pydantic AI 2.55.0.

```bash
pip install genai-otel-instrument[pydantic-ai]
```

```python
import genai_otel

genai_otel.instrument(
    service_name="my-pydantic-ai-app",
    enabled_instrumentors=["pydantic_ai", "openai"],
)

from pydantic import BaseModel
from pydantic_ai import Agent


class CityInfo(BaseModel):
    city: str
    country: str


agent = Agent(
    "openai:gpt-4o-mini",
    name="city_agent",
    instructions="Call get_country exactly once, then give the final answer.",
    output_type=CityInfo,
)


@agent.tool_plain
def get_country(city: str) -> str:
    """Return the country a city is in."""
    return {"paris": "France", "rome": "Italy"}.get(city.lower(), "Unknown")


result = agent.run_sync("Which country is Paris in?")
print(result.output)
# pydantic_ai.agent.run_sync -> pydantic_ai.agent.run -> openai.responses (one per model request)
```

**Span types:**

| Call | Spans |
|---|---|
| `agent.run_sync()` | `pydantic_ai.agent.run_sync` -> `pydantic_ai.agent.run` -> LLM spans |
| `await agent.run()`, `agent.run_stream_events()` | `pydantic_ai.agent.run` -> LLM spans |
| `agent.run_stream()`, `agent.run_stream_sync()` | `pydantic_ai.agent.run_stream`, open for the whole block |
| `agent.iter()` | `pydantic_ai.agent.iter` -> LLM spans |

**What to expect:**

- Agent spans carry the agent name, model, tool names, output type, response model and
  finish reason.
- Tokens and cost are on the LLM spans (`openai.responses` for `openai:` models,
  `openai.chat.completion` for `openai-chat:`). With only `pydantic_ai` enabled, the
  agent span carries the run totals itself.
- Tool executions have no span of their own.
- Concurrent runs (`asyncio.gather`) are separate traces.
- Pydantic AI's own instrumentation (`Agent.instrument_all()`) is off by default. If you
  turn it on as well, its `invoke_agent`, `chat` and `execute_tool` spans appear in the
  same trace and every model request is reported twice. Enable one or the other.
- Pydantic AI prints a one-time `observability: off` banner because it does not detect
  TraceVerde; `PYDANTIC_AI_NO_BANNER=1` hides it.

See [Pydantic AI example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pydantic_ai_example.py).

## Strands Harness

Native instrumentation for the optional Strands Harness runtime. It uses the
Harness hook API to preserve agent, delegated-agent, model, tool, MCP, and
context-compaction relationships while allowing the Strands SDK's own provider
spans to remain in place.

```bash
pip install 'genai-otel-instrument[strands]'
```

```python
import genai_otel

genai_otel.instrument(
    service_name="my-strands-app",
    enabled_instrumentors=["strands"],
    enable_content_capture=False,
)

from strands_harness import create_harness

agent = create_harness(instructions="Answer in one paragraph.", session=True)
result = agent("Explain OpenTelemetry.")
```

The logical hierarchy is `harness.run` -> `agent.run`, with `llm.request`,
`tool.call`, `mcp.call`, `context.compaction`, and `subagent.run` children when
those operations are exposed by the installed Harness version. Prompts, tool
arguments/results, MCP payloads, memory contents, and model output are omitted
unless the existing GenAI content-capture setting is enabled. Provider-level
spans are enriched when active, avoiding duplicate model/tool spans. The extra
currently targets `strands-harness>=0.1.1` on Python 3.10+.

See the [Strands Harness example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/strands_harness/example.py).

## Anthropic Claude Agent SDK

The Claude Agent SDK is distinct from Anthropic's Messages API client. This integration
uses the maintained OpenInference adapter to trace `query()` and `ClaudeSDKClient` turns,
tool hooks, and delegated subagents.

```bash
pip install 'genai-otel-instrument[claude-agent-sdk]'
```

```python
import genai_otel

genai_otel.instrument(
    service_name="claude-agent-app",
    enabled_instrumentors=["claude_agent_sdk"],
    enable_content_capture=False,
)

import asyncio
from claude_agent_sdk import ClaudeAgentOptions, query


async def main():
    async for message in query(
        "List Python files in the current directory.",
        options=ClaudeAgentOptions(allowed_tools=["Glob"]),
    ):
        if hasattr(message, "result"):
            print(message.result)


asyncio.run(main())
```

Agent and tool spans include SDK-reported token usage when present; the configured cost
enricher prices those counts when the model is in the pricing catalog. The Claude Agent
SDK runs the model through its own CLI process, so the ordinary Anthropic Messages
instrumentor does not see its internal requests. The adapter's agent span carries usage;
it does not claim separate LLM request spans. Do not enable a second Claude Agent SDK
instrumentor alongside this integration.

See the [Claude Agent SDK example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/claude_agent_sdk/example.py).

## Strands Agents SDK

This is the current `strands-agents` runtime, separate from the legacy Strands Harness
integration above. Strands emits native OpenTelemetry agent, cycle, model, and tool spans
through the configured global provider. The integration does not add a second wrapper
span or a conversion processor.

```bash
pip install 'genai-otel-instrument[strands-agents]'
```

```python
import genai_otel

genai_otel.instrument(
    service_name="strands-agent-app",
    enabled_instrumentors=["strands_agents"],
    enable_content_capture=False,
)

from strands import Agent

agent = Agent()
print(agent("In one sentence, what does OpenTelemetry do?"))
```

Model and token attributes on Strands' native LLM spans are available to the normal cost
enrichment path. Call `genai_otel.instrument()` before importing Strands so its telemetry
uses the configured global OpenTelemetry provider.

See the [Strands Agents example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/strands_agents/example.py).

## Microsoft Agent Framework

Microsoft Agent Framework already emits GenAI traces, logs, and metrics through
OpenTelemetry. The integration activates its native telemetry on TraceVerde's global
provider; it does not patch the framework or configure a second exporter pipeline.

```bash
pip install 'genai-otel-instrument[microsoft-agent-framework]'
```

```python
import asyncio

import genai_otel

genai_otel.instrument(
    service_name="microsoft-agent-app",
    enabled_instrumentors=["microsoft_agent_framework"],
    enable_content_capture=False,
)

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Answer in one concise sentence.",
    )
    print((await agent.run("What does OpenTelemetry do?")).text)


asyncio.run(main())
```

Native GenAI model spans carry token counts and model identity for TraceVerde's cost
enrichment. Sensitive prompt and tool content remains disabled by default. Configure
exporters once through TraceVerde; do not also call the framework's
`configure_otel_providers()`, which would compete for global providers.

See the [Microsoft Agent Framework example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/microsoft_agent_framework/example.py).

## Agno

Agno runs are traced through the maintained OpenInference adapter. The integration
captures agent and model activity, and the model spans' token usage is passed through the
standard cost enrichment path.

```bash
pip install 'genai-otel-instrument[agno]'
```

```python
import genai_otel

genai_otel.instrument(
    service_name="agno-agent-app",
    enabled_instrumentors=["agno"],
    enable_content_capture=False,
)

from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), markdown=False)
agent.print_response("In one sentence, what does OpenTelemetry do?")
```

See the [Agno example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/agno/example.py).

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
