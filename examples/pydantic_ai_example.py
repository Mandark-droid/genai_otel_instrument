"""
Example demonstrating OpenTelemetry instrumentation for the Pydantic AI framework.

This example shows:
1. An agent with a tool and a structured (Pydantic) output
2. Two runs gathered concurrently
3. A streamed run

Written for Pydantic AI 2.x (`output_type`, `result.output`, `@agent.tool_plain`).
Everything runs in one event loop: Pydantic AI reuses its HTTP client, so mixing
`run_sync()` with a later `asyncio.run()` in the same process can stall.

Requirements:
    pip install "genai-otel-instrument[pydantic-ai]"

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import asyncio

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()  # Load .env file from project root
except ImportError:
    pass  # python-dotenv not installed, that's okay

# Step 1: Set up OpenTelemetry instrumentation
# This should be done BEFORE importing Pydantic AI
from genai_otel import instrument

instrument(
    service_name="pydantic-ai-example",
    # endpoint="http://localhost:4318",  # OTLP endpoint
    enabled_instrumentors=["pydantic_ai", "openai"],
    enable_mcp_instrumentation=False,  # Disable MCP to avoid optional deps
    enable_gpu_metrics=False,  # Disable GPU metrics to avoid warnings
)

# Step 2: Import Pydantic AI after instrumentation is set up
from pydantic import BaseModel
from pydantic_ai import Agent

MODEL = "openai:gpt-4o-mini"


class CityInfo(BaseModel):
    """Structured answer the agent must return."""

    city: str
    country: str


# An agent with instructions, a structured output type and one tool.
city_agent = Agent(
    MODEL,
    name="city_agent",
    instructions="Call get_country exactly once, then give the final answer.",
    output_type=CityInfo,
)


@city_agent.tool_plain
def get_country(city: str) -> str:
    """Return the country a city is in."""
    return {"paris": "France", "rome": "Italy", "madrid": "Spain"}.get(city.lower(), "Unknown")


# A plain-text agent for the streaming example.
greeter = Agent(MODEL, name="greeter", instructions="Reply with one short greeting.")


async def run_example():
    """Trace: pydantic_ai.agent.run -> one LLM span per model request."""
    print("\n" + "=" * 80)
    print("Example 1: a run with a tool and a structured output")
    print("=" * 80)
    result = await city_agent.run("Which country is Rome in?")
    print("Output: {0}".format(result.output))
    print("Usage:  {0}".format(result.usage))


async def run_gathered_example():
    """Gathered runs are separate traces, each with its own LLM spans."""
    print("\n" + "=" * 80)
    print("Example 2: two runs gathered")
    print("=" * 80)
    first, second = await asyncio.gather(
        city_agent.run("Which country is Madrid in?"),
        city_agent.run("Which country is Paris in?"),
    )
    print("Gathered: {0} / {1}".format(first.output, second.output))


async def run_stream_example():
    """Trace: pydantic_ai.agent.run_stream, open for the whole `async with` block."""
    print("\n" + "=" * 80)
    print("Example 3: streamed run")
    print("=" * 80)
    async with greeter.run_stream("Say hello.") as stream:
        async for text in stream.stream_text():
            print("Streamed so far: {0}".format(text))


async def main():
    await run_example()
    await run_gathered_example()
    await run_stream_example()


if __name__ == "__main__":
    asyncio.run(main())

    print("\n" + "=" * 80)
    print("Telemetry")
    print("=" * 80)
    print("""
Spans created by the instrumentation:

- pydantic_ai.agent.run / pydantic_ai.agent.run_stream
  (agent.run_sync() adds a pydantic_ai.agent.run_sync span around the run span)
  gen_ai.system=pydantic_ai, gen_ai.agent.name, gen_ai.request.model,
  pydantic_ai.tools, pydantic_ai.result_type, gen_ai.response.model
- openai.responses (one per model request)
  gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, gen_ai.usage.cost.total

Tokens and cost are reported on the LLM spans. Tool executions have no span of
their own; the tool call is visible on the LLM span that requested it.
""")
