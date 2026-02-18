"""Google Agent Development Kit (ADK) Instrumentation Example.

This example demonstrates automatic OpenTelemetry instrumentation for
Google ADK, an open-source framework for building AI agents.

Requirements:
    pip install genai-otel-instrument
    pip install google-adk>=1.17.0
    export GOOGLE_API_KEY=your_api_key
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import asyncio
import os

import genai_otel

# Initialize instrumentation - Google ADK is enabled automatically
genai_otel.instrument(
    service_name="google-adk-example",
    # endpoint="http://localhost:4318",
)

print("\n" + "=" * 80)
print("Google ADK OpenTelemetry Instrumentation Example")
print("=" * 80 + "\n")

# Import Google ADK
try:
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner
except ImportError:
    print("ERROR: Google ADK not installed. Install with:")
    print("  pip install google-adk")
    exit(1)

# Check for API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY environment variable not set")
    print("Get your key from: https://aistudio.google.com/apikey")
    exit(1)


# Define a simple tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72F."


# Create an agent
weather_agent = Agent(
    name="weather_assistant",
    model="gemini-2.5-flash",
    description="A helpful weather assistant",
    instruction="You are a weather assistant. Use the get_weather tool to answer weather questions.",
    tools=[get_weather],
)

# Create runner
runner = InMemoryRunner(agent=weather_agent, app_name="weather_app")


async def main():
    """Run the agent with instrumentation."""
    print("1. Simple Agent Execution...")
    print("-" * 80)

    # This call is automatically instrumented with OpenTelemetry spans
    result = await runner.run_debug("What's the weather in San Francisco?")
    print(f"Result: {result}")
    print()

    print("2. Multi-turn Conversation...")
    print("-" * 80)

    result = await runner.run_debug("How about New York?")
    print(f"Result: {result}")
    print()


# Run the async main function
asyncio.run(main())

print("=" * 80)
print("Telemetry Data Collected:")
print("=" * 80)
print(
    """
For each agent execution, the following data is automatically collected:

TRACES (Spans):
- Span name: google_adk.runner.run or google_adk.runner.run_debug
- Attributes:
  - gen_ai.system: "google_adk"
  - gen_ai.operation.name: "runner.run" or "runner.run_debug"
  - gen_ai.request.model: Model identifier (e.g., "gemini-2.5-flash")
  - google_adk.app_name: Application name
  - google_adk.agent.name: Agent name
  - google_adk.agent.description: Agent description
  - google_adk.sub_agent_count: Number of sub-agents
  - google_adk.tool_count: Number of tools
  - google_adk.tools: List of tool names
  - google_adk.user_id: User identifier
  - google_adk.session_id: Session identifier
  - google_adk.input_message: Input message (truncated)

View these traces in your observability platform (Grafana, Jaeger, etc.)
"""
)

print("=" * 80)
print("Example complete! Check your OTLP collector/Grafana for traces and metrics.")
print("=" * 80 + "\n")
