"""OpenAI Agents SDK Instrumentation Example.

This example demonstrates automatic OpenTelemetry instrumentation for
OpenAI's Agents SDK, which provides production-ready agent orchestration
with handoffs, sessions, and guardrails.

The OpenAI Agents SDK is the production upgrade of the experimental Swarm
framework, released March 2025.

Requirements:
    pip install genai-otel-instrument[openai]
    pip install openai-agents
    export OPENAI_API_KEY=your_api_key
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

import genai_otel

# Initialize instrumentation - OpenAI Agents SDK is enabled automatically
genai_otel.instrument(
    service_name="openai-agents-example",
    endpoint="http://localhost:4318",
)

print("\n" + "=" * 80)
print("OpenAI Agents SDK OpenTelemetry Instrumentation Example")
print("=" * 80 + "\n")

# Import OpenAI Agents SDK
try:
    from agents import Agent, Runner
    from agents.tools import function_tool
except ImportError:
    print("ERROR: OpenAI Agents SDK not installed. Install with:")
    print("  pip install openai-agents")
    exit(1)

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Get your API key from: https://platform.openai.com/api-keys")
    exit(1)

print("1. Simple Agent with Tool...")
print("-" * 80)


# Define a custom tool using function_tool decorator
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72F",
        "New York": "Cloudy, 65F",
        "London": "Rainy, 55F",
        "Tokyo": "Clear, 68F",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


# Create an agent with a tool
weather_agent = Agent(
    name="WeatherAssistant",
    instructions="You are a helpful weather assistant. Use the get_weather tool to provide weather information.",
    tools=[get_weather],
)

# Run the agent
result = Runner.run_sync(weather_agent, "What's the weather in San Francisco?")
print(f"User: What's the weather in San Francisco?")
print(f"Agent: {result.final_output}")
print()

print("2. Multi-Agent System with Handoffs...")
print("-" * 80)

# Create specialized agents
spanish_agent = Agent(
    name="SpanishAgent",
    instructions="You are a Spanish language expert. Respond only in Spanish.",
)

english_agent = Agent(
    name="EnglishAgent",
    instructions="You are an English language expert. Respond only in English.",
)

# Create a triage agent that hands off to specialized agents
triage_agent = Agent(
    name="TriageAgent",
    instructions="You route conversations to the appropriate language expert based on the user's language preference.",
    handoffs=[spanish_agent, english_agent],
)

# Run with handoff
result = Runner.run_sync(triage_agent, "Hola, como estas?")
print(f"User: Hola, como estas?")
print(f"Agent: {result.final_output}")
print()

print("3. Agent with Session (Conversation Memory)...")
print("-" * 80)

try:
    from agents.sessions import SQLiteSession

    # Create a session for conversation history
    session = SQLiteSession("user_123")

    # Create a conversational agent
    chat_agent = Agent(
        name="ChatAssistant",
        instructions="You are a helpful assistant that remembers previous conversations.",
    )

    # First message
    result = Runner.run_sync(chat_agent, "My name is Alice", session=session)
    print(f"User: My name is Alice")
    print(f"Agent: {result.final_output}")
    print()

    # Second message (agent should remember the name)
    result = Runner.run_sync(chat_agent, "What is my name?", session=session)
    print(f"User: What is my name?")
    print(f"Agent: {result.final_output}")
    print()

except ImportError:
    print("Note: SQLiteSession requires additional dependencies for session support")
    print()

print("4. Agent with Guardrails...")
print("-" * 80)

try:
    from agents.guardrails import Guardrail

    # Define a simple content filter guardrail
    def content_filter(input_text: str) -> bool:
        """Simple content filter that blocks certain keywords."""
        blocked_words = ["spam", "scam", "malware"]
        return not any(word in input_text.lower() for word in blocked_words)

    # Create guardrail
    safety_guardrail = Guardrail(
        name="ContentFilter",
        validate_input=content_filter,
    )

    # Create agent with guardrail
    safe_agent = Agent(
        name="SafeAssistant",
        instructions="You are a helpful assistant.",
        guardrails=[safety_guardrail],
    )

    # Test with safe input
    result = Runner.run_sync(safe_agent, "Tell me about artificial intelligence")
    print(f"User: Tell me about artificial intelligence")
    print(f"Agent: {result.final_output}")
    print()

except ImportError:
    print("Note: Guardrails require additional configuration")
    print()

print("=" * 80)
print("Telemetry Data Collected:")
print("=" * 80)
print(
    """
For each agent execution, the following data is automatically collected:

TRACES (Spans):
- Span name: openai_agents.runner.run or openai_agents.runner.run_sync
- Attributes:
  - gen_ai.system: "openai_agents"
  - gen_ai.operation.name: "agent.run"
  - openai.agent.name: Agent name
  - openai.agent.model: Model used (e.g., "gpt-4")
  - openai.agent.instructions: Agent instructions (truncated)
  - openai.agent.tools: List of tool names
  - openai.agent.handoffs: List of handoff targets
  - openai.agent.guardrails_enabled: Boolean
  - openai.session.id: Session identifier (if using sessions)
  - openai.handoff.occurred: Boolean if handoff happened
  - openai.handoff.to_agent: Target agent name
  - openai.guardrail.violations: Count of guardrail violations

METRICS:
- genai.requests: Request count by agent and model
- genai.tokens: Token usage (underlying LLM calls)
- genai.latency: Agent execution duration
- genai.cost: Costs for LLM calls (via OpenAI instrumentor)

View these metrics in your observability platform (Grafana, Jaeger, etc.)
"""
)

print("=" * 80)
print("OpenAI Agents SDK Features:")
print("=" * 80)
print(
    """
Key Features Instrumented:
- Agent Execution: Tracks Runner.run() and Runner.run_sync() calls
- Handoffs: Captures agent-to-agent delegation
- Sessions: Monitors conversation history and session IDs
- Guardrails: Records validation results and violations
- Tools: Tracks which tools are available to agents
- Multi-Agent Workflows: Full visibility into agent orchestration

Benefits:
- Production-ready agent framework from OpenAI
- Automatic conversation history management (sessions)
- Built-in safety with guardrails
- Seamless handoffs between specialized agents
- Complete observability with zero code changes
"""
)

print("=" * 80)
print("Example complete! Check your OTLP collector/Grafana for traces and metrics.")
print("=" * 80 + "\n")
