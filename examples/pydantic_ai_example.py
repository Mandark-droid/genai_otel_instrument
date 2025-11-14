"""
Example demonstrating OpenTelemetry instrumentation for Pydantic AI framework.

This example shows:
1. Basic type-safe agent with automatic instrumentation
2. Agent with tools/functions
3. Structured output with Pydantic models
4. Multi-provider support (OpenAI, Anthropic, etc.)

Pydantic AI is a new framework (Dec 2024) by the Pydantic team that provides
type-safe agent development with full Pydantic validation.

Requirements:
    pip install genai-otel pydantic-ai openai anthropic

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
    # Or for other providers:
    # export ANTHROPIC_API_KEY=your_key_here
"""

import os
from typing import Optional

# Step 1: Set up OpenTelemetry instrumentation
# This should be done BEFORE importing Pydantic AI
from genai_otel import instrument

instrument(
    service_name="pydantic-ai-example",
    endpoint="http://localhost:4318",  # OTLP endpoint
)

# Step 2: Import Pydantic AI after instrumentation is set up
from pydantic import BaseModel
from pydantic_ai import Agent


def example_basic_agent():
    """
    Example 1: Basic Type-Safe Agent

    This demonstrates a simple agent with type-safe responses.

    Telemetry Captured:
    - Span: pydantic_ai.agent.run
    - Attributes:
        - gen_ai.system: "pydantic_ai"
        - gen_ai.operation.name: "agent.run"
        - gen_ai.request.model: Model name (e.g., "gpt-4")
        - pydantic_ai.agent.name: Agent name if provided
        - pydantic_ai.model.provider: Model provider (e.g., "OpenAIModel")
        - pydantic_ai.user_prompt: User's input prompt
        - pydantic_ai.result.data: Agent's response
        - gen_ai.response.model: Response model used
    """
    print("=" * 80)
    print("Example 1: Basic Type-Safe Agent")
    print("=" * 80)

    # Create a simple agent - This will be automatically instrumented
    agent = Agent(
        "openai:gpt-4",
        system_prompt="You are a helpful assistant. Be concise and accurate.",
    )

    # Run the agent
    result = agent.run_sync("What is the capital of France? Answer in one sentence.")

    print(f"\nAgent response: {result.data}")

    print("\n✓ Agent run completed. Check your telemetry backend for traces!")
    print("  - Trace shows agent execution")
    print("  - Model provider and name captured")
    print("  - User prompt and response data recorded")
    print("  - Token usage tracked from underlying OpenAI calls\n")


def example_structured_output():
    """
    Example 2: Agent with Structured Output

    This demonstrates using Pydantic models for type-safe structured responses.

    Telemetry Captured:
    - All basic attributes from Example 1
    - pydantic_ai.result_type: The expected result type
    - pydantic_ai.result.data: Structured response as dict
    """
    print("=" * 80)
    print("Example 2: Agent with Structured Output")
    print("=" * 80)

    # Define a Pydantic model for structured output
    class CityInfo(BaseModel):
        """Information about a city."""

        name: str
        country: str
        population: Optional[int] = None
        famous_for: str

    # Create agent with result_type - This will be automatically instrumented
    agent = Agent(
        "openai:gpt-4",
        result_type=CityInfo,
        system_prompt="You provide structured information about cities.",
    )

    # Run agent with structured output
    result = agent.run_sync("Tell me about Paris")

    # Result is type-safe!
    city_info: CityInfo = result.data
    print(f"\nCity: {city_info.name}")
    print(f"Country: {city_info.country}")
    print(f"Famous for: {city_info.famous_for}")
    if city_info.population:
        print(f"Population: {city_info.population:,}")

    print("\n✓ Structured output completed. Check your telemetry backend!")
    print("  - Trace shows result_type in attributes")
    print("  - Structured data captured in span")
    print("  - Type safety enforced by Pydantic\n")


def example_agent_with_tools():
    """
    Example 3: Agent with Tools/Functions

    This demonstrates agents that can use tools to accomplish tasks.

    Telemetry Captured:
    - All basic attributes
    - pydantic_ai.tools: List of available tool names
    - pydantic_ai.tools.count: Number of tools available
    - Tool execution captured via function calls
    """
    print("=" * 80)
    print("Example 3: Agent with Tools/Functions")
    print("=" * 80)

    # Create agent with tools
    agent = Agent(
        "openai:gpt-4",
        system_prompt="You are a calculator assistant. Use the provided tools to perform calculations.",
    )

    # Define tools
    @agent.tool
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @agent.tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    @agent.tool
    def calculate_percentage(value: float, total: float) -> float:
        """Calculate what percentage 'value' is of 'total'."""
        return (value / total) * 100

    # Run agent with tool usage
    result = agent.run_sync("What is 25% of 80 plus 5 multiplied by 3?")

    print(f"\nAgent response: {result.data}")

    print("\n✓ Tool usage completed. Check your telemetry backend!")
    print("  - Trace shows available tools in attributes")
    print("  - Tool count captured")
    print("  - Individual tool executions visible in trace\n")


def example_multi_provider():
    """
    Example 4: Multi-Provider Support

    This demonstrates using different AI providers with the same code.

    Telemetry Captured:
    - pydantic_ai.model.provider: Provider name (OpenAIModel, AnthropicModel, etc.)
    - Different provider-specific attributes
    """
    print("=" * 80)
    print("Example 4: Multi-Provider Support")
    print("=" * 80)

    providers = []

    # Try OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(("openai:gpt-4", "OpenAI GPT-4"))

    # Try Anthropic (if available)
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(("anthropic:claude-3-sonnet", "Anthropic Claude 3 Sonnet"))

    if not providers:
        print("⚠ Warning: No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    prompt = "What is 2+2? Answer with just the number."

    for model_name, display_name in providers:
        print(f"\nTrying {display_name}...")

        try:
            # Create agent with specific provider
            agent = Agent(model_name, system_prompt="You are a helpful assistant.")

            # Run agent
            result = agent.run_sync(prompt)

            print(f"Response: {result.data}")

        except Exception as e:
            print(f"Error with {display_name}: {e}")

    print("\n✓ Multi-provider test completed. Check your telemetry backend!")
    print("  - Different traces for different providers")
    print("  - Provider information captured in spans")
    print("  - Token usage from each provider tracked\n")


def example_streaming_agent():
    """
    Example 5: Streaming Agent Response

    This demonstrates streaming responses from the agent.

    Telemetry Captured:
    - Span: pydantic_ai.agent.run_stream
    - All standard attributes
    - Streaming chunks aggregated in final response
    """
    print("=" * 80)
    print("Example 5: Streaming Agent Response")
    print("=" * 80)

    # Create agent
    agent = Agent(
        "openai:gpt-4",
        system_prompt="You are a creative writer. Write concisely.",
    )

    print("\nStreaming response:")
    print("-" * 40)

    # Stream the response
    try:
        with agent.run_stream("Write a 2-sentence story about a robot.") as response:
            for chunk in response.stream_text():
                print(chunk, end="", flush=True)
    except AttributeError:
        # Fallback if streaming not available
        result = agent.run_sync("Write a 2-sentence story about a robot.")
        print(result.data)

    print()
    print("-" * 40)

    print("\n✓ Streaming completed. Check your telemetry backend!")
    print("  - Trace shows streaming operation")
    print("  - Complete response captured in span\n")


def example_agent_with_message_history():
    """
    Example 6: Agent with Conversation History

    This demonstrates maintaining conversation history across turns.

    Telemetry Captured:
    - pydantic_ai.message_history.count: Number of previous messages
    - Multiple agent.run spans showing conversation flow
    """
    print("=" * 80)
    print("Example 6: Agent with Conversation History")
    print("=" * 80)

    # Create agent
    agent = Agent(
        "openai:gpt-4",
        system_prompt="You are a helpful assistant with memory of the conversation.",
    )

    # First turn
    print("\nTurn 1: Setting context...")
    result1 = agent.run_sync("My favorite color is blue.")
    print(f"Agent: {result1.data}")

    # Second turn with history
    print("\nTurn 2: Asking follow-up with history...")
    result2 = agent.run_sync("What is my favorite color?", message_history=result1.new_messages())
    print(f"Agent: {result2.data}")

    print("\n✓ Conversation completed. Check your telemetry backend!")
    print("  - Multiple spans showing conversation flow")
    print("  - Message history count captured")
    print("  - Context maintenance visible in traces\n")


def example_model_settings():
    """
    Example 7: Agent with Custom Model Settings

    This demonstrates configuring model parameters.

    Telemetry Captured:
    - gen_ai.request.temperature: Temperature setting
    - gen_ai.request.max_tokens: Max tokens setting
    - gen_ai.request.top_p: Top P setting
    """
    print("=" * 80)
    print("Example 7: Agent with Custom Model Settings")
    print("=" * 80)

    # Create agent
    agent = Agent(
        "openai:gpt-4",
        system_prompt="You are a creative writer.",
    )

    # Run with custom settings
    result = agent.run_sync(
        "Write a creative name for a space robot.",
        model_settings={"temperature": 1.0, "max_tokens": 50, "top_p": 0.95},
    )

    print(f"\nCreative name: {result.data}")

    print("\n✓ Custom settings applied. Check your telemetry backend!")
    print("  - Temperature, max_tokens, top_p captured in span")
    print("  - Model parameters visible in trace attributes\n")


def main():
    """
    Run all Pydantic AI instrumentation examples.
    """
    print("\n" + "=" * 80)
    print("Pydantic AI OpenTelemetry Instrumentation Examples")
    print("=" * 80)
    print("\nThese examples demonstrate automatic tracing of Pydantic AI agents.")
    print("Make sure you have:")
    print("  1. OTEL_EXPORTER_OTLP_ENDPOINT configured (default: http://localhost:4318)")
    print("  2. OPENAI_API_KEY set in environment (or other provider keys)")
    print("  3. An OTLP-compatible backend running (Jaeger, Grafana, etc.)\n")

    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠ Warning: No API keys found. Examples may fail.")
        print("  Set OPENAI_API_KEY or ANTHROPIC_API_KEY\n")

    try:
        # Run examples
        example_basic_agent()
        example_structured_output()
        example_agent_with_tools()
        example_multi_provider()
        example_streaming_agent()
        example_agent_with_message_history()
        example_model_settings()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nWhat to look for in your telemetry:")
        print("  📊 Spans:")
        print("    - pydantic_ai.agent.run: Synchronous agent execution")
        print("    - pydantic_ai.agent.run_sync: Explicit sync execution")
        print("    - pydantic_ai.agent.run_stream: Streaming execution")
        print("    - openai.chat.completions: Underlying LLM calls")
        print("\n  🏷️  Attributes:")
        print("    - gen_ai.system: 'pydantic_ai'")
        print("    - pydantic_ai.agent.name: Agent identifier")
        print("    - pydantic_ai.model.provider: Provider (OpenAIModel, etc.)")
        print("    - gen_ai.request.model: Model name (gpt-4, claude-3, etc.)")
        print("    - pydantic_ai.user_prompt: User's input")
        print("    - pydantic_ai.result_type: Expected result type")
        print("    - pydantic_ai.tools: List of available tools")
        print("    - pydantic_ai.tools.count: Number of tools")
        print("    - pydantic_ai.system_prompts: System prompts used")
        print("    - pydantic_ai.result.data: Agent's response")
        print("    - gen_ai.request.temperature/max_tokens/top_p: Model settings")
        print("\n  💰 Cost Tracking:")
        print("    - Token usage from provider calls aggregated")
        print("    - Cost calculated automatically per agent run")
        print("    - Multi-turn conversations tracked separately")
        print("\n  🔍 Trace Visualization:")
        print("    - Type-safe structured outputs visible")
        print("    - Tool usage and function calls tracked")
        print("    - Multi-provider usage distinguished")
        print("    - Streaming vs. sync operations clear")
        print("    - Conversation history flow visible\n")

        print("📚 Key Features of Pydantic AI:")
        print("  - Type-safe responses with Pydantic validation")
        print("  - Multi-provider support (OpenAI, Anthropic, Gemini, etc.)")
        print("  - Tools/functions with automatic parameter validation")
        print("  - Structured outputs with full IDE support")
        print("  - Async and streaming support")
        print("  - Built-in conversation history management\n")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you have installed: pip install pydantic-ai openai")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Check your configuration and try again.")


if __name__ == "__main__":
    main()
