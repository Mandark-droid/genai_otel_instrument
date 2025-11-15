"""
Example demonstrating OpenTelemetry instrumentation for Microsoft AutoGen framework.

This example shows:
1. Basic ConversableAgent chat with automatic instrumentation
2. GroupChat with multiple agents and orchestration
3. Agent-to-agent conversations with tracing

AutoGen Note: AutoGen is entering maintenance mode and merging with Semantic Kernel
into the Microsoft Agent Framework (public preview Oct 2025). This example uses the
current AutoGen release.

Requirements:
    pip install genai-otel pyautogen

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

# Step 1: Set up OpenTelemetry instrumentation
# This should be done BEFORE importing AutoGen
from genai_otel import instrument

instrument(
    service_name="autogen-example",
    endpoint="http://localhost:4318",  # OTLP endpoint
)

# Step 2: Import AutoGen after instrumentation is set up
import autogen

# AutoGen configuration for LLM
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}


def example_basic_conversation():
    """
    Example 1: Basic Agent-to-Agent Conversation

    This demonstrates a simple conversation between two agents.

    Telemetry Captured:
    - Span: autogen.initiate_chat
    - Attributes:
        - gen_ai.system: "autogen"
        - gen_ai.operation.name: "conversation.initiate"
        - autogen.agent.name: Sender agent name
        - autogen.conversation.sender: Sender agent name
        - autogen.conversation.recipient: Recipient agent name
        - autogen.message: Initial message content
        - autogen.conversation.max_turns: Max conversation turns
        - autogen.conversation.messages: Total messages exchanged
        - autogen.conversation.last_message: Final message content
    """
    print("=" * 80)
    print("Example 1: Basic Agent-to-Agent Conversation")
    print("=" * 80)

    # Create assistant agent
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message="You are a helpful AI assistant. Provide concise answers.",
    )

    # Create user proxy agent (simulates human user)
    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",  # Don't wait for human input
        max_consecutive_auto_reply=2,
        code_execution_config=False,
    )

    # Initiate conversation - This will be automatically instrumented
    user_proxy.initiate_chat(
        assistant,
        message="What is the capital of France? Please be brief.",
        max_turns=2,
    )

    print("\n✓ Conversation completed. Check your telemetry backend for traces!")
    print("  - Trace shows agent names, message content, and conversation flow")
    print("  - Token usage aggregated from underlying OpenAI calls\n")


def example_group_chat():
    """
    Example 2: Multi-Agent Group Chat

    This demonstrates a group chat with multiple specialized agents coordinated
    by a GroupChatManager.

    Telemetry Captured:
    - Span: autogen.group_chat.run (GroupChatManager execution)
    - Span: autogen.group_chat.select_speaker (Speaker selection)
    - Span: autogen.initiate_chat (Individual agent interactions)
    - Attributes:
        - gen_ai.system: "autogen"
        - gen_ai.operation.name: "group_chat.run" or "group_chat.select_speaker"
        - autogen.manager.name: Manager agent name
        - autogen.group_chat.agent_count: Number of agents
        - autogen.group_chat.agents: List of agent names
        - autogen.group_chat.selection_mode: Speaker selection method
        - autogen.group_chat.max_round: Maximum conversation rounds
    """
    print("=" * 80)
    print("Example 2: Multi-Agent Group Chat")
    print("=" * 80)

    # Create specialized agents
    researcher = autogen.AssistantAgent(
        name="researcher",
        llm_config=llm_config,
        system_message="""You are a research specialist. Your role is to gather
        information and provide factual data. Be concise.""",
    )

    writer = autogen.AssistantAgent(
        name="writer",
        llm_config=llm_config,
        system_message="""You are a content writer. Your role is to take research
        and create clear, engaging content. Be concise.""",
    )

    critic = autogen.AssistantAgent(
        name="critic",
        llm_config=llm_config,
        system_message="""You are a quality critic. Your role is to review content
        and provide constructive feedback. Be brief and specific.""",
    )

    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        system_message="You initiate and coordinate the task.",
    )

    # Create group chat - This enables multi-agent collaboration
    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, writer, critic],
        messages=[],
        max_round=8,
        speaker_selection_method="auto",  # Automatic speaker selection
    )

    # Create manager to orchestrate the group chat
    # This will be automatically instrumented
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Initiate group conversation
    user_proxy.initiate_chat(
        manager,
        message="""Create a brief 2-sentence description of quantum computing.
        Researcher: provide key facts. Writer: create the description.
        Critic: review it.""",
    )

    print("\n✓ Group chat completed. Check your telemetry backend for traces!")
    print("  - Trace shows orchestration of multiple agents")
    print("  - Speaker selection decisions captured")
    print("  - Individual agent contributions tracked")
    print("  - Complete conversation flow visible\n")


def example_sequential_chat():
    """
    Example 3: Sequential Multi-Agent Chat

    This demonstrates sequential chats where output of one agent
    becomes input for the next.

    Telemetry Captured:
    - Multiple autogen.initiate_chat spans
    - Each span captures individual agent interactions
    - Spans show sequential flow through agents
    """
    print("=" * 80)
    print("Example 3: Sequential Multi-Agent Chat")
    print("=" * 80)

    # Create agents for sequential workflow
    data_analyst = autogen.AssistantAgent(
        name="data_analyst",
        llm_config=llm_config,
        system_message="""You analyze data and provide insights.
        Respond with one key insight.""",
    )

    report_writer = autogen.AssistantAgent(
        name="report_writer",
        llm_config=llm_config,
        system_message="""You write professional reports based on analysis.
        Create a brief 1-sentence summary.""",
    )

    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
    )

    # Sequential chat 1: User -> Data Analyst
    print("\nStep 1: User requests analysis...")
    chat_result_1 = user_proxy.initiate_chat(
        data_analyst,
        message="Analyze this trend: AI adoption increased 50% in 2024.",
        max_turns=2,
    )

    # Sequential chat 2: Data Analyst -> Report Writer
    print("\nStep 2: Analysis sent to report writer...")
    if hasattr(chat_result_1, "summary") and chat_result_1.summary:
        analysis = chat_result_1.summary
    else:
        # Fallback if summary not available
        analysis = "AI adoption growth indicates strong market demand"

    chat_result_2 = user_proxy.initiate_chat(
        report_writer,
        message=f"Write a brief summary based on this analysis: {analysis}",
        max_turns=2,
    )

    print("\n✓ Sequential chats completed. Check your telemetry backend!")
    print("  - Trace shows sequential agent flow")
    print("  - Each conversation tracked separately")
    print("  - Parent-child span relationships visible\n")


def example_function_calling_agents():
    """
    Example 4: Agents with Function Calling

    This demonstrates agents that can execute functions/tools.

    Telemetry Captured:
    - autogen.initiate_chat spans
    - Function execution captured via code execution
    - Tool usage and results tracked
    """
    print("=" * 80)
    print("Example 4: Agents with Function Calling")
    print("=" * 80)

    # Define a simple function for the agent to use
    def calculate_roi(investment: float, return_value: float) -> dict:
        """Calculate return on investment percentage."""
        roi = ((return_value - investment) / investment) * 100
        return {
            "investment": investment,
            "return": return_value,
            "roi_percent": round(roi, 2),
            "status": "profit" if roi > 0 else "loss",
        }

    # Create assistant with function
    assistant = autogen.AssistantAgent(
        name="financial_assistant",
        llm_config=llm_config,
        system_message="""You are a financial assistant. You can calculate ROI
        using the calculate_roi function. Be concise.""",
    )

    # Create user proxy that can execute functions
    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # Set to True if you have Docker
        },
    )

    # Register the function
    autogen.register_function(
        calculate_roi,
        caller=assistant,
        executor=user_proxy,
        name="calculate_roi",
        description="Calculate return on investment (ROI) percentage",
    )

    # Initiate conversation that will use the function
    user_proxy.initiate_chat(
        assistant,
        message="""Calculate the ROI for an investment of $10,000 that returned $12,500.
        Use the calculate_roi function.""",
        max_turns=3,
    )

    print("\n✓ Function calling completed. Check your telemetry backend!")
    print("  - Trace shows agent interactions")
    print("  - Function execution tracked")
    print("  - Results captured in spans\n")


def main():
    """
    Run all AutoGen instrumentation examples.
    """
    print("\n" + "=" * 80)
    print("AutoGen OpenTelemetry Instrumentation Examples")
    print("=" * 80)
    print("\nThese examples demonstrate automatic tracing of AutoGen conversations.")
    print("Make sure you have:")
    print("  1. OTEL_EXPORTER_OTLP_ENDPOINT configured (default: http://localhost:4318)")
    print("  2. OPENAI_API_KEY set in environment")
    print("  3. An OTLP-compatible backend running (Jaeger, Grafana, etc.)\n")

    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠ Warning: OPENAI_API_KEY not set. Examples may fail.")
        print("  Set it with: export OPENAI_API_KEY=your_key_here\n")

    try:
        # Run examples
        example_basic_conversation()
        example_group_chat()
        example_sequential_chat()
        example_function_calling_agents()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nWhat to look for in your telemetry:")
        print("  📊 Spans:")
        print("    - autogen.initiate_chat: Agent-to-agent conversations")
        print("    - autogen.group_chat.run: Group chat orchestration")
        print("    - autogen.group_chat.select_speaker: Speaker selection")
        print("    - openai.chat.completions: Underlying LLM calls")
        print("\n  🏷️  Attributes:")
        print("    - gen_ai.system: 'autogen'")
        print("    - autogen.agent.name: Agent names")
        print("    - autogen.conversation.sender/recipient: Conversation parties")
        print("    - autogen.message: Message content")
        print("    - autogen.group_chat.agents: List of agents in group")
        print("    - autogen.group_chat.selection_mode: Speaker selection method")
        print("\n  💰 Cost Tracking:")
        print("    - Token usage from OpenAI calls aggregated")
        print("    - Cost calculated automatically per conversation")
        print("\n  🔍 Trace Visualization:")
        print("    - Parent-child relationships show conversation flow")
        print("    - Group chat orchestration visible")
        print("    - Speaker selection decisions tracked")
        print("    - Sequential agent workflows clear\n")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you have installed: pip install pyautogen")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Check your configuration and try again.")


if __name__ == "__main__":
    main()
