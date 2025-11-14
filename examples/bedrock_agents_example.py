"""
Example demonstrating OpenTelemetry instrumentation for AWS Bedrock Agents.

This example shows:
1. Agent invocation with automatic instrumentation
2. Knowledge base retrieval
3. Retrieve and Generate (RAG) with Bedrock
4. Session management and tracing

AWS Bedrock Agents is a managed service that helps you build and deploy
generative AI applications with agents that can reason, take actions, and
access knowledge bases.

Requirements:
    pip install genai-otel boto3

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key
    export AWS_DEFAULT_REGION=us-east-1

Prerequisites:
    - AWS Bedrock Agents set up in your AWS account
    - Knowledge base created and configured
    - Agent configured with appropriate permissions
"""

import os
import uuid

# Step 1: Set up OpenTelemetry instrumentation
# This should be done BEFORE importing boto3
from genai_otel import instrument

instrument(
    service_name="bedrock-agents-example",
    endpoint="http://localhost:4318",  # OTLP endpoint
)

# Step 2: Import boto3 after instrumentation is set up
import boto3


def example_invoke_agent():
    """
    Example 1: Basic Agent Invocation

    This demonstrates invoking a Bedrock Agent with automatic instrumentation.

    Telemetry Captured:
    - Span: bedrock.agents.invoke_agent
    - Attributes:
        - gen_ai.system: "bedrock_agents"
        - gen_ai.operation.name: "invoke_agent"
        - bedrock.agent.id: Agent identifier
        - bedrock.agent.alias_id: Agent alias ID
        - bedrock.agent.session_id: Session identifier
        - bedrock.agent.input_text: User input
        - bedrock.agent.enable_trace: Tracing flag
        - bedrock.agent.response.session_id: Response session
        - bedrock.agent.response.content_type: Response content type
    """
    print("=" * 80)
    print("Example 1: Basic Agent Invocation")
    print("=" * 80)

    # Create Bedrock Agents Runtime client
    client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

    # Agent configuration (replace with your agent IDs)
    agent_id = os.environ.get("BEDROCK_AGENT_ID", "AXXXXXX")
    agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "YYYYYYYY")

    # Generate unique session ID
    session_id = str(uuid.uuid4())

    print(f"\nInvoking agent {agent_id} with session {session_id[:8]}...")

    try:
        # Invoke agent - This will be automatically instrumented
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText="What are the key features of AWS Bedrock Agents?",
            enableTrace=True,  # Enable trace for debugging
        )

        # Process streaming response
        completion = ""
        for event in response.get("completion", []):
            chunk = event.get("chunk")
            if chunk:
                completion += chunk.get("bytes", b"").decode("utf-8")

        print(f"\nAgent response: {completion[:200]}...")

        print("\n✓ Agent invocation completed. Check your telemetry backend for traces!")
        print("  - Trace shows agent invocation")
        print("  - Agent ID and alias captured")
        print("  - Session tracking enabled")
        print("  - Input and response metadata tracked\n")

    except client.exceptions.ResourceNotFoundException:
        print("\n⚠ Warning: Agent not found. Set BEDROCK_AGENT_ID and BEDROCK_AGENT_ALIAS_ID")
        print("  This is a demonstration example. Configure your agent first.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


def example_knowledge_base_retrieve():
    """
    Example 2: Knowledge Base Retrieval

    This demonstrates retrieving documents from a Bedrock knowledge base.

    Telemetry Captured:
    - Span: bedrock.agents.retrieve
    - Attributes:
        - gen_ai.operation.name: "retrieve"
        - bedrock.knowledge_base.id: Knowledge base identifier
        - bedrock.retrieval.query: Search query
        - bedrock.retrieval.number_of_results: Number of documents
        - bedrock.retrieval.search_type: Search type (HYBRID, SEMANTIC, etc.)
        - bedrock.retrieval.results_count: Retrieved documents count
    """
    print("=" * 80)
    print("Example 2: Knowledge Base Retrieval")
    print("=" * 80)

    # Create Bedrock Agents Runtime client
    client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

    # Knowledge base configuration
    knowledge_base_id = os.environ.get("BEDROCK_KB_ID", "KBXXXXXX")

    print(f"\nRetrieving from knowledge base {knowledge_base_id}...")

    try:
        # Retrieve documents - This will be automatically instrumented
        response = client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={"text": "What is the pricing model for AWS services?"},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    "overrideSearchType": "HYBRID",  # Use both semantic and keyword search
                }
            },
        )

        # Process retrieval results
        retrieval_results = response.get("retrievalResults", [])
        print(f"\nRetrieved {len(retrieval_results)} documents")

        for i, result in enumerate(retrieval_results[:3], 1):
            content = result.get("content", {}).get("text", "")
            score = result.get("score", 0)
            print(f"\nDocument {i} (score: {score:.3f}):")
            print(f"  {content[:150]}...")

        print("\n✓ Knowledge base retrieval completed. Check your telemetry backend!")
        print("  - Trace shows retrieval operation")
        print("  - Query and search parameters captured")
        print("  - Retrieved document count tracked\n")

    except client.exceptions.ResourceNotFoundException:
        print("\n⚠ Warning: Knowledge base not found. Set BEDROCK_KB_ID")
        print("  Create a knowledge base in AWS Bedrock first.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


def example_retrieve_and_generate():
    """
    Example 3: Retrieve and Generate (RAG)

    This demonstrates the RAG pattern with Bedrock Agents.

    Telemetry Captured:
    - Span: bedrock.agents.retrieve_and_generate
    - Attributes:
        - gen_ai.operation.name: "retrieve_and_generate"
        - bedrock.rag.input_text: User query
        - bedrock.rag.session_id: RAG session ID
        - bedrock.rag.type: Configuration type
        - bedrock.knowledge_base.id: Knowledge base used
        - gen_ai.request.model: Foundation model ARN
        - gen_ai.request.temperature: Model temperature
        - gen_ai.request.max_tokens: Maximum tokens
        - bedrock.rag.output_text: Generated response
        - bedrock.rag.citations_count: Number of citations
    """
    print("=" * 80)
    print("Example 3: Retrieve and Generate (RAG)")
    print("=" * 80)

    # Create Bedrock Agents Runtime client
    client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

    # Configuration
    knowledge_base_id = os.environ.get("BEDROCK_KB_ID", "KBXXXXXX")
    model_arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2"

    # Generate unique session ID for RAG
    session_id = str(uuid.uuid4())

    print(f"\nPerforming RAG with knowledge base {knowledge_base_id}...")

    try:
        # Retrieve and Generate - This will be automatically instrumented
        response = client.retrieve_and_generate(
            input={
                "text": "Explain the benefits of using AWS Bedrock for generative AI applications."
            },
            sessionId=session_id,
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": knowledge_base_id,
                    "modelArn": model_arn,
                    "generationConfiguration": {
                        "inferenceConfig": {
                            "temperature": 0.7,
                            "maxTokens": 500,
                            "topP": 0.9,
                        }
                    },
                },
            },
        )

        # Extract response
        output = response.get("output", {})
        generated_text = output.get("text", "")
        print(f"\nGenerated response:\n{generated_text[:300]}...")

        # Extract citations
        citations = response.get("citations", [])
        print(f"\nCitations: {len(citations)} sources")

        for i, citation in enumerate(citations[:3], 1):
            references = citation.get("retrievedReferences", [])
            if references:
                ref = references[0]
                location = ref.get("location", {})
                print(f"  [{i}] {location.get('s3Location', {}).get('uri', 'N/A')}")

        print("\n✓ RAG operation completed. Check your telemetry backend!")
        print("  - Trace shows complete RAG workflow")
        print("  - Retrieval and generation tracked separately")
        print("  - Model parameters and configuration captured")
        print("  - Citations and sources tracked\n")

    except client.exceptions.ResourceNotFoundException:
        print("\n⚠ Warning: Resource not found. Set BEDROCK_KB_ID")
        print("  Configure knowledge base and verify model access.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


def example_agent_with_session_state():
    """
    Example 4: Agent with Session State

    This demonstrates using session state for context preservation.

    Telemetry Captured:
    - Multiple bedrock.agents.invoke_agent spans
    - Session state attributes
    - Prompt and session attributes captured
    - Conversation flow visible across requests
    """
    print("=" * 80)
    print("Example 4: Agent with Session State")
    print("=" * 80)

    # Create Bedrock Agents Runtime client
    client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

    # Agent configuration
    agent_id = os.environ.get("BEDROCK_AGENT_ID", "AXXXXXX")
    agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "YYYYYYYY")

    # Use same session ID for conversation continuity
    session_id = str(uuid.uuid4())

    print(f"\nStarting conversational session {session_id[:8]}...")

    try:
        # First turn - Set context
        print("\nTurn 1: Setting context...")
        response1 = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText="I'm interested in serverless architectures.",
            sessionState={
                "sessionAttributes": {"topic": "serverless", "expertise_level": "intermediate"},
                "promptSessionAttributes": {"focus": "AWS_services"},
            },
        )

        # Process response
        completion1 = ""
        for event in response1.get("completion", []):
            chunk = event.get("chunk")
            if chunk:
                completion1 += chunk.get("bytes", b"").decode("utf-8")

        print(f"Agent: {completion1[:150]}...")

        # Second turn - Follow-up question
        print("\nTurn 2: Follow-up question with context...")
        response2 = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,  # Same session
            inputText="What are the cost optimization strategies?",
        )

        # Process response
        completion2 = ""
        for event in response2.get("completion", []):
            chunk = event.get("chunk")
            if chunk:
                completion2 += chunk.get("bytes", b"").decode("utf-8")

        print(f"Agent: {completion2[:150]}...")

        print("\n✓ Conversational session completed. Check your telemetry backend!")
        print("  - Multiple invocations with same session traced")
        print("  - Session state attributes captured")
        print("  - Conversation flow visible in trace hierarchy\n")

    except client.exceptions.ResourceNotFoundException:
        print("\n⚠ Warning: Agent not found.")
        print("  This example requires a configured Bedrock Agent.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


def main():
    """
    Run all AWS Bedrock Agents instrumentation examples.
    """
    print("\n" + "=" * 80)
    print("AWS Bedrock Agents OpenTelemetry Instrumentation Examples")
    print("=" * 80)
    print("\nThese examples demonstrate automatic tracing of Bedrock Agents operations.")
    print("Make sure you have:")
    print("  1. OTEL_EXPORTER_OTLP_ENDPOINT configured (default: http://localhost:4318)")
    print("  2. AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    print("  3. AWS_DEFAULT_REGION set (e.g., us-east-1)")
    print("  4. Bedrock Agents and Knowledge Bases created in your AWS account")
    print("  5. An OTLP-compatible backend running (Jaeger, Grafana, etc.)\n")

    # Check if AWS credentials are set
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("⚠ Warning: AWS_ACCESS_KEY_ID not set. Examples may fail.")
        print("  Set AWS credentials to run these examples.\n")

    try:
        # Run examples
        example_invoke_agent()
        example_knowledge_base_retrieve()
        example_retrieve_and_generate()
        example_agent_with_session_state()

        print("=" * 80)
        print("All examples completed!")
        print("=" * 80)
        print("\nWhat to look for in your telemetry:")
        print("  📊 Spans:")
        print("    - bedrock.agents.invoke_agent: Agent invocations")
        print("    - bedrock.agents.retrieve: Knowledge base retrieval")
        print("    - bedrock.agents.retrieve_and_generate: RAG operations")
        print(
            "    - bedrock.runtime.invoke_model: Underlying model calls (from Bedrock instrumentor)"
        )
        print("\n  🏷️  Attributes:")
        print("    - gen_ai.system: 'bedrock_agents'")
        print("    - bedrock.agent.id: Agent identifier")
        print("    - bedrock.agent.session_id: Session tracking")
        print("    - bedrock.agent.input_text: User input")
        print("    - bedrock.knowledge_base.id: Knowledge base ID")
        print("    - bedrock.retrieval.query: Search query")
        print("    - bedrock.retrieval.number_of_results: Retrieved documents")
        print("    - bedrock.rag.output_text: Generated response")
        print("    - bedrock.rag.citations_count: Number of citations")
        print("    - gen_ai.request.model: Foundation model ARN")
        print("    - gen_ai.request.temperature/max_tokens: Model parameters")
        print("\n  💰 Cost Tracking:")
        print("    - Token usage from underlying Bedrock model calls aggregated")
        print("    - Cost calculated via AWS Bedrock instrumentor")
        print("    - Per-operation costs visible")
        print("\n  🔍 Trace Visualization:")
        print("    - Complete RAG workflow traced (retrieve → generate)")
        print("    - Knowledge base queries and results tracked")
        print("    - Agent orchestration visible")
        print("    - Session-based conversation flow clear")
        print("    - Multi-turn interactions tracked with same session ID\n")

        print("📚 Key Features of AWS Bedrock Agents:")
        print("  - Fully managed agent runtime")
        print("  - Integration with knowledge bases")
        print("  - Built-in RAG capabilities")
        print("  - Support for multiple foundation models")
        print("  - Action groups for tool integration")
        print("  - Session management for conversations")
        print("  - Enterprise security and compliance\n")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you have installed: pip install boto3")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Check your configuration and AWS setup.")


if __name__ == "__main__":
    main()
