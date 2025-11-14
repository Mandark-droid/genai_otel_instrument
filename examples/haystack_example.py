"""
Example demonstrating OpenTelemetry instrumentation for Haystack NLP framework.

This example shows:
1. Basic pipeline execution with automatic instrumentation
2. RAG (Retrieval-Augmented Generation) pipeline
3. Text generation with Generators
4. Document retrieval with Retrievers

Haystack is a modular NLP framework for building production-ready search and
question-answering systems with support for various LLMs and retrievers.

Requirements:
    pip install genai-otel haystack-ai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

# Step 1: Set up OpenTelemetry instrumentation
# This should be done BEFORE importing Haystack
from genai_otel import instrument

instrument(
    service_name="haystack-example",
    endpoint="http://localhost:4318",  # OTLP endpoint
)

# Step 2: Import Haystack after instrumentation is set up
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIChatGenerator, OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore


def example_basic_pipeline():
    """
    Example 1: Basic Text Generation Pipeline

    This demonstrates a simple pipeline with a generator component.

    Telemetry Captured:
    - Span: haystack.pipeline.run
    - Span: haystack.generator.run
    - Attributes:
        - gen_ai.system: "haystack"
        - gen_ai.operation.name: "pipeline.run" or "generator.run"
        - gen_ai.request.model: Model name
        - haystack.pipeline.components.count: Number of components
        - haystack.pipeline.components: List of component names
        - haystack.generator.prompt: Input prompt
        - haystack.pipeline.output.keys: Output keys
    """
    print("=" * 80)
    print("Example 1: Basic Text Generation Pipeline")
    print("=" * 80)

    # Create a simple pipeline with a generator
    pipeline = Pipeline()

    # Add generator component
    generator = OpenAIGenerator(model="gpt-3.5-turbo", generation_kwargs={"max_tokens": 200})
    pipeline.add_component("generator", generator)

    # Run the pipeline - This will be automatically instrumented
    result = pipeline.run({"generator": {"prompt": "What is Haystack? Answer in 2 sentences."}})

    print(f"\nGenerated text: {result['generator']['replies'][0]}")

    print("\n✓ Pipeline completed. Check your telemetry backend for traces!")
    print("  - Trace shows pipeline execution")
    print("  - Generator component instrumented")
    print("  - Model information and parameters captured")
    print("  - Token usage tracked from OpenAI calls\n")


def example_chat_generator():
    """
    Example 2: Chat Generation Pipeline

    This demonstrates using chat models with conversation history.

    Telemetry Captured:
    - Span: haystack.pipeline.run
    - Span: haystack.chat_generator.run
    - Attributes:
        - haystack.component.type: "chat_generator"
        - haystack.chat_generator.messages.count: Number of messages
        - haystack.chat_generator.last_message: Last message content
        - haystack.chat_generator.last_role: Last message role
        - gen_ai.request.model: Chat model name
    """
    print("=" * 80)
    print("Example 2: Chat Generation Pipeline")
    print("=" * 80)

    # Create pipeline with chat generator
    pipeline = Pipeline()

    # Add chat generator
    chat_generator = OpenAIChatGenerator(
        model="gpt-3.5-turbo", generation_kwargs={"temperature": 0.7}
    )
    pipeline.add_component("chat_generator", chat_generator)

    # Create conversation messages
    messages = [
        ChatMessage.from_system("You are a helpful AI assistant. Be concise."),
        ChatMessage.from_user("What is the capital of France?"),
    ]

    # Run chat pipeline
    result = pipeline.run({"chat_generator": {"messages": messages}})

    print(f"\nChat response: {result['chat_generator']['replies'][0].content}")

    print("\n✓ Chat pipeline completed. Check your telemetry backend!")
    print("  - Trace shows chat generator execution")
    print("  - Message history captured")
    print("  - Conversation flow visible\n")


def example_rag_pipeline():
    """
    Example 3: RAG (Retrieval-Augmented Generation) Pipeline

    This demonstrates a complete RAG pipeline with retriever and generator.

    Telemetry Captured:
    - Span: haystack.pipeline.run
    - Span: haystack.retriever.run
    - Span: haystack.generator.run
    - Attributes:
        - haystack.pipeline.connections.count: Number of edges in graph
        - haystack.retriever.query: Search query
        - haystack.retriever.top_k: Number of documents to retrieve
        - haystack.output.retriever.documents.count: Retrieved documents
        - haystack.output.generator.replies.count: Generated responses
    """
    print("=" * 80)
    print("Example 3: RAG Pipeline")
    print("=" * 80)

    # Create document store and add documents
    document_store = InMemoryDocumentStore()

    # Sample documents about Haystack
    from haystack import Document

    documents = [
        Document(
            content="Haystack is an open-source NLP framework for building production-ready applications."
        ),
        Document(
            content="Haystack supports various LLM providers including OpenAI, Anthropic, and Cohere."
        ),
        Document(content="Haystack uses a pipeline architecture with modular components."),
        Document(
            content="Haystack is ideal for building RAG (Retrieval-Augmented Generation) systems."
        ),
    ]

    # Write documents to store
    writer = DocumentWriter(document_store=document_store)
    writer.run(documents=documents)

    # Create RAG pipeline
    pipeline = Pipeline()

    # Add retriever
    retriever = InMemoryBM25Retriever(document_store=document_store)
    pipeline.add_component("retriever", retriever)

    # Add prompt builder
    template = """
    Answer the question based on the provided context.

    Context:
    {% for doc in documents %}
    - {{ doc.content }}
    {% endfor %}

    Question: {{ query }}
    Answer (be concise):
    """
    prompt_builder = PromptBuilder(template=template)
    pipeline.add_component("prompt_builder", prompt_builder)

    # Add generator
    generator = OpenAIGenerator(model="gpt-3.5-turbo", generation_kwargs={"max_tokens": 100})
    pipeline.add_component("generator", generator)

    # Connect components
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")

    # Run RAG pipeline
    query = "What is Haystack used for?"
    result = pipeline.run(
        {
            "retriever": {"query": query, "top_k": 3},
            "prompt_builder": {"query": query},
        }
    )

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(result['retriever']['documents'])} documents")
    print(f"Answer: {result['generator']['replies'][0]}")

    print("\n✓ RAG pipeline completed. Check your telemetry backend!")
    print("  - Trace shows complete RAG workflow")
    print("  - Retriever and generator spans visible")
    print("  - Document retrieval metrics captured")
    print("  - Pipeline connections tracked\n")


def example_complex_pipeline():
    """
    Example 4: Complex Multi-Component Pipeline

    This demonstrates a pipeline with multiple generators and conditional logic.

    Telemetry Captured:
    - Multiple haystack.generator.run spans
    - Pipeline graph structure with nodes and edges
    - Component-level instrumentation
    - Nested span hierarchy
    """
    print("=" * 80)
    print("Example 4: Complex Multi-Component Pipeline")
    print("=" * 80)

    # Create complex pipeline
    pipeline = Pipeline()

    # Add multiple generators
    generator1 = OpenAIGenerator(
        model="gpt-3.5-turbo", generation_kwargs={"max_tokens": 50, "temperature": 0.9}
    )
    pipeline.add_component("creative_generator", generator1)

    generator2 = OpenAIGenerator(
        model="gpt-3.5-turbo", generation_kwargs={"max_tokens": 50, "temperature": 0.1}
    )
    pipeline.add_component("factual_generator", generator2)

    # Run both generators (in practice, you'd use routing/branching)
    print("\nGenerating creative and factual responses...")

    result1 = pipeline.run(
        {"creative_generator": {"prompt": "Write a creative name for a coffee shop."}}
    )

    result2 = pipeline.run({"factual_generator": {"prompt": "What are the ingredients in coffee?"}})

    print(f"\nCreative: {result1['creative_generator']['replies'][0]}")
    print(f"Factual: {result2['factual_generator']['replies'][0]}")

    print("\n✓ Complex pipeline completed. Check your telemetry backend!")
    print("  - Multiple generator executions traced")
    print("  - Different temperature settings captured")
    print("  - Component names distinguish operations\n")


def example_pipeline_with_metadata():
    """
    Example 5: Pipeline with Metadata

    This demonstrates pipelines with custom metadata for better observability.

    Telemetry Captured:
    - haystack.pipeline.metadata.*: Custom metadata fields
    - Pipeline identification and versioning
    - Enhanced trace context
    """
    print("=" * 80)
    print("Example 5: Pipeline with Metadata")
    print("=" * 80)

    # Create pipeline with metadata
    pipeline = Pipeline(
        metadata={
            "name": "qa_pipeline",
            "version": "1.0.0",
            "author": "test_user",
            "description": "Question answering pipeline with metadata",
        }
    )

    # Add generator
    generator = OpenAIGenerator(model="gpt-3.5-turbo")
    pipeline.add_component("generator", generator)

    # Run pipeline
    result = pipeline.run({"generator": {"prompt": "What is 2+2?"}})

    print(f"\nAnswer: {result['generator']['replies'][0]}")

    print("\n✓ Pipeline with metadata completed. Check your telemetry backend!")
    print("  - Custom metadata fields captured in span attributes")
    print("  - Pipeline identification visible in traces")
    print("  - Version information tracked\n")


def main():
    """
    Run all Haystack instrumentation examples.
    """
    print("\n" + "=" * 80)
    print("Haystack OpenTelemetry Instrumentation Examples")
    print("=" * 80)
    print("\nThese examples demonstrate automatic tracing of Haystack pipelines.")
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
        example_basic_pipeline()
        example_chat_generator()
        example_rag_pipeline()
        example_complex_pipeline()
        example_pipeline_with_metadata()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nWhat to look for in your telemetry:")
        print("  📊 Spans:")
        print("    - haystack.pipeline.run: Pipeline execution")
        print("    - haystack.generator.run: Text generation")
        print("    - haystack.chat_generator.run: Chat generation")
        print("    - haystack.retriever.run: Document retrieval")
        print("    - openai.chat.completions: Underlying OpenAI calls")
        print("\n  🏷️  Attributes:")
        print("    - gen_ai.system: 'haystack'")
        print("    - haystack.pipeline.components.count: Number of components")
        print("    - haystack.pipeline.components: List of component names")
        print("    - haystack.pipeline.connections.count: Pipeline edges")
        print("    - haystack.component.type: Component type (generator, retriever, etc.)")
        print("    - gen_ai.request.model: LLM model used")
        print("    - haystack.generator.prompt: Input prompt")
        print("    - haystack.retriever.query: Search query")
        print("    - haystack.retriever.top_k: Documents to retrieve")
        print("    - haystack.pipeline.metadata.*: Custom metadata")
        print("\n  💰 Cost Tracking:")
        print("    - Token usage from OpenAI calls aggregated")
        print("    - Cost calculated automatically per pipeline run")
        print("    - Per-component costs visible")
        print("\n  🔍 Trace Visualization:")
        print("    - Pipeline graph structure visible")
        print("    - Component execution order clear")
        print("    - RAG workflow fully traced (retrieve → build → generate)")
        print("    - Multi-component pipelines show parallel/sequential execution")
        print("    - Nested spans show component hierarchy\n")

        print("📚 Key Features of Haystack:")
        print("  - Modular pipeline architecture")
        print("  - Support for multiple LLM providers")
        print("  - Built-in RAG components (retrievers, generators, etc.)")
        print("  - Document stores for vector/keyword search")
        print("  - Production-ready with monitoring and logging")
        print("  - Flexible component composition\n")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you have installed: pip install haystack-ai")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Check your configuration and try again.")


if __name__ == "__main__":
    main()
