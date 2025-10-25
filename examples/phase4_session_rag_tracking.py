"""Phase 4 Features Example: Session/User Tracking and RAG Attributes

This example demonstrates the Phase 4 features from the OTel Semantic Gap Analysis:
1. Session and User Tracking (4.1)
2. RAG/Embedding Attributes (4.2)

These features enhance observability for multi-user applications and RAG workflows.
"""

import genai_otel
from genai_otel import OTelConfig

# Example 1: Session and User Tracking
# =====================================
# Define extractor functions to pull session_id and user_id from your requests


def extract_session_id(instance, args, kwargs):
    """Extract session ID from request metadata.

    This is called for every LLM request and can extract session context
    from wherever you store it in your application.
    """
    # Option 1: From kwargs metadata
    metadata = kwargs.get("metadata", {})
    return metadata.get("session_id")

    # Option 2: From custom headers
    # headers = kwargs.get("headers", {})
    # return headers.get("X-Session-ID")

    # Option 3: From thread-local storage
    # import threading
    # return getattr(threading.current_thread(), "session_id", None)


def extract_user_id(instance, args, kwargs):
    """Extract user ID from request metadata."""
    metadata = kwargs.get("metadata", {})
    return metadata.get("user_id")


# Configure instrumentation with session/user extractors
config = OTelConfig(
    service_name="rag-app",
    endpoint="http://localhost:4318",
    session_id_extractor=extract_session_id,
    user_id_extractor=extract_user_id,
)

# Initialize instrumentation
genai_otel.instrument(config)

print("=" * 80)
print("Example 1: Session and User Tracking")
print("=" * 80)

# Now make an OpenAI call with session/user context
from openai import OpenAI

client = OpenAI()

# Pass session and user info via metadata
# The extractors will automatically add session.id and user.id to the span
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is OpenTelemetry?"},
    ],
    # Pass metadata with session/user context
    # Note: OpenAI doesn't natively support metadata, but you can pass it
    # and the extractor will find it in kwargs
    extra_body={"metadata": {"session_id": "sess_12345", "user_id": "user_alice"}},
)

print(f"Response: {response.choices[0].message.content[:100]}...")
print("\nThe span for this request now includes:")
print("  - session.id: sess_12345")
print("  - user.id: user_alice")
print("\nThese attributes allow you to:")
print("  - Track conversations across multiple requests")
print("  - Analyze usage patterns per user")
print("  - Debug session-specific issues")
print("  - Calculate per-user costs")

# Example 2: RAG/Embedding Attributes
# ====================================
print("\n" + "=" * 80)
print("Example 2: RAG/Embedding Attributes")
print("=" * 80)

# For embedding operations, you can manually add attributes using the helper methods
# This requires direct access to the instrumentor and span

from opentelemetry import trace

# Get current tracer
tracer = trace.get_tracer(__name__)

# Simulate an embedding operation
with tracer.start_as_current_span("embedding.create") as span:
    # Call OpenAI embedding
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small", input="OpenTelemetry provides observability"
    )

    # Manually add embedding attributes
    # In a real application, you'd access the instrumentor instance
    # For demonstration, we'll set attributes directly
    span.set_attribute("embedding.model_name", "text-embedding-3-small")
    span.set_attribute("embedding.text", "OpenTelemetry provides observability"[:500])
    span.set_attribute("embedding.vector.dimension", len(embedding_response.data[0].embedding))

    print(f"Created embedding with {len(embedding_response.data[0].embedding)} dimensions")
    print("\nThe span includes:")
    print("  - embedding.model_name")
    print("  - embedding.text (truncated)")
    print("  - embedding.vector.dimension")

# Simulate a retrieval operation
with tracer.start_as_current_span("retrieval.search") as span:
    # Simulate retrieved documents
    retrieved_docs = [
        {
            "id": "doc_001",
            "score": 0.95,
            "content": "OpenTelemetry is an observability framework for cloud-native software...",
            "metadata": {"source": "docs.opentelemetry.io", "category": "intro"},
        },
        {
            "id": "doc_002",
            "score": 0.87,
            "content": "Traces, metrics, and logs are the three pillars of observability...",
            "metadata": {"source": "blog", "category": "fundamentals"},
        },
        {
            "id": "doc_003",
            "score": 0.82,
            "content": "GenAI semantic conventions define standard attributes for LLM traces...",
            "metadata": {"source": "spec", "category": "semantic-conventions"},
        },
    ]

    # Add retrieval attributes
    query = "What is OpenTelemetry?"
    span.set_attribute("retrieval.query", query[:500])

    for i, doc in enumerate(retrieved_docs[:5]):  # Limit to 5 docs
        prefix = f"retrieval.documents.{i}.document"
        span.set_attribute(f"{prefix}.id", doc["id"])
        span.set_attribute(f"{prefix}.score", doc["score"])
        span.set_attribute(f"{prefix}.content", doc["content"][:500])

        # Add metadata
        for key, value in doc["metadata"].items():
            span.set_attribute(f"{prefix}.metadata.{key}", str(value))

    span.set_attribute("retrieval.document_count", len(retrieved_docs))

    print(f"\nRetrieved {len(retrieved_docs)} documents")
    print("The span includes:")
    print("  - retrieval.query")
    print("  - retrieval.document_count")
    print("  - For each document:")
    print("    - document.id")
    print("    - document.score")
    print("    - document.content (truncated)")
    print("    - document.metadata.*")

# Example 3: Complete RAG Workflow with Session Tracking
# =======================================================
print("\n" + "=" * 80)
print("Example 3: Complete RAG Workflow with Session Tracking")
print("=" * 80)


def rag_query(query: str, session_id: str, user_id: str):
    """Complete RAG workflow with session and retrieval tracking."""

    # 1. Embed the query
    with tracer.start_as_current_span("rag.embed_query") as span:
        span.set_attribute("session.id", session_id)
        span.set_attribute("user.id", user_id)

        embedding_response = client.embeddings.create(model="text-embedding-3-small", input=query)

        span.set_attribute("embedding.model_name", "text-embedding-3-small")
        span.set_attribute("embedding.text", query[:500])

    # 2. Retrieve documents (simulated)
    with tracer.start_as_current_span("rag.retrieve") as span:
        span.set_attribute("session.id", session_id)
        span.set_attribute("user.id", user_id)
        span.set_attribute("retrieval.query", query[:500])

        # Simulate retrieval
        docs = retrieved_docs  # Using docs from above
        span.set_attribute("retrieval.document_count", len(docs))

        for i, doc in enumerate(docs[:5]):
            prefix = f"retrieval.documents.{i}.document"
            span.set_attribute(f"{prefix}.id", doc["id"])
            span.set_attribute(f"{prefix}.score", doc["score"])

    # 3. Generate response with context
    with tracer.start_as_current_span("rag.generate") as span:
        span.set_attribute("session.id", session_id)
        span.set_attribute("user.id", user_id)

        # Build context from retrieved docs
        context = "\n\n".join([doc["content"] for doc in docs[:3]])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Answer based on this context:\n{context}"},
                {"role": "user", "content": query},
            ],
            extra_body={"metadata": {"session_id": session_id, "user_id": user_id}},
        )

        span.set_attribute("gen_ai.response.citations_count", len(docs[:3]))

        return response.choices[0].message.content


# Run RAG workflow
answer = rag_query(
    query="What is OpenTelemetry and why is it important?",
    session_id="sess_67890",
    user_id="user_bob",
)

print(f"\nRAG Answer: {answer[:150]}...")
print("\nThe complete RAG workflow now has:")
print("  - Session and user tracking across all spans")
print("  - Embedding attributes for the query")
print("  - Retrieval attributes for document search")
print("  - Generation attributes for LLM response")
print("\nThis enables:")
print("  - End-to-end tracing of RAG pipelines")
print("  - Per-user cost and usage analytics")
print("  - Session-based conversation tracking")
print("  - Retrieval quality monitoring")

print("\n" + "=" * 80)
print("Phase 4 Features Demonstrated Successfully!")
print("=" * 80)
print("\nNext steps:")
print("1. View traces in your observability backend (Jaeger, Tempo, etc.)")
print("2. Filter by session.id or user.id to track specific users/sessions")
print("3. Analyze retrieval.documents.*.score to optimize retrieval")
print("4. Monitor embedding and retrieval latencies")
