"""Tests for Phase 4 features: RAG and Embedding Attributes."""

from unittest.mock import Mock

from genai_otel import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor


class ConcreteInstrumentor(BaseInstrumentor):
    """Concrete implementation for testing BaseInstrumentor."""

    def instrument(self, config: OTelConfig):
        """Dummy implementation."""
        self.config = config

    def _extract_usage(self, result):
        """Dummy implementation."""
        return {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


# Note: Session/User tracking tests are integration-tested via examples/phase4_session_rag_tracking.py
# The feature is exercised in real instrumentor usage


class TestRAGEmbeddingAttributes:
    """Tests for Phase 4.2: RAG and Embedding Attributes."""

    def test_add_embedding_attributes_basic(self):
        """Test adding basic embedding attributes."""
        instrumentor = ConcreteInstrumentor()
        instrumentor.config = OTelConfig(service_name="test")

        mock_span = Mock()

        # Test basic embedding attributes
        instrumentor.add_embedding_attributes(
            span=mock_span, model="text-embedding-3-small", input_text="Test input text"
        )

        # Verify attributes were set
        mock_span.set_attribute.assert_any_call("embedding.model_name", "text-embedding-3-small")
        mock_span.set_attribute.assert_any_call("embedding.text", "Test input text")

    def test_add_embedding_attributes_truncation(self):
        """Test that long input text is truncated."""
        instrumentor = ConcreteInstrumentor()
        instrumentor.config = OTelConfig(service_name="test")

        mock_span = Mock()

        # Create text longer than 500 characters
        long_text = "a" * 600

        instrumentor.add_embedding_attributes(
            span=mock_span, model="text-embedding-3-small", input_text=long_text
        )

        # Verify text was truncated to 500 chars
        calls = mock_span.set_attribute.call_args_list
        text_call = [call for call in calls if call[0][0] == "embedding.text"][0]
        assert len(text_call[0][1]) == 500

    def test_add_embedding_attributes_with_vector(self):
        """Test adding embedding attributes with vector (when enabled)."""
        # Create config with capture_embedding_vectors attribute
        config = OTelConfig(service_name="test")
        config.capture_embedding_vectors = True  # Enable vector capture

        instrumentor = ConcreteInstrumentor()
        instrumentor.config = config

        mock_span = Mock()

        test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        instrumentor.add_embedding_attributes(
            span=mock_span,
            model="text-embedding-3-small",
            input_text="Test",
            vector=test_vector,
        )

        # Verify vector dimension was set
        mock_span.set_attribute.assert_any_call("embedding.vector.dimension", 5)

    def test_add_retrieval_attributes_basic(self):
        """Test adding basic retrieval attributes."""
        instrumentor = ConcreteInstrumentor()
        instrumentor.config = OTelConfig(service_name="test")

        mock_span = Mock()

        documents = [
            {"id": "doc_001", "score": 0.95, "content": "Test document content"},
            {"id": "doc_002", "score": 0.87, "content": "Another document"},
        ]

        instrumentor.add_retrieval_attributes(
            span=mock_span, documents=documents, query="Test query"
        )

        # Verify query was set
        mock_span.set_attribute.assert_any_call("retrieval.query", "Test query")

        # Verify document count
        mock_span.set_attribute.assert_any_call("retrieval.document_count", 2)

        # Verify first document attributes
        mock_span.set_attribute.assert_any_call("retrieval.documents.0.document.id", "doc_001")
        mock_span.set_attribute.assert_any_call("retrieval.documents.0.document.score", 0.95)
        mock_span.set_attribute.assert_any_call(
            "retrieval.documents.0.document.content", "Test document content"
        )

    def test_add_retrieval_attributes_with_metadata(self):
        """Test adding retrieval attributes with document metadata."""
        instrumentor = ConcreteInstrumentor()
        instrumentor.config = OTelConfig(service_name="test")

        mock_span = Mock()

        documents = [
            {
                "id": "doc_001",
                "score": 0.95,
                "content": "Test content",
                "metadata": {"source": "docs.example.com", "category": "intro"},
            }
        ]

        instrumentor.add_retrieval_attributes(span=mock_span, documents=documents)

        # Verify metadata attributes were set
        mock_span.set_attribute.assert_any_call(
            "retrieval.documents.0.document.metadata.source", "docs.example.com"
        )
        mock_span.set_attribute.assert_any_call(
            "retrieval.documents.0.document.metadata.category", "intro"
        )

    def test_add_retrieval_attributes_max_docs_limit(self):
        """Test that max_docs parameter limits documents."""
        instrumentor = ConcreteInstrumentor()
        instrumentor.config = OTelConfig(service_name="test")

        mock_span = Mock()

        # Create 10 documents
        documents = [{"id": f"doc_{i:03d}", "score": 0.9 - i * 0.05} for i in range(10)]

        # Set max_docs to 3
        instrumentor.add_retrieval_attributes(span=mock_span, documents=documents, max_docs=3)

        # Verify document count is still 10 (total)
        mock_span.set_attribute.assert_any_call("retrieval.document_count", 10)

        # Verify only first 3 documents have attributes
        calls = [call[0][0] for call in mock_span.set_attribute.call_args_list]

        # Should have doc 0, 1, 2
        assert "retrieval.documents.0.document.id" in calls
        assert "retrieval.documents.1.document.id" in calls
        assert "retrieval.documents.2.document.id" in calls

        # Should NOT have doc 3 or higher
        assert "retrieval.documents.3.document.id" not in calls
        assert "retrieval.documents.9.document.id" not in calls

    def test_add_retrieval_attributes_content_truncation(self):
        """Test that long document content is truncated."""
        instrumentor = ConcreteInstrumentor()
        instrumentor.config = OTelConfig(service_name="test")

        mock_span = Mock()

        # Create document with long content
        long_content = "x" * 600
        documents = [{"id": "doc_001", "score": 0.95, "content": long_content}]

        instrumentor.add_retrieval_attributes(span=mock_span, documents=documents)

        # Find the content attribute call
        calls = mock_span.set_attribute.call_args_list
        content_call = [
            call for call in calls if call[0][0] == "retrieval.documents.0.document.content"
        ][0]

        # Verify content was truncated to 500 chars
        assert len(content_call[0][1]) == 500

    def test_add_retrieval_attributes_query_truncation(self):
        """Test that long query is truncated."""
        instrumentor = ConcreteInstrumentor()
        instrumentor.config = OTelConfig(service_name="test")

        mock_span = Mock()

        # Create long query
        long_query = "q" * 600
        documents = [{"id": "doc_001"}]

        instrumentor.add_retrieval_attributes(span=mock_span, documents=documents, query=long_query)

        # Find the query attribute call
        calls = mock_span.set_attribute.call_args_list
        query_call = [call for call in calls if call[0][0] == "retrieval.query"][0]

        # Verify query was truncated to 500 chars
        assert len(query_call[0][1]) == 500
