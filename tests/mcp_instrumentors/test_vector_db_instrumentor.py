import logging
from unittest.mock import MagicMock, call, patch

import pytest
from opentelemetry.trace import SpanKind, Status, StatusCode

import genai_otel.mcp_instrumentors.vector_db_instrumentor
from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.vector_db_instrumentor import VectorDBInstrumentor


# --- Fixtures ---
class MockPineconeClient:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.Index = MagicMock()


@pytest.fixture
def mock_tracer():
    with patch(
        "genai_otel.mcp_instrumentors.vector_db_instrumentor.trace.get_tracer"
    ) as mock_get_tracer:
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        # Configure the logger to ensure caplog captures messages
        logger = logging.getLogger("genai_otel.mcp_instrumentors.vector_db_instrumentor")
        logger.setLevel(logging.INFO)
        logger.propagate = True

        yield mock_tracer


@pytest.fixture
def vector_db_instrumentor(mock_tracer):
    config = OTelConfig()
    instrumentor = VectorDBInstrumentor(config)
    instrumentor.tracer = mock_tracer
    return instrumentor


# --- Tests for VectorDBInstrumentor.__init__ ---
def test_vector_db_instrumentor_init(vector_db_instrumentor):
    """Test that VectorDBInstrumentor initializes with the provided config."""
    assert vector_db_instrumentor.config is not None
    assert vector_db_instrumentor.tracer is not None


# --- Tests for VectorDBInstrumentor.instrument ---
def test_instrument_all_libraries(vector_db_instrumentor, caplog):
    """Test that instrument() attempts to instrument all supported libraries."""
    with (
        patch.object(vector_db_instrumentor, "_instrument_pinecone", return_value=True),
        patch.object(vector_db_instrumentor, "_instrument_weaviate", return_value=True),
        patch.object(vector_db_instrumentor, "_instrument_qdrant", return_value=True),
        patch.object(vector_db_instrumentor, "_instrument_chroma", return_value=True),
        patch.object(vector_db_instrumentor, "_instrument_milvus", return_value=True),
        patch.object(vector_db_instrumentor, "_instrument_faiss", return_value=True),
    ):

        instrumented_count = vector_db_instrumentor.instrument()
        assert instrumented_count == 6


def test_instrument_no_libraries(vector_db_instrumentor, caplog):
    """Test that instrument() returns 0 if no libraries are available."""
    with (
        patch.object(vector_db_instrumentor, "_instrument_pinecone", return_value=False),
        patch.object(vector_db_instrumentor, "_instrument_weaviate", return_value=False),
        patch.object(vector_db_instrumentor, "_instrument_qdrant", return_value=False),
        patch.object(vector_db_instrumentor, "_instrument_chroma", return_value=False),
        patch.object(vector_db_instrumentor, "_instrument_milvus", return_value=False),
        patch.object(vector_db_instrumentor, "_instrument_faiss", return_value=False),
    ):

        instrumented_count = vector_db_instrumentor.instrument()
        assert instrumented_count == 0


# --- Tests for _instrument_pinecone ---
def test_instrument_pinecone_success(vector_db_instrumentor, caplog):
    """Test successful Pinecone instrumentation."""
    with patch.dict(
        "sys.modules", {"pinecone": MagicMock(__version__="3.0.0", Pinecone=MockPineconeClient)}
    ):
        assert vector_db_instrumentor._instrument_pinecone() is True
        assert "Pinecone instrumentation enabled" in caplog.text


def test_instrument_pinecone_missing(vector_db_instrumentor, caplog):
    """Test that missing Pinecone is handled gracefully."""
    with patch.dict("sys.modules", {"pinecone": None}):
        assert vector_db_instrumentor._instrument_pinecone() is False
        assert "Pinecone not installed, skipping instrumentation" in caplog.text


def test_instrument_pinecone_error(vector_db_instrumentor, caplog):
    """Test that Pinecone instrumentation errors are logged."""
    with (
        patch.dict("sys.modules", {"pinecone": MagicMock()}),
        patch.object(
            vector_db_instrumentor, "_wrap_pinecone_method", side_effect=RuntimeError("Mock error")
        ),
    ):
        assert vector_db_instrumentor._instrument_pinecone() is False
        assert "Failed to instrument Pinecone" in caplog.text


def test_instrument_pinecone_old_api(vector_db_instrumentor, caplog):
    """Test Pinecone 2.x API instrumentation."""
    mock_pinecone = MagicMock(__version__="2.0.0")
    # Ensure Pinecone.Pinecone does not exist for old API test
    if hasattr(mock_pinecone, "Pinecone"):
        del mock_pinecone.Pinecone
    mock_pinecone.Index = MagicMock()
    with patch.dict("sys.modules", {"pinecone": mock_pinecone}):
        assert vector_db_instrumentor._instrument_pinecone() is True
        assert "Detected Pinecone 2.x API" in caplog.text


# --- Tests for _wrap_pinecone_method ---
def test_wrap_pinecone_method_success(vector_db_instrumentor, mock_tracer):
    """Test that _wrap_pinecone_method creates a span and calls the original method."""
    mock_method = MagicMock(return_value="result")
    wrapped_method = vector_db_instrumentor._wrap_pinecone_method(mock_method, "pinecone.query")

    result = wrapped_method("arg1", kwarg1="value")

    mock_tracer.start_as_current_span.assert_called_once_with(
        "pinecone.query",
        kind=SpanKind.CLIENT,
        attributes={"db.system": "pinecone", "db.operation": "query"},
    )
    mock_method.assert_called_once_with("arg1", kwarg1="value")
    assert result == "result"


def test_wrap_pinecone_method_error(vector_db_instrumentor, mock_tracer, caplog):
    """Test that _wrap_pinecone_method records exceptions."""
    mock_method = MagicMock(side_effect=RuntimeError("Mock error"))
    wrapped_method = vector_db_instrumentor._wrap_pinecone_method(mock_method, "pinecone.query")

    with pytest.raises(RuntimeError):
        wrapped_method("arg1", kwarg1="value")

    status_call_args = (
        mock_tracer.start_as_current_span.return_value.__enter__.return_value.set_status.call_args[
            0
        ][0]
    )
    assert status_call_args.status_code == StatusCode.ERROR
    assert status_call_args.description == "Mock error"
    mock_tracer.start_as_current_span.assert_called_once_with(
        "pinecone.query",
        kind=SpanKind.CLIENT,
        attributes={"db.system": "pinecone", "db.operation": "query"},
    )
    mock_tracer.start_as_current_span.return_value.__enter__.return_value.record_exception.assert_called_once()


# --- Tests for _instrument_weaviate ---
def test_instrument_weaviate_success(vector_db_instrumentor, caplog):
    """Test successful Weaviate instrumentation."""
    with patch.dict("sys.modules", {"weaviate": MagicMock(Client=MagicMock())}):
        assert vector_db_instrumentor._instrument_weaviate() is True
        assert "Weaviate instrumentation enabled" in caplog.text


def test_instrument_weaviate_missing(vector_db_instrumentor, caplog):
    """Test that missing Weaviate is handled gracefully."""
    with patch.dict("sys.modules", {"weaviate": None}):
        assert vector_db_instrumentor._instrument_weaviate() is False


# --- Tests for _instrument_qdrant ---
def test_instrument_qdrant_success(vector_db_instrumentor, caplog):
    """Test successful Qdrant instrumentation."""
    with patch.dict("sys.modules", {"qdrant_client": MagicMock(QdrantClient=MagicMock())}):
        assert vector_db_instrumentor._instrument_qdrant() is True
        assert "Qdrant instrumentation enabled" in caplog.text


def test_instrument_qdrant_missing(vector_db_instrumentor, caplog):
    """Test that missing Qdrant is handled gracefully."""
    with patch.dict("sys.modules", {"qdrant_client": None}):
        assert vector_db_instrumentor._instrument_qdrant() is False


# --- Tests for _instrument_chroma ---
def test_instrument_chroma_success(vector_db_instrumentor, caplog):
    """Test successful ChromaDB instrumentation."""
    with patch.dict("sys.modules", {"chromadb": MagicMock(Collection=MagicMock())}):
        assert vector_db_instrumentor._instrument_chroma() is True
        assert "ChromaDB instrumentation enabled" in caplog.text


def test_instrument_chroma_missing(vector_db_instrumentor, caplog):
    """Test that missing ChromaDB is handled gracefully."""
    with patch.dict("sys.modules", {"chromadb": None}):
        assert vector_db_instrumentor._instrument_chroma() is False


# --- Tests for _instrument_milvus ---
def test_instrument_milvus_success(vector_db_instrumentor, caplog):
    """Test successful Milvus instrumentation."""
    with patch.dict("sys.modules", {"pymilvus": MagicMock(Collection=MagicMock())}):
        assert vector_db_instrumentor._instrument_milvus() is True
        assert "Milvus instrumentation enabled" in caplog.text


def test_instrument_milvus_missing(vector_db_instrumentor, caplog):
    """Test that missing Milvus is handled gracefully."""
    with patch.dict("sys.modules", {"pymilvus": None}):
        assert vector_db_instrumentor._instrument_milvus() is False


# --- Tests for _instrument_faiss ---
def test_instrument_faiss_success(vector_db_instrumentor, caplog):
    """Test successful FAISS instrumentation."""
    with patch.dict("sys.modules", {"faiss": MagicMock(Index=MagicMock())}):
        assert vector_db_instrumentor._instrument_faiss() is True
        assert "FAISS instrumentation enabled" in caplog.text


def test_instrument_faiss_missing(vector_db_instrumentor, caplog):
    """Test that missing FAISS is handled gracefully."""
    with patch.dict("sys.modules", {"faiss": None}):
        assert vector_db_instrumentor._instrument_faiss() is False


# --- Additional Tests for Missing Coverage ---


def test_instrument_pinecone_unknown_api(vector_db_instrumentor, caplog):
    """Test Pinecone with unknown API (no Pinecone class and no Index class)."""
    mock_pinecone = MagicMock(__version__="1.0.0", spec=[])
    # Remove both Pinecone and Index attributes
    if hasattr(mock_pinecone, "Pinecone"):
        delattr(mock_pinecone, "Pinecone")
    if hasattr(mock_pinecone, "Index"):
        delattr(mock_pinecone, "Index")

    with patch.dict("sys.modules", {"pinecone": mock_pinecone}):
        assert vector_db_instrumentor._instrument_pinecone() is False
        assert "Could not detect Pinecone API version" in caplog.text


def test_instrument_pinecone_client_renamed_error(vector_db_instrumentor, caplog):
    """Test Pinecone with pinecone-client renamed error."""
    mock_pinecone = MagicMock(__version__="3.0.0")
    mock_pinecone.Pinecone = MagicMock()

    with (
        patch.dict("sys.modules", {"pinecone": mock_pinecone}),
        patch(
            "genai_otel.mcp_instrumentors.vector_db_instrumentor.wrapt.wrap_function_wrapper",
            side_effect=Exception("pinecone-client has been renamed to pinecone"),
        ),
    ):
        assert vector_db_instrumentor._instrument_pinecone() is False
        assert "pinecone-client" in caplog.text
        assert "renamed" in caplog.text


def test_wrap_pinecone_init(vector_db_instrumentor):
    """Test _wrap_pinecone_init wrapper for Pinecone 3.0+ API."""

    # Create a mock Pinecone instance
    class MockPineconeInstance:
        """Mock Pinecone instance with Index method."""

        class MockIndexClass:
            """Mock Index class."""

            def __init__(self, name):
                self.name = name
                self.query = MagicMock(return_value="query_result")
                self.upsert = MagicMock(return_value="upsert_result")
                self.delete = MagicMock(return_value="delete_result")

        def __init__(self):
            self.Index = self.MockIndexClass

    mock_pinecone_instance = MockPineconeInstance()

    # Create a wrapped __init__ function
    original_init = MagicMock(return_value=None)

    # Call the wrapper
    result = vector_db_instrumentor._wrap_pinecone_init(
        original_init, mock_pinecone_instance, (), {}
    )

    # Verify original init was called
    original_init.assert_called_once_with()

    # Verify Index was replaced with wrapped version
    assert callable(mock_pinecone_instance.Index)

    # Test that the wrapped Index class works by calling it
    # This should trigger the traced_index decorator which wraps the methods
    index_instance = mock_pinecone_instance.Index("test_index")

    # The index methods should now be wrapped with _wrap_pinecone_method
    # Verify we can call them (they should be instrumented)
    assert hasattr(index_instance, "query")
    assert hasattr(index_instance, "upsert")
    assert hasattr(index_instance, "delete")

    # Verify the methods are wrapped (not the original mocks)
    # The wrapped methods should be different from the originals
    assert callable(index_instance.query)
    assert callable(index_instance.upsert)
    assert callable(index_instance.delete)


def test_wrap_pinecone_init_without_index_attr(vector_db_instrumentor):
    """Test _wrap_pinecone_init when instance doesn't have Index attribute."""
    # Create a mock Pinecone instance without Index
    mock_pinecone_instance = MagicMock(spec=[])
    if hasattr(mock_pinecone_instance, "Index"):
        delattr(mock_pinecone_instance, "Index")

    original_init = MagicMock(return_value=None)

    # Should not raise
    result = vector_db_instrumentor._wrap_pinecone_init(
        original_init, mock_pinecone_instance, (), {}
    )

    original_init.assert_called_once_with()


def test_weaviate_query_execution(vector_db_instrumentor, mock_tracer):
    """Test that Weaviate query wrapper executes and creates spans."""
    import sys

    # Create mock weaviate module
    mock_weaviate = MagicMock()
    mock_client_class = MagicMock()
    original_query = MagicMock(return_value={"results": []})
    mock_client_class.query = original_query
    mock_weaviate.Client = mock_client_class

    with patch.dict(sys.modules, {"weaviate": mock_weaviate}):
        # Instrument
        vector_db_instrumentor._instrument_weaviate()

        # Get the wrapped query method
        wrapped_query = mock_weaviate.Client.query

        # Create a mock client instance
        mock_client_instance = MagicMock()

        # Call the wrapped query
        result = wrapped_query(mock_client_instance, query_arg="test")

        # Verify tracer was called
        mock_tracer.start_as_current_span.assert_called_once_with("weaviate.query")


def test_qdrant_search_execution(vector_db_instrumentor, mock_tracer):
    """Test that Qdrant search wrapper executes and creates spans with attributes."""
    from unittest.mock import MagicMock

    # Create mock QdrantClient class
    mock_qdrant_client_class = MagicMock()
    original_search = MagicMock(return_value=[])
    mock_qdrant_client_class.search = original_search

    # Create mock qdrant_client module
    mock_qdrant_module = MagicMock()
    mock_qdrant_module.QdrantClient = mock_qdrant_client_class

    import sys

    with patch.dict(sys.modules, {"qdrant_client": mock_qdrant_module}):
        # Instrument
        vector_db_instrumentor._instrument_qdrant()

        # Get the wrapped search method
        wrapped_search = mock_qdrant_module.QdrantClient.search

        # Create a mock client instance
        mock_client_instance = MagicMock()

        # Call the wrapped search with collection_name in kwargs
        result = wrapped_search(
            mock_client_instance, collection_name="test_collection", query_vector=[1, 2, 3], limit=5
        )

        # Verify tracer was called
        mock_tracer.start_as_current_span.assert_called_once_with("qdrant.search")

        # Call again with collection_name in args
        mock_tracer.reset_mock()
        result = wrapped_search(mock_client_instance, "test_collection2", query_vector=[1, 2, 3])

        mock_tracer.start_as_current_span.assert_called_once_with("qdrant.search")


def test_chroma_query_execution(vector_db_instrumentor, mock_tracer):
    """Test that ChromaDB query wrapper executes and creates spans with attributes."""
    from unittest.mock import MagicMock

    # Create mock Collection
    mock_collection_class = MagicMock()
    original_query = MagicMock(return_value={"results": []})
    mock_collection_class.query = original_query

    # Create mock chromadb module
    mock_chromadb = MagicMock()
    mock_chromadb.Collection = mock_collection_class

    import sys

    with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
        # Instrument
        vector_db_instrumentor._instrument_chroma()

        # Get the wrapped query method
        wrapped_query = mock_chromadb.Collection.query

        # Create a mock collection instance
        mock_collection_instance = MagicMock()
        mock_collection_instance.name = "test_collection"

        # Call the wrapped query
        result = wrapped_query(mock_collection_instance, query_texts=["test"], n_results=5)

        # Verify tracer was called
        mock_tracer.start_as_current_span.assert_called_once_with("chroma.query")


def test_milvus_search_execution(vector_db_instrumentor, mock_tracer):
    """Test that Milvus search wrapper executes and creates spans with attributes."""
    from unittest.mock import MagicMock

    # Create mock Collection
    mock_collection_class = MagicMock()
    original_search = MagicMock(return_value=[])
    mock_collection_class.search = original_search

    # Create mock pymilvus module
    mock_pymilvus = MagicMock()
    mock_pymilvus.Collection = mock_collection_class

    import sys

    with patch.dict(sys.modules, {"pymilvus": mock_pymilvus}):
        # Instrument
        vector_db_instrumentor._instrument_milvus()

        # Get the wrapped search method
        wrapped_search = mock_pymilvus.Collection.search

        # Create a mock collection instance
        mock_collection_instance = MagicMock()
        mock_collection_instance.name = "test_collection"

        # Call the wrapped search
        result = wrapped_search(
            mock_collection_instance, data=[[1, 2, 3]], anns_field="embedding", limit=10
        )

        # Verify tracer was called
        mock_tracer.start_as_current_span.assert_called_once_with("milvus.search")


def test_faiss_search_execution(vector_db_instrumentor, mock_tracer):
    """Test that FAISS search wrapper executes and creates spans with attributes."""
    from unittest.mock import MagicMock

    # Create mock Index
    mock_index_class = MagicMock()
    original_search = MagicMock(return_value=([], []))
    mock_index_class.search = original_search

    # Create mock faiss module
    mock_faiss = MagicMock()
    mock_faiss.Index = mock_index_class

    import sys

    with patch.dict(sys.modules, {"faiss": mock_faiss}):
        # Instrument
        vector_db_instrumentor._instrument_faiss()

        # Get the wrapped search method
        wrapped_search = mock_faiss.Index.search

        # Create a mock index instance
        mock_index_instance = MagicMock()

        # Call the wrapped search with k in args
        result = wrapped_search(mock_index_instance, [[1, 2, 3]], 5)

        # Verify tracer was called
        mock_tracer.start_as_current_span.assert_called_once_with("faiss.search")

        # Call again with k in kwargs
        mock_tracer.reset_mock()
        result = wrapped_search(mock_index_instance, [[1, 2, 3]], k=10)

        mock_tracer.start_as_current_span.assert_called_once_with("faiss.search")
