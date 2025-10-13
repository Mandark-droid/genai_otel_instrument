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
    with patch.object(
        vector_db_instrumentor, "_instrument_pinecone", return_value=True
    ), patch.object(
        vector_db_instrumentor, "_instrument_weaviate", return_value=True
    ), patch.object(
        vector_db_instrumentor, "_instrument_qdrant", return_value=True
    ), patch.object(
        vector_db_instrumentor, "_instrument_chroma", return_value=True
    ), patch.object(
        vector_db_instrumentor, "_instrument_milvus", return_value=True
    ), patch.object(
        vector_db_instrumentor, "_instrument_faiss", return_value=True
    ):

        instrumented_count = vector_db_instrumentor.instrument()
        assert instrumented_count == 6


def test_instrument_no_libraries(vector_db_instrumentor, caplog):
    """Test that instrument() returns 0 if no libraries are available."""
    with patch.object(
        vector_db_instrumentor, "_instrument_pinecone", return_value=False
    ), patch.object(
        vector_db_instrumentor, "_instrument_weaviate", return_value=False
    ), patch.object(
        vector_db_instrumentor, "_instrument_qdrant", return_value=False
    ), patch.object(
        vector_db_instrumentor, "_instrument_chroma", return_value=False
    ), patch.object(
        vector_db_instrumentor, "_instrument_milvus", return_value=False
    ), patch.object(
        vector_db_instrumentor, "_instrument_faiss", return_value=False
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
    with patch.dict("sys.modules", {"pinecone": MagicMock()}), patch.object(
        vector_db_instrumentor, "_wrap_pinecone_method", side_effect=RuntimeError("Mock error")
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
