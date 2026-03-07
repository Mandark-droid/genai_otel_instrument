import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.elasticsearch_instrumentor import ElasticsearchInstrumentor


# --- Fixtures ---
@pytest.fixture
def mock_otel_es_instrumentor():
    """Mock the OTel Elasticsearch instrumentor by injecting a fake module into sys.modules."""
    mock_instrumentor_class = MagicMock()
    mock_instrumentor_instance = MagicMock()
    mock_instrumentor_class.return_value = mock_instrumentor_instance

    # Create a fake module to inject
    fake_module = types.ModuleType("opentelemetry.instrumentation.elasticsearch")
    fake_module.ElasticsearchInstrumentor = mock_instrumentor_class

    with patch.dict(sys.modules, {"opentelemetry.instrumentation.elasticsearch": fake_module}):
        yield mock_instrumentor_class, mock_instrumentor_instance


@pytest.fixture
def es_instrumentor():
    config = OTelConfig()
    return ElasticsearchInstrumentor(config)


# --- Tests ---
def test_init():
    """Test that ElasticsearchInstrumentor respects custom OTelConfig."""
    config = OTelConfig()
    instrumentor = ElasticsearchInstrumentor(config)
    assert instrumentor.config is config


def test_instrument_success(es_instrumentor, mock_otel_es_instrumentor, caplog):
    """Test successful Elasticsearch instrumentation."""
    mock_instrumentor_class, mock_instrumentor_instance = mock_otel_es_instrumentor

    caplog.set_level(logging.INFO)
    es_instrumentor.instrument()

    mock_instrumentor_class.assert_called_once()
    mock_instrumentor_instance.instrument.assert_called_once()
    assert "Elasticsearch instrumentation enabled" in caplog.text


def test_instrument_missing_elasticsearch_py(es_instrumentor, caplog):
    """Test behavior when elasticsearch-py (or OTelElasticsearchInstrumentor) is not available."""
    with patch.dict(sys.modules, {"opentelemetry.instrumentation.elasticsearch": None}):
        caplog.set_level(logging.DEBUG)
        es_instrumentor.instrument()
        assert any(
            "elasticsearch-py not installed, skipping instrumentation." in r.message
            and r.levelno == logging.DEBUG
            for r in caplog.records
        )


def test_instrument_logs_info_on_success(es_instrumentor, mock_otel_es_instrumentor, caplog):
    """Test that INFO log is emitted on successful instrumentation."""
    caplog.set_level(logging.INFO)
    es_instrumentor.instrument()
    assert "Elasticsearch instrumentation enabled" in caplog.text


def test_instrument_logs_debug_on_import_error(es_instrumentor, caplog):
    """Test that DEBUG log is emitted if elasticsearch-py is missing."""
    with patch.dict(sys.modules, {"opentelemetry.instrumentation.elasticsearch": None}):
        caplog.set_level(logging.DEBUG)
        es_instrumentor.instrument()
        assert any(
            "elasticsearch-py not installed, skipping instrumentation." in r.message
            and r.levelno == logging.DEBUG
            for r in caplog.records
        )


def test_instrument_logs_warning_on_failure(es_instrumentor, caplog):
    """Test that WARNING log is emitted on instrumentation failure."""
    mock_instrumentor_class = MagicMock(side_effect=RuntimeError("Mock error"))
    fake_module = types.ModuleType("opentelemetry.instrumentation.elasticsearch")
    fake_module.ElasticsearchInstrumentor = mock_instrumentor_class

    with patch.dict(sys.modules, {"opentelemetry.instrumentation.elasticsearch": fake_module}):
        caplog.set_level(logging.WARNING)
        es_instrumentor.instrument()
        assert any(
            "Elasticsearch instrumentation failed" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )
