import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.opensearch_instrumentor import OpenSearchInstrumentor


# --- Fixtures ---
@pytest.fixture
def mock_otel_opensearch_instrumentor():
    """Mock the opentelemetry OpenSearch instrumentor module for lazy import."""
    mock_instrumentor_instance = MagicMock()
    mock_instrumentor_class = MagicMock(return_value=mock_instrumentor_instance)

    mock_module = MagicMock()
    mock_module.OpenSearchInstrumentor = mock_instrumentor_class

    with patch.dict(sys.modules, {"opentelemetry.instrumentation.opensearch": mock_module}):
        yield mock_instrumentor_class, mock_instrumentor_instance


@pytest.fixture
def opensearch_instrumentor():
    config = OTelConfig()
    return OpenSearchInstrumentor(config)


# --- Tests ---
def test_instrument_success(opensearch_instrumentor, mock_otel_opensearch_instrumentor, caplog):
    """Test successful OpenSearch instrumentation."""
    mock_instrumentor_class, mock_instrumentor_instance = mock_otel_opensearch_instrumentor

    caplog.set_level(logging.INFO)
    opensearch_instrumentor.instrument()

    mock_instrumentor_class.assert_called_once()
    mock_instrumentor_instance.instrument.assert_called_once()
    assert "OpenSearch instrumentation enabled" in caplog.text


def test_instrument_missing_opensearch_py(opensearch_instrumentor, caplog):
    """Test behavior when opensearch-py (or OTelOpenSearchInstrumentor) is not available."""
    # Remove the module from sys.modules to trigger ImportError
    with patch.dict(sys.modules, {"opentelemetry.instrumentation.opensearch": None}):
        caplog.set_level(logging.DEBUG)
        opensearch_instrumentor.instrument()
        assert any(
            "opensearch-py not installed, skipping instrumentation." in r.message
            and r.levelno == logging.DEBUG
            for r in caplog.records
        )


def test_instrument_with_custom_config():
    """Test that OpenSearchInstrumentor respects custom OTelConfig."""
    config = OTelConfig()
    instrumentor = OpenSearchInstrumentor(config)
    assert instrumentor.config is config


def test_instrument_logs_info_on_success(
    opensearch_instrumentor, mock_otel_opensearch_instrumentor, caplog
):
    """Test that INFO log is emitted on successful instrumentation."""
    caplog.set_level(logging.INFO)
    opensearch_instrumentor.instrument()
    assert "OpenSearch instrumentation enabled" in caplog.text


def test_instrument_logs_debug_on_import_error(opensearch_instrumentor, caplog):
    """Test that DEBUG log is emitted if opensearch-py is missing."""
    with patch.dict(sys.modules, {"opentelemetry.instrumentation.opensearch": None}):
        caplog.set_level(logging.DEBUG)
        opensearch_instrumentor.instrument()
        assert any(
            "opensearch-py not installed, skipping instrumentation." in r.message
            and r.levelno == logging.DEBUG
            for r in caplog.records
        )


def test_instrument_logs_warning_on_failure(
    opensearch_instrumentor, mock_otel_opensearch_instrumentor, caplog
):
    """Test that WARNING log is emitted on instrumentation failure."""
    mock_instrumentor_class, mock_instrumentor_instance = mock_otel_opensearch_instrumentor
    mock_instrumentor_instance.instrument.side_effect = RuntimeError("Mock error")

    caplog.set_level(logging.WARNING)
    opensearch_instrumentor.instrument()
    assert any(
        "OpenSearch instrumentation failed" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    )
