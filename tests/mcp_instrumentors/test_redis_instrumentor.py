import logging
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.redis_instrumentor import RedisInstrumentor


# --- Fixtures ---
@pytest.fixture
def mock_otel_redis_instrumentor():
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.RedisInstrumentor"
    ) as mock_instrumentor:
        yield mock_instrumentor


@pytest.fixture
def redis_instrumentor(mock_otel_redis_instrumentor):
    config = OTelConfig()
    return RedisInstrumentor(config)


# --- Tests for RedisInstrumentor.__init__ ---
def test_redis_instrumentor_init(redis_instrumentor):
    """Test that RedisInstrumentor initializes with the provided config."""
    assert redis_instrumentor.config is not None


# --- Tests for RedisInstrumentor.instrument ---
def test_instrument_success(redis_instrumentor, mock_otel_redis_instrumentor, caplog):
    """Test successful Redis instrumentation."""
    mock_instrumentor_instance = MagicMock()
    mock_otel_redis_instrumentor.return_value = mock_instrumentor_instance

    redis_instrumentor.instrument()

    mock_otel_redis_instrumentor.assert_called_once()
    mock_instrumentor_instance.instrument.assert_called_once()
    assert "Redis instrumentation enabled" in caplog.text


def test_instrument_missing_redis_py(redis_instrumentor, caplog):
    """Test that missing redis-py is handled gracefully."""
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.RedisInstrumentor", side_effect=ImportError
    ):
        redis_instrumentor.instrument()
        assert "Redis-py not installed, skipping instrumentation." in caplog.text


def test_instrument_unexpected_error(redis_instrumentor, caplog):
    """Test that unexpected errors are logged as warnings."""
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.RedisInstrumentor",
        side_effect=RuntimeError("Mock error"),
    ):
        redis_instrumentor.instrument()
        assert "Redis instrumentation failed: Mock error" in caplog.text


def test_instrument_with_custom_config():
    """Test that RedisInstrumentor respects custom OTelConfig."""
    config = OTelConfig(redis_enabled=True)
    instrumentor = RedisInstrumentor(config)
    assert instrumentor.config.redis_enabled is True


# --- Tests for logging ---
def test_instrument_logs_info_on_success(redis_instrumentor, caplog):
    """Test that INFO log is emitted on successful instrumentation."""
    with patch("genai_otel.mcp_instrumentors.redis_instrumentor.RedisInstrumentor"):
        caplog.set_level(logging.INFO)
        redis_instrumentor.instrument()
        assert "Redis instrumentation enabled" in caplog.text


def test_instrument_logs_debug_on_import_error(redis_instrumentor, caplog):
    """Test that DEBUG log is emitted if redis-py is missing."""
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.RedisInstrumentor", side_effect=ImportError
    ):
        caplog.set_level(logging.DEBUG)
        redis_instrumentor.instrument()
        assert "Redis-py not installed, skipping instrumentation." in caplog.text


def test_instrument_logs_warning_on_failure(redis_instrumentor, caplog):
    """Test that WARNING log is emitted on instrumentation failure."""
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.RedisInstrumentor",
        side_effect=RuntimeError("Mock error"),
    ):
        caplog.set_level(logging.WARNING)
        redis_instrumentor.instrument()
        assert "Redis instrumentation failed: Mock error" in caplog.text
