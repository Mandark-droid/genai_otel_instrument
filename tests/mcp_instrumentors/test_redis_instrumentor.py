import logging
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.redis_instrumentor import RedisInstrumentor


# --- Fixtures ---
@pytest.fixture
def mock_otel_redis_instrumentor():
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.OTelRedisInstrumentor"
    ) as mock_instrumentor_class:
        mock_instrumentor_instance = MagicMock()
        mock_instrumentor_class.return_value = mock_instrumentor_instance
        yield mock_instrumentor_class, mock_instrumentor_instance


@pytest.fixture(autouse=True)
def reset_otel_redis_instrumentor_mock(mock_otel_redis_instrumentor):
    mock_instrumentor_class, mock_instrumentor_instance = mock_otel_redis_instrumentor
    yield
    mock_instrumentor_class.reset_mock()
    mock_instrumentor_instance.uninstrument()
    mock_instrumentor_instance.reset_mock()


@pytest.fixture
def redis_instrumentor(mock_otel_redis_instrumentor):
    mock_instrumentor_class, mock_instrumentor_instance = mock_otel_redis_instrumentor
    config = OTelConfig()
    return RedisInstrumentor(config)


# --- Tests ---
def test_instrument_success(redis_instrumentor, mock_otel_redis_instrumentor, caplog):
    """Test successful Redis instrumentation."""
    mock_instrumentor_class, mock_instrumentor_instance = mock_otel_redis_instrumentor

    caplog.set_level(logging.INFO)
    redis_instrumentor.instrument()

    mock_instrumentor_class.assert_called_once()
    mock_instrumentor_instance.instrument.assert_called_once()
    assert "Redis instrumentation enabled" in caplog.text


def test_instrument_missing_redis_py(redis_instrumentor, caplog):
    """Test behavior when redis-py (or OTelRedisInstrumentor) is not available."""
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.OTelRedisInstrumentor",
        side_effect=ImportError("redis-py not installed"),
    ):
        caplog.set_level(logging.DEBUG)
        redis_instrumentor.instrument()
        assert any(
            "Redis-py not installed, skipping instrumentation." in r.message
            and r.levelno == logging.DEBUG
            for r in caplog.records
        )


def test_instrument_with_custom_config():
    """Test that RedisInstrumentor respects custom OTelConfig."""
    config = OTelConfig()
    instrumentor = RedisInstrumentor(config)
    assert instrumentor.config is config


def test_instrument_logs_info_on_success(redis_instrumentor, caplog):
    """Test that INFO log is emitted on successful instrumentation."""
    with patch("genai_otel.mcp_instrumentors.redis_instrumentor.OTelRedisInstrumentor"):
        caplog.set_level(logging.INFO)
        redis_instrumentor.instrument()
        assert "Redis instrumentation enabled" in caplog.text


def test_instrument_logs_debug_on_import_error(redis_instrumentor, caplog):
    """Test that DEBUG log is emitted if redis-py is missing."""
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.OTelRedisInstrumentor",
        side_effect=ImportError("redis-py not installed"),
    ):
        caplog.set_level(logging.DEBUG)
        redis_instrumentor.instrument()
        assert any(
            "Redis-py not installed, skipping instrumentation." in r.message
            and r.levelno == logging.DEBUG
            for r in caplog.records
        )


def test_instrument_logs_warning_on_failure(redis_instrumentor, caplog):
    """Test that WARNING log is emitted on instrumentation failure."""
    with patch(
        "genai_otel.mcp_instrumentors.redis_instrumentor.OTelRedisInstrumentor",
        side_effect=RuntimeError("Mock error"),
    ):
        caplog.set_level(logging.WARNING)
        redis_instrumentor.instrument()
        assert any(
            "Redis instrumentation failed: Mock error" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )
