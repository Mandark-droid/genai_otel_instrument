import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.rabbitmq_instrumentor import RabbitMQInstrumentor


@pytest.fixture
def config():
    return OTelConfig()


@pytest.fixture
def instrumentor(config):
    return RabbitMQInstrumentor(config)


def test_init(config):
    """Test that config is stored and tracer is created."""
    instrumentor = RabbitMQInstrumentor(config)
    assert instrumentor.config is config
    assert instrumentor.tracer is not None


@patch("genai_otel.mcp_instrumentors.rabbitmq_instrumentor.wrapt")
def test_instrument_success(mock_wrapt, instrumentor, caplog):
    """Test successful instrumentation when pika is available."""
    mock_pika = MagicMock()
    with patch.dict("sys.modules", {"pika": mock_pika}):
        caplog.set_level(logging.INFO)
        result = instrumentor.instrument()

        assert result is True
        assert "RabbitMQ instrumentation enabled" in caplog.text
        # Verify wrapt.wrap_function_wrapper was called for all methods
        # 1 publish + 2 consume + 4 admin = 7 total
        assert mock_wrapt.wrap_function_wrapper.call_count == 7


def test_instrument_missing(instrumentor, caplog):
    """Test graceful handling when pika is not installed."""
    with patch.dict("sys.modules", {"pika": None}):
        caplog.set_level(logging.DEBUG)
        result = instrumentor.instrument()

        assert result is False
        assert any(
            "pika not installed, skipping RabbitMQ instrumentation" in r.message
            and r.levelno == logging.DEBUG
            for r in caplog.records
        )


@patch("genai_otel.mcp_instrumentors.rabbitmq_instrumentor.wrapt")
def test_instrument_error(mock_wrapt, instrumentor, caplog):
    """Test error logging on general exception."""
    mock_pika = MagicMock()
    mock_wrapt.wrap_function_wrapper.side_effect = RuntimeError("wrap error")
    with patch.dict("sys.modules", {"pika": mock_pika}):
        caplog.set_level(logging.ERROR)
        result = instrumentor.instrument()

        assert result is False
        assert any(
            "Failed to instrument RabbitMQ" in r.message and r.levelno == logging.ERROR
            for r in caplog.records
        )


def test_publish_wrapper_creates_span(config):
    """Test that basic_publish creates PRODUCER span with messaging attributes."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

    instrumentor = RabbitMQInstrumentor(config)
    instrumentor.tracer = mock_tracer

    # Capture the wrapper that gets passed to wrapt
    wrappers = {}
    mock_pika = MagicMock()

    def capture_wrapper(module, method, wrapper):
        wrappers[method] = wrapper

    with patch.dict("sys.modules", {"pika": mock_pika}):
        with patch("genai_otel.mcp_instrumentors.rabbitmq_instrumentor.wrapt") as mock_wrapt:
            mock_wrapt.wrap_function_wrapper.side_effect = capture_wrapper
            instrumentor.instrument()

    # Call the basic_publish wrapper
    assert "basic_publish" in wrappers
    wrapper = wrappers["basic_publish"]
    mock_wrapped = MagicMock(return_value="ok")
    wrapper(mock_wrapped, None, [], {"exchange": "test_exchange", "routing_key": "test.key"})

    # Verify span was created with correct attributes
    mock_tracer.start_as_current_span.assert_called()
    call_args = mock_tracer.start_as_current_span.call_args
    assert call_args[0][0] == "rabbitmq.basic_publish"
    mock_span.set_attribute.assert_any_call("messaging.system", "rabbitmq")
    mock_span.set_attribute.assert_any_call("messaging.operation", "basic_publish")
    mock_span.set_attribute.assert_any_call("messaging.destination", "test_exchange")
    mock_span.set_attribute.assert_any_call("messaging.rabbitmq.routing_key", "test.key")


def test_consume_wrapper_creates_span(config):
    """Test that basic_consume creates CONSUMER span."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

    instrumentor = RabbitMQInstrumentor(config)
    instrumentor.tracer = mock_tracer

    wrappers = {}
    mock_pika = MagicMock()

    def capture_wrapper(module, method, wrapper):
        wrappers[method] = wrapper

    with patch.dict("sys.modules", {"pika": mock_pika}):
        with patch("genai_otel.mcp_instrumentors.rabbitmq_instrumentor.wrapt") as mock_wrapt:
            mock_wrapt.wrap_function_wrapper.side_effect = capture_wrapper
            instrumentor.instrument()

    assert "basic_consume" in wrappers
    wrapper = wrappers["basic_consume"]
    mock_wrapped = MagicMock(return_value="consumer_tag")
    wrapper(mock_wrapped, None, [], {"queue": "my_queue"})

    mock_tracer.start_as_current_span.assert_called()
    call_args = mock_tracer.start_as_current_span.call_args
    assert call_args[0][0] == "rabbitmq.basic_consume"
    mock_span.set_attribute.assert_any_call("messaging.system", "rabbitmq")
    mock_span.set_attribute.assert_any_call("messaging.destination", "my_queue")


def test_admin_wrapper_creates_span(config):
    """Test that queue_declare creates CLIENT span."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

    instrumentor = RabbitMQInstrumentor(config)
    instrumentor.tracer = mock_tracer

    wrappers = {}
    mock_pika = MagicMock()

    def capture_wrapper(module, method, wrapper):
        wrappers[method] = wrapper

    with patch.dict("sys.modules", {"pika": mock_pika}):
        with patch("genai_otel.mcp_instrumentors.rabbitmq_instrumentor.wrapt") as mock_wrapt:
            mock_wrapt.wrap_function_wrapper.side_effect = capture_wrapper
            instrumentor.instrument()

    assert "queue_declare" in wrappers
    wrapper = wrappers["queue_declare"]
    mock_wrapped = MagicMock(return_value="result")
    wrapper(mock_wrapped, None, [], {"queue": "test_queue"})

    mock_tracer.start_as_current_span.assert_called()
    call_args = mock_tracer.start_as_current_span.call_args
    assert call_args[0][0] == "rabbitmq.queue_declare"
    mock_span.set_attribute.assert_any_call("messaging.system", "rabbitmq")
    mock_span.set_attribute.assert_any_call("messaging.destination", "test_queue")


def test_wrapper_handles_exception(config):
    """Test that exceptions are recorded on spans."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

    instrumentor = RabbitMQInstrumentor(config)
    instrumentor.tracer = mock_tracer

    wrappers = {}
    mock_pika = MagicMock()

    def capture_wrapper(module, method, wrapper):
        wrappers[method] = wrapper

    with patch.dict("sys.modules", {"pika": mock_pika}):
        with patch("genai_otel.mcp_instrumentors.rabbitmq_instrumentor.wrapt") as mock_wrapt:
            mock_wrapt.wrap_function_wrapper.side_effect = capture_wrapper
            instrumentor.instrument()

    wrapper = wrappers["basic_publish"]
    test_error = ConnectionError("Connection lost")
    mock_wrapped = MagicMock(side_effect=test_error)

    with pytest.raises(ConnectionError, match="Connection lost"):
        wrapper(mock_wrapped, None, [], {"exchange": "ex", "routing_key": "rk"})

    mock_span.record_exception.assert_called_once_with(test_error)
    mock_span.set_status.assert_called()
