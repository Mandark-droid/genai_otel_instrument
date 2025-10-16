import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from genai_otel.instrumentors.ollama_instrumentor import OllamaInstrumentor


@pytest.fixture
def instrumentor():
    return OllamaInstrumentor()


def test_init_available():
    """Test initialization when ollama is available"""
    # Create a fresh instrumentor with ollama available
    with patch.dict("sys.modules", {"ollama": MagicMock()}):
        # Re-import to get a fresh instrumentor that sees ollama as available
        from genai_otel.instrumentors.ollama_instrumentor import OllamaInstrumentor

        fresh_instrumentor = OllamaInstrumentor()
        assert fresh_instrumentor._ollama_available is True


def test_init_not_available():
    """Test initialization when ollama is not available"""
    # Create a fresh instrumentor without ollama
    with patch.dict("sys.modules", {"ollama": None}):
        # Force reload by removing the module if it exists
        if "genai_otel.instrumentors.ollama_instrumentor" in sys.modules:
            del sys.modules["genai_otel.instrumentors.ollama_instrumentor"]

        from genai_otel.instrumentors.ollama_instrumentor import OllamaInstrumentor

        fresh_instrumentor = OllamaInstrumentor()
        assert fresh_instrumentor._ollama_available is False


def test_instrument_available(instrumentor):
    """Test instrumentation when ollama is available"""
    mock_config = Mock()

    # Create a proper mock ollama module
    mock_ollama_module = MagicMock()
    mock_ollama_module.generate = Mock(return_value={"response": "test"})
    mock_ollama_module.chat = Mock(return_value={"response": "chat test"})

    # Mock the context manager for the span
    mock_span = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)

    # Set up the instrumentor state
    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.request_counter = Mock()

    # Perform instrumentation
    instrumentor.instrument(mock_config)

    # Verify config was set
    assert instrumentor.config == mock_config

    # Test that wrapped generate function works
    result = mock_ollama_module.generate(model="test_model")

    # Verify tracing was called
    instrumentor.tracer.start_as_current_span.assert_called_once_with("ollama.generate")
    mock_span.set_attribute.assert_any_call("gen_ai.system", "ollama")
    mock_span.set_attribute.assert_any_call("gen_ai.request.model", "test_model")
    instrumentor.request_counter.add.assert_called_once_with(
        1, {"model": "test_model", "provider": "ollama"}
    )
    # Verify original function was called using the stored reference
    instrumentor._original_generate.assert_called_once_with(model="test_model")
    assert result == {"response": "test"}


def test_instrument_not_available(instrumentor):
    """Test instrumentation when ollama is not available"""
    mock_config = Mock()

    # Set ollama as not available
    instrumentor._ollama_available = False
    instrumentor._ollama_module = None
    instrumentor.tracer = Mock()
    instrumentor.request_counter = Mock()

    # This should not raise an exception and should not attempt instrumentation
    instrumentor.instrument(mock_config)

    assert instrumentor.config == mock_config
    # Verify no tracing was set up
    instrumentor.tracer.start_as_current_span.assert_not_called()


def test_wrapped_generate_no_model(instrumentor):
    """Test wrapped generate function when no model is specified"""
    mock_ollama_module = MagicMock()
    mock_ollama_module.generate = Mock(return_value={"response": "test"})

    # Mock the context manager for the span
    mock_span = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.request_counter = Mock()

    # Instrument first
    instrumentor.instrument(Mock())

    # Call wrapped without model
    result = mock_ollama_module.generate()

    # Should use "unknown" as model name
    mock_span.set_attribute.assert_any_call("gen_ai.request.model", "unknown")
    instrumentor.request_counter.add.assert_called_once_with(
        1, {"model": "unknown", "provider": "ollama"}
    )


def test_wrapped_chat(instrumentor):
    """Test wrapped chat function"""
    mock_ollama_module = MagicMock()
    mock_ollama_module.chat = Mock(return_value={"response": "chat test"})

    # Mock the context manager for the span
    mock_span = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.request_counter = Mock()

    instrumentor.instrument(Mock())

    # Call wrapped chat
    result = mock_ollama_module.chat(model="test_model")

    instrumentor.tracer.start_as_current_span.assert_called_once_with("ollama.chat")
    mock_span.set_attribute.assert_any_call("gen_ai.system", "ollama")
    mock_span.set_attribute.assert_any_call("gen_ai.request.model", "test_model")
    instrumentor.request_counter.add.assert_called_once_with(
        1, {"model": "test_model", "provider": "ollama"}
    )


def test_wrapped_chat_no_model(instrumentor):
    """Test wrapped chat function when no model is specified"""
    mock_ollama_module = MagicMock()
    mock_ollama_module.chat = Mock(return_value={"response": "chat test"})

    # Mock the context manager for the span
    mock_span = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.request_counter = Mock()

    instrumentor.instrument(Mock())

    # Call wrapped chat without model
    result = mock_ollama_module.chat()

    mock_span.set_attribute.assert_any_call("gen_ai.request.model", "unknown")


def test_extract_usage(instrumentor):
    """Test usage extraction (returns None as implemented)"""
    assert instrumentor._extract_usage(None) is None
    assert instrumentor._extract_usage({"foo": "bar"}) is None
