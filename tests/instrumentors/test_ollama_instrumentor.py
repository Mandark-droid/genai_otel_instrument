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
    original_generate = Mock(
        return_value={"response": "test", "prompt_eval_count": 10, "eval_count": 20}
    )
    original_chat = Mock(
        return_value={"response": "chat test", "prompt_eval_count": 15, "eval_count": 25}
    )
    mock_ollama_module.generate = original_generate
    mock_ollama_module.chat = original_chat

    # Set up the instrumentor state
    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module
    instrumentor.config = mock_config

    # Mock the tracer and metrics
    mock_span = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.request_counter = Mock()
    instrumentor.token_counter = Mock()
    instrumentor.latency_histogram = Mock()
    instrumentor.cost_gauge = Mock()

    # Perform instrumentation
    instrumentor.instrument(mock_config)

    # Verify config was set
    assert instrumentor.config == mock_config

    # Test that wrapped generate function works
    result = mock_ollama_module.generate(model="test_model")

    # Verify the original function was called
    instrumentor._original_generate.assert_called_once()
    # Result should be returned
    assert result == {"response": "test", "prompt_eval_count": 10, "eval_count": 20}


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
    original_generate = Mock(return_value={"response": "test"})
    mock_ollama_module.generate = original_generate

    # Mock the tracer and metrics
    mock_span = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.request_counter = Mock()
    instrumentor.token_counter = Mock()
    instrumentor.latency_histogram = Mock()
    instrumentor.cost_gauge = Mock()

    # Instrument first
    instrumentor.instrument(Mock())

    # Call wrapped without model
    result = mock_ollama_module.generate()

    # Verify the original function was called
    instrumentor._original_generate.assert_called_once()


def test_wrapped_chat(instrumentor):
    """Test wrapped chat function"""
    mock_ollama_module = MagicMock()
    original_chat = Mock(
        return_value={"response": "chat test", "prompt_eval_count": 15, "eval_count": 25}
    )
    mock_ollama_module.chat = original_chat

    # Mock the tracer and metrics
    mock_span = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.request_counter = Mock()
    instrumentor.token_counter = Mock()
    instrumentor.latency_histogram = Mock()
    instrumentor.cost_gauge = Mock()

    instrumentor.instrument(Mock())

    # Call wrapped chat
    result = mock_ollama_module.chat(
        model="test_model", messages=[{"role": "user", "content": "test"}]
    )

    # Verify the original function was called
    instrumentor._original_chat.assert_called_once()
    assert result == {"response": "chat test", "prompt_eval_count": 15, "eval_count": 25}


def test_wrapped_chat_no_model(instrumentor):
    """Test wrapped chat function when no model is specified"""
    mock_ollama_module = MagicMock()
    original_chat = Mock(return_value={"response": "chat test"})
    mock_ollama_module.chat = original_chat

    # Mock the tracer and metrics
    mock_span = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.request_counter = Mock()
    instrumentor.token_counter = Mock()
    instrumentor.latency_histogram = Mock()
    instrumentor.cost_gauge = Mock()

    instrumentor.instrument(Mock())

    # Call wrapped chat without model
    result = mock_ollama_module.chat()

    # Verify the original function was called
    instrumentor._original_chat.assert_called_once()


def test_extract_usage(instrumentor):
    """Test usage extraction from Ollama response"""
    # Test with None
    assert instrumentor._extract_usage(None) is None

    # Test with missing usage fields
    assert instrumentor._extract_usage({"foo": "bar"}) is None

    # Test with dict response
    result_dict = {"response": "test", "prompt_eval_count": 10, "eval_count": 20}
    usage = instrumentor._extract_usage(result_dict)
    assert usage is not None
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30

    # Test with object-like response
    class MockResponse:
        def __init__(self):
            self.prompt_eval_count = 15
            self.eval_count = 25

    usage = instrumentor._extract_usage(MockResponse())
    assert usage is not None
    assert usage["prompt_tokens"] == 15
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 40

    # Test with zero tokens (should return None)
    result_zero = {"response": "test", "prompt_eval_count": 0, "eval_count": 0}
    assert instrumentor._extract_usage(result_zero) is None
