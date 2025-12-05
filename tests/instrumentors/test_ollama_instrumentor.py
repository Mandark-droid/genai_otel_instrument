import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from genai_otel.instrumentors.ollama_instrumentor import OllamaInstrumentor

# Ollama server metrics poller requires Python 3.11+
POLLER_TESTS_SUPPORTED = sys.version_info >= (3, 11)
skipif_poller_not_supported = pytest.mark.skipif(
    not POLLER_TESTS_SUPPORTED, reason="Ollama server metrics poller requires Python 3.11+"
)


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
    mock_span.set_attribute = Mock()
    mock_span.set_status = Mock()
    mock_span.end = Mock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)
    instrumentor.tracer = Mock()
    instrumentor.tracer.start_as_current_span = Mock(return_value=mock_context_manager)
    instrumentor.tracer.start_span = Mock(return_value=mock_span)
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


@skipif_poller_not_supported
def test_instrument_starts_server_metrics_poller():
    """Test that instrumentation starts the server metrics poller by default."""
    from unittest.mock import patch

    # Reset global state first
    import genai_otel.instrumentors.ollama_server_metrics_poller as poller_module

    poller_module._global_poller = None

    mock_config = Mock()
    mock_config.fail_on_error = False

    # Create a fresh instrumentor
    instrumentor = OllamaInstrumentor()

    # Set up the instrumentor state
    mock_ollama_module = MagicMock()
    mock_ollama_module.generate = Mock(return_value={"response": "test"})
    mock_ollama_module.chat = Mock(return_value={"response": "test"})

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module

    # Mock create_span_wrapper to avoid wrapt complexity in this test
    # We're testing poller startup, not span creation
    def mock_wrapper_factory(*args, **kwargs):
        # Return a simple passthrough decorator
        def decorator(func):
            return func

        return decorator

    # Mock the poller start function
    with patch(
        "genai_otel.instrumentors.ollama_instrumentor.start_ollama_metrics_poller"
    ) as mock_start_poller:
        with patch.dict("os.environ", {"GENAI_ENABLE_OLLAMA_SERVER_METRICS": "true"}, clear=False):
            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=mock_wrapper_factory
            ):
                instrumentor.instrument(mock_config)

                # Verify poller was started
                mock_start_poller.assert_called_once()
                # Verify it was called with correct defaults
                call_kwargs = mock_start_poller.call_args[1]
                assert call_kwargs["base_url"] == "http://localhost:11434"
                assert call_kwargs["interval"] == 5.0
                assert call_kwargs["max_vram_gb"] is None


@skipif_poller_not_supported
def test_instrument_starts_poller_with_custom_config():
    """Test that instrumentation uses custom poller configuration."""
    from unittest.mock import patch

    # Reset global state first
    import genai_otel.instrumentors.ollama_server_metrics_poller as poller_module

    poller_module._global_poller = None

    mock_config = Mock()
    mock_config.fail_on_error = False

    instrumentor = OllamaInstrumentor()

    mock_ollama_module = MagicMock()
    mock_ollama_module.generate = Mock(return_value={"response": "test"})
    mock_ollama_module.chat = Mock(return_value={"response": "test"})

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module

    # Mock create_span_wrapper to avoid wrapt complexity in this test
    def mock_wrapper_factory(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    # Mock environment with custom config
    with patch(
        "genai_otel.instrumentors.ollama_instrumentor.start_ollama_metrics_poller"
    ) as mock_start_poller:
        with patch.dict(
            "os.environ",
            {
                "GENAI_ENABLE_OLLAMA_SERVER_METRICS": "true",
                "OLLAMA_BASE_URL": "http://192.168.1.100:11434",
                "GENAI_OLLAMA_METRICS_INTERVAL": "3.0",
                "GENAI_OLLAMA_MAX_VRAM_GB": "24",
            },
            clear=False,
        ):
            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=mock_wrapper_factory
            ):
                instrumentor.instrument(mock_config)

                # Verify poller was started with custom config
                mock_start_poller.assert_called_once()
                call_kwargs = mock_start_poller.call_args[1]
                assert call_kwargs["base_url"] == "http://192.168.1.100:11434"
                assert call_kwargs["interval"] == 3.0
                assert call_kwargs["max_vram_gb"] == 24.0


@skipif_poller_not_supported
def test_instrument_doesnt_start_poller_when_disabled():
    """Test that poller is not started when disabled via env var."""
    from unittest.mock import patch

    # Reset global state first
    import genai_otel.instrumentors.ollama_server_metrics_poller as poller_module

    poller_module._global_poller = None

    mock_config = Mock()
    mock_config.fail_on_error = False

    instrumentor = OllamaInstrumentor()

    mock_ollama_module = MagicMock()
    mock_ollama_module.generate = Mock(return_value={"response": "test"})
    mock_ollama_module.chat = Mock(return_value={"response": "test"})

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module

    # Mock create_span_wrapper to avoid wrapt complexity in this test
    def mock_wrapper_factory(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    # Disable server metrics
    with patch(
        "genai_otel.instrumentors.ollama_instrumentor.start_ollama_metrics_poller"
    ) as mock_start_poller:
        with patch.dict("os.environ", {"GENAI_ENABLE_OLLAMA_SERVER_METRICS": "false"}, clear=False):
            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=mock_wrapper_factory
            ):
                instrumentor.instrument(mock_config)

                # Verify poller was NOT started
                mock_start_poller.assert_not_called()


@skipif_poller_not_supported
def test_instrument_poller_start_failure_continues():
    """Test that instrumentation continues even if poller fails to start."""
    from unittest.mock import patch

    # Reset global state first
    import genai_otel.instrumentors.ollama_server_metrics_poller as poller_module

    poller_module._global_poller = None

    mock_config = Mock()
    mock_config.fail_on_error = False

    instrumentor = OllamaInstrumentor()

    mock_ollama_module = MagicMock()
    mock_ollama_module.generate = Mock(return_value={"response": "test"})
    mock_ollama_module.chat = Mock(return_value={"response": "test"})

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module

    # Mock create_span_wrapper to avoid wrapt complexity in this test
    def mock_wrapper_factory(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    # Make poller start fail
    with patch(
        "genai_otel.instrumentors.ollama_instrumentor.start_ollama_metrics_poller",
        side_effect=Exception("Poller failed"),
    ):
        with patch.dict("os.environ", {"GENAI_ENABLE_OLLAMA_SERVER_METRICS": "true"}, clear=False):
            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=mock_wrapper_factory
            ):
                # Should not raise exception (fail_on_error is False)
                instrumentor.instrument(mock_config)

                # Instrumentation should still succeed
                assert instrumentor._instrumented is True


@skipif_poller_not_supported
def test_instrument_poller_start_failure_with_fail_on_error():
    """Test that instrumentation fails if poller fails and fail_on_error is True."""
    from unittest.mock import patch

    # Reset global state first
    import genai_otel.instrumentors.ollama_server_metrics_poller as poller_module

    poller_module._global_poller = None

    mock_config = Mock()
    mock_config.fail_on_error = True

    instrumentor = OllamaInstrumentor()

    mock_ollama_module = MagicMock()
    mock_ollama_module.generate = Mock(return_value={"response": "test"})
    mock_ollama_module.chat = Mock(return_value={"response": "test"})

    instrumentor._ollama_available = True
    instrumentor._ollama_module = mock_ollama_module

    # Mock create_span_wrapper to avoid wrapt complexity in this test
    def mock_wrapper_factory(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    # Make poller start fail
    with patch(
        "genai_otel.instrumentors.ollama_instrumentor.start_ollama_metrics_poller",
        side_effect=Exception("Poller failed"),
    ):
        with patch.dict("os.environ", {"GENAI_ENABLE_OLLAMA_SERVER_METRICS": "true"}, clear=False):
            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=mock_wrapper_factory
            ):
                # Should raise exception
                with pytest.raises(Exception, match="Poller failed"):
                    instrumentor.instrument(mock_config)


def test_extract_response_attributes_dict(instrumentor):
    """Test response attributes extraction from dict response"""
    # Test with complete response
    result_dict = {
        "model": "llama2",
        "response": "This is a test response",
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 20,
    }
    attrs = instrumentor._extract_response_attributes(result_dict)
    assert attrs["gen_ai.response.model"] == "llama2"
    assert attrs["gen_ai.response.finish_reason"] == "stop"
    assert attrs["gen_ai.response.length"] == len("This is a test response")

    # Test with chat response (message format)
    chat_result = {
        "model": "llama2",
        "message": {"role": "assistant", "content": "Chat response"},
        "done_reason": "stop",
    }
    attrs = instrumentor._extract_response_attributes(chat_result)
    assert attrs["gen_ai.response.model"] == "llama2"
    assert attrs["gen_ai.response.finish_reason"] == "stop"
    assert attrs["gen_ai.response.length"] == len("Chat response")

    # Test with missing optional fields
    minimal_result = {"model": "llama2"}
    attrs = instrumentor._extract_response_attributes(minimal_result)
    assert attrs["gen_ai.response.model"] == "llama2"
    assert "gen_ai.response.finish_reason" not in attrs
    assert "gen_ai.response.length" not in attrs

    # Test with empty dict
    attrs = instrumentor._extract_response_attributes({})
    assert attrs == {}


def test_extract_response_attributes_object(instrumentor):
    """Test response attributes extraction from object response"""

    class MockResponse:
        def __init__(self):
            self.model = "llama2"
            self.response = "Test response"
            self.done_reason = "stop"
            self.prompt_eval_count = 10
            self.eval_count = 20

    attrs = instrumentor._extract_response_attributes(MockResponse())
    assert attrs["gen_ai.response.model"] == "llama2"
    assert attrs["gen_ai.response.finish_reason"] == "stop"
    assert attrs["gen_ai.response.length"] == len("Test response")

    # Test with chat message format
    class MockChatResponse:
        def __init__(self):
            self.model = "llama2"
            self.done_reason = "stop"

            class Message:
                content = "Chat content"

            self.message = Message()

    attrs = instrumentor._extract_response_attributes(MockChatResponse())
    assert attrs["gen_ai.response.model"] == "llama2"
    assert attrs["gen_ai.response.finish_reason"] == "stop"
    assert attrs["gen_ai.response.length"] == len("Chat content")


def test_extract_finish_reason_dict(instrumentor):
    """Test finish reason extraction from dict response"""
    # With done_reason
    result = {"done_reason": "stop", "model": "llama2"}
    assert instrumentor._extract_finish_reason(result) == "stop"

    # With length reason
    result = {"done_reason": "length", "model": "llama2"}
    assert instrumentor._extract_finish_reason(result) == "length"

    # Without done_reason
    result = {"model": "llama2"}
    assert instrumentor._extract_finish_reason(result) is None

    # Empty dict
    assert instrumentor._extract_finish_reason({}) is None


def test_extract_finish_reason_object(instrumentor):
    """Test finish reason extraction from object response"""

    class MockResponse:
        def __init__(self, done_reason=None):
            self.model = "llama2"
            self.done_reason = done_reason

    # With done_reason
    assert instrumentor._extract_finish_reason(MockResponse("stop")) == "stop"
    assert instrumentor._extract_finish_reason(MockResponse("length")) == "length"

    # Without done_reason (None)
    assert instrumentor._extract_finish_reason(MockResponse(None)) is None


def test_extract_finish_reason_none(instrumentor):
    """Test finish reason extraction with None input"""
    assert instrumentor._extract_finish_reason(None) is None
