import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.mistralai_instrumentor import MistralAIInstrumentor


class TestMistralAIInstrumentor(unittest.TestCase):
    """Tests for MistralAIInstrumentor"""

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_instrument_with_mistralai_available(self, mock_logger):
        """Test that instrument works when MistralAI is available."""
        # Create mock mistralai module
        mock_mistralai = MagicMock()
        mock_client = MagicMock()
        mock_mistralai.client.MistralClient = mock_client

        with patch.dict(
            "sys.modules", {"mistralai": mock_mistralai, "mistralai.client": mock_mistralai.client}
        ):
            instrumentor = MistralAIInstrumentor()
            config = OTelConfig()

            # Mock the instrumentation methods
            instrumentor._instrument_chat = MagicMock()
            instrumentor._instrument_embeddings = MagicMock()

            instrumentor.instrument(config)

            # Verify methods were called
            instrumentor._instrument_chat.assert_called_once()
            instrumentor._instrument_embeddings.assert_called_once()
            mock_logger.info.assert_called_with("MistralAI instrumentation enabled")

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_instrument_with_mistralai_not_available(self, mock_logger):
        """Test that instrument handles missing MistralAI gracefully."""
        instrumentor = MistralAIInstrumentor()
        config = OTelConfig()

        # Mock import to fail
        with patch("builtins.__import__", side_effect=ImportError("No module named 'mistralai'")):
            instrumentor.instrument(config)

            mock_logger.warning.assert_called_with(
                "mistralai package not available, skipping instrumentation"
            )

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_instrument_with_exception(self, mock_logger):
        """Test that instrument handles exceptions during instrumentation."""
        mock_mistralai = MagicMock()
        mock_mistralai.client.MistralClient = MagicMock()

        with patch.dict(
            "sys.modules", {"mistralai": mock_mistralai, "mistralai.client": mock_mistralai.client}
        ):
            instrumentor = MistralAIInstrumentor()
            config = OTelConfig()

            # Make _instrument_chat raise an exception
            def raise_error():
                raise RuntimeError("Test error")

            instrumentor._instrument_chat = raise_error

            instrumentor.instrument(config)

            # Should log error
            self.assertTrue(
                any(
                    "Failed to instrument mistralai" in str(call)
                    for call in mock_logger.error.call_args_list
                )
            )

    def test_instrument_chat_success(self):
        """Test that _instrument_chat successfully wraps the chat method."""
        mock_mistralai = MagicMock()
        mock_client = MagicMock()
        mock_mistralai.client.MistralClient = mock_client

        # Create mock wrapt
        mock_wrapt = MagicMock()
        captured_wrapper = None

        def capture_wrapper(module, func_name):
            def decorator(wrapper_func):
                nonlocal captured_wrapper
                captured_wrapper = wrapper_func
                return wrapper_func

            return decorator

        mock_wrapt.patch_function_wrapper = capture_wrapper

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.client": mock_mistralai.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()
            instrumentor.cost_counter = MagicMock()

            # Call _instrument_chat
            instrumentor._instrument_chat()

            # Verify wrapper was captured
            self.assertIsNotNone(captured_wrapper)

            # Create mock response with usage
            mock_response = MagicMock()
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 20
            mock_usage.total_tokens = 30
            mock_response.usage = mock_usage

            # Create mock wrapped function
            def mock_wrapped(*args, **kwargs):
                return mock_response

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Call the wrapper
            result = captured_wrapper(mock_wrapped, None, (), {"model": "mistral-tiny"})

            # Verify span was created
            instrumentor.tracer.start_as_current_span.assert_called_with("mistralai.chat")

            # Verify attributes were set
            mock_span.set_attribute.assert_any_call("gen_ai.system", "mistralai")
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "mistral-tiny")
            mock_span.set_attribute.assert_any_call("llm.request.type", "chat")

            # Verify request counter was called
            instrumentor.request_counter.add.assert_called()

            # Verify result was returned
            self.assertEqual(result, mock_response)

    def test_instrument_chat_error(self):
        """Test that _instrument_chat handles errors correctly."""
        mock_mistralai = MagicMock()
        mock_client = MagicMock()
        mock_mistralai.client.MistralClient = mock_client

        # Create mock wrapt
        mock_wrapt = MagicMock()
        captured_wrapper = None

        def capture_wrapper(module, func_name):
            def decorator(wrapper_func):
                nonlocal captured_wrapper
                captured_wrapper = wrapper_func
                return wrapper_func

            return decorator

        mock_wrapt.patch_function_wrapper = capture_wrapper

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.client": mock_mistralai.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()

            # Call _instrument_chat
            instrumentor._instrument_chat()

            # Create mock wrapped function that raises error
            def mock_wrapped(*args, **kwargs):
                raise ValueError("Test error")

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Call the wrapper - should raise
            with self.assertRaises(ValueError):
                captured_wrapper(mock_wrapped, None, (), {"model": "mistral-tiny"})

            # Verify error attributes were set
            mock_span.set_attribute.assert_any_call("error", True)
            mock_span.set_attribute.assert_any_call("error.message", "Test error")

    def test_instrument_chat_with_default_model(self):
        """Test that _instrument_chat uses 'unknown' when model is not provided."""
        mock_mistralai = MagicMock()
        mock_client = MagicMock()
        mock_mistralai.client.MistralClient = mock_client

        # Create mock wrapt
        mock_wrapt = MagicMock()
        captured_wrapper = None

        def capture_wrapper(module, func_name):
            def decorator(wrapper_func):
                nonlocal captured_wrapper
                captured_wrapper = wrapper_func
                return wrapper_func

            return decorator

        mock_wrapt.patch_function_wrapper = capture_wrapper

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.client": mock_mistralai.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()

            # Call _instrument_chat
            instrumentor._instrument_chat()

            # Create mock response
            mock_response = MagicMock()
            mock_response.usage = None

            # Create mock wrapped function
            def mock_wrapped(*args, **kwargs):
                return mock_response

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Call the wrapper without model
            captured_wrapper(mock_wrapped, None, (), {})

            # Verify 'unknown' was used as model
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "unknown")

    def test_instrument_embeddings_success(self):
        """Test that _instrument_embeddings successfully wraps the embeddings method."""
        mock_mistralai = MagicMock()
        mock_client = MagicMock()
        mock_mistralai.client.MistralClient = mock_client

        # Create mock wrapt
        mock_wrapt = MagicMock()
        captured_wrapper = None

        def capture_wrapper(module, func_name):
            def decorator(wrapper_func):
                nonlocal captured_wrapper
                captured_wrapper = wrapper_func
                return wrapper_func

            return decorator

        mock_wrapt.patch_function_wrapper = capture_wrapper

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.client": mock_mistralai.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()

            # Call _instrument_embeddings
            instrumentor._instrument_embeddings()

            # Verify wrapper was captured
            self.assertIsNotNone(captured_wrapper)

            # Create mock response with usage
            mock_response = MagicMock()
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 15
            mock_usage.total_tokens = 15
            mock_response.usage = mock_usage

            # Create mock wrapped function
            def mock_wrapped(*args, **kwargs):
                return mock_response

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Call the wrapper
            result = captured_wrapper(mock_wrapped, None, (), {"model": "mistral-embed"})

            # Verify span was created
            instrumentor.tracer.start_as_current_span.assert_called_with("mistralai.embeddings")

            # Verify attributes were set
            mock_span.set_attribute.assert_any_call("gen_ai.system", "mistralai")
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "mistral-embed")
            mock_span.set_attribute.assert_any_call("llm.request.type", "embeddings")

            # Verify result was returned
            self.assertEqual(result, mock_response)

    def test_instrument_embeddings_error(self):
        """Test that _instrument_embeddings handles errors correctly."""
        mock_mistralai = MagicMock()
        mock_client = MagicMock()
        mock_mistralai.client.MistralClient = mock_client

        # Create mock wrapt
        mock_wrapt = MagicMock()
        captured_wrapper = None

        def capture_wrapper(module, func_name):
            def decorator(wrapper_func):
                nonlocal captured_wrapper
                captured_wrapper = wrapper_func
                return wrapper_func

            return decorator

        mock_wrapt.patch_function_wrapper = capture_wrapper

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.client": mock_mistralai.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()

            # Call _instrument_embeddings
            instrumentor._instrument_embeddings()

            # Create mock wrapped function that raises error
            def mock_wrapped(*args, **kwargs):
                raise ValueError("Embedding error")

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Call the wrapper - should raise
            with self.assertRaises(ValueError):
                captured_wrapper(mock_wrapped, None, (), {"model": "mistral-embed"})

            # Verify error attributes were set
            mock_span.set_attribute.assert_any_call("error", True)
            mock_span.set_attribute.assert_any_call("error.message", "Embedding error")

    def test_instrument_embeddings_with_default_model(self):
        """Test that _instrument_embeddings uses 'mistral-embed' as default."""
        mock_mistralai = MagicMock()
        mock_client = MagicMock()
        mock_mistralai.client.MistralClient = mock_client

        # Create mock wrapt
        mock_wrapt = MagicMock()
        captured_wrapper = None

        def capture_wrapper(module, func_name):
            def decorator(wrapper_func):
                nonlocal captured_wrapper
                captured_wrapper = wrapper_func
                return wrapper_func

            return decorator

        mock_wrapt.patch_function_wrapper = capture_wrapper

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.client": mock_mistralai.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()

            # Call _instrument_embeddings
            instrumentor._instrument_embeddings()

            # Create mock response
            mock_response = MagicMock()
            mock_response.usage = None

            # Create mock wrapped function
            def mock_wrapped(*args, **kwargs):
                return mock_response

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Call the wrapper without model
            captured_wrapper(mock_wrapped, None, (), {})

            # Verify 'mistral-embed' was used as default model
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "mistral-embed")

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_instrument_chat_exception_during_wrapping(self, mock_logger):
        """Test that _instrument_chat handles exceptions during wrapping."""
        mock_mistralai = MagicMock()
        mock_mistralai.client.MistralClient = MagicMock()

        # Create mock wrapt that raises exception
        mock_wrapt = MagicMock()
        mock_wrapt.patch_function_wrapper.side_effect = RuntimeError("Wrapping failed")

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.client": mock_mistralai.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()

            # Should not raise, just log
            instrumentor._instrument_chat()

            mock_logger.debug.assert_called()

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_instrument_embeddings_exception_during_wrapping(self, mock_logger):
        """Test that _instrument_embeddings handles exceptions during wrapping."""
        mock_mistralai = MagicMock()
        mock_mistralai.client.MistralClient = MagicMock()

        # Create mock wrapt that raises exception
        mock_wrapt = MagicMock()
        mock_wrapt.patch_function_wrapper.side_effect = RuntimeError("Wrapping failed")

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.client": mock_mistralai.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()

            # Should not raise, just log
            instrumentor._instrument_embeddings()

            mock_logger.debug.assert_called()

    def test_record_result_metrics_with_usage(self):
        """Test that _record_result_metrics records metrics correctly."""
        instrumentor = MistralAIInstrumentor()
        instrumentor.token_counter = MagicMock()
        instrumentor.cost_counter = MagicMock()

        # Create mock span and result
        mock_span = MagicMock()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_result.usage = mock_usage

        # Call method
        instrumentor._record_result_metrics(mock_span, mock_result, 0.05)

        # Verify span attributes
        mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 10)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 20)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 30)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.cost", 0.05)

        # Verify token counter
        instrumentor.token_counter.add.assert_any_call(
            10, {"type": "input", "provider": "mistralai"}
        )
        instrumentor.token_counter.add.assert_any_call(
            20, {"type": "output", "provider": "mistralai"}
        )

        # Verify cost counter
        instrumentor.cost_counter.add.assert_called_with(0.05, {"provider": "mistralai"})

    def test_record_result_metrics_without_usage(self):
        """Test that _record_result_metrics handles missing usage."""
        instrumentor = MistralAIInstrumentor()
        instrumentor.token_counter = MagicMock()
        instrumentor.cost_counter = MagicMock()

        # Create mock span and result without usage
        mock_span = MagicMock()
        mock_result = MagicMock(spec=[])  # No attributes

        # Call method - should not raise
        instrumentor._record_result_metrics(mock_span, mock_result, 0.0)

        # Should not set any attributes or record metrics
        # (since there's no usage to extract)

    def test_record_result_metrics_with_zero_cost(self):
        """Test that _record_result_metrics doesn't record cost when cost is zero."""
        instrumentor = MistralAIInstrumentor()
        instrumentor.token_counter = MagicMock()
        instrumentor.cost_counter = MagicMock()

        # Create mock span and result
        mock_span = MagicMock()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_result.usage = mock_usage

        # Call method with zero cost
        instrumentor._record_result_metrics(mock_span, mock_result, 0)

        # Verify tokens were recorded but cost was not
        instrumentor.token_counter.add.assert_called()
        instrumentor.cost_counter.add.assert_not_called()

    def test_record_embedding_metrics_with_usage(self):
        """Test that _record_embedding_metrics records metrics correctly."""
        instrumentor = MistralAIInstrumentor()
        instrumentor.token_counter = MagicMock()

        # Create mock span and result
        mock_span = MagicMock()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.total_tokens = 15
        mock_result.usage = mock_usage

        # Call method
        instrumentor._record_embedding_metrics(mock_span, mock_result)

        # Verify span attributes
        mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 15)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 15)

        # Verify token counter
        instrumentor.token_counter.add.assert_called_with(
            15, {"type": "input", "provider": "mistralai", "operation": "embeddings"}
        )

    def test_record_embedding_metrics_without_usage(self):
        """Test that _record_embedding_metrics handles missing usage."""
        instrumentor = MistralAIInstrumentor()
        instrumentor.token_counter = MagicMock()

        # Create mock span and result without usage
        mock_span = MagicMock()
        mock_result = MagicMock(spec=[])  # No attributes

        # Call method - should not raise
        instrumentor._record_embedding_metrics(mock_span, mock_result)

    def test_extract_usage_with_usage_object(self):
        """Test that _extract_usage extracts token counts from response."""
        instrumentor = MistralAIInstrumentor()

        # Create mock result with usage
        result = MagicMock()
        result.usage = MagicMock()
        result.usage.prompt_tokens = 10
        result.usage.completion_tokens = 20
        result.usage.total_tokens = 30

        usage = instrumentor._extract_usage(result)

        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(usage["completion_tokens"], 20)
        self.assertEqual(usage["total_tokens"], 30)

    def test_extract_usage_without_usage_attribute(self):
        """Test that _extract_usage returns None when result has no usage attribute."""
        instrumentor = MistralAIInstrumentor()

        # Create mock result without usage attribute
        result = MagicMock(spec=[])  # spec=[] means no attributes

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_extract_usage_with_exception(self, mock_logger):
        """Test that _extract_usage handles exceptions gracefully."""
        instrumentor = MistralAIInstrumentor()

        # Create mock result that raises exception when accessing usage
        result = MagicMock()
        type(result).usage = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
        )

        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)
        mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
