import sys
import unittest
from unittest.mock import MagicMock, call, create_autospec, patch

from genai_otel.instrumentors.huggingface_instrumentor import HuggingFaceInstrumentor


class TestHuggingFaceInstrumentor(unittest.TestCase):
    """All tests for HuggingFaceInstrumentor"""

    def setUp(self):
        """Reset sys.modules before each test."""
        self.original_sys_modules = dict(sys.modules)
        sys.modules.pop("transformers", None)
        sys.modules.pop("huggingface_hub", None)

    def tearDown(self):
        """Restore sys.modules after each test."""
        sys.modules.clear()
        sys.modules.update(self.original_sys_modules)

    # ------------------------------------------------------------------
    # 1. Transformers NOT installed → instrumentation is a no-op
    # ------------------------------------------------------------------
    def test_instrument_when_transformers_missing(self):
        with patch.dict("sys.modules", {"transformers": None}):
            instrumentor = HuggingFaceInstrumentor()
            config = MagicMock()

            # Act - should not raise any exception
            instrumentor.instrument(config)

            # Assert - transformers module is not available
            self.assertFalse(instrumentor._transformers_available)

    # ------------------------------------------------------------------
    # 2. Transformers IS installed → pipeline is wrapped correctly
    # ------------------------------------------------------------------
    def test_instrument_when_transformers_present(self):
        # Create a mock pipe class that simulates a real pipeline
        class MockPipe:
            def __init__(self, task, model_name):
                self.task = task
                self.model = MagicMock()
                self.model.name_or_path = model_name

            def __call__(self, *args, **kwargs):
                return "generated text"

        # Mock the original pipeline function
        def mock_original_pipeline(task, model=None):
            return MockPipe(task, model)

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.pipeline = mock_original_pipeline

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            instrumentor = HuggingFaceInstrumentor()
            config = MagicMock()

            # Create a mock span context manager
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)

            # Mock the instrumentor's tracer and request_counter (set in BaseInstrumentor.__init__)
            instrumentor.tracer = MagicMock()
            instrumentor.tracer.start_span.return_value = mock_span_context
            instrumentor.request_counter = MagicMock()

            # Act - run instrumentation
            instrumentor.instrument(config)

            import transformers

            # The pipeline function should now be wrapped
            self.assertNotEqual(transformers.pipeline, mock_original_pipeline)

            # Call the wrapped pipeline
            pipe = transformers.pipeline("text-generation", model="gpt2")

            # Verify the wrapper delegates attributes correctly
            self.assertEqual(pipe.task, "text-generation")
            self.assertEqual(pipe.model.name_or_path, "gpt2")

            # Now call the pipe - this should trigger the instrumentation
            result = pipe("hello world")

            # Assertions
            self.assertEqual(result, "generated text")

            # Verify tracing was called
            instrumentor.tracer.start_span.assert_called_once_with("huggingface.pipeline")

            # Verify span attributes were set
            mock_span.set_attribute.assert_has_calls(
                [
                    call("gen_ai.system", "huggingface"),
                    call("gen_ai.request.model", "gpt2"),
                    call("gen_ai.operation.name", "text-generation"),
                    call("huggingface.task", "text-generation"),
                ]
            )

            # Verify metrics were recorded
            instrumentor.request_counter.add.assert_called_once_with(
                1, {"model": "gpt2", "provider": "huggingface"}
            )

    # ------------------------------------------------------------------
    # 3. When the pipeline does NOT expose `task` or `model.name_or_path`
    # ------------------------------------------------------------------
    def test_instrument_missing_attributes(self):
        # Create a mock pipe without task or model attributes
        class MockPipe:
            def __call__(self, *args, **kwargs):
                return "output"

        # Mock the original pipeline function
        def mock_original_pipeline(task):
            return MockPipe()

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.pipeline = mock_original_pipeline

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            instrumentor = HuggingFaceInstrumentor()
            config = MagicMock()

            # Create a mock span context manager
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)

            # Mock the instrumentor's tracer and request_counter (set in BaseInstrumentor.__init__)
            instrumentor.tracer = MagicMock()
            instrumentor.tracer.start_span.return_value = mock_span_context
            instrumentor.request_counter = MagicMock()

            # Act
            instrumentor.instrument(config)
            import transformers

            pipe = transformers.pipeline("unknown-task")
            result = pipe("input")

            # Assertions
            self.assertEqual(result, "output")

            # Verify span attributes fall back to "unknown"
            mock_span.set_attribute.assert_has_calls(
                [
                    call("gen_ai.system", "huggingface"),
                    call("gen_ai.request.model", "unknown"),
                    call("gen_ai.operation.name", "unknown"),
                    call("huggingface.task", "unknown"),
                ]
            )

            # Verify request counter
            instrumentor.request_counter.add.assert_called_once_with(
                1, {"model": "unknown", "provider": "huggingface"}
            )

    # ------------------------------------------------------------------
    # 4. _extract_usage – Transformers returns None, InferenceClient extracts tokens
    # ------------------------------------------------------------------
    def test_extract_usage_transformers_returns_none(self):
        """Transformers pipeline results have no usage info."""
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()
            self.assertIsNone(instrumentor._extract_usage("pipeline output"))
            self.assertIsNone(instrumentor._extract_usage(["list", "of", "results"]))

    def test_extract_usage_inference_client_object_response(self):
        """InferenceClient returns usage as object attributes."""
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()

            # Mock InferenceClient response with usage object
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 20
            mock_usage.total_tokens = 30

            mock_response = MagicMock()
            mock_response.usage = mock_usage

            usage = instrumentor._extract_usage(mock_response)

            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 10)
            self.assertEqual(usage["completion_tokens"], 20)
            self.assertEqual(usage["total_tokens"], 30)

    def test_extract_usage_inference_client_dict_response(self):
        """InferenceClient can also return usage as a dict."""
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()

            mock_response = MagicMock()
            mock_response.usage = {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            }

            usage = instrumentor._extract_usage(mock_response)

            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 15)
            self.assertEqual(usage["completion_tokens"], 25)
            self.assertEqual(usage["total_tokens"], 40)

    def test_extract_usage_inference_client_partial_tokens(self):
        """InferenceClient response with only prompt or completion tokens."""
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = None
            mock_usage.total_tokens = None

            mock_response = MagicMock()
            mock_response.usage = mock_usage

            usage = instrumentor._extract_usage(mock_response)

            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 10)
            self.assertEqual(usage["completion_tokens"], 0)
            self.assertEqual(usage["total_tokens"], 10)  # Calculated from prompt + completion

    # ------------------------------------------------------------------
    # 5. _check_availability – both transformers and inference_client branches
    # ------------------------------------------------------------------
    @patch("genai_otel.instrumentors.huggingface_instrumentor.logger")
    def test_check_availability_transformers_missing(self, mock_logger):
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()
            self.assertFalse(instrumentor._transformers_available)
            self.assertFalse(instrumentor._inference_client_available)

    @patch("genai_otel.instrumentors.huggingface_instrumentor.logger")
    def test_check_availability_transformers_present(self, mock_logger):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()
            self.assertTrue(instrumentor._transformers_available)
            self.assertFalse(instrumentor._inference_client_available)

    @patch("genai_otel.instrumentors.huggingface_instrumentor.logger")
    def test_check_availability_inference_client_present(self, mock_logger):
        mock_hub = MagicMock()
        mock_hub.InferenceClient = MagicMock()
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": mock_hub}):
            instrumentor = HuggingFaceInstrumentor()
            self.assertFalse(instrumentor._transformers_available)
            self.assertTrue(instrumentor._inference_client_available)

    @patch("genai_otel.instrumentors.huggingface_instrumentor.logger")
    def test_check_availability_both_present(self, mock_logger):
        mock_hub = MagicMock()
        mock_hub.InferenceClient = MagicMock()
        with patch.dict("sys.modules", {"transformers": MagicMock(), "huggingface_hub": mock_hub}):
            instrumentor = HuggingFaceInstrumentor()
            self.assertTrue(instrumentor._transformers_available)
            self.assertTrue(instrumentor._inference_client_available)

    # ------------------------------------------------------------------
    # 6. __init__ calls _check_availability
    # ------------------------------------------------------------------
    @patch.object(HuggingFaceInstrumentor, "_check_availability", autospec=True)
    def test_init_calls_check_availability(self, mock_check):
        HuggingFaceInstrumentor()
        mock_check.assert_called_once()

    # ------------------------------------------------------------------
    # 7. InferenceClient instrumentation tests
    # ------------------------------------------------------------------
    def test_instrument_inference_client_when_available(self):
        """Test that InferenceClient methods are wrapped correctly."""
        # Create mock InferenceClient class
        mock_inference_client = MagicMock()
        original_chat = MagicMock(return_value="chat response")
        original_text_gen = MagicMock(return_value="text response")
        mock_inference_client.chat_completion = original_chat
        mock_inference_client.text_generation = original_text_gen

        # Create mock huggingface_hub module
        mock_hub = MagicMock()
        mock_hub.InferenceClient = mock_inference_client

        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": mock_hub}):
            instrumentor = HuggingFaceInstrumentor()
            config = MagicMock()

            # Mock create_span_wrapper to return a simple wrapper
            def mock_wrapper_factory(span_name, extract_attributes):
                def decorator(func):
                    def wrapped(*args, **kwargs):
                        return func(*args, **kwargs)

                    return wrapped

                return decorator

            instrumentor.create_span_wrapper = mock_wrapper_factory

            # Act
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._inference_client_available)
            self.assertTrue(instrumentor._instrumented)

            # Verify the methods were replaced (wrapped)
            from huggingface_hub import InferenceClient

            self.assertIsNotNone(InferenceClient.chat_completion)
            self.assertIsNotNone(InferenceClient.text_generation)

    def test_extract_inference_client_attributes_with_model_in_kwargs(self):
        """Test attribute extraction when model is in kwargs."""
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()

            instance = MagicMock()
            args = []
            kwargs = {
                "model": "meta-llama/Llama-2-7b-hf",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            attrs = instrumentor._extract_inference_client_attributes(instance, args, kwargs)

            self.assertEqual(attrs["gen_ai.system"], "huggingface")
            self.assertEqual(attrs["gen_ai.request.model"], "meta-llama/Llama-2-7b-hf")
            self.assertEqual(attrs["gen_ai.operation.name"], "chat")
            self.assertEqual(attrs["gen_ai.request.max_tokens"], 100)
            self.assertEqual(attrs["gen_ai.request.temperature"], 0.7)
            self.assertEqual(attrs["gen_ai.request.top_p"], 0.9)

    def test_extract_inference_client_attributes_with_model_in_args(self):
        """Test attribute extraction when model is first positional argument."""
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()

            instance = MagicMock()
            args = ["gpt2"]
            kwargs = {"temperature": 0.5}

            attrs = instrumentor._extract_inference_client_attributes(instance, args, kwargs)

            self.assertEqual(attrs["gen_ai.system"], "huggingface")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt2")
            self.assertEqual(attrs["gen_ai.operation.name"], "chat")
            self.assertEqual(attrs["gen_ai.request.temperature"], 0.5)

    def test_extract_inference_client_attributes_no_model(self):
        """Test attribute extraction when no model provided."""
        with patch.dict("sys.modules", {"transformers": None, "huggingface_hub": None}):
            instrumentor = HuggingFaceInstrumentor()

            instance = MagicMock()
            args = []
            kwargs = {}

            attrs = instrumentor._extract_inference_client_attributes(instance, args, kwargs)

            self.assertEqual(attrs["gen_ai.system"], "huggingface")
            self.assertEqual(attrs["gen_ai.request.model"], "unknown")
            self.assertEqual(attrs["gen_ai.operation.name"], "chat")

    # ------------------------------------------------------------------
    # 8. Test ImportError during instrument() method
    # ------------------------------------------------------------------
    def test_instrument_importlib_fails(self):
        """Test that ImportError during instrumentation is handled gracefully."""
        # Setup: transformers is available during init but fails during instrument
        with patch.dict("sys.modules", {"transformers": MagicMock()}):
            instrumentor = HuggingFaceInstrumentor()
            self.assertTrue(instrumentor._transformers_available)

            config = MagicMock()
            config.tracer = MagicMock()
            config.fail_on_error = False  # Should handle errors gracefully

            # Mock importlib.import_module to raise ImportError
            with patch("importlib.import_module", side_effect=ImportError("Module not found")):
                # Act - should not raise, should handle gracefully
                instrumentor.instrument(config)

                # Should complete without errors (pass block executes)

    # ------------------------------------------------------------------
    # 9. Test AutoModelForCausalLM instrumentation doesn't fail
    # ------------------------------------------------------------------
    def test_instrument_model_classes_no_error(self):
        """Test that model class instrumentation attempt completes without error.

        Note: Full integration test with real transformers models is in examples.
        This test just verifies the instrumentation code doesn't crash.
        """
        # Mock transformers module
        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            instrumentor = HuggingFaceInstrumentor()
            self.assertTrue(instrumentor._transformers_available)

            config = MagicMock()
            config.fail_on_error = False

            # Act - should complete without raising even if mocks aren't perfect
            # The _instrument_model_classes method has try/except handling
            instrumentor.instrument(config)

            # If we get here, no exceptions were raised - test passes
            self.assertTrue(instrumentor._instrumented)


if __name__ == "__main__":
    unittest.main(verbosity=2)
