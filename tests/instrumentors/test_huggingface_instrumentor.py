import sys
import unittest
from unittest.mock import MagicMock, call, patch

from genai_otel.instrumentors.huggingface_instrumentor import HuggingFaceInstrumentor


class TestHuggingFaceInstrumentor(unittest.TestCase):
    """All tests for HuggingFaceInstrumentor"""

    def setUp(self):
        """Reset sys.modules before each test."""
        self.original_sys_modules = dict(sys.modules)
        # Remove transformers if it exists
        sys.modules.pop("transformers", None)

    def tearDown(self):
        """Restore sys.modules after each test."""
        sys.modules.clear()
        sys.modules.update(self.original_sys_modules)

    # ------------------------------------------------------------------
    # 1. Transformers NOT installed → instrumentation is a no-op
    # ------------------------------------------------------------------
    @patch.dict("sys.modules", {"transformers": None})
    def test_instrument_when_transformers_missing(self):
        instrumentor = HuggingFaceInstrumentor()
        config = MagicMock()

        # Act
        try:
            instrumentor.instrument(config)
        except Exception as e:
            self.fail(f"instrument() raised an exception unexpectedly: {e}")

        # Assert – transformers should not be in sys.modules
        self.assertFalse(hasattr(sys.modules, "transformers"))

    # ------------------------------------------------------------------
    # 2. Transformers IS installed → pipeline is wrapped correctly
    # ------------------------------------------------------------------
    @patch.dict("sys.modules", {"transformers": MagicMock()})
    def test_instrument_when_transformers_present(self):
        instrumentor = HuggingFaceInstrumentor()
        config = MagicMock()
        config.tracer = MagicMock()
        config.request_counter = MagicMock()

        # Track the original call
        original_call_mock = MagicMock(return_value="generated text")

        # Create a mock pipe class
        class MockPipe:
            def __init__(self):
                self.task = "text-generation"
                self.model = MagicMock()
                self.model.name_or_path = "gpt2"
                self._original_call = original_call_mock

            def __call__(self, *args, **kwargs):
                return self._original_call(*args, **kwargs)

        mock_pipe = MockPipe()
        original_pipeline = MagicMock(return_value=mock_pipe)
        sys.modules["transformers"].pipeline = original_pipeline

        # Store the original pipeline function
        original_pipeline_func = sys.modules["transformers"].pipeline

        # Act – run instrumentation
        instrumentor.instrument(config)

        # Verify pipeline was replaced with wrapped_pipeline
        import transformers

        self.assertIsNot(transformers.pipeline, original_pipeline_func)

        # Call the wrapped pipeline
        pipe = transformers.pipeline("text-generation", model="gpt2")
        result = pipe("hello world")

        # Assertions
        # a) Original pipeline was called
        original_pipeline.assert_called_once_with("text-generation", model="gpt2")

        # b) Returned pipe is our mock
        self.assertIs(pipe, mock_pipe)

        # c) Inner __call__ was invoked
        original_call_mock.assert_called_once_with("hello world")
        self.assertEqual(result, "generated text")

        # d) Span was started and attributes were set
        config.tracer.start_as_current_span.assert_called_once_with("huggingface.pipeline")
        span = config.tracer.start_as_current_span.return_value.__enter__.return_value
        span.set_attribute.assert_has_calls(
            [
                call("gen_ai.system", "huggingface"),
                call("gen_ai.request.model", "gpt2"),
                call("huggingface.task", "text-generation"),
            ],
            any_order=False
        )

        # e) Request counter was incremented
        config.request_counter.add.assert_called_once_with(
            1, {"model": "gpt2", "provider": "huggingface"}
        )

    # ------------------------------------------------------------------
    # 3. When the pipeline does NOT expose `task` or `model.name_or_path`
    # ------------------------------------------------------------------
    @patch.dict("sys.modules", {"transformers": MagicMock()})
    def test_instrument_missing_attributes(self):
        instrumentor = HuggingFaceInstrumentor()
        config = MagicMock()
        config.tracer = MagicMock()
        config.request_counter = MagicMock()

        # Track the original call
        original_call_mock = MagicMock(return_value="output")

        # Create a mock pipe class without task or model attributes
        class MockPipe:
            def __init__(self):
                self._original_call = original_call_mock

            def __call__(self, *args, **kwargs):
                return self._original_call(*args, **kwargs)

        mock_pipe = MockPipe()
        original_pipeline = MagicMock(return_value=mock_pipe)
        sys.modules["transformers"].pipeline = original_pipeline

        # Act
        instrumentor.instrument(config)
        import transformers

        pipe = transformers.pipeline("unknown-task")
        pipe("input")

        # Assertions
        original_pipeline.assert_called_once_with("unknown-task")
        original_call_mock.assert_called_once_with("input")

        # Verify span attributes fall back to "unknown"
        span = config.tracer.start_as_current_span.return_value.__enter__.return_value
        span.set_attribute.assert_has_calls(
            [
                call("gen_ai.system", "huggingface"),
                call("gen_ai.request.model", "unknown"),
                call("huggingface.task", "unknown"),
            ],
            any_order=False
        )

        # Verify request counter
        config.request_counter.add.assert_called_once_with(
            1, {"model": "unknown", "provider": "huggingface"}
        )

    # ------------------------------------------------------------------
    # 4. _extract_usage – returns None
    # ------------------------------------------------------------------
    def test_extract_usage(self):
        instrumentor = HuggingFaceInstrumentor()
        self.assertIsNone(instrumentor._extract_usage("anything"))

    # ------------------------------------------------------------------
    # 5. _check_availability – both branches
    # ------------------------------------------------------------------
    @patch.dict("sys.modules", {"transformers": None})
    @patch("genai_otel.instrumentors.huggingface_instrumentor.logger")
    def test_check_availability_missing(self, mock_logger):
        instrumentor = HuggingFaceInstrumentor()
        self.assertFalse(instrumentor._transformers_available)
        mock_logger.debug.assert_called_with(
            "Transformers library not installed, instrumentation will be skipped"
        )

    @patch.dict("sys.modules", {"transformers": MagicMock()})
    @patch("genai_otel.instrumentors.huggingface_instrumentor.logger")
    def test_check_availability_present(self, mock_logger):
        instrumentor = HuggingFaceInstrumentor()
        self.assertTrue(instrumentor._transformers_available)
        mock_logger.debug.assert_called_with(
            "Transformers library detected and available for instrumentation"
        )

    # ------------------------------------------------------------------
    # 6. __init__ calls _check_availability
    # ------------------------------------------------------------------
    @patch.object(HuggingFaceInstrumentor, "_check_availability", autospec=True)
    def test_init_calls_check_availability(self, mock_check):
        HuggingFaceInstrumentor()
        mock_check.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)