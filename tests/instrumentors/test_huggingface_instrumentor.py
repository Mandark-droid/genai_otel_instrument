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
            config.tracer = MagicMock()
            config.request_counter = MagicMock()

            # Create a mock span context manager
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            config.tracer.start_as_current_span.return_value = mock_span_context

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
            config.tracer.start_as_current_span.assert_called_once_with("huggingface.pipeline")

            # Verify span attributes were set
            mock_span.set_attribute.assert_has_calls(
                [
                    call("gen_ai.system", "huggingface"),
                    call("gen_ai.request.model", "gpt2"),
                    call("huggingface.task", "text-generation"),
                ]
            )

            # Verify metrics were recorded
            config.request_counter.add.assert_called_once_with(
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
            config.tracer = MagicMock()
            config.request_counter = MagicMock()

            # Create a mock span context manager
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            config.tracer.start_as_current_span.return_value = mock_span_context

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
                    call("huggingface.task", "unknown"),
                ]
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
    @patch("genai_otel.instrumentors.huggingface_instrumentor.logger")
    def test_check_availability_missing(self, mock_logger):
        with patch.dict("sys.modules", {"transformers": None}):
            instrumentor = HuggingFaceInstrumentor()
            self.assertFalse(instrumentor._transformers_available)
            mock_logger.debug.assert_called_with(
                "Transformers library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.huggingface_instrumentor.logger")
    def test_check_availability_present(self, mock_logger):
        with patch.dict("sys.modules", {"transformers": MagicMock()}):
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
