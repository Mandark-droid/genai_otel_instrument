import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.replicate_instrumentor import ReplicateInstrumentor


class TestReplicateInstrumentor(unittest.TestCase):
    """Tests for ReplicateInstrumentor"""

    def test_instrument_when_replicate_not_installed(self):
        """Test that instrument handles missing replicate gracefully."""
        with patch.dict("sys.modules", {"replicate": None}):
            instrumentor = ReplicateInstrumentor()
            config = MagicMock()

            # Should not raise any exception
            instrumentor.instrument(config)

            # Config should be stored
            self.assertEqual(instrumentor.config, config)

    def test_instrument_with_replicate_installed(self):
        """Test that instrument wraps replicate.run when installed."""
        # Create mock replicate module
        mock_replicate = MagicMock()
        original_run = MagicMock(return_value="model output")
        mock_replicate.run = original_run

        with patch.dict("sys.modules", {"replicate": mock_replicate}):
            instrumentor = ReplicateInstrumentor()
            config = MagicMock()

            # Create mock tracer
            mock_tracer = MagicMock()
            instrumentor.tracer = mock_tracer

            # Create mock span
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            mock_tracer.start_as_current_span.return_value = mock_span_context

            # Create mock request counter
            mock_request_counter = MagicMock()
            instrumentor.request_counter = mock_request_counter

            # Act
            instrumentor.instrument(config)

            # The run function should now be wrapped
            self.assertNotEqual(mock_replicate.run, original_run)

            # Call the wrapped run function
            result = mock_replicate.run("stability-ai/stable-diffusion:model-version")

            # Assertions
            self.assertEqual(result, "model output")

            # Verify tracing was called
            mock_tracer.start_as_current_span.assert_called_once_with("replicate.run")

            # Verify span attributes were set
            mock_span.set_attribute.assert_any_call("gen_ai.system", "replicate")
            mock_span.set_attribute.assert_any_call(
                "gen_ai.request.model", "stability-ai/stable-diffusion:model-version"
            )

            # Verify request counter was called
            mock_request_counter.add.assert_called_once_with(
                1, {"model": "stability-ai/stable-diffusion:model-version", "provider": "replicate"}
            )

    def test_wrapped_run_without_args(self):
        """Test that wrapped run handles call without args (uses 'unknown' as model)."""
        # Create mock replicate module
        mock_replicate = MagicMock()
        original_run = MagicMock(return_value="model output")
        mock_replicate.run = original_run

        with patch.dict("sys.modules", {"replicate": mock_replicate}):
            instrumentor = ReplicateInstrumentor()
            config = MagicMock()

            # Create mock tracer
            mock_tracer = MagicMock()
            instrumentor.tracer = mock_tracer

            # Create mock span
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            mock_tracer.start_as_current_span.return_value = mock_span_context

            # Create mock request counter
            mock_request_counter = MagicMock()
            instrumentor.request_counter = mock_request_counter

            # Act
            instrumentor.instrument(config)

            # Call the wrapped run function without args (only kwargs)
            result = mock_replicate.run()

            # Assertions
            self.assertEqual(result, "model output")

            # Verify span attribute was set with "unknown" model
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "unknown")

            # Verify request counter was called with "unknown" model
            mock_request_counter.add.assert_called_once_with(
                1, {"model": "unknown", "provider": "replicate"}
            )

    def test_wrapped_run_with_kwargs(self):
        """Test that wrapped run handles kwargs properly."""
        # Create mock replicate module
        mock_replicate = MagicMock()
        original_run = MagicMock(return_value="model output")
        mock_replicate.run = original_run

        with patch.dict("sys.modules", {"replicate": mock_replicate}):
            instrumentor = ReplicateInstrumentor()
            config = MagicMock()

            # Create mock tracer
            mock_tracer = MagicMock()
            instrumentor.tracer = mock_tracer

            # Create mock span
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            mock_tracer.start_as_current_span.return_value = mock_span_context

            # Create mock request counter
            mock_request_counter = MagicMock()
            instrumentor.request_counter = mock_request_counter

            # Act
            instrumentor.instrument(config)

            # Call the wrapped run function with kwargs
            result = mock_replicate.run(model="test-model", input={"prompt": "test"})

            # Assertions
            self.assertEqual(result, "model output")

            # Verify original_run was called with the kwargs
            original_run.assert_called_once_with(model="test-model", input={"prompt": "test"})

    def test_extract_usage(self):
        """Test that _extract_usage returns None."""
        instrumentor = ReplicateInstrumentor()
        result = instrumentor._extract_usage("any_result")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
