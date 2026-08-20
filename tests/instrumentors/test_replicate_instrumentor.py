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

            # Create mock tracer and metrics
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()
            instrumentor.latency_histogram = MagicMock()
            instrumentor.cost_gauge = MagicMock()

            # Act
            instrumentor.instrument(config)

            # The run function should now be wrapped (callable)
            self.assertIsNotNone(mock_replicate.run)
            self.assertTrue(callable(mock_replicate.run))

            # Call the wrapped run function
            result = mock_replicate.run("stability-ai/stable-diffusion:model-version")

            # Assertions
            self.assertEqual(result, "model output")
            # Verify original function was called
            original_run.assert_called_once_with("stability-ai/stable-diffusion:model-version")

    def test_wrapped_run_without_args(self):
        """Test that wrapped run handles call without args (uses 'unknown' as model)."""
        # Create mock replicate module
        mock_replicate = MagicMock()
        original_run = MagicMock(return_value="model output")
        mock_replicate.run = original_run

        with patch.dict("sys.modules", {"replicate": mock_replicate}):
            instrumentor = ReplicateInstrumentor()
            config = MagicMock()

            # Create mock tracer and metrics
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()
            instrumentor.latency_histogram = MagicMock()
            instrumentor.cost_gauge = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Call the wrapped run function without args (only kwargs)
            result = mock_replicate.run()

            # Assertions
            self.assertEqual(result, "model output")
            # Verify original function was called
            original_run.assert_called_once_with()

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

    def test_extract_run_attributes_chat_model(self):
        """A non-embedding model reference stays classified as a plain run."""
        instrumentor = ReplicateInstrumentor()
        attrs = instrumentor._extract_run_attributes(
            None, ("stability-ai/stable-diffusion:model-version",), {}
        )

        self.assertEqual(attrs["gen_ai.system"], "replicate")
        self.assertEqual(attrs["gen_ai.operation.name"], "run")
        self.assertNotIn("gen_ai.request.type", attrs)

    def test_extract_run_attributes_embedding_model(self):
        """A known embedding model family (here BGE) is classified as an
        embedding call, even without the literal substring "embed" - real
        Replicate embedding models are commonly named after their model
        family (bge/e5/gte/mpnet/minilm) rather than "embed" itself."""
        instrumentor = ReplicateInstrumentor()
        attrs = instrumentor._extract_run_attributes(
            None,
            ("nateraw/bge-large-en-v1.5",),
            {"input": {"text": "hello world"}},
        )

        self.assertEqual(attrs["gen_ai.operation.name"], "embeddings")
        self.assertEqual(attrs["gen_ai.request.type"], "embedding")
        self.assertEqual(attrs["gen_ai.request.input_count"], 1)

    def test_extract_run_attributes_embedding_model_literal_embed(self):
        """A model reference containing the literal substring "embed" too."""
        instrumentor = ReplicateInstrumentor()
        attrs = instrumentor._extract_run_attributes(
            None, ("nomic-ai/nomic-embed-text-v1.5",), {"input": {"text": "hi"}}
        )

        self.assertEqual(attrs["gen_ai.operation.name"], "embeddings")
        self.assertEqual(attrs["gen_ai.request.type"], "embedding")

    def test_count_inputs(self):
        instrumentor = ReplicateInstrumentor()
        self.assertEqual(instrumentor._count_inputs({"text": "hi"}), 1)
        self.assertEqual(instrumentor._count_inputs({"texts": ["a", "b", "c"]}), 3)
        self.assertEqual(instrumentor._count_inputs({"inputs": ["a", "b"]}), 2)
        self.assertEqual(instrumentor._count_inputs({"unrelated": True}), 0)
        self.assertEqual(instrumentor._count_inputs("raw string"), 1)
        self.assertEqual(instrumentor._count_inputs(None), 0)

    def test_extract_response_attributes_single_vector(self):
        """A flat list of numbers from an embedding call is one vector."""
        instrumentor = ReplicateInstrumentor()
        instrumentor._extract_run_attributes(None, ("nateraw/bge-large-en-v1.5",), {})

        attrs = instrumentor._extract_response_attributes([0.1, 0.2, 0.3, 0.4])

        self.assertEqual(attrs["gen_ai.response.embedding_count"], 1)
        self.assertEqual(attrs["gen_ai.response.vector_size"], 4)

    def test_extract_response_attributes_batch(self):
        """A list of vectors from a batched embedding call."""
        instrumentor = ReplicateInstrumentor()
        instrumentor._extract_run_attributes(None, ("nateraw/bge-large-en-v1.5",), {})

        attrs = instrumentor._extract_response_attributes([[0.1, 0.2], [0.3, 0.4]])

        self.assertEqual(attrs["gen_ai.response.embedding_count"], 2)
        self.assertEqual(attrs["gen_ai.response.vector_size"], 2)

    def test_extract_response_attributes_non_embedding_call_is_not_mislabeled(self):
        """A non-embedding model's list-shaped output (e.g. audio samples,
        bounding boxes) must not be reported as an embedding, even though the
        raw shape looks identical to an embedding vector."""
        instrumentor = ReplicateInstrumentor()
        instrumentor._extract_run_attributes(None, ("stability-ai/some-audio-model",), {})

        attrs = instrumentor._extract_response_attributes([0.1, 0.2, 0.3, 0.4])

        self.assertEqual(attrs, {})

    def test_extract_response_attributes_non_list_result(self):
        instrumentor = ReplicateInstrumentor()
        instrumentor._extract_run_attributes(None, ("nateraw/bge-large-en-v1.5",), {})

        self.assertEqual(instrumentor._extract_response_attributes("not a list"), {})
        self.assertEqual(instrumentor._extract_response_attributes([]), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
