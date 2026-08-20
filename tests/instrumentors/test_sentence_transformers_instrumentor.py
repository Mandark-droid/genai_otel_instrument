import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.sentence_transformers_instrumentor import (
    SentenceTransformersInstrumentor,
)


class TestSentenceTransformersInstrumentor(unittest.TestCase):
    """Tests for SentenceTransformersInstrumentor.instrument() and its guards."""

    def test_init_available(self):
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            instrumentor = SentenceTransformersInstrumentor()
            self.assertTrue(instrumentor._available)

    def test_init_not_available(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            instrumentor = SentenceTransformersInstrumentor()
            self.assertFalse(instrumentor._available)

    def test_instrument_skips_when_not_available(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            instrumentor = SentenceTransformersInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            self.assertFalse(instrumentor._instrumented)

    def test_instrument_wraps_encode_when_available(self):
        class MockSentenceTransformer:
            encode = MagicMock(return_value="result")

        mock_module = MagicMock()
        mock_module.SentenceTransformer = MockSentenceTransformer

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            instrumentor = SentenceTransformersInstrumentor()
            config = OTelConfig()
            original_encode = MockSentenceTransformer.encode

            instrumentor.instrument(config)

            self.assertTrue(instrumentor._instrumented)
            self.assertEqual(instrumentor.config, config)
            self.assertIsNot(MockSentenceTransformer.encode, original_encode)
            self.assertTrue(getattr(MockSentenceTransformer, "_genai_otel_st_instrumented", False))

    def test_instrument_is_idempotent(self):
        class MockSentenceTransformer:
            encode = MagicMock(return_value="result")

        mock_module = MagicMock()
        mock_module.SentenceTransformer = MockSentenceTransformer

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            instrumentor = SentenceTransformersInstrumentor()
            instrumentor.instrument(OTelConfig())
            wrapped_once = MockSentenceTransformer.encode

            second_instrumentor = SentenceTransformersInstrumentor()
            second_instrumentor.instrument(OTelConfig())

            self.assertEqual(MockSentenceTransformer.encode, wrapped_once)
            self.assertTrue(second_instrumentor._instrumented)

    def test_instrument_logs_and_swallows_error_by_default(self):
        class MockSentenceTransformer:
            encode = MagicMock(return_value="result")

        mock_module = MagicMock()
        mock_module.SentenceTransformer = MockSentenceTransformer

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            instrumentor = SentenceTransformersInstrumentor()
            config = OTelConfig()
            config.fail_on_error = False

            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=RuntimeError("boom")
            ):
                instrumentor.instrument(config)  # should not raise

    def test_instrument_reraises_when_fail_on_error(self):
        class MockSentenceTransformer:
            encode = MagicMock(return_value="result")

        mock_module = MagicMock()
        mock_module.SentenceTransformer = MockSentenceTransformer

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            instrumentor = SentenceTransformersInstrumentor()
            config = OTelConfig()
            config.fail_on_error = True

            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=RuntimeError("boom")
            ):
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_extract_embedding_attributes(self):
        instrumentor = SentenceTransformersInstrumentor()
        instance = MagicMock()
        instance.model_name_or_path = "all-MiniLM-L6-v2"
        attrs = instrumentor._extract_embedding_attributes(
            instance, (), {"sentences": ["a", "b", "c"]}
        )
        self.assertEqual(attrs["gen_ai.system"], "sentence-transformers")
        self.assertEqual(attrs["gen_ai.request.model"], "all-MiniLM-L6-v2")
        self.assertEqual(attrs["gen_ai.request.input_count"], 3)

    def test_extract_response_attributes_single_sentence(self):
        """encode('one sentence') returns a 1D array: one embedding, not `dim` of them."""
        instrumentor = SentenceTransformersInstrumentor()

        class FakeArray:
            shape = (384,)

        attrs = instrumentor._extract_response_attributes(FakeArray())
        self.assertEqual(attrs["gen_ai.response.embedding_count"], 1)
        self.assertEqual(attrs["gen_ai.response.vector_size"], 384)

    def test_extract_response_attributes_batch(self):
        instrumentor = SentenceTransformersInstrumentor()

        class FakeArray:
            shape = (3, 384)

        attrs = instrumentor._extract_response_attributes(FakeArray())
        self.assertEqual(attrs["gen_ai.response.embedding_count"], 3)
        self.assertEqual(attrs["gen_ai.response.vector_size"], 384)

    def test_extract_usage_returns_none(self):
        instrumentor = SentenceTransformersInstrumentor()
        self.assertIsNone(instrumentor._extract_usage(None))
        self.assertIsNone(instrumentor._extract_usage("anything"))


if __name__ == "__main__":
    unittest.main()
