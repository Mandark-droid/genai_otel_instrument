import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.azure_ai_inference_instrumentor import AzureAIInferenceInstrumentor


def _mock_sdk_modules(embeddings_client=None):
    """Build a sys.modules dict exposing azure.ai.inference.EmbeddingsClient.

    All parent packages must be present in sys.modules too, or `from
    azure.ai.inference import EmbeddingsClient` re-triggers real imports.
    """
    mock_azure = MagicMock()
    mock_azure.ai = MagicMock()
    mock_azure.ai.inference = MagicMock()
    if embeddings_client is not None:
        mock_azure.ai.inference.EmbeddingsClient = embeddings_client
    return {
        "azure": mock_azure,
        "azure.ai": mock_azure.ai,
        "azure.ai.inference": mock_azure.ai.inference,
    }


def _unavailable_sdk_modules():
    return {"azure": None, "azure.ai": None, "azure.ai.inference": None}


class TestAzureAIInferenceInstrumentor(unittest.TestCase):
    """Tests for AzureAIInferenceInstrumentor.instrument() and its guards."""

    def test_init_available(self):
        with patch.dict("sys.modules", _mock_sdk_modules(MagicMock())):
            instrumentor = AzureAIInferenceInstrumentor()
            self.assertTrue(instrumentor._available)

    def test_init_not_available(self):
        with patch.dict("sys.modules", _unavailable_sdk_modules()):
            instrumentor = AzureAIInferenceInstrumentor()
            self.assertFalse(instrumentor._available)

    def test_instrument_skips_when_not_available(self):
        with patch.dict("sys.modules", _unavailable_sdk_modules()):
            instrumentor = AzureAIInferenceInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            self.assertEqual(instrumentor.config, config)
            self.assertFalse(instrumentor._instrumented)

    def test_instrument_wraps_embed_when_available(self):
        class MockEmbeddingsClient:
            embed = MagicMock(return_value="result")

        with patch.dict("sys.modules", _mock_sdk_modules(MockEmbeddingsClient)):
            instrumentor = AzureAIInferenceInstrumentor()
            config = OTelConfig()
            original_embed = MockEmbeddingsClient.embed

            instrumentor.instrument(config)

            self.assertTrue(instrumentor._instrumented)
            self.assertEqual(instrumentor.config, config)
            self.assertIsNot(MockEmbeddingsClient.embed, original_embed)
            self.assertTrue(
                getattr(MockEmbeddingsClient, "_genai_otel_azure_inference_instrumented", False)
            )

    def test_instrument_is_idempotent(self):
        class MockEmbeddingsClient:
            embed = MagicMock(return_value="result")

        with patch.dict("sys.modules", _mock_sdk_modules(MockEmbeddingsClient)):
            instrumentor = AzureAIInferenceInstrumentor()
            instrumentor.instrument(OTelConfig())
            wrapped_once = MockEmbeddingsClient.embed

            second_instrumentor = AzureAIInferenceInstrumentor()
            second_instrumentor.instrument(OTelConfig())

            # A second instrument() call must not re-wrap an already-wrapped method.
            self.assertEqual(MockEmbeddingsClient.embed, wrapped_once)
            self.assertTrue(second_instrumentor._instrumented)

    def test_instrument_logs_and_swallows_error_by_default(self):
        class MockEmbeddingsClient:
            embed = MagicMock(return_value="result")

        with patch.dict("sys.modules", _mock_sdk_modules(MockEmbeddingsClient)):
            instrumentor = AzureAIInferenceInstrumentor()
            config = OTelConfig()
            config.fail_on_error = False

            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=RuntimeError("boom")
            ):
                # Should not raise.
                instrumentor.instrument(config)

    def test_instrument_reraises_when_fail_on_error(self):
        class MockEmbeddingsClient:
            embed = MagicMock(return_value="result")

        with patch.dict("sys.modules", _mock_sdk_modules(MockEmbeddingsClient)):
            instrumentor = AzureAIInferenceInstrumentor()
            config = OTelConfig()
            config.fail_on_error = True

            with patch.object(
                instrumentor, "create_span_wrapper", side_effect=RuntimeError("boom")
            ):
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_extract_embedding_attributes(self):
        instrumentor = AzureAIInferenceInstrumentor()
        instance = MagicMock()
        attrs = instrumentor._extract_embedding_attributes(
            instance, (), {"model": "Cohere-embed-v3-english", "input": ["a", "b"]}
        )
        self.assertEqual(attrs["gen_ai.system"], "azure_ai_inference")
        self.assertEqual(attrs["gen_ai.request.model"], "Cohere-embed-v3-english")
        self.assertEqual(attrs["gen_ai.operation.name"], "embeddings")
        self.assertEqual(attrs["gen_ai.request.type"], "embedding")
        self.assertEqual(attrs["gen_ai.request.input_count"], 2)

    def test_extract_usage(self):
        instrumentor = AzureAIInferenceInstrumentor()
        self.assertIsNone(instrumentor._extract_usage(None))

        usage = instrumentor._extract_usage({"usage": {"prompt_tokens": 5, "total_tokens": 5}})
        self.assertEqual(usage["prompt_tokens"], 5)
        self.assertEqual(usage["total_tokens"], 5)
        self.assertEqual(usage["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
