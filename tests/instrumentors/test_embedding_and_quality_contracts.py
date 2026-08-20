from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.aws_bedrock_instrumentor import AWSBedrockInstrumentor
from genai_otel.instrumentors.azure_ai_inference_instrumentor import AzureAIInferenceInstrumentor
from genai_otel.instrumentors.base import BaseInstrumentor
from genai_otel.instrumentors.cohere_instrumentor import CohereInstrumentor
from genai_otel.instrumentors.google_ai_instrumentor import GoogleAIInstrumentor
from genai_otel.instrumentors.huggingface_instrumentor import HuggingFaceInstrumentor
from genai_otel.instrumentors.liquid_audio_instrumentor import LiquidAudioInstrumentor
from genai_otel.instrumentors.litellm_latency_instrumentor import LiteLLMLatencyInstrumentor
from genai_otel.instrumentors.ollama_instrumentor import OllamaInstrumentor
from genai_otel.instrumentors.sentence_transformers_instrumentor import (
    SentenceTransformersInstrumentor,
)
from genai_otel.instrumentors.togetherai_instrumentor import TogetherAIInstrumentor
from genai_otel.mcp_instrumentors.vector_db_instrumentor import VectorDBInstrumentor


class _ConcreteBaseInstrumentor(BaseInstrumentor):
    def instrument(self, config):
        return None

    def _extract_usage(self, result):
        return None


def test_provider_embedding_extractors_use_common_contract():
    cases = [
        (
            CohereInstrumentor(),
            {"model": "embed-english-v3.0", "texts": ["a", "b"]},
            "cohere",
        ),
        (
            GoogleAIInstrumentor(),
            {"model": "text-embedding-005", "content": ["a", "b"]},
            "google",
        ),
        (
            OllamaInstrumentor(),
            {"model": "nomic-embed-text", "input": ["a", "b"]},
            "ollama",
        ),
        (
            TogetherAIInstrumentor(),
            {"model": "BAAI/bge-base-en-v1.5", "input": ["a", "b"]},
            "together",
        ),
        (
            AzureAIInferenceInstrumentor(),
            {"model": "text-embedding-3-small", "input": ["a", "b"]},
            "azure_ai_inference",
        ),
        (
            SentenceTransformersInstrumentor(),
            {"sentences": ["a", "b"]},
            "sentence-transformers",
        ),
    ]
    for instrumentor, kwargs, provider in cases:
        extractor = instrumentor._extract_embedding_attributes
        attrs = extractor(SimpleNamespace(model_name_or_path="model-a"), (), kwargs)
        assert attrs["gen_ai.system"] == provider
        assert attrs["gen_ai.operation.name"] == "embeddings"
        assert attrs["gen_ai.request.type"] == "embedding"
        assert attrs["gen_ai.request.input_count"] == 2


def test_embedding_response_dimensions_are_recorded():
    assert (
        CohereInstrumentor()._extract_response_attributes(
            SimpleNamespace(
                embeddings=SimpleNamespace(float=[[1.0, 2.0], [3.0, 4.0]]), generations=[]
            )
        )["gen_ai.response.vector_size"]
        == 2
    )
    assert (
        OllamaInstrumentor()._extract_response_attributes(
            {"model": "nomic", "embeddings": [[1.0, 2.0], [3.0, 4.0]]}
        )["gen_ai.response.embedding_count"]
        == 2
    )
    assert (
        TogetherAIInstrumentor()._extract_response_attributes(
            {"data": [{"embedding": [1.0, 2.0, 3.0]}]}
        )["gen_ai.response.vector_size"]
        == 3
    )
    assert (
        AzureAIInferenceInstrumentor()._extract_response_attributes(
            {"data": [{"embedding": [1.0, 2.0, 3.0]}]}
        )["gen_ai.response.embedding_count"]
        == 1
    )


def test_bedrock_embedding_request_and_response_contract():
    instrumentor = AWSBedrockInstrumentor()
    attrs = instrumentor._extract_aws_bedrock_attributes(
        None,
        (),
        {"modelId": "amazon.titan-embed-text-v2", "body": '{"inputText":"hello"}'},
    )
    assert attrs["gen_ai.operation.name"] == "embeddings"
    assert attrs["gen_ai.request.type"] == "embedding"
    assert attrs["gen_ai.request.input_count"] == 1
    response = instrumentor._extract_response_attributes(
        {"body": '{"embedding":[0.1,0.2,0.3]}', "contentType": "application/json"}
    )
    assert response["gen_ai.response.embedding_count"] == 1
    assert response["gen_ai.response.vector_size"] == 3


def test_litellm_embedding_response_contract():
    instrumentor = LiteLLMLatencyInstrumentor()
    instrumentor.tracer = MagicMock()
    span, model = instrumentor._start_span(
        "litellm.embeddings", {"model": "text-embedding-3-small", "input": ["a", "b"]}
    )
    assert model == "text-embedding-3-small"
    span.set_attribute.assert_any_call("gen_ai.operation.name", "embeddings")
    span.set_attribute.assert_any_call("gen_ai.request.type", "embedding")
    span.set_attribute.assert_any_call("gen_ai.request.input_count", 2)
    assert (
        instrumentor._extract_response_attributes({"data": [{"embedding": [1.0, 2.0]}]})[
            "gen_ai.response.vector_size"
        ]
        == 2
    )


def test_retrieval_quality_helper_derives_scores_and_model_match():
    instrumentor = object.__new__(_ConcreteBaseInstrumentor)
    span = SimpleNamespace(
        set_attribute=lambda *args: calls.append(args), add_event=lambda *args, **kwargs: None
    )
    calls = []
    instrumentor.add_retrieval_quality_attributes(
        span,
        embedding_model="model-a",
        index_embedding_model="model-b",
        scores=[0.9, 0.7, 0.4],
        context_truncated=False,
    )
    attrs = dict(calls)
    assert attrs["rag.embedding.model_match"] is False
    assert attrs["rag.result.score_max"] == 0.9
    assert attrs["rag.result.score_min"] == 0.4
    assert attrs["rag.result.score_margin"] == pytest.approx(0.2)
    assert attrs["rag.context.truncated"] is False


def test_degradation_event_contract():
    events = []
    span = SimpleNamespace(add_event=lambda *args, **kwargs: events.append((args, kwargs)))
    BaseInstrumentor.record_degradation(
        span, "retriever", "hybrid", "dense", "sparse index unavailable", True
    )
    assert events == [
        (
            ("gen_ai.degraded",),
            {
                "attributes": {
                    "gen_ai.degraded.component": "retriever",
                    "gen_ai.degraded.from": "hybrid",
                    "gen_ai.degraded.to": "dense",
                    "gen_ai.degraded.reason": "sparse index unavailable",
                    "gen_ai.degraded.recoverable": True,
                }
            },
        )
    ]


def test_vector_score_extraction_contract():
    result = SimpleNamespace(points=[SimpleNamespace(score=0.9), SimpleNamespace(score=0.6)])
    assert VectorDBInstrumentor._extract_scores(result) == [0.9, 0.6]


def test_huggingface_task_attributes_cover_asr_and_feature_extraction():
    asr_calls = []
    asr_span = SimpleNamespace(set_attribute=lambda *args: asr_calls.append(args))
    HuggingFaceInstrumentor._record_pipeline_task_attributes(
        asr_span,
        "automatic-speech-recognition",
        ({"array": [0] * 16000, "sampling_rate": 16000},),
        {},
        {"text": "hello", "language": "en", "confidence": 0.95},
        0.5,
    )
    assert ("gen_ai.response.transcript_length", 5) in asr_calls
    assert ("gen_ai.usage.audio_duration_seconds", 1.0) in asr_calls
    assert ("gen_ai.audio.real_time_factor", 0.5) in asr_calls


def test_liquid_audio_detects_audio_input_for_asr_contract():
    audio = SimpleNamespace(shape=(128, 16000))
    assert LiquidAudioInstrumentor._has_audio_input({"audio_in": audio}) is True
    assert LiquidAudioInstrumentor._audio_seconds({"audio_in": audio}) == 1.0
    assert (
        LiquidAudioInstrumentor._has_audio_input({"audio_in": SimpleNamespace(shape=(128, 0))})
        is False
    )
