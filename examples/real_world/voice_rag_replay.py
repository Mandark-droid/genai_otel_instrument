"""Replay one real TraceSense banking voice-RAG turn through this package.

This offline example uses an actual WAV recording and the TraceSense
knowledge-base corpus. It emits speech-to-text, local embedding, retrieval,
and grounded-answer spans without requiring paid LLM credentials. The audio
content part is emitted only on the speech span and must be typed as ``audio``.

Example::

    python examples/real_world/voice_rag_replay.py \
      --audio D:/Projects/traceverse-chaos-lab/apps/tracesense/data/samples/audio/audio_001_account_balance.wav \
      --corpus D:/Projects/traceverse-chaos-lab/apps/tracesense/data/kb_corpus.py \
      --transcript "Please check my account balance and recent transactions."

The corpus loader accepts TraceSense ``kb_corpus.py`` or JSONL records with a
``text`` field and optional ``passage_id``/``title`` fields.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import sys
import wave
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor


class _ConcreteBaseInstrumentor(BaseInstrumentor):
    def instrument(self, config):
        return None

    def _extract_usage(self, result):
        return None


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[\w]+", value.lower()) if len(token) > 2}


def _hash_embedding(value: str, dimensions: int = 32) -> list[float]:
    """Create a deterministic local feature vector for offline replay."""
    vector = [0.0] * dimensions
    for token in sorted(_tokens(value)):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:2], "big") % dimensions] += 1.0
    norm = sum(component * component for component in vector) ** 0.5 or 1.0
    return [round(component / norm, 6) for component in vector]


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("voice_rag_corpus", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import corpus module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        passages = module.generate_corpus()
        return [
            item.as_payload() if hasattr(item, "as_payload") else dict(item) for item in passages
        ]

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _retrieve(query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_tokens = _tokens(query)
    ranked = []
    for index, record in enumerate(records):
        text = str(record.get("text", ""))
        overlap = len(query_tokens & _tokens(text))
        score = overlap / max(len(query_tokens), 1)
        if score > 0:
            ranked.append((score, index, record))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    results = []
    for score, _, record in ranked[:top_k]:
        result = dict(record)
        result["score"] = round(score, 6)
        result.setdefault("id", result.get("passage_id", "unknown"))
        results.append(result)
    return results


def _audio_metadata(path: Path) -> tuple[bytes, int, float, str]:
    with wave.open(str(path), "rb") as audio:
        payload = audio.readframes(audio.getnframes())
        sample_rate = audio.getframerate()
        duration = audio.getnframes() / float(sample_rate or 1)
    return payload, sample_rate, duration, "wav"


def run(audio_path: Path, corpus_path: Path, transcript: str, endpoint: str) -> dict[str, Any]:
    audio_bytes, sample_rate, audio_seconds, audio_format = _audio_metadata(audio_path)
    records = _load_corpus(corpus_path)
    if not records:
        raise ValueError(f"Corpus is empty: {corpus_path}")

    config = OTelConfig()
    config.media_capture_mode = "reference_only"
    config.media_store = "none"
    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "tracesense-voice-rag-real-world",
                "deployment.environment": "validation",
            }
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("examples.real_world.voice_rag")
    base = object.__new__(_ConcreteBaseInstrumentor)
    base.config = config
    media = OpenAIInstrumentor()
    media.config = config
    media._instrumented = True

    with tracer.start_as_current_span("tracesense.voice_rag.turn") as turn:
        turn.set_attribute("rag.mode", "real_world_offline_replay")
        turn.set_attribute("rag.corpus.id", corpus_path.stem)
        turn.set_attribute("rag.input.audio.filename", audio_path.name)
        turn.set_attribute("rag.input.audio.bytes", len(audio_bytes))

        with tracer.start_as_current_span("tracesense.speech_to_text") as speech:
            speech.set_attribute("gen_ai.system", "openai-compatible")
            speech.set_attribute("gen_ai.request.model", "voice-replay-transcript")
            speech.set_attribute("gen_ai.operation.name", "speech_to_text")
            speech.set_attribute("gen_ai.request.type", "speech_to_text")
            speech.set_attribute("gen_ai.request.audio.sample_rate", sample_rate)
            speech.set_attribute("gen_ai.usage.audio_duration_seconds", audio_seconds)
            speech.set_attribute("gen_ai.response.transcript_length", len(transcript))
            media._emit_media_attributes(
                speech,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": transcript},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": base64.b64encode(audio_bytes).decode("ascii"),
                                        "format": audio_format,
                                    },
                                },
                            ],
                        }
                    ]
                },
                result=None,
            )

        vector = _hash_embedding(transcript)
        with tracer.start_as_current_span("local.embedding") as embedding:
            embedding.set_attribute("gen_ai.system", "local")
            embedding.set_attribute("gen_ai.request.model", "hashing-tfidf-replay-v1")
            embedding.set_attribute("gen_ai.operation.name", "embeddings")
            embedding.set_attribute("gen_ai.request.type", "embedding")
            embedding.set_attribute("gen_ai.request.input_count", 1)
            embedding.set_attribute("gen_ai.response.embedding_count", 1)
            embedding.set_attribute("gen_ai.response.vector_size", len(vector))
            base.add_embedding_attributes(embedding, "hashing-tfidf-replay-v1", transcript)

        hits = _retrieve(transcript, records, top_k=5)
        with tracer.start_as_current_span("local.vector_search") as retrieval:
            retrieval.set_attribute("db.system", "local_lexical_replay")
            retrieval.set_attribute("db.operation", "query")
            retrieval.set_attribute("db.collection.name", corpus_path.stem)
            retrieval.set_attribute("db.vector.top_k", 5)
            base.add_retrieval_attributes(retrieval, hits, query=transcript)
            base.add_retrieval_quality_attributes(
                retrieval,
                embedding_model="hashing-tfidf-replay-v1",
                index_embedding_model="hashing-tfidf-replay-v1",
                embedding_dim=len(vector),
                scores=[hit["score"] for hit in hits],
                distance="lexical-overlap",
                corpus_version=corpus_path.stem,
                context_truncated=False,
            )

        answer = (
            f"Based on the support guide: {hits[0].get('text')}"
            if hits
            else "I could not find verified information in the support knowledge base."
        )
        with tracer.start_as_current_span("tracesense.grounded_answer") as answer_span:
            answer_span.set_attribute("gen_ai.operation.name", "chat")
            answer_span.set_attribute("gen_ai.response.text", answer[:1000])
            answer_span.set_attribute("rag.answer.refused", not bool(hits))
            answer_span.set_attribute("rag.grounded.method", "retrieved_passage")

        trace_id = f"{turn.get_span_context().trace_id:032x}"

    provider.force_flush(timeout_millis=15000)
    provider.shutdown()
    spans = memory_exporter.get_finished_spans()
    speech_span = next(span for span in spans if span.name == "tracesense.speech_to_text")
    media_type = speech_span.attributes.get("gen_ai.prompt.0.content.1.type")
    if media_type != "audio":
        raise AssertionError(f"Expected audio content type, got {media_type!r}")
    print(f"trace_id={trace_id}")
    print(f"span_count={len(spans)}")
    print(f"speech_content_type={media_type}")
    print(
        f"speech_mime_type={speech_span.attributes.get('gen_ai.prompt.0.content.1.media_mime_type')}"
    )
    print(f"retrieved_passages={len(hits)}")
    return {"trace_id": trace_id, "span_count": len(spans), "media_type": media_type}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="Path to a real WAV recording")
    parser.add_argument(
        "--corpus", required=True, type=Path, help="TraceSense kb_corpus.py or JSONL corpus"
    )
    parser.add_argument(
        "--transcript", required=True, help="Transcript from the real voice recording"
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
        help="OTLP HTTP collector base endpoint",
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 2
    if not args.corpus.is_file():
        print(f"Corpus file not found: {args.corpus}", file=sys.stderr)
        return 2
    run(args.audio, args.corpus, args.transcript, args.endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
