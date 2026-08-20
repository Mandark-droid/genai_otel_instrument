# genai-otel-instrument v1.20.0

This release closes the supported SDK embedding-coverage slice tracked by
upstream issue #23 and adds the voice-RAG telemetry needed by the upstream
issues draft.

Highlights:

- Embedding spans for Cohere, Google GenAI, Ollama, Together, Bedrock, LiteLLM,
  Azure AI Inference, Hugging Face feature extraction, and SentenceTransformers.
- Retrieval-quality helpers, vector score summaries, and awaited async Qdrant
  coverage.
- ASR attributes for Hugging Face and Liquid Audio, shared Sarvam/ElevenLabs
  TTS fields, and streaming TTFT measurement.
- The public `gen_ai.degraded` span event helper.

## Fixed during review

A pre-merge code review caught several correctness issues in the above, all
fixed before this release:

- Bedrock Titan Text generation calls were misclassified as embeddings and
  priced off the wrong table, because the classifier keyed on a body field
  Titan Text and Titan Embed both send.
- Azure AI Inference embedding calls resolved to no pricing entry (billed as
  $0) since the pricing table only had the catalog-prefixed model names, not
  the bare deployment name the SDK reports.
- `capture_embedding_vectors` behaved inconsistently across providers.
- SentenceTransformers `encode()` on a single string reported the vector
  dimension as the embedding count instead of `1`.
- Weaviate query spans never got `rag.result.score_*` attributes; the score
  extractor didn't match Weaviate's actual response shape.
- Added the missing `instrument()`/idempotency/`fail_on_error` test coverage
  for the three new instrumentors (Azure AI Inference, Liquid Audio,
  SentenceTransformers).

Validation: 1,951 tests passed, 12 skipped, 15 subtests passed; `black`/`isort`
clean; the v1.20.0 sdist and wheel passed `twine check` and imported
successfully from the wheel.
