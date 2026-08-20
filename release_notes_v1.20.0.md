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

Validation: 1,906 tests passed, 15 skipped, 15 subtests passed; the v1.20.0
sdist and wheel passed `twine check` and imported successfully from the wheel.
