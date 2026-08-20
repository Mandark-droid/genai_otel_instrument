# Real-world application examples

Release validation should use application-shaped inputs, not only spans
assembled from isolated attribute fixtures. The repository includes a
vendor-key-free replay of a real TraceSense banking voice-RAG turn:

[`examples/real_world/voice_rag_replay.py`](https://github.com/Mandark-droid/genai_otel_instrument/blob/main/examples/real_world/voice_rag_replay.py)

The replay reads an actual WAV recording and the TraceSense knowledge-base
corpus. It emits one trace containing separate speech-to-text, local
embedding, retrieval, and grounded-answer spans. Audio is attached only to the
speech span and must be represented as `type=audio` with
`media_mime_type=audio/wav`.

Run it against the remote collector:

```powershell
python examples/real_world/voice_rag_replay.py `
  --audio REDACTED-PATH/apps/tracesense/data/samples/audio/audio_001_account_balance.wav `
  --corpus REDACTED-PATH/apps/tracesense/data/kb_corpus.py `
  --transcript "Please check my account balance and recent transactions." `
  --endpoint http://192.168.18.128:4318
```

The example does not claim to call an ASR, embedding, vector database, or LLM
vendor. It validates the telemetry path with the application payload shape
while remaining runnable without paid credentials. The checked-in Chaos Lab
WAV and corpus are deterministic lab fixtures, not production customer data.

## TraceSense document OCR

The companion replay covers the other multimodal workflow already implemented
in the Chaos Lab: PDF/image ingestion and structured document extraction.

```powershell
python examples/real_world/document_ocr_replay.py `
  --document REDACTED-PATH/apps/tracesense/data/samples/documents/invoice_001.pdf `
  --endpoint http://192.168.18.128:4318
```

The emitted OCR span must report `type=document` and
`media_mime_type=application/pdf` for a PDF. An image input must report
`type=image` and its image MIME type. Use `--no-export` for a local shape-only
check. This replay validates file handling and telemetry; it does not claim
that a provider-backed OCR model ran. For the provider-backed path, run the
Chaos Lab's `apps/tracesense/tools/extract_document.py` with Ollama available.

The broader application workflow is documented in the Chaos Lab's
`apps/tracesense/agents/voice_retrieval_agent.py` and
`apps/tracesense/tools/extract_document.py`: Voice RAG runs ASR → embedding →
Qdrant retrieval → reranking → grounded answer → TTS, while OCR runs
document bytes through multimodal extraction. The lab's `apps/traceinsure`
document processor is a separate insurance workflow and currently uses
simulated claim-field extraction, so it should not be presented as a live OCR
provider trace.
