# Real-world application examples

These examples use application-shaped workflows and real file inputs while
keeping unrelated media out of the same trace. The TraceVerse Chaos Lab
fixtures are deterministic lab assets, not production customer recordings or
documents; a platform-captured run must be labeled accordingly.

## TraceSense banking voice RAG

`voice_rag_replay.py` replays one voice-RAG turn from the TraceVerse
TraceSense lab. It reads an actual WAV recording and the TraceSense
knowledge-base corpus, then emits speech-to-text, local embedding, retrieval,
and grounded-answer spans. It uses a deterministic local retrieval fallback,
so it runs without paid LLM credentials.

```powershell
python examples/real_world/voice_rag_replay.py `
  --audio REDACTED-PATH/apps/tracesense/data/samples/audio/audio_001_account_balance.wav `
  --corpus REDACTED-PATH/apps/tracesense/data/kb_corpus.py `
  --transcript "Please check my account balance and recent transactions." `
  --endpoint http://192.168.18.128:4318
```

The speech span must contain `type=audio` and `media_mime_type=audio/wav`.
Use the printed trace ID to query the `traceverse-jaeger-span-*` OpenSearch
indices. The trace must not contain an image part unless the application
actually sent an image.

## TraceSense document OCR

`document_ocr_replay.py` replays the lab's PDF/image document-extraction
shape. PDFs are emitted as `type=document` with `application/pdf`; PNG/JPEG
inputs are emitted as `type=image` with their actual image MIME type.

```powershell
python examples/real_world/document_ocr_replay.py `
  --document REDACTED-PATH/apps/tracesense/data/samples/documents/invoice_001.pdf `
  --endpoint http://192.168.18.128:4318
```

For a local shape-only check, add `--no-export`. This replay reads the file
and may extract local PDF text when `pypdf` is installed; it does not claim to
have run Ollama OCR. The full provider workflow is implemented in
`apps/tracesense/tools/extract_document.py` in the Chaos Lab and requires the
lab's Ollama multimodal service.
