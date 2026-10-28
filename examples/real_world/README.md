# Real-world application examples

These examples use application-shaped workflows and real file inputs while
keeping unrelated media out of the same trace. Use deterministic lab fixtures
as inputs, not production customer recordings or documents, and label a
captured run accordingly.

## Banking voice RAG

`voice_rag_replay.py` replays one voice-RAG turn. It reads an actual WAV
recording and a knowledge-base corpus, then emits speech-to-text, local
embedding, retrieval, and grounded-answer spans. It uses a deterministic local
retrieval fallback, so it runs without paid LLM credentials.

```powershell
python examples/real_world/voice_rag_replay.py `
  --audio <path-to>/account_balance.wav `
  --corpus <path-to>/kb_corpus.py `
  --transcript "Please check my account balance and recent transactions." `
  --endpoint http://<collector-host>:4318
```

The speech span must contain `type=audio` and `media_mime_type=audio/wav`.
Use the printed trace ID to look the trace up in your backend. The trace must
not contain an image part unless the application actually sent an image.

## Document OCR

`document_ocr_replay.py` replays a PDF/image document-extraction workflow.
PDFs are emitted as `type=document` with `application/pdf`; PNG/JPEG inputs
are emitted as `type=image` with their actual image MIME type.

```powershell
python examples/real_world/document_ocr_replay.py `
  --document <path-to>/invoice.pdf `
  --endpoint http://<collector-host>:4318
```

For a local shape-only check, add `--no-export`. This replay reads the file
and may extract local PDF text when `pypdf` is installed; it does not claim to
have run a provider-backed OCR model.
