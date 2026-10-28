# Real-world application examples

Release validation should use application-shaped inputs, not only spans
assembled from isolated attribute fixtures. The repository includes a
vendor-key-free replay of a banking voice-RAG turn:

[`examples/real_world/voice_rag_replay.py`](https://github.com/Mandark-droid/genai_otel_instrument/blob/main/examples/real_world/voice_rag_replay.py)

The replay reads an actual WAV recording and a knowledge-base corpus. It emits
one trace containing separate speech-to-text, local embedding, retrieval, and
grounded-answer spans. Audio is attached only to the speech span and must be
represented as `type=audio` with `media_mime_type=audio/wav`.

Run it against the remote collector:

```powershell
python examples/real_world/voice_rag_replay.py `
  --audio <path-to>/account_balance.wav `
  --corpus <path-to>/kb_corpus.py `
  --transcript "Please check my account balance and recent transactions." `
  --endpoint http://<collector-host>:4318
```

The example does not claim to call an ASR, embedding, vector database, or LLM
vendor. It validates the telemetry path with the application payload shape
while remaining runnable without paid credentials. Use deterministic lab
fixtures for the WAV and corpus, not production customer data.

## Document OCR

The companion replay covers the other multimodal workflow: PDF/image ingestion
and structured document extraction.

```powershell
python examples/real_world/document_ocr_replay.py `
  --document <path-to>/invoice.pdf `
  --endpoint http://<collector-host>:4318
```

The emitted OCR span must report `type=document` and
`media_mime_type=application/pdf` for a PDF. An image input must report
`type=image` and its image MIME type. Use `--no-export` for a local shape-only
check. This replay validates file handling and telemetry; it does not claim
that a provider-backed OCR model ran.

A full voice-RAG application runs ASR → embedding → vector retrieval →
reranking → grounded answer → TTS, and an OCR application runs document bytes
through multimodal extraction. These replays cover the telemetry shape of both
without the providers.
