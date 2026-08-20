# Real-world application examples

These examples use real application inputs and keep unrelated media out of
the same trace.

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
