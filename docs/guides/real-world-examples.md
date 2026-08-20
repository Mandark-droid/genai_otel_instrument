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
  --audio D:/Projects/traceverse-chaos-lab/apps/tracesense/data/samples/audio/audio_001_account_balance.wav `
  --corpus D:/Projects/traceverse-chaos-lab/apps/tracesense/data/kb_corpus.py `
  --transcript "Please check my account balance and recent transactions." `
  --endpoint http://192.168.18.128:4318
```

The example does not claim to call an ASR, embedding, vector database, or LLM
vendor. It validates the telemetry path with the actual application payloads
while remaining runnable without paid credentials. A live deployment can
replace the local adapters with its provider SDKs and retain the same span
contract.
