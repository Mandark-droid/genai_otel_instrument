# Speech telemetry

Speech spans use the same operation vocabulary across local and hosted
instrumentors:

- ASR: `gen_ai.operation.name` and `gen_ai.request.type` are
  `speech_to_text`.
- TTS: `gen_ai.operation.name` and `gen_ai.request.type` are
  `text_to_speech`.
- `gen_ai.request.model`, `gen_ai.request.language_code`,
  `gen_ai.request.voice_id`, `gen_ai.request.audio.sample_rate`,
  `gen_ai.request.streamed`, and `gen_ai.response.output_format` are shared
  where the provider exposes them.
- `gen_ai.usage.audio_duration_seconds` records measured or provider-reported
  audio duration. ASR spans may also include
  `gen_ai.response.transcript_length`, `gen_ai.response.language_code`, and
  `gen_ai.audio.real_time_factor`.
- Streaming TTS and audio-generation spans record
  `gen_ai.server.time_to_first_token` when the first audio chunk is observed.
  Non-streaming calls leave this attribute unset; zero is not a valid proxy for
  an unmeasured first-byte time.

The Hugging Face instrumentor covers the
`automatic-speech-recognition` pipeline and direct audio model generation.
The optional `liquid_audio` instrumentor covers Liquid AI's
`LFM2AudioModel` generator methods. Install it with:

```bash
pip install "genai-otel-instrument[liquid-audio]"
```

Sarvam-specific `sarvam.tts.*` and `sarvam.stt.*` attributes remain available
alongside the portable attributes.
