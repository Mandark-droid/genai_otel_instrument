# xAI SDK and API examples

Install the optional integration and set `XAI_API_KEY`:

```bash
pip install "genai-otel-instrument[xai]"
```

Run the chat example with `python chat.py`. Run `python image_video.py` for a
billable image request; the video request is commented out until you choose to
enable it. `python voice_tts.py` demonstrates the current xAI Text to Speech REST
API and writes the returned audio to `xai-speech.mp3`.
Set `XAI_STT_FILE` to a local audio file to also run a billable STT request.

The SDK examples emit request cost from xAI's response. TTS uses the current
published rate of $15 per million input characters because the REST response is
audio bytes and does not include a cost field. TTS text is never written to the
span; only its character count is used for cost tracking.
