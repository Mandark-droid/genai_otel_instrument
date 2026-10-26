"""Trace xAI Text to Speech over its REST endpoint.

The current official Python xai-sdk package does not yet expose a released
`Client.voice` helper, so this example calls the documented TTS endpoint directly.
The xAI instrumentor records input character count and estimates cost from the
vendor's published per-character rate without capturing the text in telemetry.
"""

import os
from pathlib import Path

import requests
from xai_sdk import Client  # noqa: F401 - installed by the optional [xai] extra

import genai_otel


def main():
    genai_otel.instrument()
    text = "Hello from the xAI Text to Speech API."
    response = requests.post(
        "https://api.x.ai/v1/tts",
        headers={"Authorization": "Bearer " + os.environ["XAI_API_KEY"]},
        json={
            "text": text,
            "voice_id": "eve",
            "language": "en",
        },
        timeout=120,
    )
    response.raise_for_status()
    output = Path("xai-speech.mp3")
    output.write_bytes(response.content)
    print("Saved %s" % output.resolve())
    print("Estimated TTS cost: $%.8f" % (len(text) * 0.015 / 1000))

    # Optionally transcribe a local recording; STT cost is priced from the
    # response duration and the published $0.10/audio-hour REST rate.
    recording = os.getenv("XAI_STT_FILE")
    if recording:
        audio_path = Path(recording)
        with audio_path.open("rb") as audio_file:
            transcription = requests.post(
                "https://api.x.ai/v1/stt",
                headers={"Authorization": "Bearer " + os.environ["XAI_API_KEY"]},
                data={"model": "grok-voice-transcribe-2.0"},
                files={"file": (audio_path.name, audio_file, "application/octet-stream")},
                timeout=300,
            )
        transcription.raise_for_status()
        result = transcription.json()
        print("Transcript duration: %ss" % result["duration"])


if __name__ == "__main__":
    main()
