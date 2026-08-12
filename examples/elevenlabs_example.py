"""ElevenLabs Example with genai-otel-instrument

Demonstrates auto-instrumentation of ElevenLabs text-to-speech and Scribe
speech-to-text. Call genai_otel.instrument() before constructing the client.

ElevenLabs is billed by media rather than tokens - TTS per character of input
text, Scribe per second of audio - so the spans carry gen_ai.usage.characters
and gen_ai.usage.audio_duration_seconds instead of token counts.

Audio bytes are never attached to spans. For voice workloads the audio is
frequently personal data, so only sizes and durations are recorded.

Requires ELEVENLABS_API_KEY. Text-to-speech additionally needs a paid plan -
free accounts cannot use library voices over the API - so that half is skipped
with a message rather than failing the run.
"""

import io
import math
import os
import struct
import wave

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import genai_otel

# Instrument BEFORE creating the client so the wrapper is installed on __init__.
genai_otel.instrument()

from elevenlabs import ElevenLabs  # noqa: E402

if not os.getenv("ELEVENLABS_API_KEY"):
    raise SystemExit("ELEVENLABS_API_KEY is not set")

client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])


def sample_wav(seconds: int = 3) -> io.BytesIO:
    """Generate a short mono WAV so the transcription demo needs no asset file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * t / 16000)))
                for t in range(16000 * seconds)
            )
        )
    buf.seek(0)
    return buf


# --- Text to speech -------------------------------------------------------
# Listing voices needs the voices_read permission, which a least-privilege
# service key scoped only to synthesis will not have.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
if not voice_id:
    try:
        found = client.voices.search(page_size=1)
        voice_id = found.voices[0].voice_id
        print(f"Using voice from account: {found.voices[0].name} ({voice_id})")
    except Exception as e:  # noqa: BLE001
        voice_id = DEFAULT_VOICE_ID
        print(f"Could not list voices ({type(e).__name__}); using stock voice {voice_id}")
else:
    print(f"Using voice from ELEVENLABS_VOICE_ID: {voice_id}")

text = "Open semantic conventions keep telemetry portable across backends."
audio = None
try:
    # convert() returns an iterator of audio bytes; draining it is what completes
    # the span and records time-to-first-byte.
    audio = b"".join(
        client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
    )
    print(f"TTS: {len(text)} characters -> {len(audio)} bytes of audio")
except Exception as e:  # noqa: BLE001
    print(f"TTS skipped ({type(e).__name__}) - a paid plan is required for library voices")

# --- Speech to text (Scribe) ---------------------------------------------
source = io.BytesIO(audio) if audio else sample_wav()
transcript = client.speech_to_text.convert(model_id="scribe_v1", file=source)
print(f"STT: transcript = {transcript.text!r}")
print(f"STT: audio duration = {getattr(transcript, 'audio_duration_secs', 'n/a')}s")

print("Traces and metrics have been sent to your OTLP endpoint.")
