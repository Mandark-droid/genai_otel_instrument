"""Sarvam AI Example with genai-otel-instrument

Demonstrates OpenTelemetry instrumentation for Sarvam AI's multi-modal
Indian language AI platform, including:
- Chat completions (sarvam-m model)
- Text translation (22+ Indian languages)
- Text-to-speech (Bulbul v3) with audio playback
- Speech-to-text (Saarika model)
- Language detection

Prerequisites:
    pip install sarvamai genai-otel-instrument
    pip install playsound  # Optional: for audio playback

Environment variables:
    SARVAM_API_KEY: Your Sarvam AI API subscription key
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4318)
"""

import base64
import os
import tempfile

import genai_otel

# Initialize instrumentation BEFORE importing the Sarvam AI client
genai_otel.instrument(
    service_name="sarvam-ai-example",
    enable_cost_tracking=True,
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
)

from sarvamai import SarvamAI


def play_audio(audio_base64: str):
    """Decode base64 audio and play it.

    Args:
        audio_base64: Base64-encoded WAV audio string
    """
    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        played = False

        # Try different audio playback methods
        try:
            # Method 1: playsound (cross-platform, pip install playsound)
            from playsound import playsound

            playsound(temp_path)
            played = True
        except ImportError:
            pass

        if not played:
            try:
                # Method 2: winsound (Windows only)
                import winsound

                winsound.PlaySound(temp_path, winsound.SND_FILENAME)
                played = True
            except (ImportError, RuntimeError):
                pass

        if not played:
            # Method 3: Just save the file
            output_path = "sarvam_tts_output.wav"
            with open(output_path, "wb") as out:
                out.write(audio_bytes)
            print(f"Audio saved to: {output_path}")
            print("Install 'playsound' to play audio: pip install playsound")

        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    except Exception as e:
        print(f"Audio playback error: {e}")


client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))

# --- 1. Chat Completion ---
print("--- Chat Completion (sarvam-m) ---")
response = client.chat.completions(
    messages=[{"role": "user", "content": "What is India's AI mission?"}],
)
print(f"Chat: {response}")

# --- 2. Translation (English -> Hindi) ---
print("\n--- Translation (en -> hi) ---")
translated = client.text.translate(
    input="Artificial intelligence is transforming India's technology landscape.",
    source_language_code="en-IN",
    target_language_code="hi-IN",
)
print(f"Translated: {translated}")

# --- 3. Language Detection ---
print("\n--- Language Detection ---")
detected = client.text.identify_language(
    input="Namaste, aap kaise hain?",
)
print(f"Detected language: {detected}")

# --- 4. Transliteration ---
print("\n--- Transliteration ---")
transliterated = client.text.transliterate(
    input="Namaste",
    source_language_code="en-IN",
    target_language_code="hi-IN",
)
print(f"Transliterated: {transliterated}")

# --- 5. Text-to-Speech (Bulbul v3) with Audio Playback ---
print("\n--- Text-to-Speech (Bulbul v3) ---")
audio_response = client.text_to_speech.convert(
    text="Bharat mein AI ka bhavishya bahut ujjwal hai.",
    target_language_code="hi-IN",
    speaker="abhilash",
)
print(f"TTS response received (type: {type(audio_response).__name__})")

# Play the audio if we got a base64 response
if hasattr(audio_response, "audios") and audio_response.audios:
    print("Playing audio...")
    play_audio(audio_response.audios[0])
elif hasattr(audio_response, "audio_base64"):
    print("Playing audio...")
    play_audio(audio_response.audio_base64)
elif isinstance(audio_response, str):
    # Direct base64 string
    print("Playing audio...")
    play_audio(audio_response)
else:
    print(f"TTS result: {audio_response}")

print("\nAll Sarvam AI API calls instrumented with OpenTelemetry!")
