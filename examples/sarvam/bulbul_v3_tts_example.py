"""Sarvam AI Bulbul v3 Text-to-Speech Example with genai-otel-instrument

Demonstrates Bulbul v3, Sarvam AI's latest text-to-speech model with
30+ voices across 11 Indian languages. All API calls are automatically
instrumented with OpenTelemetry for full observability.

Bulbul v3 features:
    - 48 speakers (male and female voices)
    - 11 Indian languages
    - Pace control (0.5 to 2.0)
    - Temperature control for expressiveness
    - Default sample rate: 24000 Hz
    - Max input: 2500 characters

Note: Bulbul v3 does NOT support pitch or loudness (v2 only).

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
import time

from genai_otel import instrument

# Single line of instrumentation - that's it!
instrument(
    service_name="sarvam-bulbul-v3-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_cost_tracking=True,
    enable_content_capture=True,
)

from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))


def play_audio(audio_base64: str):
    """Decode base64 audio and play it."""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        played = False
        try:
            from playsound import playsound

            playsound(temp_path)
            played = True
        except ImportError:
            pass

        if not played:
            try:
                import winsound

                winsound.PlaySound(temp_path, winsound.SND_FILENAME)
                played = True
            except (ImportError, RuntimeError):
                pass

        if not played:
            output_path = f"bulbul_v3_audio_{time.time():.0f}.wav"
            with open(output_path, "wb") as out:
                out.write(audio_bytes)
            print(f"  Audio saved to: {output_path}")

        try:
            os.unlink(temp_path)
        except OSError:
            pass
    except Exception as e:
        print(f"  Audio playback error: {e}")


def extract_and_play(audio_result):
    """Extract audio from TTS response and play it."""
    if hasattr(audio_result, "audios") and audio_result.audios:
        play_audio(audio_result.audios[0])
    elif hasattr(audio_result, "audio_base64"):
        play_audio(audio_result.audio_base64)
    elif isinstance(audio_result, str):
        play_audio(audio_result)


# --- Bulbul v3 speaker samples across languages ---

# Bulbul v3 speakers (subset - 48 total available)
V3_SAMPLES = [
    {
        "speaker": "shubh",
        "language": "hi-IN",
        "text": "Namaste, main Shubh hoon. Bulbul version teen mein meri awaaz suniye.",
        "description": "Shubh (Hindi) - v3 default speaker",
    },
    {
        "speaker": "priya",
        "language": "hi-IN",
        "text": "Namaste, main Priya hoon. Yeh Bulbul v3 ka demo hai.",
        "description": "Priya (Hindi) - female voice",
    },
    {
        "speaker": "aditya",
        "language": "ta-IN",
        "text": "Vanakkam, naan Aditya. Indha Bulbul moondram pathippu.",
        "description": "Aditya (Tamil) - male voice",
    },
    {
        "speaker": "ritu",
        "language": "bn-IN",
        "text": "Namaskar, ami Ritu. Ei holo Bulbul version tin.",
        "description": "Ritu (Bengali) - female voice",
    },
    {
        "speaker": "kavya",
        "language": "te-IN",
        "text": "Namaskaram, nenu Kavya. Idi Bulbul v3 demo.",
        "description": "Kavya (Telugu) - female voice",
    },
    {
        "speaker": "dev",
        "language": "gu-IN",
        "text": "Namaskar, hu Dev chhu. Aa Bulbul v3 no demo chhe.",
        "description": "Dev (Gujarati) - male voice",
    },
]


def demo_v3_speakers():
    """Demonstrate multiple Bulbul v3 speakers across languages."""
    print("\n--- Demo 1: Bulbul v3 Speakers Across Languages ---\n")

    for sample in V3_SAMPLES:
        print(f"  Speaker: {sample['description']}")
        print(f"  Text: {sample['text'][:80]}...")
        try:
            result = client.text_to_speech.convert(
                text=sample["text"],
                target_language_code=sample["language"],
                speaker=sample["speaker"],
                model="bulbul:v3",
            )
            print("  Status: Audio generated")
            extract_and_play(result)
        except Exception as e:
            print(f"  Error: {e}")
        print()


def demo_v3_pace_control():
    """Demonstrate Bulbul v3 pace control (0.5 to 2.0)."""
    print("\n--- Demo 2: Bulbul v3 Pace Control ---\n")

    text = "Artificial intelligence is transforming India's future."
    paces = [0.7, 1.0, 1.5]

    for pace in paces:
        label = "slow" if pace < 1.0 else "normal" if pace == 1.0 else "fast"
        print(f"  Pace: {pace} ({label})")
        try:
            result = client.text_to_speech.convert(
                text=text,
                target_language_code="en-IN",
                speaker="shubh",
                model="bulbul:v3",
                pace=pace,
            )
            print("  Status: Audio generated")
            extract_and_play(result)
        except Exception as e:
            print(f"  Error: {e}")
        print()


def demo_v3_temperature():
    """Demonstrate Bulbul v3 temperature control for expressiveness."""
    print("\n--- Demo 3: Bulbul v3 Temperature Control ---\n")

    text = "Yeh bahut acchi baat hai! Main bahut khush hoon."

    for temp in [0.2, 0.7, 1.0]:
        label = "calm" if temp < 0.5 else "balanced" if temp < 0.9 else "expressive"
        print(f"  Temperature: {temp} ({label})")
        try:
            result = client.text_to_speech.convert(
                text=text,
                target_language_code="hi-IN",
                speaker="neha",
                model="bulbul:v3",
                temperature=temp,
            )
            print("  Status: Audio generated")
            extract_and_play(result)
        except Exception as e:
            print(f"  Error: {e}")
        print()


def demo_v3_all_languages():
    """Demonstrate Bulbul v3 across all 11 supported languages."""
    print("\n--- Demo 4: Bulbul v3 Across All 11 Languages ---\n")

    language_samples = [
        ("hi-IN", "Hindi", "Bharat mein AI ka bhavishya bahut ujjwal hai."),
        ("en-IN", "English", "India's AI ecosystem is growing rapidly."),
        ("ta-IN", "Tamil", "Indiyavin AI ethirkalam migavum pirkasamanadhaga irukkiradhu."),
        ("bn-IN", "Bengali", "Bharoter AI bhobishyot onek ujjwal."),
        ("te-IN", "Telugu", "Bharatadeshpu AI bhavishyatthu chala ujwalanga undi."),
        ("gu-IN", "Gujarati", "Bharat nu AI bhavishya khub ujjwal chhe."),
        ("mr-IN", "Marathi", "Bharatache AI bhavishya khup ujjwal aahe."),
        ("kn-IN", "Kannada", "Bharatada AI bhavishya thumba ujwalavaagi ide."),
        ("ml-IN", "Malayalam", "Bharathathinte AI bhavi valare prakasamaanathaanu."),
        ("pa-IN", "Punjabi", "Bharat vich AI da bhavishya bahut chamakdar hai."),
        ("od-IN", "Odia", "Bharatare AI bhabishya bahut ujjwala."),
    ]

    for lang_code, lang_name, text in language_samples:
        print(f"  {lang_name} ({lang_code}): {text[:60]}...")
        try:
            result = client.text_to_speech.convert(
                text=text,
                target_language_code=lang_code,
                speaker="shubh",
                model="bulbul:v3",
            )
            print("  Status: Audio generated")
            extract_and_play(result)
        except Exception as e:
            print(f"  Error: {e}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("Sarvam AI Bulbul v3 Text-to-Speech Demo")
    print("with TraceVerde OpenTelemetry Instrumentation")
    print("=" * 70)
    print()
    print("Bulbul v3 highlights:")
    print("  - 48 speakers (male and female)")
    print("  - 11 Indian languages")
    print("  - Pace control (0.5 - 2.0)")
    print("  - Temperature control for expressiveness")
    print("  - 24000 Hz sample rate")
    print()

    start_time = time.time()

    demo_v3_speakers()
    demo_v3_pace_control()
    demo_v3_temperature()
    demo_v3_all_languages()

    elapsed = time.time() - start_time

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total elapsed time: {elapsed:.2f}s")
    print(f"  TTS calls made:     {len(V3_SAMPLES) + 3 + 3 + 11}")
    print(f"  Model:              bulbul:v3")
    print()
    print("OpenTelemetry spans captured for each TTS call:")
    print("  - gen_ai.system = 'sarvam'")
    print("  - gen_ai.request.model = 'bulbul-v3'")
    print("  - gen_ai.operation.name = 'text_to_speech'")
    print("  - sarvam.target_language = <language code>")
    print("  - sarvam.speaker = <speaker name>")
    print("  - sarvam.input_text_length = <character count>")
    print("  - sarvam.tts.pace = <pace value> (when specified)")
    print("  - sarvam.tts.temperature = <temp value> (when specified)")
    print("  - gen_ai.usage.characters = <char count>")
    print("  - gen_ai.usage.cost.total = <cost in USD>")
    print()
    print("View traces at your OTLP endpoint (Jaeger, Grafana, etc.)")
