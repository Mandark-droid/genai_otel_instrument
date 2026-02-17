"""Sarvam AI Mayura v1 Translation Example with genai-otel-instrument

Demonstrates Sarvam AI's translation capabilities using the Mayura v1
model with full OpenTelemetry instrumentation. Mayura v1 is Sarvam's
latest translation model supporting 22+ Indian languages with advanced
features like speaker gender adaptation and numerals formatting.

This example covers:
    - Translation with mayura:v1 model (default)
    - Translation with sarvam-translate:v1 (legacy model)
    - Translation modes (classic-formal, classic-colloquial)
    - Speaker gender adaptation
    - Numerals format control
    - Language detection + translation pipeline
    - Transliteration across scripts
    - Multi-language round-trip translation

All API calls are automatically instrumented - no manual tracing code needed.

Prerequisites:
    pip install sarvamai genai-otel-instrument

Environment variables:
    SARVAM_API_KEY: Your Sarvam AI API subscription key
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4318)
"""

import os
import time

from genai_otel import instrument

# Single line of instrumentation - that's it!
instrument(
    service_name="sarvam-mayura-translate-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_cost_tracking=True,
    enable_content_capture=True,
)

from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))


def demo_basic_translation():
    """Basic translation using Mayura v1 (default model)."""
    print("\n--- Demo 1: Basic Translation with Mayura v1 ---\n")

    translations = [
        (
            "en-IN",
            "hi-IN",
            "English -> Hindi",
            "Artificial intelligence is transforming India's technology landscape.",
        ),
        ("en-IN", "ta-IN", "English -> Tamil", "The future of AI in India looks very promising."),
        (
            "en-IN",
            "bn-IN",
            "English -> Bengali",
            "Machine learning is helping farmers predict crop yields.",
        ),
        ("hi-IN", "en-IN", "Hindi -> English", "Bharat mein AI ka bhavishya bahut ujjwal hai."),
        (
            "ta-IN",
            "en-IN",
            "Tamil -> English",
            "Indiyavin AI ethirkalam migavum pirkasamanadhaga irukkiradhu.",
        ),
    ]

    for src, tgt, label, text in translations:
        print(f"  {label}:")
        print(f"    Input:  {text[:80]}")
        try:
            result = client.text.translate(
                input=text,
                source_language_code=src,
                target_language_code=tgt,
                # model defaults to mayura:v1
            )
            translated = getattr(result, "translated_text", str(result))
            print(f"    Output: {translated[:80]}")
        except Exception as e:
            print(f"    Error: {e}")
        print()


def demo_translation_modes():
    """Demonstrate translation modes: formal vs colloquial."""
    print("\n--- Demo 2: Translation Modes (Formal vs Colloquial) ---\n")

    text = "Please help me with my order. I need to return this product."

    modes = ["formal", "classic-colloquial"]
    for mode in modes:
        print(f"  Mode: {mode}")
        print(f"    Input (en): {text}")
        try:
            result = client.text.translate(
                input=text,
                source_language_code="en-IN",
                target_language_code="hi-IN",
                mode=mode,
            )
            translated = getattr(result, "translated_text", str(result))
            print(f"    Output (hi): {translated[:100]}")
        except Exception as e:
            print(f"    Error: {e}")
        print()


def demo_speaker_gender():
    """Demonstrate speaker gender adaptation in translation."""
    print("\n--- Demo 3: Speaker Gender Adaptation ---\n")

    text = "I am happy to help you with your request."

    for gender in ["Male", "Female"]:
        print(f"  Speaker gender: {gender}")
        print(f"    Input (en): {text}")
        try:
            result = client.text.translate(
                input=text,
                source_language_code="en-IN",
                target_language_code="hi-IN",
                speaker_gender=gender,
            )
            translated = getattr(result, "translated_text", str(result))
            print(f"    Output (hi): {translated[:100]}")
        except Exception as e:
            print(f"    Error: {e}")
        print()


def demo_numerals_format():
    """Demonstrate numerals format control."""
    print("\n--- Demo 4: Numerals Format Control ---\n")

    text = "The total cost is 12,345 rupees and delivery takes 7 days."

    for fmt in ["international", "native"]:
        print(f"  Numerals format: {fmt}")
        print(f"    Input (en): {text}")
        try:
            result = client.text.translate(
                input=text,
                source_language_code="en-IN",
                target_language_code="hi-IN",
                numerals_format=fmt,
            )
            translated = getattr(result, "translated_text", str(result))
            print(f"    Output (hi): {translated[:100]}")
        except Exception as e:
            print(f"    Error: {e}")
        print()


def demo_model_comparison():
    """Compare mayura:v1 vs sarvam-translate:v1 models."""
    print("\n--- Demo 5: Model Comparison (mayura:v1 vs sarvam-translate:v1) ---\n")

    text = "India's digital transformation is accelerating with AI adoption."

    for model in ["mayura:v1", "sarvam-translate:v1"]:
        print(f"  Model: {model}")
        print(f"    Input (en): {text}")
        try:
            result = client.text.translate(
                input=text,
                source_language_code="en-IN",
                target_language_code="hi-IN",
                model=model,
            )
            translated = getattr(result, "translated_text", str(result))
            print(f"    Output (hi): {translated[:100]}")
        except Exception as e:
            print(f"    Error: {e}")
        print()


def demo_detect_and_translate():
    """Detect language first, then translate."""
    print("\n--- Demo 6: Detect Language + Translate Pipeline ---\n")

    queries = [
        "Mera order kab aayega?",
        "Ennudaiya order eppo varum?",
        "Amar product bhenge gechhe.",
        "My subscription has expired.",
    ]

    for query in queries:
        print(f"  Input: {query}")
        try:
            # Step 1: Detect language
            lang_result = client.text.identify_language(input=query)
            detected = getattr(lang_result, "language_code", "unknown")
            print(f"    Detected language: {detected}")

            # Step 2: Translate to English (if not already English)
            if detected != "en-IN":
                result = client.text.translate(
                    input=query,
                    source_language_code=detected,
                    target_language_code="en-IN",
                )
                translated = getattr(result, "translated_text", str(result))
                print(f"    English: {translated[:100]}")
            else:
                print(f"    Already in English")
        except Exception as e:
            print(f"    Error: {e}")
        print()


def demo_transliteration():
    """Demonstrate transliteration (script conversion)."""
    print("\n--- Demo 7: Transliteration (Script Conversion) ---\n")

    samples = [
        ("Namaste duniya", "en-IN", "hi-IN", "Latin -> Devanagari"),
        ("Vanakkam ulagam", "en-IN", "ta-IN", "Latin -> Tamil"),
        ("Namaskar jagat", "en-IN", "bn-IN", "Latin -> Bengali"),
        ("Namaskaram prapancham", "en-IN", "te-IN", "Latin -> Telugu"),
    ]

    for text, src, tgt, label in samples:
        print(f"  {label}: '{text}'")
        try:
            result = client.text.transliterate(
                input=text,
                source_language_code=src,
                target_language_code=tgt,
            )
            transliterated = getattr(result, "transliterated_text", str(result))
            print(f"    Result: {transliterated}")
        except Exception as e:
            print(f"    Error: {e}")
        print()


def demo_round_trip_translation():
    """Translate English -> Hindi -> English to test quality."""
    print("\n--- Demo 8: Round-Trip Translation (en -> hi -> en) ---\n")

    original = "The monsoon season brings life to the parched fields of India."
    print(f"  Original (en): {original}")

    try:
        # English -> Hindi
        hi_result = client.text.translate(
            input=original,
            source_language_code="en-IN",
            target_language_code="hi-IN",
        )
        hindi = getattr(hi_result, "translated_text", str(hi_result))
        print(f"  Hindi:         {hindi[:100]}")

        # Hindi -> English
        en_result = client.text.translate(
            input=hindi,
            source_language_code="hi-IN",
            target_language_code="en-IN",
        )
        back_to_en = getattr(en_result, "translated_text", str(en_result))
        print(f"  Back to en:    {back_to_en[:100]}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("Sarvam AI Mayura v1 Translation Demo")
    print("with TraceVerde OpenTelemetry Instrumentation")
    print("=" * 70)
    print()
    print("Translation models available:")
    print("  - mayura:v1 (latest, default)")
    print("  - sarvam-translate:v1 (legacy)")
    print()
    print("Supported languages: 22+ Indian languages")
    print("Features: modes, speaker gender, numerals format")
    print()

    start_time = time.time()

    demo_basic_translation()
    demo_translation_modes()
    demo_speaker_gender()
    demo_numerals_format()
    demo_model_comparison()
    demo_detect_and_translate()
    demo_transliteration()
    demo_round_trip_translation()

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total elapsed time: {elapsed:.2f}s")
    print(f"  APIs exercised: translate, identify_language, transliterate")
    print()
    print("OpenTelemetry spans captured for each API call:")
    print()
    print("  sarvam.text.translate:")
    print("    - gen_ai.request.model = 'mayura:v1' or 'sarvam-translate:v1'")
    print("    - sarvam.source_language / sarvam.target_language")
    print("    - sarvam.translate.mode (when specified)")
    print("    - sarvam.translate.speaker_gender (when specified)")
    print("    - sarvam.translate.numerals_format (when specified)")
    print("    - sarvam.translated_text (response)")
    print("    - gen_ai.usage.characters / gen_ai.usage.cost.total")
    print()
    print("  sarvam.text.identify_language:")
    print("    - gen_ai.request.model = 'sarvam-detect-language'")
    print("    - sarvam.detected_language")
    print("    - gen_ai.usage.characters / gen_ai.usage.cost.total")
    print()
    print("  sarvam.text.transliterate:")
    print("    - gen_ai.request.model = 'sarvam-transliterate'")
    print("    - sarvam.source_language / sarvam.target_language")
    print("    - gen_ai.usage.characters / gen_ai.usage.cost.total")
    print()
    print("View traces at your OTLP endpoint (Jaeger, Grafana, etc.)")
