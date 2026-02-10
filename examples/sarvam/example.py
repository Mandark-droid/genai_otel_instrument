"""Sarvam AI Example with genai-otel-instrument

Demonstrates OpenTelemetry instrumentation for Sarvam AI's multi-modal
Indian language AI platform, including:
- Chat completions (sarvam-m model)
- Text translation (22+ Indian languages)
- Text-to-speech (Bulbul v3)
- Speech-to-text (Saarika model)
- Language detection

Prerequisites:
    pip install sarvamai genai-otel-instrument

Environment variables:
    SARVAM_API_KEY: Your Sarvam AI API subscription key
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4318)
"""

import os

import genai_otel

# Initialize instrumentation BEFORE importing the Sarvam AI client
genai_otel.instrument(
    service_name="sarvam-ai-example",
    enable_cost_tracking=True,
)

from sarvamai import SarvamAI

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

# --- 5. Text-to-Speech (Bulbul v3) ---
print("\n--- Text-to-Speech (Bulbul v3) ---")
audio = client.text_to_speech.convert(
    text="Bharat mein AI ka bhavishya bahut ujjwal hai.",
    target_language_code="hi-IN",
    speaker="shubh",
)
print(f"TTS result: {audio}")

print("\nAll Sarvam AI API calls instrumented with OpenTelemetry!")
