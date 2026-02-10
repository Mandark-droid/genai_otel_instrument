"""Sarvam AI Multilingual Pipeline Example with genai-otel-instrument

Demonstrates a complex real-world pipeline using Sarvam AI's sovereign Indian
language AI platform with full OpenTelemetry observability. This example
simulates a multilingual customer support pipeline that:

1. Detects the incoming language of customer queries
2. Translates queries to English for processing
3. Generates responses using sarvam-m chat model
4. Translates responses back to the customer's language
5. Converts the response to speech using Bulbul v3 TTS
6. Processes multiple customers across different Indian languages

This demonstrates how TraceVerde captures distributed traces across the full
pipeline, enabling cost tracking, latency analysis, and quality monitoring
across all Sarvam AI API calls.

Prerequisites:
    pip install sarvamai genai-otel-instrument

Environment variables:
    SARVAM_API_KEY: Your Sarvam AI API subscription key
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4318)

Sarvam AI Drop Week (Feb 2026):
    - Drop 1: Sarvam Dub (multilingual dubbing)
    - Drop 2: Live Budget Speech dubbing
    - Drop 3: Sarvam Audio (ASR benchmarks)
    - Drop 4: Sarvam Vision (OCR/document intelligence)
    - Drop 5: Bulbul V3 (text-to-speech, 35 voices, 11 languages)
    - Drop 6: Government partnerships (Odisha & Tamil Nadu)
    - Drop 7: Sarvam Arya (agent orchestration stack)
"""

import os
import time

from genai_otel import instrument

# Initialize instrumentation BEFORE importing the Sarvam AI client
# Enable cost tracking and content capture for full observability
instrument(
    service_name="sarvam-multilingual-pipeline",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_cost_tracking=True,
    enable_content_capture=True,
)

from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))


# Simulated customer queries in different Indian languages
CUSTOMER_QUERIES = [
    {
        "name": "Priya (Tamil)",
        "query": "Ennudaiya order eppo varum? Order number 12345.",
        "expected_lang": "ta-IN",
        "tts_speaker": "meera",
    },
    {
        "name": "Rahul (Hindi)",
        "query": "Mera refund kab milega? Maine 3 din pehle request kiya tha.",
        "expected_lang": "hi-IN",
        "tts_speaker": "shubh",
    },
    {
        "name": "Ananya (Bengali)",
        "query": "Ami amar account e login korte parchi na. Ki korbo?",
        "expected_lang": "bn-IN",
        "tts_speaker": "meera",
    },
    {
        "name": "Vikram (Gujarati)",
        "query": "Maru product kharab aavyu chhe. Exchange karavi shakay?",
        "expected_lang": "gu-IN",
        "tts_speaker": "shubh",
    },
    {
        "name": "Lakshmi (Telugu)",
        "query": "Naa subscription eppudu expire avutundi? Renew cheyyadam ela?",
        "expected_lang": "te-IN",
        "tts_speaker": "meera",
    },
]


def process_customer_query(customer):
    """Process a single customer query through the full multilingual pipeline.

    Pipeline steps:
    1. Language Detection -> identify the customer's language
    2. Translation to English -> translate for processing
    3. AI Response Generation -> generate support response via sarvam-m
    4. Translation back to customer language -> localize the response
    5. Text-to-Speech -> generate audio response with Bulbul v3
    """
    print(f"\n{'='*70}")
    print(f"Processing: {customer['name']}")
    print(f"Query: {customer['query']}")
    print(f"{'='*70}")

    # Step 1: Detect language
    print("\n  [Step 1] Detecting language...")
    try:
        lang_result = client.text.identify_language(
            input=customer["query"],
        )
        detected_lang = getattr(lang_result, "language_code", customer["expected_lang"])
        print(f"  Detected: {detected_lang} (expected: {customer['expected_lang']})")
    except Exception as e:
        print(f"  Language detection failed: {e}")
        detected_lang = customer["expected_lang"]

    # Step 2: Translate to English
    print("\n  [Step 2] Translating to English...")
    try:
        en_translation = client.text.translate(
            input=customer["query"],
            source_language_code=detected_lang,
            target_language_code="en-IN",
        )
        english_query = getattr(en_translation, "translated_text", customer["query"])
        print(f"  English: {english_query}")
    except Exception as e:
        print(f"  Translation to English failed: {e}")
        english_query = customer["query"]

    # Step 3: Generate AI response
    print("\n  [Step 3] Generating AI response...")
    try:
        ai_response = client.chat.completions(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer support agent. "
                        "Be concise, empathetic, and provide actionable solutions. "
                        "Keep responses under 3 sentences."
                    ),
                },
                {"role": "user", "content": english_query},
            ],
        )
        # Extract response text
        if hasattr(ai_response, "choices") and ai_response.choices:
            response_text = ai_response.choices[0].message.content
        else:
            response_text = str(ai_response)
        print(f"  Response (EN): {response_text[:200]}")
    except Exception as e:
        print(f"  Chat completion failed: {e}")
        response_text = "We apologize for the inconvenience. Please contact support."

    # Step 4: Translate response back to customer's language
    print(f"\n  [Step 4] Translating response to {detected_lang}...")
    try:
        localized_response = client.text.translate(
            input=response_text,
            source_language_code="en-IN",
            target_language_code=detected_lang,
        )
        local_text = getattr(localized_response, "translated_text", response_text)
        print(f"  Localized: {local_text[:200]}")
    except Exception as e:
        print(f"  Translation back failed: {e}")
        local_text = response_text

    # Step 5: Convert to speech
    print(f"\n  [Step 5] Converting to speech (Bulbul v3, speaker: {customer['tts_speaker']})...")
    try:
        audio_result = client.text_to_speech.convert(
            text=local_text,
            target_language_code=detected_lang,
            speaker=customer["tts_speaker"],
        )
        print(f"  TTS: Audio generated successfully")
    except Exception as e:
        print(f"  TTS conversion failed: {e}")

    print(f"\n  Pipeline complete for {customer['name']}")
    return {
        "customer": customer["name"],
        "detected_lang": detected_lang,
        "english_query": english_query,
        "response": response_text,
        "localized_response": local_text,
    }


def run_transliteration_demo():
    """Demonstrate transliteration across multiple scripts."""
    print(f"\n{'='*70}")
    print("Bonus: Transliteration Demo (Script Conversion)")
    print(f"{'='*70}")

    transliteration_tests = [
        ("Namaste duniya", "en-IN", "hi-IN", "Latin -> Devanagari"),
        ("Vanakkam ulagam", "en-IN", "ta-IN", "Latin -> Tamil"),
        ("Namaskar jagat", "en-IN", "bn-IN", "Latin -> Bengali"),
    ]

    for text, src, tgt, desc in transliteration_tests:
        print(f"\n  {desc}: '{text}'")
        try:
            result = client.text.transliterate(
                input=text,
                source_language_code=src,
                target_language_code=tgt,
            )
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("Sarvam AI Multilingual Customer Support Pipeline")
    print("with TraceVerde OpenTelemetry Instrumentation")
    print("=" * 70)
    print(f"\nProcessing {len(CUSTOMER_QUERIES)} customer queries across 5 Indian languages")
    print("Each query goes through a 5-step pipeline (detect -> translate -> chat -> translate -> TTS)")
    print(f"Total expected API calls: {len(CUSTOMER_QUERIES) * 5} + transliteration demos")

    start_time = time.time()
    results = []

    # Process all customer queries
    for customer in CUSTOMER_QUERIES:
        try:
            result = process_customer_query(customer)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR processing {customer['name']}: {e}")

    # Run transliteration demo
    run_transliteration_demo()

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print("Pipeline Summary")
    print(f"{'='*70}")
    print(f"Customers processed: {len(results)}/{len(CUSTOMER_QUERIES)}")
    print(f"Total elapsed time:  {elapsed:.2f}s")
    print(f"Languages covered:   Tamil, Hindi, Bengali, Gujarati, Telugu")
    print(f"APIs exercised:      chat, translate, transliterate, identify_language, text_to_speech")

    print(f"\n{'='*70}")
    print("OpenTelemetry Traces Captured:")
    print(f"{'='*70}")
    print("""
Each pipeline execution generates spans with these attributes:

  sarvam.chat.completions:
    - gen_ai.system = 'sarvam'
    - gen_ai.request.model = 'sarvam-m'
    - gen_ai.usage.prompt_tokens / completion_tokens / total_tokens
    - gen_ai.usage.cost.total (free tier)

  sarvam.text.translate:
    - sarvam.source_language = 'ta-IN' / 'hi-IN' / etc.
    - sarvam.target_language = 'en-IN' / target lang
    - sarvam.translated_text = <translated output>

  sarvam.text.identify_language:
    - sarvam.detected_language = 'ta-IN' / 'hi-IN' / etc.

  sarvam.text.transliterate:
    - sarvam.source_language / sarvam.target_language

  sarvam.text_to_speech.convert:
    - gen_ai.request.model = 'bulbul'
    - sarvam.target_language / sarvam.speaker
    - sarvam.input_text_length

View traces at your OTLP endpoint (Jaeger, Grafana, etc.) to see the
full distributed trace across all pipeline steps.
""")
