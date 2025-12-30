"""
Example: Category-Specific Toxicity Detection

This example demonstrates the different toxicity categories that are detected
and how they appear in telemetry. Each category represents a different type
of toxic content.

Categories:
- toxicity: General toxic language
- severe_toxicity: Extremely harmful content
- identity_attack: Attacks on protected characteristics
- insult: Insulting or demeaning language
- profanity: Swearing and obscene language
- threat: Threatening or violent language

Requirements:
    pip install genai-otel-instrument openai
    pip install detoxify

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation
instrument(
    service_name="toxicity-categories-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Category-Specific Toxicity Detection")
print("=" * 80)
print("\nToxicity detection supports multiple categories:")
print("  - toxicity: General toxic language")
print("  - severe_toxicity: Extremely toxic content")
print("  - identity_attack: Attacks on protected characteristics")
print("  - insult: Insulting or demeaning language")
print("  - profanity: Swearing and obscene language")
print("  - threat: Threatening language")

# Example texts and their likely categories
examples = [
    ("You're an idiot", "insult"),
    ("I hate all people from that country", "identity_attack"),
    ("F*** you!", "profanity"),
    ("I will hurt you", "threat"),
    ("You are worthless trash", "severe_toxicity"),
]

print("\n" + "=" * 80)
print("Testing different toxicity categories:")
print("=" * 80)

for text, expected_category in examples:
    print(f"\nText: '{text}'")
    print(f"Expected category: {expected_category}")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
        )
        print(f"Response: {response.choices[0].message.content[:80]}...")
    except Exception as e:
        print(f"Note: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.toxicity.prompt.categories = [list of detected categories]")
print("  - evaluation.toxicity.prompt.<category>_score for each category")
print("  - Example: evaluation.toxicity.prompt.insult_score = 0.95")
print("\nMetrics:")
print("  - genai.evaluation.toxicity.categories (counter)")
print("    Tagged by category: insult, threat, profanity, etc.")
print("\nEach prompt may trigger multiple categories simultaneously")
print("=" * 80)
