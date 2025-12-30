"""
Example: Custom PII Detection Threshold

This example demonstrates how to adjust the confidence threshold for PII detection.
- Lower threshold (e.g., 0.5): More sensitive, may have false positives
- Higher threshold (e.g., 0.9): Less sensitive, higher confidence required

Requirements:
    pip install genai-otel-instrument openai
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_lg

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# High confidence threshold - only detect high-confidence PII
instrument(
    service_name="pii-threshold-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="detect",
    pii_threshold=0.9,  # Only detect with 90%+ confidence
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Custom Detection Threshold")
print("=" * 80)
print("\nYou can adjust the confidence threshold:")
print("  - Lower threshold (e.g., 0.5): More sensitive, may have false positives")
print("  - Higher threshold (e.g., 0.9): Less sensitive, higher confidence required")
print("\nCurrent threshold: 0.9 (high confidence required)")
print("\nSending prompt with clear PII (email)...")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Contact me at john.smith@example.com for more information",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nNote: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.pii.prompt.detected = true")
print("  - evaluation.pii.prompt.score >= 0.9")
print("\nWith threshold=0.9, only high-confidence PII is detected")
print("Ambiguous patterns below 0.9 confidence will be ignored")
print("=" * 80)
print("\nRecommendations:")
print("  - Use 0.7-0.8 for balanced detection (default: 0.7)")
print("  - Use 0.5-0.6 for sensitive applications (more false positives)")
print("  - Use 0.9+ for low false-positive tolerance (may miss some PII)")
print("=" * 80)
