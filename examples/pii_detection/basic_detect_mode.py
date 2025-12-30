"""
Example: Basic PII Detection in Detect Mode

This example demonstrates basic PII detection without modification.
The detector identifies PII entities (email, phone, SSN, etc.) and records
them in telemetry attributes, but does not modify the prompt or block requests.

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

# Set up OpenTelemetry instrumentation with PII detection in detect mode
instrument(
    service_name="pii-detection-basic-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="detect",  # Only detect, don't modify
    pii_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Basic PII Detection - Detect Mode")
print("=" * 80)
print("\nSending prompt with PII (email, phone)...")
print("Prompt: 'My email is john.doe@example.com and phone is 555-123-4567'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "My email is john.doe@example.com and phone is 555-123-4567. Can you help me?",
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
print("  - evaluation.pii.prompt.entity_count = 2")
print("  - evaluation.pii.prompt.entity_types = ['EMAIL_ADDRESS', 'PHONE_NUMBER']")
print("  - evaluation.pii.prompt.EMAIL_ADDRESS_count = 1")
print("  - evaluation.pii.prompt.PHONE_NUMBER_count = 1")
print("  - evaluation.pii.prompt.score = <confidence>")
print("\nMetrics:")
print("  - genai.evaluation.pii.detections (counter)")
print("  - genai.evaluation.pii.entities (counter, by entity_type)")
print("=" * 80)
