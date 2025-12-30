"""
Example: PII Detection with Redaction Mode

This example demonstrates PII detection with redaction enabled.
Detected PII entities are replaced with asterisks in the telemetry data,
protecting sensitive information while still allowing monitoring and analysis.

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

# Set up OpenTelemetry instrumentation with PII redaction
instrument(
    service_name="pii-redaction-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="redact",  # Detect and redact
    pii_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("PII Redaction Mode")
print("=" * 80)
print("\nWith redaction mode enabled, PII is detected and redacted in telemetry...")
print("Prompt: 'My SSN is 123-45-6789 and credit card is 1234-5678-9012-3456'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "My SSN is 123-45-6789 and credit card is 1234-5678-9012-3456",
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
print("  - evaluation.pii.prompt.entity_types = ['US_SSN', 'CREDIT_CARD']")
print(
    "  - evaluation.pii.prompt.redacted = 'My SSN is ********* and credit card is ****************'"
)
print("\nNote: Original PII is replaced with asterisks in the redacted attribute")
print("=" * 80)
