"""
Example: PII Detection with Blocking Mode

This example demonstrates PII detection with blocking enabled.
When PII is detected, the span status is set to ERROR and blocked metrics
are recorded. This allows you to filter or alert on requests containing PII.

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

# Set up OpenTelemetry instrumentation with PII blocking
instrument(
    service_name="pii-blocking-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="block",  # Block requests with PII
    pii_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("PII Blocking Mode")
print("=" * 80)
print("\nWith blocking mode enabled, requests with PII are marked as ERROR in telemetry...")
print("Prompt: 'My passport number is AB1234567'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "My passport number is AB1234567",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nNote: Request would be blocked in production: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.pii.prompt.detected = true")
print("  - evaluation.pii.prompt.blocked = true")
print("  - Span status = ERROR")
print("  - evaluation.pii.prompt.entity_types = ['US_PASSPORT']")
print("\nMetrics:")
print("  - genai.evaluation.pii.blocked (counter)")
print("\nNote: The actual request is NOT blocked by the library.")
print("The ERROR status allows you to filter/alert on PII-containing requests.")
print("=" * 80)
