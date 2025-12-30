"""
Example: PII Detection in LLM Responses

This example demonstrates PII detection applied to LLM responses.
If the LLM accidentally includes PII in its response (e.g., generating
sample email addresses, phone numbers), it will be detected and flagged
in telemetry.

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

# Set up OpenTelemetry instrumentation for response detection
instrument(
    service_name="pii-response-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="detect",
    pii_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Response PII Detection")
print("=" * 80)
print("\nPII detection also checks LLM responses...")
print("If the LLM accidentally includes PII in its response,")
print("it will be detected and flagged in telemetry.")
print("\nAsking LLM to generate a sample email address...")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Generate a sample email address for testing",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
    print("\nIf the response contains an email, it will be detected!")
except Exception as e:
    print(f"\nNote: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.pii.response.detected = true (if email in response)")
print("  - evaluation.pii.response.entity_types = ['EMAIL_ADDRESS']")
print("  - evaluation.pii.response.entity_count = <count>")
print("  - evaluation.pii.response.score = <confidence>")
print("\nNote: Both prompts AND responses are checked for PII")
print("=" * 80)
