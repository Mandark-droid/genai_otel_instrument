"""
Example: PII Detection with HIPAA Compliance Mode

This example demonstrates HIPAA compliance mode for healthcare data protection.
In addition to standard PII entities, HIPAA mode enables detection of:
- Medical license numbers
- US Passport numbers
- Date/time (Protected Health Information)
- Other healthcare-specific data types

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

# Enable HIPAA mode for healthcare data
instrument(
    service_name="pii-hipaa-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="redact",
    pii_hipaa_mode=True,  # Enable healthcare-specific entity types
    pii_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("HIPAA Compliance Mode")
print("=" * 80)
print("\nHIPAA mode enables detection of:")
print("  - Medical license numbers")
print("  - US Passport numbers")
print("  - Date/time (PHI)")
print("  - Plus all standard PII types")
print("\nPrompt with medical data: 'Patient DOB: 01/15/1980, Med License: MD123456'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Patient DOB: 01/15/1980, Med License: MD123456",
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
print("  - evaluation.pii.prompt.entity_types = ['DATE_TIME', 'MEDICAL_LICENSE']")
print("  - evaluation.pii.prompt.redacted = '<redacted text with PHI replaced>'")
print("\nHIPAA-specific entities are detected and redacted")
print("=" * 80)
