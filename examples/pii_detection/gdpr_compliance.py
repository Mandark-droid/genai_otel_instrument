"""
Example: PII Detection with GDPR Compliance Mode

This example demonstrates GDPR compliance mode for EU data protection.
In addition to standard PII entities, GDPR mode enables detection of:
- IBAN codes (European bank accounts)
- UK NHS numbers
- Named Recognized Persons (NRP)
- Other EU-specific data types

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

# Enable GDPR mode for EU data protection
instrument(
    service_name="pii-gdpr-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="detect",
    pii_gdpr_mode=True,  # Enable EU-specific entity types
    pii_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("GDPR Compliance Mode")
print("=" * 80)
print("\nGDPR mode enables detection of:")
print("  - IBAN codes (European bank accounts)")
print("  - UK NHS numbers")
print("  - Named Recognized Persons (NRP)")
print("  - Plus all standard PII types")
print("\nPrompt with IBAN: 'My IBAN is GB29NWBK60161331926819'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "My IBAN is GB29NWBK60161331926819. Can you process this?",
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
print("  - evaluation.pii.prompt.entity_types = ['IBAN_CODE']")
print("  - evaluation.pii.prompt.IBAN_CODE_count = 1")
print("\nGDPR-specific entities are now detected in addition to standard PII")
print("=" * 80)
