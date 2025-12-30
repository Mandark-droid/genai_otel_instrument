"""
Example: Combined PII Compliance Features

This example demonstrates enabling multiple compliance modes simultaneously
(GDPR, HIPAA, PCI-DSS) to ensure comprehensive protection across all
regulatory requirements.

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

# Enable all compliance modes for comprehensive protection
instrument(
    service_name="pii-combined-compliance-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="redact",
    pii_threshold=0.7,
    pii_gdpr_mode=True,  # EU data protection
    pii_hipaa_mode=True,  # Healthcare data
    pii_pci_dss_mode=True,  # Payment card data
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Combined Compliance Features")
print("=" * 80)
print("\nAll compliance modes enabled simultaneously:")
print("  GDPR Mode: European data protection")
print("  HIPAA Mode: Healthcare data protection")
print("  PCI-DSS Mode: Payment card data protection")
print("\nThis ensures comprehensive detection across all entity types:")
print("  - Standard PII: email, phone, SSN, address")
print("  - GDPR: IBAN, NHS numbers, EU identifiers")
print("  - HIPAA: Medical licenses, PHI, dates")
print("  - PCI-DSS: Credit cards, bank accounts")

# Test with mixed PII types
test_cases = [
    ("Email & Phone", "Contact me at john@example.com or 555-1234"),
    ("Healthcare PHI", "Patient DOB: 01/15/1980, MRN: 12345678"),
    ("Payment Data", "Card ending in 1234, account 987654321"),
    ("EU Data", "IBAN: GB29NWBK60161331926819"),
]

for name, prompt in test_cases:
    print(f"\n{'-' * 80}")
    print(f"Test: {name}")
    print(f"Prompt: {prompt}")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"Response: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"Note: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - Comprehensive entity detection across all compliance modes")
print("  - Redacted attributes protecting all PII types")
print("  - Detailed entity counts by type")
print("  - Compliance-specific attributes for audit trails")
print("\nThis approach is recommended for:")
print("  - Multi-jurisdiction applications")
print("  - Healthcare + payment processing systems")
print("  - Enterprise applications with global users")
print("=" * 80)
