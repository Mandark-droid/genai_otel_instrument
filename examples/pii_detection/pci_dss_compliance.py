"""
Example: PII Detection with PCI-DSS Compliance Mode

This example demonstrates PCI-DSS compliance mode for payment card data protection.
PCI-DSS mode ensures rigorous detection of:
- Credit card numbers (all major card types)
- US bank account numbers
- Other payment-related sensitive data

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

# Enable PCI-DSS mode for payment card data
instrument(
    service_name="pii-pci-dss-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="block",
    pii_pci_dss_mode=True,  # Ensure credit card detection
    pii_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("PCI-DSS Compliance Mode")
print("=" * 80)
print("\nPCI-DSS mode ensures detection of:")
print("  - Credit card numbers (all major types)")
print("  - US bank account numbers")
print("  - Plus all standard PII types")
print("\nPrompt with card data: 'Card: 4532-1234-5678-9010, CVV: 123'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Card: 4532-1234-5678-9010, CVV: 123",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nNote: Payment card data would be blocked: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.pii.prompt.detected = true")
print("  - evaluation.pii.prompt.blocked = true")
print("  - evaluation.pii.prompt.entity_types = ['CREDIT_CARD']")
print("  - Span status = ERROR")
print("\nPCI-DSS compliance ensures no payment data leaks to LLMs")
print("=" * 80)
