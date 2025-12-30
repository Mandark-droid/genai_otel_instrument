"""
Example demonstrating PII Detection with OpenTelemetry instrumentation.

This example shows:
1. Basic PII detection in detect mode
2. PII redaction in redact mode
3. PII blocking in block mode
4. GDPR compliance mode
5. HIPAA compliance mode
6. PCI-DSS compliance mode
7. Custom entity types and thresholds

PII detection helps ensure compliance with data protection regulations like
GDPR, HIPAA, and PCI-DSS by detecting and handling personally identifiable
information in LLM prompts and responses.

Requirements:
    pip install genai-otel-instrument openai
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_lg

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
    export GENAI_ENABLE_PII_DETECTION=true
"""

import os

# Example 1: Basic PII Detection (Detect Mode)
print("=" * 80)
print("Example 1: Basic PII Detection - Detect Mode")
print("=" * 80)

# Set up OpenTelemetry instrumentation with PII detection
from genai_otel import instrument

instrument(
    service_name="pii-detection-example",
    endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="detect",  # Only detect, don't modify
    pii_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

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

print("\n✓ Check your telemetry backend for:")
print("  - evaluation.pii.prompt.detected = true")
print("  - evaluation.pii.prompt.entity_count = 2")
print("  - evaluation.pii.prompt.entity_types = ['EMAIL_ADDRESS', 'PHONE_NUMBER']")
print("  - evaluation.pii.prompt.score = <confidence>")
print("  - Metrics: genai.evaluation.pii.detections")
print("  - Metrics: genai.evaluation.pii.entities (by type)")


# Example 2: PII Redaction Mode
print("\n" + "=" * 80)
print("Example 2: PII Redaction Mode")
print("=" * 80)

from genai_otel import instrument

# Re-instrument with redaction mode
instrument(
    service_name="pii-redaction-example",
    # endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="redact",  # Detect and redact
    pii_threshold=0.7,
)

print("\nWith redaction mode enabled...")
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

print("\n✓ Check your telemetry backend for:")
print("  - evaluation.pii.prompt.detected = true")
print(
    "  - evaluation.pii.prompt.redacted = 'My SSN is ********* and credit card is ****************'"
)
print("  - PII entities are replaced with asterisks in the redacted attribute")


# Example 3: PII Blocking Mode
print("\n" + "=" * 80)
print("Example 3: PII Blocking Mode")
print("=" * 80)

from genai_otel import instrument

# Re-instrument with blocking mode
instrument(
    service_name="pii-blocking-example",
    endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="block",  # Block requests with PII
    pii_threshold=0.7,
)

print("\nWith blocking mode enabled...")
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

print("\n✓ Check your telemetry backend for:")
print("  - evaluation.pii.prompt.detected = true")
print("  - evaluation.pii.prompt.blocked = true")
print("  - Span status = ERROR")
print("  - Metrics: genai.evaluation.pii.blocked")


# Example 4: GDPR Compliance Mode
print("\n" + "=" * 80)
print("Example 4: GDPR Compliance Mode")
print("=" * 80)

from genai_otel import instrument

# Enable GDPR mode for EU data protection
instrument(
    service_name="pii-gdpr-example",
    endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="detect",
    pii_gdpr_mode=True,  # Enable EU-specific entity types
    pii_threshold=0.7,
)

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

print("\n✓ GDPR-specific entities are now detected in addition to standard PII")


# Example 5: HIPAA Compliance Mode
print("\n" + "=" * 80)
print("Example 5: HIPAA Compliance Mode")
print("=" * 80)

from genai_otel import instrument

# Enable HIPAA mode for healthcare data
instrument(
    service_name="pii-hipaa-example",
    endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="redact",
    pii_hipaa_mode=True,  # Enable healthcare-specific entity types
    pii_threshold=0.7,
)

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

print("\n✓ HIPAA-specific entities are detected and redacted")


# Example 6: PCI-DSS Compliance Mode
print("\n" + "=" * 80)
print("Example 6: PCI-DSS Compliance Mode")
print("=" * 80)

from genai_otel import instrument

# Enable PCI-DSS mode for payment card data
instrument(
    service_name="pii-pci-dss-example",
    endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="block",
    pii_pci_dss_mode=True,  # Ensure credit card detection
    pii_threshold=0.7,
)

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

print("\n✓ PCI-DSS compliance ensures no payment data leaks to LLMs")


# Example 7: Environment Variable Configuration
print("\n" + "=" * 80)
print("Example 7: Environment Variable Configuration")
print("=" * 80)

print("\nYou can also configure PII detection via environment variables:")
print(
    """
# Enable PII detection
export GENAI_ENABLE_PII_DETECTION=true

# Set detection mode (detect, redact, or block)
export GENAI_PII_MODE=redact

# Set confidence threshold (0.0-1.0)
export GENAI_PII_THRESHOLD=0.8

# Enable compliance modes
export GENAI_PII_GDPR_MODE=true
export GENAI_PII_HIPAA_MODE=true
export GENAI_PII_PCI_DSS_MODE=true
"""
)

print("\nThen simply instrument without parameters:")
print(
    """
from genai_otel import instrument
instrument(service_name="my-app")
"""
)


# Example 8: Response PII Detection
print("\n" + "=" * 80)
print("Example 8: Response PII Detection")
print("=" * 80)

from genai_otel import instrument

# Re-instrument for response detection
instrument(
    service_name="pii-response-example",
    endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="detect",
    pii_threshold=0.7,
)

print("\nPII detection also checks LLM responses...")
print("If the LLM accidentally includes PII in its response,")
print("it will be detected and flagged in telemetry.")

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
    print("\n✓ If the response contains an email, it will be detected")
    print("  - evaluation.pii.response.detected = true")
    print("  - evaluation.pii.response.entity_types = ['EMAIL_ADDRESS']")
except Exception as e:
    print(f"\nNote: {e}")


# Example 9: Custom Threshold
print("\n" + "=" * 80)
print("Example 9: Custom Detection Threshold")
print("=" * 80)

print("\nYou can adjust the confidence threshold:")
print("  - Lower threshold (e.g., 0.5): More sensitive, may have false positives")
print("  - Higher threshold (e.g., 0.9): Less sensitive, higher confidence required")

from genai_otel import instrument

# High confidence threshold
instrument(
    service_name="pii-threshold-example",
    endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="detect",
    pii_threshold=0.9,  # Only detect with 90%+ confidence
)

print("\nWith threshold=0.9, only high-confidence PII is detected")


# Summary
print("\n" + "=" * 80)
print("Summary: PII Detection Features")
print("=" * 80)

print(
    """
✓ Detection Modes:
  - detect: Only detect and report PII
  - redact: Detect and redact PII in telemetry
  - block: Block requests/responses containing PII

✓ Compliance Modes:
  - GDPR: EU data protection (IBAN, NHS, etc.)
  - HIPAA: Healthcare data (medical licenses, PHI)
  - PCI-DSS: Payment card data (credit cards, bank accounts)

✓ Detected Entity Types:
  - Email addresses
  - Phone numbers
  - Social Security Numbers (SSN)
  - Credit card numbers
  - IP addresses
  - Passport numbers
  - Bank account numbers
  - Medical information
  - And many more...

✓ Telemetry Attributes:
  - evaluation.pii.prompt.detected
  - evaluation.pii.response.detected
  - evaluation.pii.*.entity_count
  - evaluation.pii.*.entity_types
  - evaluation.pii.*.score
  - evaluation.pii.*.redacted (in redact mode)
  - evaluation.pii.*.blocked (in block mode)

✓ Metrics:
  - genai.evaluation.pii.detections
  - genai.evaluation.pii.entities (by type)
  - genai.evaluation.pii.blocked

✓ Technology:
  - Microsoft Presidio for ML-based detection
  - Regex fallback patterns when Presidio unavailable
  - Zero-code integration via instrumentation
  - Async processing for minimal latency impact

For more information, see:
https://docs.microsoft.com/presidio
"""
)

print("\n" + "=" * 80)
print("All examples completed!")
print("=" * 80)
