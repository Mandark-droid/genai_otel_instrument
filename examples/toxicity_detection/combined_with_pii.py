"""
Example: Combining Toxicity and PII Detection

This example demonstrates enabling multiple safety features simultaneously:
- PII Detection: Protect sensitive data (emails, SSN, credit cards)
- Toxicity Detection: Prevent harmful content (insults, threats)

This comprehensive approach ensures both data privacy and content safety.

Requirements:
    pip install genai-otel-instrument openai
    pip install detoxify
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_lg

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Enable both PII and toxicity detection
instrument(
    service_name="combined-safety-features",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="redact",
    pii_threshold=0.7,
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Combined Safety Features: PII + Toxicity Detection")
print("=" * 80)
print("\nMultiple safety features enabled simultaneously:")
print("  PII Detection (redact mode): Protect sensitive data")
print("  Toxicity Detection: Prevent harmful content")

# Test cases combining PII and toxicity
test_cases = [
    {
        "name": "PII + Toxicity",
        "prompt": "You idiot, my email is john@example.com",
        "expected": "Both PII (email) and toxicity (insult) detected",
    },
    {
        "name": "PII only",
        "prompt": "My phone number is 555-1234, please call me",
        "expected": "PII detected, no toxicity",
    },
    {
        "name": "Toxicity only",
        "prompt": "This is absolute garbage and you're terrible",
        "expected": "Toxicity detected, no PII",
    },
    {
        "name": "Clean content",
        "prompt": "Can you help me with my project?",
        "expected": "Neither PII nor toxicity",
    },
]

for test in test_cases:
    print(f"\n{'-' * 80}")
    print(f"Test: {test['name']}")
    print(f"Prompt: {test['prompt']}")
    print(f"Expected: {test['expected']}")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": test["prompt"]}],
        )
        print(f"Response: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"Note: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("PII Attributes:")
print("  - evaluation.pii.prompt.detected")
print("  - evaluation.pii.prompt.entity_types")
print("  - evaluation.pii.prompt.redacted (redacted text)")
print("\nToxicity Attributes:")
print("  - evaluation.toxicity.prompt.detected")
print("  - evaluation.toxicity.prompt.categories")
print("  - evaluation.toxicity.prompt.max_score")
print("\nBoth sets of attributes coexist on the same span!")
print("=" * 80)

print("\n" + "=" * 80)
print("Comprehensive Safety Stack:")
print("=" * 80)
print("  Available Features:")
print("  - PII Detection (GDPR, HIPAA, PCI-DSS)")
print("  - Toxicity Detection")
print("  - Bias Detection (coming soon)")
print("  - Prompt Injection Detection (coming soon)")
print("  - Hallucination Detection (coming soon)")
print("\nEnable multiple features for comprehensive protection:")
print(
    """
instrument(
    service_name="my-app",
    enable_pii_detection=True,
    pii_mode="redact",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    enable_bias_detection=True,  # Coming soon
)
"""
)
print("=" * 80)
