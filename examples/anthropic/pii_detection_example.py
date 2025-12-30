"""
Example: PII Detection with Anthropic Claude

This example demonstrates PII detection working with Anthropic's Claude API.
Shows that evaluation features work across all supported LLM providers,
not just OpenAI.

Requirements:
    pip install genai-otel-instrument anthropic

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export ANTHROPIC_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with PII detection
instrument(
    service_name="anthropic-pii-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="detect",
    pii_threshold=0.5,
)

# Now import Anthropic after instrumentation is set up
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("PII Detection with Anthropic Claude")
print("=" * 80)
print("\nDemonstrating that evaluation features work with Anthropic...\n")

# Test cases
test_cases = [
    {
        "name": "Email and Phone PII",
        "prompt": "Please send the invoice to john.doe@company.com and call me at 555-123-4567",
        "has_pii": True,
    },
    {
        "name": "SSN in Context",
        "prompt": "My social security number is 123-45-6789",
        "has_pii": True,
    },
    {
        "name": "No PII",
        "prompt": "What is the weather like today?",
        "has_pii": False,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Expected PII: {test['has_pii']})")
    print("-" * 80)
    print(f"Prompt: '{test['prompt']}'")

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": test["prompt"]}],
        )
        print(f"Response: {response.content[0].text[:100]}...")
    except Exception as e:
        print(f"Note: {e}")

    print()

print("=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("Provider: Anthropic Claude")
print("Evaluation Attributes:")
print("  - evaluation.pii.prompt.detected = true/false")
print("  - evaluation.pii.prompt.entity_count = <number>")
print("  - evaluation.pii.prompt.entity_types = [...]")
print("  - evaluation.pii.prompt.max_score = <0.0-1.0>")
print("\nMetrics:")
print("  - genai.evaluation.pii.detections (counter)")
print("  - genai.evaluation.pii.entities (counter)")
print("  - genai.evaluation.pii.score (histogram)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'anthropic'")
print("  - gen_ai.request.model = 'claude-3-haiku-20240307'")
print("=" * 80)
