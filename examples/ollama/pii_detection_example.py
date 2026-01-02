"""
Example: PII Detection with Ollama

This example demonstrates PII detection working with locally-hosted Ollama models.
Shows that evaluation features work with self-hosted LLMs, not just cloud APIs.

Requirements:
    pip install genai-otel-instrument ollama
    # Install and run Ollama: https://ollama.ai
    # Pull a model: ollama pull llama2

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    # Ollama runs locally by default on http://localhost:11434
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with PII detection
instrument(
    service_name="ollama-pii-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="detect",
    enable_content_capture=True,  # Required for evaluation features
)

# Now import Ollama after instrumentation is set up
try:
    import ollama
except ImportError:
    print("Error: ollama package not installed. Run: pip install ollama")
    exit(1)

print("=" * 80)
print("PII Detection with Ollama (Local LLM)")
print("=" * 80)
print("\nDemonstrating PII detection with self-hosted models...\n")

# Test cases
test_cases = [
    {
        "name": "Email and Phone",
        "prompt": "Contact John Doe at john.doe@example.com or call 555-123-4567",
        "has_pii": True,
    },
    {
        "name": "SSN and Name",
        "prompt": "Sarah Johnson's SSN is 123-45-6789",
        "has_pii": True,
    },
    {
        "name": "Credit Card",
        "prompt": "Use card number 4532-1234-5678-9010 for payment",
        "has_pii": True,
    },
    {
        "name": "No PII",
        "prompt": "What are the benefits of regular exercise?",
        "has_pii": False,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Expected PII: {test['has_pii']})")
    print("-" * 80)
    print(f"Prompt: '{test['prompt']}'")

    try:
        response = ollama.chat(
            model="smollm2:360m",
            messages=[{"role": "user", "content": test["prompt"]}],
        )
        print(f"Response: {response['message']['content'][:100]}...")
    except Exception as e:
        print(f"Note: Make sure Ollama is running and llama2 model is pulled")
        print(f"Error: {e}")

    print()

print("=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("Provider: Ollama (Self-Hosted)")
print("Evaluation Attributes:")
print("  - evaluation.pii.prompt.detected = true/false")
print("  - evaluation.pii.prompt.entity_count = <number>")
print("  - evaluation.pii.prompt.entity_types = [...]")
print("  - evaluation.pii.prompt.EMAIL_ADDRESS = <count>")
print("  - evaluation.pii.prompt.PHONE_NUMBER = <count>")
print("  - evaluation.pii.prompt.SSN = <count>")
print("\nMetrics:")
print("  - genai.evaluation.pii.detections (counter)")
print("  - genai.evaluation.pii.entity_types (counter)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'ollama'")
print("  - gen_ai.request.model = 'llama2'")
print("\nNote: PII detection works identically for cloud and self-hosted LLMs!")
print("=" * 80)
