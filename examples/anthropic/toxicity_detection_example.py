"""
Example: Toxicity Detection with Anthropic Claude

This example demonstrates toxicity detection working with Anthropic's Claude API.
Shows comprehensive evaluation coverage across different LLM providers.

Requirements:
    pip install genai-otel-instrument[toxicity] anthropic

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export ANTHROPIC_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with toxicity detection
instrument(
    service_name="anthropic-toxicity-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)

# Now import Anthropic after instrumentation is set up
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Toxicity Detection with Anthropic Claude")
print("=" * 80)
print("\nDemonstrating toxicity detection across providers...\n")

# Test cases
test_cases = [
    {
        "name": "Toxic Content",
        "prompt": "You're an idiot and should go to hell",
        "is_toxic": True,
    },
    {
        "name": "Threatening Language",
        "prompt": "I'm going to hurt you",
        "is_toxic": True,
    },
    {
        "name": "Neutral Content",
        "prompt": "Can you explain how photosynthesis works?",
        "is_toxic": False,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Expected Toxic: {test['is_toxic']})")
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
print("  - evaluation.toxicity.prompt.detected = true/false")
print("  - evaluation.toxicity.prompt.max_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.categories = [...]")
print("\nMetrics:")
print("  - genai.evaluation.toxicity.detections (counter)")
print("  - genai.evaluation.toxicity.categories (counter)")
print("  - genai.evaluation.toxicity.score (histogram)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'anthropic'")
print("  - gen_ai.request.model = 'claude-3-haiku-20240307'")
print("=" * 80)
