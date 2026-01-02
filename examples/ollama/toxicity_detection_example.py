"""
Example: Toxicity Detection with Ollama

This example demonstrates toxicity detection working with locally-hosted Ollama models.
Shows that evaluation features work with self-hosted LLMs, not just cloud APIs.

Requirements:
    pip install genai-otel-instrument ollama detoxify
    # Install and run Ollama: https://ollama.ai
    # Pull a model: ollama pull llama2

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    # Ollama runs locally by default on http://localhost:11434
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with toxicity detection
instrument(
    service_name="ollama-toxicity-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    enable_content_capture=True,  # Required for evaluation features
)

# Now import Ollama after instrumentation is set up
try:
    import ollama
except ImportError:
    print("Error: ollama package not installed. Run: pip install ollama")
    exit(1)

print("=" * 80)
print("Toxicity Detection with Ollama (Local LLM)")
print("=" * 80)
print("\nDemonstrating toxicity detection with self-hosted models...\n")

# Test cases
test_cases = [
    {
        "name": "Toxic Insult",
        "prompt": "You're an idiot and your ideas are garbage",
        "is_toxic": True,
    },
    {
        "name": "Threatening Language",
        "prompt": "I'm going to make you regret this",
        "is_toxic": True,
    },
    {
        "name": "Severe Toxicity",
        "prompt": "Go kill yourself, you worthless piece of trash",
        "is_toxic": True,
    },
    {
        "name": "Constructive Criticism",
        "prompt": "I respectfully disagree with your approach",
        "is_toxic": False,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Expected Toxic: {test['is_toxic']})")
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
print("  - evaluation.toxicity.prompt.detected = true/false")
print("  - evaluation.toxicity.prompt.toxicity_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.severe_toxicity_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.obscene_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.threat_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.insult_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.identity_attack_score = <0.0-1.0>")
print("\nMetrics:")
print("  - genai.evaluation.toxicity.detections (counter)")
print("  - genai.evaluation.toxicity.score (histogram)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'ollama'")
print("  - gen_ai.request.model = 'llama2'")
print("\nNote: Toxicity detection works identically for cloud and self-hosted LLMs!")
print("=" * 80)
