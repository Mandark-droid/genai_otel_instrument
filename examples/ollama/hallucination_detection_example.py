"""
Example: Hallucination Detection with Ollama

This example demonstrates hallucination detection working with locally-hosted Ollama models.
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

# Set up OpenTelemetry instrumentation with hallucination detection
instrument(
    service_name="ollama-hallucination-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_hallucination_detection=True,
    hallucination_threshold=0.5,
    enable_content_capture=True,  # Required for evaluation features
)

# Now import Ollama after instrumentation is set up
try:
    import ollama
except ImportError:
    print("Error: ollama package not installed. Run: pip install ollama")
    exit(1)

print("=" * 80)
print("Hallucination Detection with Ollama (Local LLM)")
print("=" * 80)
print("\nDemonstrating hallucination detection with self-hosted models...\n")

# Test cases with context
test_cases = [
    {
        "name": "Factual Question",
        "prompt": "What is the capital of France?",
        "context": None,
        "expected_hallucination": False,
    },
    {
        "name": "RAG with Context",
        "prompt": "Based on the context, what is the company's policy on remote work?",
        "context": "Our company allows full-time remote work for all employees.",
        "expected_hallucination": False,
    },
    {
        "name": "Unsupported Claim",
        "prompt": "Tell me about the company's vacation policy",
        "context": "The company has an open PTO policy.",
        "expected_hallucination": False,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Expected Hallucination: {test['expected_hallucination']})")
    print("-" * 80)
    if test["context"]:
        print(f"Context: '{test['context']}'")
    print(f"Prompt: '{test['prompt']}'")

    try:
        # Include context in the prompt if available
        full_prompt = test["prompt"]
        if test["context"]:
            full_prompt = f"Context: {test['context']}\n\nQuestion: {test['prompt']}"

        response = ollama.chat(
            model="smollm2:360m",
            messages=[{"role": "user", "content": full_prompt}],
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
print("  - evaluation.hallucination.detected = true/false")
print("  - evaluation.hallucination.score = <0.0-1.0>")
print("  - evaluation.hallucination.method = 'self_check' or 'context_similarity'")
print("\nMetrics:")
print("  - genai.evaluation.hallucination.detections (counter)")
print("  - genai.evaluation.hallucination.score (histogram)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'ollama'")
print("  - gen_ai.request.model = 'llama2'")
print("\nNote: Hallucination detection works identically for cloud and self-hosted LLMs!")
print("=" * 80)
