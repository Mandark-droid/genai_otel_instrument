"""
Example: Hallucination Detection with HuggingFace

This example demonstrates hallucination detection working with HuggingFace transformers.
Shows that evaluation features work with open-source models.

Requirements:
    pip install genai-otel-instrument transformers torch

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with hallucination detection
instrument(
    service_name="huggingface-hallucination-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_hallucination_detection=True,
    hallucination_threshold=0.5,
    enable_content_capture=True,  # Required for evaluation features
)

# Now import transformers after instrumentation is set up
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers torch")
    exit(1)

print("=" * 80)
print("Hallucination Detection with HuggingFace")
print("=" * 80)
print("\nDemonstrating hallucination detection with HuggingFace models...\n")

# Load model and tokenizer
print("Loading model (this may take a moment)...")
try:
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Model loaded: {model_name}\n")
except Exception as e:
    print(f"Note: Model loading failed. This is expected in CI/demo: {e}")
    model = None
    tokenizer = None

# Test cases
test_cases = [
    {
        "name": "Factual Question",
        "prompt": "What is the capital of France?",
        "expected_hallucination": False,
    },
    {
        "name": "Specific Query",
        "prompt": "Explain the concept of machine learning",
        "expected_hallucination": False,
    },
    {
        "name": "Open-ended Question",
        "prompt": "What will happen in the future of AI?",
        "expected_hallucination": False,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Expected Hallucination: {test['expected_hallucination']})")
    print("-" * 80)
    print(f"Prompt: '{test['prompt']}'")

    if model and tokenizer:
        try:
            inputs = tokenizer(test["prompt"], return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {result[:150]}...")
        except Exception as e:
            print(f"Note: Generation failed: {e}")
    else:
        print("Note: Model not loaded (demo mode)")

    print()

print("=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("Provider: HuggingFace Transformers")
print("Evaluation Attributes:")
print("  - evaluation.hallucination.response.detected = true/false")
print("  - evaluation.hallucination.response.score = <0.0-1.0>")
print("  - evaluation.hallucination.response.citations = <count>")
print("  - evaluation.hallucination.response.hedge_words = <count>")
print("  - evaluation.hallucination.response.claims = <count>")
print("\nMetrics:")
print("  - genai.evaluation.hallucination.detections (counter)")
print("  - genai.evaluation.hallucination.score (histogram)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'huggingface'")
print(f"  - gen_ai.request.model = '{model_name}'")
print("\nNote: Hallucination detection works across all instrumented providers!")
print("=" * 80)
