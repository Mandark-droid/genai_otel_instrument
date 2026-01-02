"""
Example: Toxicity Detection with HuggingFace

This example demonstrates toxicity detection working with HuggingFace transformers.
Shows that evaluation features work with open-source models.

Requirements:
    pip install genai-otel-instrument transformers torch detoxify

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with toxicity detection
instrument(
    service_name="huggingface-toxicity-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    enable_content_capture=True,  # Required for evaluation features
)

# Now import transformers after instrumentation is set up
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers torch")
    exit(1)

print("=" * 80)
print("Toxicity Detection with HuggingFace")
print("=" * 80)
print("\nDemonstrating toxicity detection with HuggingFace models...\n")

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
        "name": "Constructive Feedback",
        "prompt": "I appreciate your input, but I see it differently",
        "is_toxic": False,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Expected Toxic: {test['is_toxic']})")
    print("-" * 80)
    print(f"Prompt: '{test['prompt']}'")

    if model and tokenizer:
        try:
            inputs = tokenizer(test["prompt"], return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {result[:100]}...")
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
print("  - evaluation.toxicity.prompt.detected = true/false")
print("  - evaluation.toxicity.prompt.max_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.categories = [...]")
print("  - evaluation.toxicity.prompt.toxicity_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.severe_toxicity_score = <0.0-1.0>")
print("  - evaluation.toxicity.prompt.insult_score = <0.0-1.0>")
print("\nMetrics:")
print("  - genai.evaluation.toxicity.detections (counter)")
print("  - genai.evaluation.toxicity.categories (counter)")
print("  - genai.evaluation.toxicity.score (histogram)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'huggingface'")
print(f"  - gen_ai.request.model = '{model_name}'")
print("\nNote: Toxicity detection works across all instrumented providers!")
print("=" * 80)
