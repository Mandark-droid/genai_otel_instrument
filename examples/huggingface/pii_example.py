"""
Example: PII Detection with HuggingFace

This example demonstrates PII detection working with HuggingFace transformers.
Shows that evaluation features work with open-source models.

Requirements:
    pip install genai-otel-instrument transformers torch

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with PII detection
instrument(
    service_name="huggingface-pii-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_pii_detection=True,
    pii_mode="detect",
    enable_content_capture=True,  # Required for evaluation features
)

# Now import transformers after instrumentation is set up
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers torch")
    exit(1)

print("=" * 80)
print("PII Detection with HuggingFace")
print("=" * 80)
print("\nDemonstrating PII detection with HuggingFace models...\n")

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
        "name": "Email and Phone",
        "prompt": "Contact John at john.doe@example.com or 555-123-4567",
        "has_pii": True,
    },
    {
        "name": "SSN",
        "prompt": "Employee SSN: 123-45-6789",
        "has_pii": True,
    },
    {
        "name": "Credit Card",
        "prompt": "Payment card: 4532-1234-5678-9010",
        "has_pii": True,
    },
    {
        "name": "No PII",
        "prompt": "The weather is nice today",
        "has_pii": False,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Expected PII: {test['has_pii']})")
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
print("  - evaluation.pii.prompt.detected = true/false")
print("  - evaluation.pii.prompt.entity_count = <number>")
print("  - evaluation.pii.prompt.entity_types = [...]")
print("  - evaluation.pii.prompt.EMAIL_ADDRESS_count = <count>")
print("  - evaluation.pii.prompt.PHONE_NUMBER_count = <count>")
print("  - evaluation.pii.prompt.SSN_count = <count>")
print("\nMetrics:")
print("  - genai.evaluation.pii.detections (counter)")
print("  - genai.evaluation.pii.entities (counter, by entity_type)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'huggingface'")
print(f"  - gen_ai.request.model = '{model_name}'")
print("\nNote: PII detection works across all instrumented providers!")
print("=" * 80)
