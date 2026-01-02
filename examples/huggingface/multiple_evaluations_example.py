"""
Example: Multiple Evaluation Types with HuggingFace

This example demonstrates running multiple evaluation types simultaneously with HuggingFace:
- Bias detection
- PII detection
- Toxicity detection

This shows that comprehensive content moderation works with open-source models.

Requirements:
    pip install genai-otel-instrument transformers torch detoxify

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

from genai_otel import instrument

# Enable all evaluation types
instrument(
    service_name="huggingface-combined-evaluations",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_bias_detection=True,
    bias_threshold=0.5,
    enable_pii_detection=True,
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
print("Combined Evaluations with HuggingFace - Bias + PII + Toxicity")
print("=" * 80)
print("\nRunning multiple evaluation checks on each request with open-source models...\n")

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

# Test cases covering different evaluation types
test_cases = [
    {
        "name": "Bias Only",
        "content": "Women are not suited for leadership positions in tech companies.",
        "expected": "Bias: DETECTED, PII: None, Toxicity: Low",
    },
    {
        "name": "PII Only",
        "content": "Contact me at john.doe@example.com or call 555-123-4567 for details.",
        "expected": "Bias: None, PII: DETECTED, Toxicity: None",
    },
    {
        "name": "Toxicity Only",
        "content": "You're an idiot and your ideas are garbage.",
        "expected": "Bias: None, PII: None, Toxicity: DETECTED",
    },
    {
        "name": "Bias + Toxicity",
        "content": "All immigrants are criminals and should be deported immediately.",
        "expected": "Bias: DETECTED, PII: None, Toxicity: DETECTED",
    },
    {
        "name": "PII + Bias",
        "content": "Sarah Johnson (SSN: 123-45-6789) was rejected because women can't handle stress.",
        "expected": "Bias: DETECTED, PII: DETECTED, Toxicity: Low",
    },
    {
        "name": "All Three",
        "content": "That stupid woman Jane Smith (jane@company.com) doesn't belong in engineering.",
        "expected": "Bias: DETECTED, PII: DETECTED, Toxicity: DETECTED",
    },
    {
        "name": "Clean Content",
        "content": "We are seeking qualified candidates with strong technical skills.",
        "expected": "Bias: None, PII: None, Toxicity: None",
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"Example {i}: {test['name']}")
    print("-" * 80)
    print(f"Content: '{test['content']}'")
    print(f"Expected: {test['expected']}")

    if model and tokenizer:
        try:
            inputs = tokenizer(test["content"], return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Status: Processed")
            print(f"Response: {result[:100]}...")
        except Exception as e:
            print(f"Note: Generation failed: {e}")
    else:
        print("Note: Model not loaded (demo mode)")

    print()

print("=" * 80)
print("Combined Evaluation Telemetry:")
print("=" * 80)
print(
    """
Each request will have attributes from all enabled evaluations:

Bias Detection:
  - evaluation.bias.prompt.detected = true/false
  - evaluation.bias.prompt.max_score = <0.0-1.0>
  - evaluation.bias.prompt.detected_biases = [types...]
  - evaluation.bias.prompt.gender_score = <score>
  - evaluation.bias.prompt.race_score = <score>

PII Detection:
  - evaluation.pii.prompt.detected = true/false
  - evaluation.pii.prompt.entity_count = <number>
  - evaluation.pii.prompt.entity_types = [types...]
  - evaluation.pii.prompt.EMAIL_ADDRESS_count = <count>
  - evaluation.pii.prompt.PHONE_NUMBER_count = <count>

Toxicity Detection:
  - evaluation.toxicity.prompt.detected = true/false
  - evaluation.toxicity.prompt.max_score = <0.0-1.0>
  - evaluation.toxicity.prompt.categories = [...]
  - evaluation.toxicity.prompt.toxicity_score = <0.0-1.0>

Provider Attributes:
  - gen_ai.system = 'huggingface'
  - gen_ai.request.model = 'Qwen/Qwen3-0.6B'

Benefits of Open-Source Model Evaluation:
- Full control over the model and data
- No vendor lock-in
- Lower cost at scale
- Same telemetry format as commercial providers
"""
)
print("=" * 80)
