"""
Example: Multiple Evaluation Types with Ollama

This example demonstrates running multiple evaluation types simultaneously with Ollama:
- Bias detection
- PII detection
- Toxicity detection

This shows that comprehensive content moderation works with self-hosted models.

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

# Enable all evaluation types
instrument(
    service_name="ollama-combined-evaluations",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_bias_detection=True,
    bias_threshold=0.5,
    enable_pii_detection=True,
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
print("Combined Evaluations with Ollama - Bias + PII + Toxicity")
print("=" * 80)
print("\nRunning multiple evaluation checks on each request with self-hosted LLM...\n")

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

    try:
        response = ollama.chat(
            model="smollm2:360m",
            messages=[{"role": "user", "content": test["content"]}],
        )
        print(f"Status: Processed")
        print(f"Response: {response['message']['content'][:100]}...")
    except Exception as e:
        print(f"Note: Make sure Ollama is running and llama2 model is pulled")
        print(f"Error: {e}")

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
  - evaluation.bias.prompt.age_score = <score>
  - evaluation.bias.prompt.religion_score = <score>
  - evaluation.bias.prompt.disability_score = <score>

PII Detection:
  - evaluation.pii.prompt.detected = true/false
  - evaluation.pii.prompt.entity_count = <number>
  - evaluation.pii.prompt.entity_types = [types...]
  - evaluation.pii.prompt.entities_redacted = <redacted text>
  - evaluation.pii.prompt.EMAIL_ADDRESS = <count>
  - evaluation.pii.prompt.PHONE_NUMBER = <count>
  - evaluation.pii.prompt.SSN = <count>

Toxicity Detection:
  - evaluation.toxicity.prompt.detected = true/false
  - evaluation.toxicity.prompt.toxicity_score = <0.0-1.0>
  - evaluation.toxicity.prompt.severe_toxicity_score = <0.0-1.0>
  - evaluation.toxicity.prompt.obscene_score = <0.0-1.0>
  - evaluation.toxicity.prompt.threat_score = <0.0-1.0>
  - evaluation.toxicity.prompt.insult_score = <0.0-1.0>
  - evaluation.toxicity.prompt.identity_attack_score = <0.0-1.0>

Provider Attributes:
  - gen_ai.system = 'ollama'
  - gen_ai.request.model = 'llama2'

Benefits of Self-Hosted Evaluation:
- Full data privacy - no data sent to cloud providers
- Lower cost - no per-evaluation API fees
- Offline capability - works without internet
- Same telemetry format as cloud providers
- Identical metrics and attributes structure
"""
)
print("=" * 80)
