"""
Example: Multiple Evaluation Types Combined

This example demonstrates running multiple evaluation types simultaneously:
- Bias detection
- PII detection
- Toxicity detection

This is useful for comprehensive content moderation where you want to check
for multiple types of problematic content in a single pass.

Requirements:
    pip install genai-otel-instrument openai detoxify

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Enable all evaluation types
instrument(
    service_name="combined-evaluations",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_bias_detection=True,
    bias_threshold=0.5,
    enable_pii_detection=True,
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Combined Evaluations - Bias + PII + Toxicity")
print("=" * 80)
print("\nRunning multiple evaluation checks on each request...")
print()

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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": test["content"]}],
        )
        print(f"Status: Processed")
        print(f"Response: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"Note: {e}")

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

Metrics (Counters):
  - genai.evaluation.bias.detections
  - genai.evaluation.bias.types (dimension: bias_type)
  - genai.evaluation.pii.detections
  - genai.evaluation.pii.entity_types (dimension: entity_type)
  - genai.evaluation.toxicity.detections

Metrics (Histograms):
  - genai.evaluation.bias.score
  - genai.evaluation.toxicity.score

Use Cases for Combined Evaluation:

1. Content Moderation Platform:
   - Check user-generated content for all types of issues
   - Block or flag content that violates any policy
   - Provide specific feedback on what was detected

2. HR/Hiring Applications:
   - Ensure job descriptions are bias-free
   - Protect candidate PII
   - Maintain professional tone (no toxicity)

3. Customer Service Chatbots:
   - Avoid biased responses
   - Don't expose customer PII in logs
   - Prevent toxic language in bot responses

4. Compliance Monitoring:
   - Track all evaluation metrics together
   - Identify patterns (e.g., bias often accompanied by toxicity)
   - Generate comprehensive compliance reports

Performance Considerations:
- Each evaluation adds minimal latency (~5-50ms depending on text length)
- Evaluations run in parallel where possible
- All evaluations use the same span, no extra OTLP exports
- Consider disabling evaluations in development if not needed

Configuration Best Practices:
- Enable only evaluations you need
- Tune thresholds per use case
- Monitor false positive rates
- Use environment variables for easy per-environment config
"""
)
print("=" * 80)
