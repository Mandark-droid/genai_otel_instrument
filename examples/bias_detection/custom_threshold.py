"""
Example: Custom Threshold Configuration

This example demonstrates how to adjust bias detection sensitivity by
configuring different threshold levels. Lower thresholds are more sensitive
and catch subtle bias, while higher thresholds only flag obvious cases.

Threshold Guidelines:
- 0.2-0.3: Very sensitive - catches subtle bias, may have false positives
- 0.4-0.5: Moderate - balanced detection for most use cases
- 0.6-0.7: Permissive - only flags obvious bias
- 0.8-1.0: Very permissive - only extreme cases

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
    export GENAI_BIAS_THRESHOLD=0.5  # Optional: override default
"""

import os

from genai_otel import instrument

# Get threshold from environment or use default
bias_threshold = float(os.getenv("GENAI_BIAS_THRESHOLD", "0.5"))

print(f"Configuring bias detection with threshold: {bias_threshold}")

instrument(
    service_name="bias-custom-threshold",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_bias_detection=True,
    bias_threshold=bias_threshold,
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Custom Threshold Configuration")
print("=" * 80)
print(f"\nCurrent threshold: {bias_threshold}")
print()

# Test content with varying levels of bias
test_cases = [
    {
        "level": "Extreme Bias",
        "expected_score": "0.9+",
        "content": "Women are inferior to men in every way and should stay home.",
    },
    {
        "level": "Strong Bias",
        "expected_score": "0.7-0.8",
        "content": "Men are naturally better leaders than women.",
    },
    {
        "level": "Moderate Bias",
        "expected_score": "0.5-0.6",
        "content": "Women tend to be more emotional in decision-making.",
    },
    {
        "level": "Subtle Bias",
        "expected_score": "0.3-0.4",
        "content": "She's pretty articulate for someone from that neighborhood.",
    },
    {
        "level": "Minimal/No Bias",
        "expected_score": "<0.3",
        "content": "We need someone with strong analytical and communication skills.",
    },
]

print("Testing various bias levels:")
print("-" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['level']} (Expected score: {test['expected_score']})")
    print(f"   Content: '{test['content']}'")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": test["content"]}],
        )
        print(f"   Status: Processed")
    except Exception as e:
        print(f"   Note: {e}")

print()
print("=" * 80)
print("Threshold Tuning Guide:")
print("=" * 80)
print(
    """
Choose your threshold based on use case:

STRICT MODE (0.2-0.3):
  Use case: Hiring, HR, public communications
  Trade-off: May flag false positives, requires manual review
  Example: "Looking for young, energetic team members" → FLAGGED

MODERATE MODE (0.4-0.5):
  Use case: General content moderation, customer service
  Trade-off: Balanced - catches most bias without too many false positives
  Example: "Men are better at math" → FLAGGED
           "We value diverse perspectives" → NOT FLAGGED

PERMISSIVE MODE (0.6-0.7):
  Use case: Internal tools, research, academic contexts
  Trade-off: Only flags obvious bias, may miss subtle cases
  Example: "All women are emotional" → FLAGGED
           "Women tend to be more emotional" → NOT FLAGGED

How to Configure:

1. Environment Variable:
   export GENAI_BIAS_THRESHOLD=0.3

2. Programmatic:
   instrument(
       service_name="my-service",
       enable_bias_detection=True,
       bias_threshold=0.3,
   )

3. Runtime Adjustment:
   Different thresholds for different contexts within same application

Monitoring:
- Track false positive rate in your telemetry
- Adjust threshold based on user feedback
- Monitor evaluation.bias.prompt.max_score distribution
- Use genai.evaluation.bias.score histogram to understand patterns

Telemetry Attributes:
  - evaluation.bias.prompt.detected = true (if score > threshold)
  - evaluation.bias.prompt.max_score = <highest score>
  - evaluation.bias.prompt.threshold = <configured threshold>
"""
)
print("=" * 80)
