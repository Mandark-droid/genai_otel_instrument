"""
Example: Basic Bias Detection

This example demonstrates basic bias detection across multiple categories.
The detector identifies demographic biases (gender, race, age, etc.) and records
them in telemetry attributes.

Bias Categories Detected:
- Gender bias
- Racial/ethnic bias
- Religious bias
- Age bias (ageism)
- Disability bias (ableism)
- Sexual orientation bias
- Political bias

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with bias detection
instrument(
    service_name="bias-detection-basic-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_bias_detection=True,
    bias_threshold=0.5,  # Sensitivity threshold
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Basic Bias Detection")
print("=" * 80)
print("\nTesting bias detection across multiple categories...")
print()

# Example 1: Gender bias
print("Example 1: Gender Bias Detection")
print("-" * 80)
print("Prompt: 'Women are always too emotional for leadership positions'")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Women are always too emotional for leadership positions. What do you think?",
            }
        ],
    )
    print(f"Response: {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 2: Age bias
print("Example 2: Age Bias Detection")
print("-" * 80)
print("Prompt: 'Millennials are always lazy and entitled'")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Millennials are always lazy and entitled. Agree or disagree?",
            }
        ],
    )
    print(f"Response: {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 3: No bias
print("Example 3: Neutral Content (No Bias)")
print("-" * 80)
print("Prompt: 'What are best practices for inclusive hiring?'")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What are best practices for creating an inclusive hiring process?",
            }
        ],
    )
    print(f"Response: {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"Note: {e}")

print()
print("=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.bias.prompt.detected = true/false")
print("  - evaluation.bias.prompt.max_score = <score>")
print("  - evaluation.bias.prompt.detected_biases = [list of bias types]")
print("  - evaluation.bias.prompt.gender_score = <score>")
print("  - evaluation.bias.prompt.age_score = <score>")
print("\nMetrics:")
print("  - genai.evaluation.bias.detections (counter)")
print("  - genai.evaluation.bias.types (counter, by bias_type)")
print("  - genai.evaluation.bias.score (histogram)")
print("=" * 80)
