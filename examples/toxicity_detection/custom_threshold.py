"""
Example: Custom Toxicity Detection Threshold

This example demonstrates how to adjust the confidence threshold for toxicity detection.
- Lower threshold (e.g., 0.5): More sensitive, catches borderline cases, may have false positives
- Higher threshold (e.g., 0.9): Less sensitive, only catches severe cases, fewer false positives

Use cases:
- 0.5-0.6: Strict content moderation (e.g., children's platforms)
- 0.7-0.8: Balanced approach (default: 0.7)
- 0.9+: Only severe toxicity (e.g., internal tools)

Requirements:
    pip install genai-otel-instrument openai
    pip install detoxify

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Example with high sensitivity (lower threshold)
print("=" * 80)
print("Custom Toxicity Threshold - High Sensitivity")
print("=" * 80)
print("\nThreshold: 0.5 (More sensitive, may have false positives)")

instrument(
    service_name="toxicity-threshold-low",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.5,  # More sensitive
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\nTesting with mildly negative content...")
print("Prompt: 'This is not very good'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "This is not very good"}],
    )
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Note: {e}")

print("\n" + "=" * 80)
print("With threshold=0.5:")
print("=" * 80)
print("  - Even borderline negative content may be detected")
print("  - Good for strict content moderation")
print("  - May have false positives")
print("\nCheck telemetry for toxicity scores between 0.5-0.7")

# Example with low sensitivity (higher threshold)
print("\n" + "=" * 80)
print("Custom Toxicity Threshold - Low Sensitivity")
print("=" * 80)
print("\nThreshold: 0.9 (Only severe toxicity detected)")

# Re-instrument with higher threshold
instrument(
    service_name="toxicity-threshold-high",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.9,  # Less sensitive
)

print("\nTesting with moderately toxic content...")
print("Prompt: 'You're being annoying'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "You're being annoying"}],
    )
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Note: {e}")

print("\n" + "=" * 80)
print("With threshold=0.9:")
print("=" * 80)
print("  - Only severe toxicity is detected")
print("  - Moderate language passes through")
print("  - Fewer false positives")
print("\nOnly toxicity scores >= 0.9 trigger detection")

print("\n" + "=" * 80)
print("Threshold Selection Guidelines:")
print("=" * 80)
print("  0.5-0.6: Strict (children's platforms, educational apps)")
print("  0.7-0.8: Balanced (recommended for most apps)")
print("  0.9+: Permissive (internal tools, developer forums)")
print("\nMonitor false positive rate and adjust accordingly")
print("=" * 80)
