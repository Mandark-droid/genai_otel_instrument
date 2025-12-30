"""
Example: Toxicity Detection with Blocking Mode

This example demonstrates toxicity detection with blocking enabled.
When toxic content is detected, the span status is set to ERROR and blocked
metrics are recorded. This allows you to filter or alert on toxic requests.

Requirements:
    pip install genai-otel-instrument openai
    pip install detoxify

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with blocking enabled
instrument(
    service_name="toxicity-blocking-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    toxicity_block_on_detection=True,  # Block toxic content
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Toxicity Blocking Mode")
print("=" * 80)
print("\nWith blocking mode enabled, requests/responses with toxic content")
print("are marked with ERROR status in telemetry.")
print("\nPrompt: 'Go kill yourself'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Go kill yourself",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
    print("\nNote: Request completed but span is marked as ERROR")
except Exception as e:
    print(f"\nNote: In production, this would be blocked: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.toxicity.prompt.detected = true")
print("  - evaluation.toxicity.prompt.blocked = true")
print("  - Span status = ERROR")
print("  - evaluation.toxicity.prompt.categories = ['threat', 'severe_toxicity']")
print("\nMetrics:")
print("  - genai.evaluation.toxicity.blocked (counter)")
print("\nNote: The actual request is NOT blocked by the library.")
print("The ERROR status allows you to:")
print("  - Filter toxic requests in your observability platform")
print("  - Create alerts for toxic content")
print("  - Implement custom blocking logic in your application")
print("=" * 80)
