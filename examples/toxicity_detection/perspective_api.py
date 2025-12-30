"""
Example: Toxicity Detection with Google Perspective API

This example demonstrates toxicity detection using Google's Perspective API.
Perspective API is a cloud-based service that provides more accurate toxicity
detection with machine learning models trained on real-world data.

Benefits:
- Higher accuracy than local models
- Continuously updated models
- Industry-standard toxicity classification

Requirements:
    pip install genai-otel-instrument openai
    pip install google-api-python-client

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
    export GENAI_TOXICITY_PERSPECTIVE_API_KEY=your_perspective_api_key

Get your Perspective API key from:
https://developers.perspectiveapi.com/
"""

import os

from genai_otel import instrument

# Check if API key is configured
perspective_api_key = os.getenv("GENAI_TOXICITY_PERSPECTIVE_API_KEY")

if not perspective_api_key:
    print("=" * 80)
    print("ERROR: Perspective API Key Required")
    print("=" * 80)
    print("\nTo run this example, you need a Perspective API key:")
    print("1. Visit: https://developers.perspectiveapi.com/")
    print("2. Sign up and create an API key")
    print("3. Set environment variable:")
    print("   export GENAI_TOXICITY_PERSPECTIVE_API_KEY=your_api_key_here")
    print("\nAlternatively, use the basic_detoxify.py example for local detection")
    print("=" * 80)
    exit(1)

# Set up OpenTelemetry instrumentation with Perspective API
instrument(
    service_name="toxicity-perspective-api",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    toxicity_use_perspective_api=True,  # Use Perspective API
    toxicity_perspective_api_key=perspective_api_key,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Toxicity Detection - Google Perspective API")
print("=" * 80)
print("\nPerspective API provides more accurate toxicity detection")
print("with machine learning models trained on real-world data.")
print("\nSending prompt with hateful content...")
print("Prompt: 'This is a hateful and offensive message'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "This is a hateful and offensive message",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nNote: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.toxicity.prompt.detected = true")
print("  - evaluation.toxicity.prompt.max_score = <score>")
print("  - evaluation.toxicity.prompt.categories = <detected categories>")
print("  - Individual category scores")
print("\nPerspective API typically provides:")
print("  - Higher accuracy than local models")
print("  - More nuanced category detection")
print("  - Better handling of context and sarcasm")
print("=" * 80)
