"""
Example: Basic Toxicity Detection with Detoxify (Local Model)

This example demonstrates toxicity detection using the Detoxify local ML model.
Detoxify runs offline without external API calls, making it privacy-friendly
and suitable for on-premise deployments.

Detected categories:
- toxicity: General toxic language
- severe_toxicity: Extremely harmful content
- identity_attack: Discrimination and hate speech
- insult: Insulting or demeaning language
- profanity: Swearing and obscene content
- threat: Threatening or violent language

Requirements:
    pip install genai-otel-instrument openai
    pip install detoxify

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with toxicity detection
instrument(
    service_name="toxicity-detection-detoxify",
    endpoint="http://192.168.206.128:55681",  # os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    toxicity_use_perspective_api=False,  # Use local Detoxify model
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Basic Toxicity Detection - Detoxify Local Model")
print("=" * 80)
print("\nDetoxify is a local ML model that runs offline")
print("No external API calls required - privacy-friendly")
print("\nSending prompt with potentially toxic content...")
print("Prompt: 'You are stupid and worthless'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "You are stupid and worthless. Can you help me?",
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
print("  - evaluation.toxicity.prompt.categories = ['toxicity', 'insult']")
print("  - evaluation.toxicity.prompt.toxicity_score = <score>")
print("  - evaluation.toxicity.prompt.insult_score = <score>")
print("\nMetrics:")
print("  - genai.evaluation.toxicity.detections (counter)")
print("  - genai.evaluation.toxicity.categories (counter, by category)")
print("  - genai.evaluation.toxicity.score (histogram)")
print("=" * 80)
