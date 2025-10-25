"""Cohere Example with genai-otel-instrument"""

import genai_otel

genai_otel.instrument()

import os

import cohere

client = cohere.Client(os.environ.get("COHERE_API_KEY"))

response = client.generate(
    model="command", prompt="Explain observability in one sentence.", max_tokens=100
)

print(f"Response: {response.generations[0].text}")
print(
    f"\nTokens used: {response.meta.billed_units.input_tokens + response.meta.billed_units.output_tokens}"
)
print("✅ Traces and metrics sent to OTLP endpoint!")
