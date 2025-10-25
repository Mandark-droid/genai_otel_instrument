"""OpenAI Example with genai-otel-instrument

This example demonstrates how to use genai-otel-instrument with OpenAI.
Simply import genai_otel and call instrument() before using OpenAI.
"""

import genai_otel

# Auto-instrument OpenAI (and all other supported libraries)
genai_otel.instrument()

# Now use OpenAI normally
from openai import OpenAI

client = OpenAI()

# Make a simple completion request
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is OpenTelemetry?"},
    ],
    max_tokens=150,
)

print(f"Response: {response.choices[0].message.content}")
print(f"\nTokens used: {response.usage.total_tokens}")
print("✅ Traces and metrics have been automatically sent to your OTLP endpoint!")
