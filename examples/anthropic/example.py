"""Anthropic Claude Example with genai-otel-instrument

This example demonstrates how to use genai-otel-instrument with Anthropic Claude.
"""

import genai_otel

# Auto-instrument Anthropic (and all other supported libraries)
genai_otel.instrument()

# Now use Anthropic normally
from anthropic import Anthropic

client = Anthropic()

# Make a simple message request
message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=150,
    messages=[{"role": "user", "content": "Explain OpenTelemetry in one sentence."}],
)

print(f"Response: {message.content[0].text}")
print(f"\nTokens used: {message.usage.input_tokens + message.usage.output_tokens}")
print("✅ Traces and metrics have been automatically sent to your OTLP endpoint!")
