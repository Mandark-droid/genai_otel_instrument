"""LiteLLM Example - Universal LLM proxy

LiteLLM allows you to call 100+ LLM APIs using a unified interface.
genai-otel-instrument uses OpenInference's LiteLLMInstrumentor for automatic tracing.
"""

import genai_otel

genai_otel.instrument()

import os

from litellm import completion

# LiteLLM supports multiple providers with a unified interface
# Example: Call OpenAI's API
response = completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain OpenTelemetry in one sentence."}],
    api_key=os.environ.get("OPENAI_API_KEY"),
)

print(f"Response: {response.choices[0].message.content}")
print("[SUCCESS] LiteLLM calls instrumented via OpenInference!")
