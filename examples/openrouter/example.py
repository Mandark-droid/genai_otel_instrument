"""OpenRouter Example with genai-otel-instrument

This example demonstrates how to use genai-otel-instrument with OpenRouter.
OpenRouter provides unified access to 300+ LLM models from 60+ providers through
an OpenAI-compatible API.

Simply import genai_otel and call instrument() before using OpenRouter.
The instrumentor automatically detects OpenRouter clients by checking the base_url.
"""

import os

import genai_otel

# Auto-instrument all supported libraries including OpenRouter
genai_otel.instrument()

# Now use OpenRouter with the OpenAI SDK
from openai import OpenAI

# OpenRouter uses the OpenAI SDK with a custom base_url
# The instrumentor automatically detects this is an OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

print("Making a request to Claude 3.5 Sonnet via OpenRouter...")

# Make a simple completion request
# OpenRouter uses provider/model format (e.g., "anthropic/claude-3.5-sonnet")
response = client.chat.completions.create(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is OpenRouter and why would I use it?"},
    ],
    max_tokens=200,
)

print(f"\nResponse: {response.choices[0].message.content}")
print(f"\nTokens used: {response.usage.total_tokens}")
print(f"  - Prompt tokens: {response.usage.prompt_tokens}")
print(f"  - Completion tokens: {response.usage.completion_tokens}")

print("\n✅ Traces and metrics have been automatically sent to your OTLP endpoint!")
print("\nWhat was instrumented:")
print("  - Span name: openrouter.chat.completion")
print("  - System: openrouter")
print(f"  - Model: {response.model}")
print("  - Token usage and cost calculation")
print("  - Response attributes (finish_reason, response_id, etc.)")

# OpenRouter-specific features
print("\n🚀 OpenRouter Features:")
print("  - Access to 300+ models from 60+ providers")
print("  - Automatic fallback routing")
print("  - Cost optimization across providers")
print("  - No vendor lock-in")

print("\n💡 Try different models:")
print("  - anthropic/claude-3-opus")
print("  - openai/gpt-4")
print("  - google/gemini-pro")
print("  - meta-llama/llama-3-70b-instruct")
print("  - mistralai/mistral-large")
print("  - deepseek/deepseek-chat")
