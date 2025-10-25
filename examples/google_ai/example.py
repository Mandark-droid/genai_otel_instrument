"""Google Gemini Example with genai-otel-instrument

This example demonstrates how to use genai-otel-instrument with Google Generative AI (Gemini).
"""

import genai_otel

# Auto-instrument Google AI
genai_otel.instrument()

import os

# Now use Google Generative AI normally
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Create a model
model = genai.GenerativeModel("gemini-pro")

# Generate content
response = model.generate_content("Explain what observability means in software engineering.")

print(f"Response: {response.text}")
print(f"\nTokens used: {response.usage_metadata.total_token_count}")
print("✅ Traces and metrics have been automatically sent to your OTLP endpoint!")
