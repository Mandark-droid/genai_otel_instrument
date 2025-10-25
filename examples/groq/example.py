"""Groq Example with genai-otel-instrument"""

import genai_otel

genai_otel.instrument()

import os

from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "What is observability?"}],
    max_tokens=100,
)
print(f"Response: {response.choices[0].message.content}")
print("✅ Instrumented!")
