"""Mistral AI Example - Using new Mistral SDK (v1.0+)"""

import genai_otel

genai_otel.instrument()

import os

from mistralai import Mistral

# New Mistral SDK (v1.0+)
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Explain distributed tracing briefly."}],
)
print(f"Response: {response.choices[0].message.content}")
print("[SUCCESS] Instrumented!")
