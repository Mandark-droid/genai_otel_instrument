"""Together AI Example"""

import genai_otel

genai_otel.instrument()

import os

import together

together.api_key = os.environ.get("TOGETHER_API_KEY")
response = together.Complete.create(
    prompt="Explain OpenTelemetry:", model="togethercomputer/llama-2-70b", max_tokens=100
)
print(f"Response: {response['output']['choices'][0]['text']}")
