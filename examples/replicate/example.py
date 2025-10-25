"""Replicate Example"""

import genai_otel

genai_otel.instrument()

import replicate

output = replicate.run("meta/llama-2-70b-chat", input={"prompt": "Explain observability"})
print(f"Response: {''.join(output)}")
print("✅ Instrumented!")
