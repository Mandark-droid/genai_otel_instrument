"""HuggingFace Transformers Example"""

import genai_otel

genai_otel.instrument()

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("OpenTelemetry is", max_length=50, num_return_sequences=1)
print(f"Response: {result[0]['generated_text']}")
print("✅ HuggingFace instrumented!")
