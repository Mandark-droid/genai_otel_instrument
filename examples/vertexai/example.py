"""Vertex AI Example"""

import genai_otel

genai_otel.instrument()

from vertexai.preview.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained("text-bison")
response = model.predict("What is distributed tracing?", max_output_tokens=100)
print(f"Response: {response.text}")
