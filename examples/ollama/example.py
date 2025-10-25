"""Ollama Example - Local LLMs"""

import genai_otel

genai_otel.instrument()

import ollama

response = ollama.generate(model="smollm2:360m", prompt="What is OpenTelemetry?")
print(f"Response: {response['response']}")
print("[SUCCESS] Local Ollama calls instrumented!")
