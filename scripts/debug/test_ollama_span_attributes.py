"""Debug script to check if span attributes are being set correctly.

This script traces the exact flow of attribute setting in Ollama instrumentor.
"""

import logging
import os

# Set logging level to DEBUG to see all details
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Enable detailed logging for our modules
logging.getLogger("genai_otel").setLevel(logging.DEBUG)
logging.getLogger("genai_otel.instrumentors").setLevel(logging.DEBUG)
logging.getLogger("genai_otel.instrumentors.base").setLevel(logging.DEBUG)
logging.getLogger("genai_otel.instrumentors.ollama_instrumentor").setLevel(logging.DEBUG)
logging.getLogger("genai_otel.cost_calculator").setLevel(logging.DEBUG)

print("=" * 80)
print("Ollama Span Attributes Debug")
print("=" * 80)

# Initialize instrumentation
print("\nInitializing instrumentation...")
import genai_otel

genai_otel.instrument()

# Make Ollama call
print("\nMaking Ollama call...")
import ollama

response = ollama.generate(model="smollm2:360m", prompt="Say hello in 3 words")

print(f"\nResponse received:")
print(f"  model: {response.model}")
print(f"  prompt_eval_count: {response.prompt_eval_count}")
print(f"  eval_count: {response.eval_count}")

# Give time for span to be exported
import time

time.sleep(2)

print("\n" + "=" * 80)
print("Check your OTLP collector for the span")
print("Expected attributes:")
print("  - gen_ai.request.model: smollm2:360m")
print("  - gen_ai.usage.prompt_tokens: 36 (or similar)")
print("  - gen_ai.usage.completion_tokens: 6 (or similar)")
print("  - gen_ai.usage.total_tokens: 42 (or similar)")
print("=" * 80)
