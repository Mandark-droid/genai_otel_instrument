"""
Debug script to test PII detection and check if span processor is working.
"""

import logging
import os
import sys

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# Enable all genai_otel logging
for logger_name in ["genai_otel", "genai_otel.evaluation", "genai_otel.auto_instrument"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

print("=" * 80)
print("PII Detection Debug Test")
print("=" * 80)

from genai_otel import instrument

# Instrument with PII detection
print("\n[1] Starting instrumentation...")
instrument(
    service_name="pii-debug-test",
    endpoint="http://192.168.206.128:4318",
    enable_pii_detection=True,
    pii_mode="detect",
    pii_threshold=0.7,
)
print("[1] Instrumentation complete")

# Check if evaluation span processor was added
from opentelemetry import trace

tracer_provider = trace.get_tracer_provider()
print(f"\n[2] Tracer provider: {tracer_provider}")
print(
    f"[2] Span processors: {tracer_provider._active_span_processor._span_processors if hasattr(tracer_provider, '_active_span_processor') else 'N/A'}"
)

# Now test with OpenAI
print("\n[3] Importing OpenAI...")
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\n[4] Sending prompt with PII...")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "My email is test@example.com and my SSN is 123-45-6789",
            }
        ],
    )
    print(f"[4] Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"[4] Error: {e}")

print("\n[5] Waiting for span export...")
import time

time.sleep(3)  # Wait for spans to export

print("\n" + "=" * 80)
print("Check Jaeger for evaluation.pii.* attributes")
print("=" * 80)
