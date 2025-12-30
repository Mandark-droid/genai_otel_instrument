"""
Test PII detection with the CORRECT collector ports.
HTTP: 55681
gRPC: 55680
"""

import logging
import os

logging.basicConfig(level=logging.INFO)

from genai_otel import instrument

# Use correct HTTP port: 55681
instrument(
    service_name="pii-correct-port",
    endpoint="http://192.168.206.128:55681",
    enable_pii_detection=True,
    pii_mode="detect",
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\n" + "=" * 80)
print("Testing PII Detection with CORRECT Collector Port (55681)")
print("=" * 80)

print("\nSending prompt with PII...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "My email is john@example.com and SSN is 123-45-6789"}],
)

print(f"\nResponse: {response.choices[0].message.content}")

import time

print("\nWaiting 5 seconds for metrics export...")
time.sleep(5)

print("\n" + "=" * 80)
print("SUCCESS! Check Prometheus for metrics:")
print("  http://192.168.206.128:9091")
print("  Query: genai_evaluation_pii_detections_total")
print("=" * 80 + "\n")
