"""
Test PII detection with gRPC protocol instead of HTTP.
This should fix the 404 metrics error.
"""

import logging
import os

logging.basicConfig(level=logging.INFO)

from genai_otel import instrument

# Use gRPC endpoint (port 4317) instead of HTTP (port 4318)
instrument(
    service_name="pii-grpc-test",
    endpoint="http://192.168.206.128:4317",  # gRPC port
    enable_pii_detection=True,
    pii_mode="detect",
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\n" + "=" * 80)
print("Testing PII Detection with gRPC Protocol")
print("=" * 80)

print("\nSending prompt with PII (email, SSN)...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "My email is john@example.com and SSN is 123-45-6789"}],
)

print(f"\nResponse: {response.choices[0].message.content}")

import time

print("\nWaiting 3 seconds for data export...")
time.sleep(3)

print("\n" + "=" * 80)
print("SUCCESS! Check Jaeger and Prometheus:")
print("  Jaeger (http://192.168.206.128:16686): Look for evaluation.pii.* attributes")
print("  Prometheus (http://192.168.206.128:9091): Query genai_evaluation_pii_detections_total")
print("=" * 80 + "\n")
