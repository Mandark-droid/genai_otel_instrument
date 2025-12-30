"""
Test PII detection with gRPC by setting environment variable.
"""

import logging
import os

# SET GRPC PROTOCOL BEFORE IMPORTING genai_otel
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://192.168.206.128:4317"

logging.basicConfig(level=logging.INFO)

from genai_otel import instrument

instrument(
    service_name="pii-grpc-env-test",
    enable_pii_detection=True,
    pii_mode="detect",
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\n" + "=" * 80)
print("Testing PII Detection with gRPC Protocol (via env var)")
print("=" * 80)

print("\nSending prompt with PII...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "My email is john@example.com"}],
)

print(f"\nResponse: {response.choices[0].message.content}")

import time

print("\nWaiting 3 seconds for export...")
time.sleep(3)

print("\n" + "=" * 80)
print("Check for errors in stderr. If no 404 or BadStatusLine, gRPC works!")
print("=" * 80 + "\n")
