"""
Test PII detection with the gRPC fix.
Should automatically detect port 4317 and use gRPC protocol.
"""

import logging
import os

logging.basicConfig(level=logging.INFO)

from genai_otel import instrument

# Port 4317 will automatically trigger gRPC protocol
instrument(
    service_name="pii-grpc-fixed",
    endpoint="http://192.168.206.128:4317",
    enable_pii_detection=True,
    pii_mode="detect",
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\n" + "=" * 80)
print("Testing PII Detection with gRPC Auto-Detection Fix")
print("=" * 80)

print("\nSending prompt with PII...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "My email is test@example.com and SSN is 123-45-6789"}],
)

print(f"\nResponse: {response.choices[0].message.content}")

import time

print("\nWaiting 3 seconds for export...")
time.sleep(3)

print("\n" + "=" * 80)
print("Check for success messages:")
print("  1. Should see 'Using OTLP gRPC protocol' in logs")
print("  2. NO 404 errors or BadStatusLine errors")
print("  3. Metrics should export successfully")
print("=" * 80 + "\n")
