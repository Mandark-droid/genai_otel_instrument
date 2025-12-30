"""
Final test to verify PII attributes are being set and exported.
"""

import logging
import os

logging.basicConfig(level=logging.INFO)

# Patch to add detailed debug
import genai_otel.instrumentors.base as base_module

original_run = base_module.BaseInstrumentor._run_evaluation_checks


def debug_run(self, span, args, kwargs, result):
    print("\n" + "=" * 80)
    print("[DEBUG] _run_evaluation_checks called")
    print(f"[DEBUG] Span name: {span.name}")
    print(
        f'[DEBUG] Span attributes count: {len(span.attributes) if hasattr(span, "attributes") else "N/A"}'
    )

    # Call original
    try:
        original_run(self, span, args, kwargs, result)
        print("[DEBUG] Original method completed")
    except Exception as e:
        print(f"[DEBUG] Exception in original: {e}")
        import traceback

        traceback.print_exc()

    # Check if attributes were set
    if hasattr(span, "attributes"):
        attrs = dict(span.attributes)
        pii_attrs = {k: v for k, v in attrs.items() if "pii" in k.lower()}
        if pii_attrs:
            print(f"[DEBUG] PII attributes found: {list(pii_attrs.keys())}")
        else:
            print("[DEBUG] NO PII attributes set!")
    print("=" * 80 + "\n")


base_module.BaseInstrumentor._run_evaluation_checks = debug_run

from genai_otel import instrument

instrument(
    service_name="pii-final-test",
    endpoint="http://192.168.206.128:55681",
    enable_pii_detection=True,
    pii_mode="detect",
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\nSending prompt with PII...\n")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "My email is test@example.com and SSN is 123-45-6789"}],
)

print(f"\nResponse: {response.choices[0].message.content}")

import time

time.sleep(3)
print("\nCheck Jaeger for PII attributes!")
