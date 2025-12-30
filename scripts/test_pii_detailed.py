"""
Detailed debug to check PII detection in span processor.
"""

import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Patch the span processor to add detailed logging
import genai_otel.evaluation.span_processor as sp

original_on_end = sp.EvaluationSpanProcessor.on_end


def debug_on_end(self, span):
    print(f"\n{'='*80}")
    print(f"[SPAN PROCESSOR] on_end called for span: {span.name}")

    # Get attributes
    if hasattr(span, "attributes"):
        attrs = dict(span.attributes)
        print(f"[SPAN PROCESSOR] Span has {len(attrs)} attributes")

        # Check for prompt attributes
        for key in attrs:
            if "message" in key.lower() or "prompt" in key.lower() or "content" in key.lower():
                print(f"[SPAN PROCESSOR]   - {key}: {str(attrs[key])[:100]}...")

        # Try to extract prompt
        prompt = self._extract_prompt(attrs)
        print(f"[SPAN PROCESSOR] Extracted prompt: {prompt[:100] if prompt else 'None'}...")

        if prompt and self.pii_detector:
            print(f"[SPAN PROCESSOR] Running PII detection...")
            from genai_otel.evaluation.pii_detector import PIIDetector

            result = self.pii_detector.detect(prompt)
            print(f"[SPAN PROCESSOR] PII detected: {result.has_pii}")
            if result.has_pii:
                print(f"[SPAN PROCESSOR] Entities: {result.entity_types}")
                print(f"[SPAN PROCESSOR] Entity count: {result.entity_counts}")

    # Call original
    original_on_end(self, span)
    print(f"{'='*80}\n")


sp.EvaluationSpanProcessor.on_end = debug_on_end

# Now run the instrumentation
from genai_otel import instrument

instrument(
    service_name="pii-detailed-debug",
    endpoint="http://192.168.206.128:4318",
    enable_pii_detection=True,
    pii_mode="detect",
    pii_threshold=0.7,
)

import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\nSending prompt with PII...\n")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "My email is test@example.com"}],
)

print(f"\nResponse: {response.choices[0].message.content}")

import time

time.sleep(3)
print("\nDone - check output above for span processor details")
