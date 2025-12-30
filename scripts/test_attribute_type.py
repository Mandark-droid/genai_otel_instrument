"""Check the actual type of gen_ai.request.first_message attribute"""

# Patch span processor
import genai_otel.evaluation.span_processor as sp
from genai_otel import instrument

original_on_end = sp.EvaluationSpanProcessor.on_end


def debug_on_end(self, span):
    if hasattr(span, "attributes"):
        attrs = dict(span.attributes)
        if "gen_ai.request.first_message" in attrs:
            value = attrs["gen_ai.request.first_message"]
            print(f"\nAttribute value: {value}")
            print(f"Attribute type: {type(value)}")
            print(f"Is string: {isinstance(value, str)}")
            print(f"Is dict: {isinstance(value, dict)}")

            # If it's a string, try to parse it
            if isinstance(value, str):
                import json

                try:
                    parsed = json.loads(value.replace("'", '"'))
                    print(f"Parsed as JSON: {parsed}")
                    print(f"Content: {parsed.get('content', 'N/A')}")
                except Exception as e:
                    print(f"JSON parse failed: {e}")
                    # Try eval (careful!)
                    try:
                        import ast

                        parsed = ast.literal_eval(value)
                        print(f"Parsed with ast: {parsed}")
                        print(f"Content: {parsed.get('content', 'N/A')}")
                    except Exception as e2:
                        print(f"ast parse failed: {e2}")

    original_on_end(self, span)


sp.EvaluationSpanProcessor.on_end = debug_on_end

instrument(
    service_name="attribute-type-test",
    endpoint="http://192.168.206.128:4318",
    enable_pii_detection=True,
)

import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
)

print(f"\nResponse: {response.choices[0].message.content}")

import time

time.sleep(2)
