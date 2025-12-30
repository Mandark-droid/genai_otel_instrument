"""
Example: Toxicity Detection in LLM Responses

This example demonstrates toxicity detection applied to LLM responses.
While LLMs are typically designed to be helpful and harmless, they may
occasionally generate content that could be perceived as toxic, especially
when:
- Responding to adversarial prompts
- Generating fictional content
- Explaining controversial topics

This feature ensures comprehensive safety monitoring.

Requirements:
    pip install genai-otel-instrument openai
    pip install detoxify

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation for response detection
instrument(
    service_name="toxicity-response-example",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Response Toxicity Detection")
print("=" * 80)
print("\nToxicity detection checks both prompts AND responses.")
print("If the LLM accidentally generates toxic content,")
print("it will be detected and flagged in telemetry.")

# Test with a prompt that might elicit strong language in response
print("\nTest 1: Asking LLM to write an insulting message...")
print("Prompt: 'Write an insulting message'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Write an insulting message",
            }
        ],
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\nLLM likely refused, but if it complied, toxicity would be detected")
except Exception as e:
    print(f"Note: {e}")

# Test with controversial topic explanation
print("\n" + "-" * 80)
print("\nTest 2: Explaining a controversial topic...")
print("Prompt: 'Explain why some people use hate speech'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Explain why some people use hate speech",
            }
        ],
    )
    print(f"Response: {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"Note: {e}")

print("\n" + "=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("  - evaluation.toxicity.response.detected (if toxic response generated)")
print("  - evaluation.toxicity.response.max_score")
print("  - evaluation.toxicity.response.categories")
print("  - evaluation.toxicity.response.<category>_score")
print("\nNote: Both prompt AND response attributes are available")
print("  - evaluation.toxicity.prompt.* - Input toxicity")
print("  - evaluation.toxicity.response.* - Output toxicity")
print("\nThis comprehensive monitoring ensures:")
print("  - Toxic prompts are flagged")
print("  - Toxic responses are caught (even if unintended)")
print("  - Full audit trail for safety compliance")
print("=" * 80)
