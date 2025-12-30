"""
Example: Hallucination Detection with Mistral AI

This example demonstrates hallucination detection working with Mistral AI's API.
Shows that advanced evaluation features work across different LLM providers.

Requirements:
    pip install genai-otel-instrument mistralai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export MISTRAL_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with hallucination detection
instrument(
    service_name="mistral-hallucination-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_hallucination_detection=True,
    hallucination_threshold=0.6,
)

# Now import Mistral after instrumentation is set up
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    print("Error: mistralai not installed. Run: pip install mistralai")
    exit(1)

api_key = os.getenv("MISTRAL_API_KEY", "dummy-key-for-demo")
client = MistralClient(api_key=api_key)

print("=" * 80)
print("Hallucination Detection with Mistral AI")
print("=" * 80)
print("\nDemonstrating hallucination detection across providers...\n")

# Test cases - responses that might contain hallucinations
test_cases = [
    {
        "name": "Request with Citations",
        "prompt": "According to the latest research, what are the benefits of meditation? Please cite sources.",
        "risk": "Medium - May fabricate citations",
    },
    {
        "name": "Factual Question",
        "prompt": "What is the speed of light in vacuum?",
        "risk": "Low - Well-established fact",
    },
    {
        "name": "Vague Request",
        "prompt": "Tell me about that thing scientists discovered recently.",
        "risk": "High - Likely to hallucinate details",
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']} (Hallucination Risk: {test['risk']})")
    print("-" * 80)
    print(f"Prompt: '{test['prompt']}'")

    try:
        messages = [ChatMessage(role="user", content=test["prompt"])]
        response = client.chat(model="mistral-tiny", messages=messages, max_tokens=150)
        print(f"Response: {response.choices[0].message.content[:150]}...")
    except Exception as e:
        print(f"Note: API call failed (expected in demo mode): {e}")

    print()

print("=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("Provider: Mistral AI")
print("Evaluation Attributes:")
print("  - evaluation.hallucination.response.detected = true/false")
print("  - evaluation.hallucination.response.score = <0.0-1.0>")
print("  - evaluation.hallucination.response.indicators = [...]")
print("  - evaluation.hallucination.response.hedge_words = <count>")
print("\nMetrics:")
print("  - genai.evaluation.hallucination.detections (counter)")
print("  - genai.evaluation.hallucination.indicators (counter)")
print("  - genai.evaluation.hallucination.score (histogram)")
print("\nProvider Attributes:")
print("  - gen_ai.system = 'mistralai'")
print("  - gen_ai.request.model = 'mistral-tiny'")
print("\nNote: Hallucination detection analyzes RESPONSES, not prompts!")
print("=" * 80)
