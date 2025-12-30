"""
Example: Basic Hallucination Detection

This example demonstrates how to detect potential hallucinations in LLM responses,
including unsupported claims, lack of citations, and inconsistencies.

Hallucination Indicators:
- Lack of citations for factual claims
- Excessive hedge words ("might", "could be", "possibly")
- Specific numbers/dates without sources
- Contradictions within the response
- Overconfidence in uncertain information

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Set up OpenTelemetry instrumentation with hallucination detection
instrument(
    service_name="hallucination-detection-basic",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_hallucination_detection=True,
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Basic Hallucination Detection")
print("=" * 80)
print("\nDetecting potential hallucinations in LLM responses...")
print()

# Example 1: Factual query (may hallucinate)
print("Example 1: Factual Query")
print("-" * 80)
print("Prompt: 'When was the Eiffel Tower built?'")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "When was the Eiffel Tower built?"}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\nAnalysis: Check for citations, specific dates")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 2: Specific statistics (high hallucination risk)
print("Example 2: Specific Statistics")
print("-" * 80)
print("Prompt: 'What percentage of Fortune 500 companies use AI?'")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What percentage of Fortune 500 companies use AI?",
            }
        ],
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\nAnalysis: Check for sources, hedge words, date context")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 3: Recent events (may be outdated)
print("Example 3: Recent Events")
print("-" * 80)
print("Prompt: 'Who won the latest World Cup?'")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Who won the latest World Cup?"}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\nAnalysis: Check knowledge cutoff date, confidence level")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 4: Open-ended opinion (lower risk)
print("Example 4: Open-Ended Opinion")
print("-" * 80)
print("Prompt: 'What makes a good team leader?'")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What makes a good team leader?"}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\nAnalysis: Opinions have lower hallucination risk")
except Exception as e:
    print(f"Note: {e}")

print()
print("=" * 80)
print("Check your telemetry backend for:")
print("=" * 80)
print("""
Hallucination Indicators:
  - evaluation.hallucination.response.detected = true/false
  - evaluation.hallucination.response.score = <0.0-1.0>
  - evaluation.hallucination.response.citations = <count>
  - evaluation.hallucination.response.hedge_words = <count>
  - evaluation.hallucination.response.claims = <count>
  - evaluation.hallucination.response.indicators = [list of indicators]

Common Indicators:
  - no_citations: Response makes factual claims without sources
  - excessive_hedging: Too many "might", "could", "possibly"
  - specific_numbers: Precise statistics without attribution
  - temporal_claims: Dates/events that may be outdated
  - contradictions: Inconsistent statements within response

Metrics:
  - genai.evaluation.hallucination.detections (counter)
  - genai.evaluation.hallucination.score (histogram)
  - genai.evaluation.hallucination.indicators (counter, by indicator)
""")
print("=" * 80)
