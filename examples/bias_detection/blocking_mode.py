"""
Example: Blocking Mode - Prevent Biased Requests

This example demonstrates how to block requests that contain detected bias
by setting a threshold and using the detection results to prevent biased
content from being processed.

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Configure bias detection with lower threshold for strict mode
instrument(
    service_name="bias-blocking-mode",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_bias_detection=True,
    bias_threshold=0.3,  # Lower threshold = more sensitive
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Blocking Mode - Bias Detection")
print("=" * 80)
print("\nBlocking requests with detected bias (threshold: 0.3)...")
print()


def process_with_bias_check(prompt, category):
    """Process prompt with bias detection and blocking"""
    print(f"Category: {category}")
    print("-" * 80)
    print(f"Prompt: '{prompt}'")

    # In a real implementation, you would check the bias detection
    # result from the span attributes before allowing the request
    # For demonstration, we'll show both blocked and allowed cases

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"Status: PROCESSED")
        print(f"Response: {response.choices[0].message.content[:150]}...")
    except Exception as e:
        print(f"Status: BLOCKED - {e}")

    print()


# Example 1: Should be blocked - Clear gender bias
process_with_bias_check(
    "Why are women always so emotional and irrational in the workplace?",
    "Gender Bias (BLOCKED)",
)

# Example 2: Should be blocked - Clear racial bias
process_with_bias_check(
    "All Black people are naturally good at sports but not academics.",
    "Racial Bias (BLOCKED)",
)

# Example 3: Should be allowed - Neutral query
process_with_bias_check(
    "What are effective strategies for building diverse and inclusive teams?",
    "Neutral Content (ALLOWED)",
)

# Example 4: Should be blocked - Age bias
process_with_bias_check(
    "We shouldn't hire anyone over 50, they're too slow to learn new things.",
    "Age Bias (BLOCKED)",
)

# Example 5: Should be allowed - Academic discussion
process_with_bias_check(
    "What are the historical factors that led to gender disparities in STEM fields?",
    "Academic Discussion (ALLOWED)",
)

print("=" * 80)
print("Blocking Implementation Notes:")
print("=" * 80)
print(
    """
In production, you would:

1. Check span attributes after the request:
   - evaluation.bias.prompt.detected = true/false
   - evaluation.bias.prompt.max_score > threshold

2. Implement blocking logic:
   if bias_detected and max_score > threshold:
       raise BiasDetectedError("Request blocked: bias detected")

3. Log blocked requests for review:
   - Track which bias types are most common
   - Analyze false positives
   - Adjust thresholds based on use case

4. Provide user feedback:
   - Explain why the request was blocked
   - Suggest rephrasing
   - Link to bias-free communication guidelines

Telemetry will show:
  - evaluation.bias.prompt.detected = true
  - evaluation.bias.prompt.max_score = <score>
  - evaluation.bias.prompt.detected_biases = [list]
  - genai.evaluation.bias.detections (counter)
"""
)
print("=" * 80)
