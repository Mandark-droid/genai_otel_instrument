"""
Example demonstrating Toxicity Detection with OpenTelemetry instrumentation.

This example shows:
1. Basic toxicity detection with Detoxify local model
2. Toxicity detection with Google Perspective API
3. Blocking mode for toxic content
4. Category-specific toxicity detection
5. Threshold configuration
6. Batch processing for efficiency

Toxicity detection helps ensure safe and respectful interactions in LLM applications
by identifying and handling toxic, harmful, or inappropriate content in prompts and responses.

Requirements:
    pip install genai-otel-instrument openai

    # Choose one or both detection methods:
    # 1. Local model (Detoxify - recommended for offline/privacy)
    pip install detoxify

    # 2. Cloud API (Perspective API - more accurate, requires API key)
    pip install google-api-python-client

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
    export GENAI_ENABLE_TOXICITY_DETECTION=true

    # Optional: For Perspective API
    export GENAI_TOXICITY_PERSPECTIVE_API_KEY=your_perspective_api_key
"""

import os

# Example 1: Basic Toxicity Detection with Detoxify (Local Model)
print("=" * 80)
print("Example 1: Basic Toxicity Detection - Detoxify Local Model")
print("=" * 80)

# Set up OpenTelemetry instrumentation with toxicity detection
from genai_otel import instrument

instrument(
    service_name="toxicity-detection-example",
    endpoint="http://localhost:4318",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    toxicity_use_perspective_api=False,  # Use local Detoxify model
)

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("\nSending prompt with potentially toxic content...")
print("Prompt: 'You are stupid and worthless'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "You are stupid and worthless. Can you help me?",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nNote: {e}")

print("\n✓ Check your telemetry backend for:")
print("  - evaluation.toxicity.prompt.detected = true")
print("  - evaluation.toxicity.prompt.max_score = <score>")
print("  - evaluation.toxicity.prompt.categories = ['toxicity', 'insult']")
print("  - evaluation.toxicity.prompt.toxicity_score = <score>")
print("  - evaluation.toxicity.prompt.insult_score = <score>")
print("  - Metrics: genai.evaluation.toxicity.detections")
print("  - Metrics: genai.evaluation.toxicity.categories (by category)")
print("  - Metrics: genai.evaluation.toxicity.score (histogram)")


# Example 2: Toxicity Detection with Perspective API
print("\n" + "=" * 80)
print("Example 2: Toxicity Detection - Google Perspective API")
print("=" * 80)

from genai_otel import instrument

# Re-instrument with Perspective API (requires API key)
instrument(
    service_name="toxicity-perspective-example",
    endpoint="http://localhost:4318",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    toxicity_use_perspective_api=True,  # Use Perspective API
    toxicity_perspective_api_key=os.getenv("GENAI_TOXICITY_PERSPECTIVE_API_KEY"),
)

print("\nPerspective API provides more accurate toxicity detection")
print("with machine learning models trained on real-world data.")
print("\nNote: Requires GENAI_TOXICITY_PERSPECTIVE_API_KEY environment variable")
print("Get your API key from: https://developers.perspectiveapi.com/")

if os.getenv("GENAI_TOXICITY_PERSPECTIVE_API_KEY"):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "This is a hateful and offensive message",
                }
            ],
        )
        print(f"\nResponse: {response.choices[0].message.content}")
    except Exception as e:
        print(f"\nNote: {e}")
else:
    print("\n⚠ Perspective API key not configured, skipping this example")


# Example 3: Blocking Mode
print("\n" + "=" * 80)
print("Example 3: Blocking Mode")
print("=" * 80)

from genai_otel import instrument

# Re-instrument with blocking enabled
instrument(
    service_name="toxicity-blocking-example",
    endpoint="http://localhost:4318",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
    toxicity_block_on_detection=True,  # Block toxic content
)

print("\nWith blocking mode enabled, requests/responses with toxic content")
print("are blocked and marked with ERROR status in telemetry.")

print("\nPrompt: 'Go kill yourself'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Go kill yourself",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nNote: In production, this would be blocked: {e}")

print("\n✓ Check your telemetry backend for:")
print("  - evaluation.toxicity.prompt.detected = true")
print("  - evaluation.toxicity.prompt.blocked = true")
print("  - Span status = ERROR")
print("  - Metrics: genai.evaluation.toxicity.blocked")


# Example 4: Category-Specific Detection
print("\n" + "=" * 80)
print("Example 4: Category-Specific Toxicity Detection")
print("=" * 80)

print("\nToxicity detection supports multiple categories:")
print("  - toxicity: General toxic language")
print("  - severe_toxicity: Extremely toxic content")
print("  - identity_attack: Attacks on protected characteristics")
print("  - insult: Insulting or demeaning language")
print("  - profanity: Swearing and obscene language")
print("  - threat: Threatening language")

from genai_otel import instrument

instrument(
    service_name="toxicity-categories-example",
    endpoint="http://localhost:4318",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)

print("\nExample texts and their likely categories:")

examples = [
    ("You're an idiot", "insult"),
    ("I hate all people from that country", "identity_attack"),
    ("F*** you!", "profanity"),
    ("I will hurt you", "threat"),
    ("You are worthless trash", "severe_toxicity"),
]

for text, expected_category in examples:
    print(f"\n  Text: '{text}'")
    print(f"  Expected: {expected_category} category")


# Example 5: Threshold Configuration
print("\n" + "=" * 80)
print("Example 5: Threshold Configuration")
print("=" * 80)

print("\nYou can adjust the toxicity threshold:")
print("  - Lower threshold (e.g., 0.5): More sensitive, may have false positives")
print("  - Higher threshold (e.g., 0.9): Less sensitive, only catches severe cases")

from genai_otel import instrument

# High sensitivity (lower threshold)
instrument(
    service_name="toxicity-threshold-low",
    endpoint="http://localhost:4318",
    enable_toxicity_detection=True,
    toxicity_threshold=0.5,  # More sensitive
)

print("\nWith threshold=0.5, even borderline cases are detected")

# Low sensitivity (higher threshold)
instrument(
    service_name="toxicity-threshold-high",
    endpoint="http://localhost:4318",
    enable_toxicity_detection=True,
    toxicity_threshold=0.9,  # Less sensitive
)

print("\nWith threshold=0.9, only severe toxicity is detected")


# Example 6: Response Toxicity Detection
print("\n" + "=" * 80)
print("Example 6: Response Toxicity Detection")
print("=" * 80)

from genai_otel import instrument

# Re-instrument for response detection
instrument(
    service_name="toxicity-response-example",
    endpoint="http://localhost:4318",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)

print("\nToxicity detection also checks LLM responses...")
print("If the LLM accidentally generates toxic content,")
print("it will be detected and flagged in telemetry.")

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
    print(f"\nResponse: {response.choices[0].message.content}")
    print("\n✓ If the response is toxic, you'll see:")
    print("  - evaluation.toxicity.response.detected = true")
    print("  - evaluation.toxicity.response.categories = [...]")
except Exception as e:
    print(f"\nNote: {e}")


# Example 7: Environment Variable Configuration
print("\n" + "=" * 80)
print("Example 7: Environment Variable Configuration")
print("=" * 80)

print("\nYou can also configure toxicity detection via environment variables:")
print(
    """
# Enable toxicity detection
export GENAI_ENABLE_TOXICITY_DETECTION=true

# Set detection threshold (0.0-1.0)
export GENAI_TOXICITY_THRESHOLD=0.8

# Use Perspective API (requires API key)
export GENAI_TOXICITY_USE_PERSPECTIVE_API=true
export GENAI_TOXICITY_PERSPECTIVE_API_KEY=your_api_key_here

# Block toxic content
export GENAI_TOXICITY_BLOCK_ON_DETECTION=true
"""
)

print("\nThen simply instrument without parameters:")
print(
    """
from genai_otel import instrument
instrument(service_name="my-app")
"""
)


# Example 8: Combining with PII Detection
print("\n" + "=" * 80)
print("Example 8: Combining Toxicity and PII Detection")
print("=" * 80)

from genai_otel import instrument

# Enable both PII and toxicity detection
instrument(
    service_name="combined-detection-example",
    endpoint="http://localhost:4318",
    enable_pii_detection=True,
    pii_mode="redact",
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)

print("\nYou can enable multiple safety features simultaneously:")
print("  - PII Detection: Protect sensitive data")
print("  - Toxicity Detection: Prevent harmful content")
print("  - (Coming Soon) Bias Detection, Prompt Injection, etc.")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "You idiot, my email is john@example.com",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
    print("\n✓ This will detect:")
    print("  - PII (email address)")
    print("  - Toxicity (insult)")
except Exception as e:
    print(f"\nNote: {e}")


# Summary
print("\n" + "=" * 80)
print("Summary: Toxicity Detection Features")
print("=" * 80)

print(
    """
✓ Detection Methods:
  - Detoxify: Local ML model (offline, privacy-friendly)
  - Perspective API: Cloud-based (more accurate, requires API key)
  - Automatic fallback: Perspective API → Detoxify on errors

✓ Toxicity Categories:
  - toxicity: General toxic language
  - severe_toxicity: Extremely harmful content
  - identity_attack: Discrimination and hate speech
  - insult: Insulting or demeaning language
  - profanity: Swearing and obscene content
  - threat: Threatening or violent language

✓ Detection Modes:
  - Monitor: Detect and report only (default)
  - Block: Prevent toxic content from being processed

✓ Configuration Options:
  - Threshold: Sensitivity (0.0-1.0)
  - Block on detection: Stop processing toxic content
  - Category selection: Choose which categories to check
  - Detection method: Perspective API, Detoxify, or both

✓ Telemetry Attributes:
  - evaluation.toxicity.prompt.detected
  - evaluation.toxicity.response.detected
  - evaluation.toxicity.*.max_score
  - evaluation.toxicity.*.categories
  - evaluation.toxicity.*.<category>_score
  - evaluation.toxicity.*.blocked

✓ Metrics:
  - genai.evaluation.toxicity.detections
  - genai.evaluation.toxicity.categories (by category)
  - genai.evaluation.toxicity.blocked
  - genai.evaluation.toxicity.score (histogram)

✓ Use Cases:
  - Content moderation for user-facing chatbots
  - Workplace collaboration tools
  - Educational platforms
  - Social media and community forums
  - Customer support systems

✓ Dependencies:
  - Detoxify: pip install detoxify
  - Perspective API: pip install google-api-python-client
  - Get Perspective API key: https://developers.perspectiveapi.com/

For more information, see:
- Detoxify: https://github.com/unitaryai/detoxify
- Perspective API: https://perspectiveapi.com/
"""
)

print("\n" + "=" * 80)
print("All examples completed!")
print("=" * 80)
