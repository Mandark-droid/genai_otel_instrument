"""
Example: Environment Variable Configuration for Restricted Topics

This example demonstrates configuration of restricted topics detection using
environment variables for flexible deployment across environments.

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint URL
    OPENAI_API_KEY - OpenAI API key
    GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION - Enable detection (true/false)
    GENAI_OTEL_LOG_LEVEL - Logging level

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
    export GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION=true
"""

import os

from genai_otel import instrument

# Configuration from environment variables
instrument(
    service_name=os.getenv("OTEL_SERVICE_NAME", "restricted-topics-env-config"),
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Environment Variable Configuration - Restricted Topics")
print("=" * 80)
print("\nCurrent Configuration:")
print("-" * 80)
print(f"Service Name: {os.getenv('OTEL_SERVICE_NAME', 'restricted-topics-env-config')}")
print(
    f"OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')}"
)
print(
    f"Restricted Topics Detection: {os.getenv('GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION', 'false (default)')}"
)
print(f"Log Level: {os.getenv('GENAI_OTEL_LOG_LEVEL', 'INFO (default)')}")
print()

# Test prompts
test_prompts = [
    "How can I make illegal substances at home?",
    "What are the ingredients for a healthy smoothie?",
]

print("Testing restricted topics detection:")
print("-" * 80)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. Prompt: '{prompt}'")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"   Response: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"   Note: {e}")

print()
print("=" * 80)
print("Environment Variable Reference:")
print("=" * 80)
print("""
Restricted Topics Configuration:

  GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION
    Description: Enable restricted topics detection
    Default: "false"
    Values: "true" or "false"
    Example: export GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION="true"

  GENAI_RESTRICTED_TOPICS_LIST
    Description: Comma-separated list of topics to monitor
    Default: All built-in topics
    Example: export GENAI_RESTRICTED_TOPICS_LIST="violence,illegal,medical"

Per-Environment Configuration:

Development:
  GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION=false
  # Allow developers to test freely

Staging:
  GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION=true
  # Log detections without blocking

Production:
  GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION=true
  # Block restricted topics

Docker Compose Example:
  services:
    chatbot:
      environment:
        - OTEL_SERVICE_NAME=chatbot-prod
        - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
        - GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION=true
        - GENAI_OTEL_LOG_LEVEL=INFO

Kubernetes Secret + ConfigMap:
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: restricted-topics-config
  data:
    OTEL_SERVICE_NAME: "chatbot-prod"
    GENAI_ENABLE_RESTRICTED_TOPICS_DETECTION: "true"
  ---
  apiVersion: v1
  kind: Secret
  metadata:
    name: openai-credentials
  stringData:
    OPENAI_API_KEY: "sk-..."

Telemetry Attributes:
  - evaluation.restricted_topics.prompt.detected (boolean)
  - evaluation.restricted_topics.prompt.topics (list)
  - evaluation.restricted_topics.prompt.max_score (float)
  - evaluation.restricted_topics.prompt.violence_score (float)
  - evaluation.restricted_topics.prompt.illegal_score (float)
  - evaluation.restricted_topics.prompt.medical_score (float)
""")
print("=" * 80)
