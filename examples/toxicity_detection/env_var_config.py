"""
Example: Toxicity Detection via Environment Variables

This example demonstrates how to configure toxicity detection entirely through
environment variables, without passing parameters to instrument().

This is useful for:
- Container/cloud deployments (Kubernetes, Docker)
- CI/CD pipelines
- Environment-specific configurations
- Zero-code configuration changes

Requirements:
    pip install genai-otel-instrument openai
    pip install detoxify  # For local detection
    pip install google-api-python-client  # Optional: For Perspective API

Environment Setup:
    # Enable toxicity detection
    export GENAI_ENABLE_TOXICITY_DETECTION=true

    # Set detection threshold (0.0-1.0)
    export GENAI_TOXICITY_THRESHOLD=0.8

    # Use Perspective API (requires API key)
    export GENAI_TOXICITY_USE_PERSPECTIVE_API=true
    export GENAI_TOXICITY_PERSPECTIVE_API_KEY=your_api_key_here

    # Block toxic content
    export GENAI_TOXICITY_BLOCK_ON_DETECTION=true

    # OpenTelemetry configuration
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Simple instrumentation - all configuration comes from environment variables
instrument(service_name="toxicity-env-config-example")

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Environment Variable Configuration")
print("=" * 80)
print("\nConfiguration loaded from environment variables:")
print(
    f"  - GENAI_ENABLE_TOXICITY_DETECTION: {os.getenv('GENAI_ENABLE_TOXICITY_DETECTION', 'false')}"
)
print(f"  - GENAI_TOXICITY_THRESHOLD: {os.getenv('GENAI_TOXICITY_THRESHOLD', '0.7')}")
print(
    f"  - GENAI_TOXICITY_USE_PERSPECTIVE_API: {os.getenv('GENAI_TOXICITY_USE_PERSPECTIVE_API', 'false')}"
)
print(
    f"  - GENAI_TOXICITY_BLOCK_ON_DETECTION: {os.getenv('GENAI_TOXICITY_BLOCK_ON_DETECTION', 'false')}"
)

print("\nSending test prompt with toxic content...")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "You're terrible at your job",
            }
        ],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nNote: {e}")

print("\n" + "=" * 80)
print("Benefits of Environment Variable Configuration:")
print("=" * 80)
print("  1. Zero code changes for different environments")
print("  2. Easy integration with container orchestration")
print("  3. Centralized configuration management")
print("  4. Secure credential handling (API keys not in code)")
print("  5. Easy A/B testing and gradual rollout")
print("\nExample Docker Compose configuration:")
print(
    """
services:
  app:
    environment:
      - GENAI_ENABLE_TOXICITY_DETECTION=true
      - GENAI_TOXICITY_THRESHOLD=0.8
      - GENAI_TOXICITY_BLOCK_ON_DETECTION=true
"""
)
print("\nTo change configuration, just update environment variables and restart")
print("=" * 80)
