"""
Example: PII Detection via Environment Variables

This example demonstrates how to configure PII detection entirely through
environment variables, without passing parameters to instrument().

This is useful for:
- Container/cloud deployments (Kubernetes, Docker)
- CI/CD pipelines
- Environment-specific configurations
- Zero-code configuration changes

Requirements:
    pip install genai-otel-instrument openai
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_lg

Environment Setup:
    # Enable PII detection
    export GENAI_ENABLE_PII_DETECTION=true

    # Set detection mode (detect, redact, or block)
    export GENAI_PII_MODE=redact

    # Set confidence threshold (0.0-1.0)
    export GENAI_PII_THRESHOLD=0.8

    # Enable compliance modes
    export GENAI_PII_GDPR_MODE=true
    export GENAI_PII_HIPAA_MODE=true
    export GENAI_PII_PCI_DSS_MODE=true

    # OpenTelemetry configuration
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Simple instrumentation - all configuration comes from environment variables
instrument(service_name="pii-env-config-example")

# Now import OpenAI after instrumentation is set up
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Environment Variable Configuration")
print("=" * 80)
print("\nConfiguration loaded from environment variables:")
print(f"  - GENAI_ENABLE_PII_DETECTION: {os.getenv('GENAI_ENABLE_PII_DETECTION', 'false')}")
print(f"  - GENAI_PII_MODE: {os.getenv('GENAI_PII_MODE', 'detect')}")
print(f"  - GENAI_PII_THRESHOLD: {os.getenv('GENAI_PII_THRESHOLD', '0.7')}")
print(f"  - GENAI_PII_GDPR_MODE: {os.getenv('GENAI_PII_GDPR_MODE', 'false')}")
print(f"  - GENAI_PII_HIPAA_MODE: {os.getenv('GENAI_PII_HIPAA_MODE', 'false')}")
print(f"  - GENAI_PII_PCI_DSS_MODE: {os.getenv('GENAI_PII_PCI_DSS_MODE', 'false')}")

print("\nSending test prompt with PII...")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "My email is test@example.com and SSN is 123-45-6789",
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
print("  4. Secure credential handling (no hardcoded values)")
print("  5. Easy A/B testing and rollout")
print("\nTo change configuration, just update environment variables and restart")
print("=" * 80)
