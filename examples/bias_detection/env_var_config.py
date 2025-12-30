"""
Example: Environment Variable Configuration

This example demonstrates how to configure bias detection entirely through
environment variables, which is useful for:
- Container deployments (Docker, Kubernetes)
- CI/CD pipelines
- Different configurations per environment (dev/staging/prod)
- No code changes needed for configuration

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint URL
    OPENAI_API_KEY - OpenAI API key
    GENAI_ENABLE_BIAS_DETECTION - Enable bias detection (true/false)
    GENAI_BIAS_THRESHOLD - Detection threshold (0.0-1.0)
    GENAI_OTEL_LOG_LEVEL - Logging level (DEBUG/INFO/WARNING/ERROR)

Requirements:
    pip install genai-otel-instrument openai

Environment Setup (Linux/Mac):
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
    export GENAI_ENABLE_BIAS_DETECTION=true
    export GENAI_BIAS_THRESHOLD=0.5
    export GENAI_OTEL_LOG_LEVEL=INFO

Environment Setup (Windows):
    set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    set OPENAI_API_KEY=your_api_key_here
    set GENAI_ENABLE_BIAS_DETECTION=true
    set GENAI_BIAS_THRESHOLD=0.5
    set GENAI_OTEL_LOG_LEVEL=INFO
"""

import os

from genai_otel import instrument

# Configuration entirely from environment variables
# instrument() will read GENAI_ENABLE_BIAS_DETECTION and GENAI_BIAS_THRESHOLD
instrument(
    service_name=os.getenv("OTEL_SERVICE_NAME", "bias-env-config-example"),
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Environment Variable Configuration")
print("=" * 80)
print("\nCurrent Configuration:")
print("-" * 80)
print(f"Service Name: {os.getenv('OTEL_SERVICE_NAME', 'bias-env-config-example')}")
print(f"OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')}")
print(f"Bias Detection: {os.getenv('GENAI_ENABLE_BIAS_DETECTION', 'true (default)')}")
print(f"Bias Threshold: {os.getenv('GENAI_BIAS_THRESHOLD', '0.5 (default)')}")
print(f"Log Level: {os.getenv('GENAI_OTEL_LOG_LEVEL', 'INFO (default)')}")
print()

# Test examples
examples = [
    "Women are too emotional to be good engineers.",
    "Older workers can't keep up with new technology.",
    "What are best practices for inclusive team management?",
]

print("Testing bias detection with environment configuration:")
print("-" * 80)

for i, prompt in enumerate(examples, 1):
    print(f"\n{i}. Prompt: '{prompt}'")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"   Response: {response.choices[0].message.content[:150]}...")
    except Exception as e:
        print(f"   Note: {e}")

print()
print("=" * 80)
print("Environment Variable Reference:")
print("=" * 80)
print(
    """
Core OpenTelemetry Settings:
  OTEL_SERVICE_NAME
    Description: Name of your service
    Default: "genai-app"
    Example: export OTEL_SERVICE_NAME="my-chatbot"

  OTEL_EXPORTER_OTLP_ENDPOINT
    Description: OTLP collector endpoint
    Default: "http://localhost:4318"
    Example: export OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4318"

  OTEL_EXPORTER_OTLP_HEADERS
    Description: Headers for OTLP exporter
    Default: None
    Example: export OTEL_EXPORTER_OTLP_HEADERS="api-key=abc123,region=us-west"

  OTEL_EXPORTER_OTLP_PROTOCOL
    Description: OTLP protocol (http/protobuf or grpc)
    Default: "http/protobuf"
    Example: export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"

  OTEL_SERVICE_INSTANCE_ID
    Description: Unique instance identifier
    Default: Auto-generated UUID
    Example: export OTEL_SERVICE_INSTANCE_ID="pod-123"

  OTEL_ENVIRONMENT
    Description: Environment name
    Default: None
    Example: export OTEL_ENVIRONMENT="production"

Bias Detection Settings:
  GENAI_ENABLE_BIAS_DETECTION
    Description: Enable/disable bias detection
    Default: "true"
    Values: "true" or "false"
    Example: export GENAI_ENABLE_BIAS_DETECTION="true"

  GENAI_BIAS_THRESHOLD
    Description: Bias detection sensitivity threshold
    Default: "0.5"
    Range: 0.0-1.0 (lower = more sensitive)
    Example: export GENAI_BIAS_THRESHOLD="0.3"

Other Evaluation Settings:
  GENAI_ENABLE_PII_DETECTION
    Description: Enable PII detection
    Default: "false"
    Example: export GENAI_ENABLE_PII_DETECTION="true"

  GENAI_ENABLE_TOXICITY_DETECTION
    Description: Enable toxicity detection
    Default: "false"
    Example: export GENAI_ENABLE_TOXICITY_DETECTION="true"

  GENAI_TOXICITY_THRESHOLD
    Description: Toxicity detection threshold
    Default: "0.7"
    Example: export GENAI_TOXICITY_THRESHOLD="0.8"

General Settings:
  GENAI_ENABLE_COST_TRACKING
    Description: Enable cost calculation
    Default: "true"
    Example: export GENAI_ENABLE_COST_TRACKING="false"

  GENAI_ENABLE_GPU_METRICS
    Description: Enable GPU metrics collection
    Default: "true"
    Example: export GENAI_ENABLE_GPU_METRICS="false"

  GENAI_OTEL_LOG_LEVEL
    Description: Logging verbosity
    Default: "INFO"
    Values: DEBUG, INFO, WARNING, ERROR
    Example: export GENAI_OTEL_LOG_LEVEL="DEBUG"

Docker Example (.env file):
  OTEL_SERVICE_NAME=my-chatbot
  OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
  OTEL_ENVIRONMENT=production
  GENAI_ENABLE_BIAS_DETECTION=true
  GENAI_BIAS_THRESHOLD=0.4
  OPENAI_API_KEY=sk-...

Kubernetes ConfigMap Example:
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: bias-detection-config
  data:
    OTEL_SERVICE_NAME: "my-chatbot"
    OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector:4318"
    GENAI_ENABLE_BIAS_DETECTION: "true"
    GENAI_BIAS_THRESHOLD: "0.4"
"""
)
print("=" * 80)
