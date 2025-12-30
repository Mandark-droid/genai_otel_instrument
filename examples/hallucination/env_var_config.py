"""
Example: Environment Variable Configuration for Hallucination Detection

This example demonstrates configuration of hallucination detection using
environment variables for flexible deployment.

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint URL
    OPENAI_API_KEY - OpenAI API key
    GENAI_ENABLE_HALLUCINATION_DETECTION - Enable detection (true/false)
    GENAI_OTEL_LOG_LEVEL - Logging level

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
    export GENAI_ENABLE_HALLUCINATION_DETECTION=true
"""

import os

from genai_otel import instrument

# Configuration from environment variables
instrument(
    service_name=os.getenv("OTEL_SERVICE_NAME", "hallucination-env-config"),
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Environment Variable Configuration - Hallucination Detection")
print("=" * 80)
print("\nCurrent Configuration:")
print("-" * 80)
print(f"Service Name: {os.getenv('OTEL_SERVICE_NAME', 'hallucination-env-config')}")
print(f"OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')}")
print(
    f"Hallucination Detection: {os.getenv('GENAI_ENABLE_HALLUCINATION_DETECTION', 'false (default)')}"
)
print(f"Log Level: {os.getenv('GENAI_OTEL_LOG_LEVEL', 'INFO (default)')}")
print()

# Test prompts
test_prompts = [
    "What percentage of the Earth is covered by water?",
    "What are some healthy habits for daily life?",
]

print("Testing hallucination detection:")
print("-" * 80)

for i, prompt in enumerate(test_prompts, 1):
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
Hallucination Detection Configuration:

  GENAI_ENABLE_HALLUCINATION_DETECTION
    Description: Enable hallucination detection
    Default: "false"
    Values: "true" or "false"
    Example: export GENAI_ENABLE_HALLUCINATION_DETECTION="true"

Per-Environment Configuration:

Development:
  GENAI_ENABLE_HALLUCINATION_DETECTION=true
  # Monitor but don't block, learn patterns

Staging:
  GENAI_ENABLE_HALLUCINATION_DETECTION=true
  # Test detection accuracy before production

Production:
  GENAI_ENABLE_HALLUCINATION_DETECTION=true
  # Full monitoring with alerts

Docker Example:
  services:
    chatbot:
      environment:
        - OTEL_SERVICE_NAME=factual-qa-bot
        - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
        - GENAI_ENABLE_HALLUCINATION_DETECTION=true
        - GENAI_OTEL_LOG_LEVEL=INFO
        - OPENAI_API_KEY=${OPENAI_API_KEY}

Kubernetes Deployment:
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: chatbot
  spec:
    template:
      spec:
        containers:
        - name: chatbot
          envFrom:
          - configMapRef:
              name: hallucination-config
          - secretRef:
              name: openai-credentials

  ---
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: hallucination-config
  data:
    OTEL_SERVICE_NAME: "factual-qa-bot"
    OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector:4318"
    GENAI_ENABLE_HALLUCINATION_DETECTION: "true"

Telemetry Attributes:
  - evaluation.hallucination.response.detected (boolean)
  - evaluation.hallucination.response.score (float, 0.0-1.0)
  - evaluation.hallucination.response.citations (integer)
  - evaluation.hallucination.response.hedge_words (integer)
  - evaluation.hallucination.response.claims (integer)
  - evaluation.hallucination.response.indicators (list)
  - evaluation.hallucination.response.unsupported_claims (list)

Metrics:
  - genai.evaluation.hallucination.detections (counter)
  - genai.evaluation.hallucination.score (histogram)
  - genai.evaluation.hallucination.indicators (counter, by indicator)

Alerting Examples:

Prometheus AlertManager:
  groups:
  - name: hallucination_alerts
    rules:
    - alert: HighHallucinationRate
      expr: rate(genai_evaluation_hallucination_detections_total[5m]) > 0.2
      labels:
        severity: warning
      annotations:
        summary: High hallucination detection rate

    - alert: CriticalHallucinationScore
      expr: genai_evaluation_hallucination_score > 0.8
      labels:
        severity: critical
      annotations:
        summary: Critical hallucination score detected

Integration with Monitoring:
  - Send high hallucination scores to incident management
  - Create tickets for manual review
  - Dashboard showing hallucination trends
  - Breakdown by query type, model version, time of day
"""
)
print("=" * 80)
