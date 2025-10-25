"""AWS Bedrock Example with genai-otel-instrument

This example demonstrates how to use genai-otel-instrument with AWS Bedrock.
"""

import genai_otel

# Auto-instrument AWS Bedrock
genai_otel.instrument()

import json

# Now use AWS Bedrock normally
import boto3

# Create Bedrock Runtime client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Invoke Claude on Bedrock
body = json.dumps(
    {
        "prompt": "\n\nHuman: Explain what distributed tracing is in one sentence.\n\nAssistant:",
        "max_tokens_to_sample": 100,
        "temperature": 0.7,
    }
)

response = bedrock.invoke_model(
    body=body,
    modelId="anthropic.claude-v2",
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
print(f"Response: {response_body.get('completion')}")
print("✅ Traces and metrics have been automatically sent to your OTLP endpoint!")
