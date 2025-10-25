"""Azure OpenAI Example with genai-otel-instrument

This example demonstrates how to use genai-otel-instrument with Azure OpenAI.
"""

import genai_otel

# Auto-instrument Azure OpenAI
genai_otel.instrument()

import os

# Now use Azure OpenAI normally
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential

# Create Azure OpenAI client
client = OpenAIClient(
    endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    credential=AzureKeyCredential(os.environ.get("AZURE_OPENAI_KEY")),
)

# Make a completion request
response = client.complete(
    deployment_id="your-deployment-name", prompt="Explain OpenTelemetry in one sentence."
)

print(f"Response: {response.choices[0].text}")
print("[SUCCESS] Traces and metrics have been automatically sent to your OTLP endpoint!")
