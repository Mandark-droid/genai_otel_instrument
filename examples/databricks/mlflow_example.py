"""Trace Databricks Model Serving calls made with MLflow's deployment client.

Install with ``pip install 'genai-otel-instrument[databricks]'`` and configure
Databricks authentication with the standard environment or config file.
"""

import genai_otel

genai_otel.instrument(enabled_instrumentors=["databricks"])

from mlflow.deployments import get_deploy_client  # noqa: E402

client = get_deploy_client("databricks")
endpoint = "my-chat-endpoint"
inputs = {"messages": [{"role": "user", "content": "What is OpenTelemetry?"}]}

# Non-streaming inference records endpoint identity and response token usage.
response = client.predict(endpoint=endpoint, inputs=inputs)
print(response)

# Streaming spans end when iteration completes and capture usage reported by
# the final chunk (when the endpoint includes usage in its stream response).
for chunk in client.predict_stream(endpoint=endpoint, inputs=inputs):
    print(chunk)
