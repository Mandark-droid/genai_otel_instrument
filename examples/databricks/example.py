"""Trace chat requests to Databricks Model Serving or Foundation Model APIs.

Install with ``pip install 'genai-otel-instrument[databricks]'`` and set
``DATABRICKS_HOST`` (workspace URL), ``DATABRICKS_TOKEN``, and
``DATABRICKS_MODEL`` (the serving endpoint or model-service name).
"""

import os

import genai_otel

# Enable the Databricks detector explicitly; it recognizes supported workspace
# routes and assigns their telemetry to Databricks instead of OpenAI.
genai_otel.instrument(enabled_instrumentors=["databricks"])

from openai import OpenAI  # noqa: E402

host = os.environ["DATABRICKS_HOST"].rstrip("/")
client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{host}/serving-endpoints",
)

response = client.chat.completions.create(
    model=os.environ["DATABRICKS_MODEL"],
    messages=[{"role": "user", "content": "What is OpenTelemetry? Answer briefly."}],
    max_tokens=100,
)
print(response.choices[0].message.content)
