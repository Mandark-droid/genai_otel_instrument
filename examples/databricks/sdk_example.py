# isort: skip_file

"""Trace a native Databricks Model Serving query through the Databricks SDK.

Install with ``pip install 'genai-otel-instrument[databricks]'`` and configure
authentication with the standard Databricks SDK environment or config file.
"""

import genai_otel

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

genai_otel.instrument(enabled_instrumentors=["databricks"])

workspace = WorkspaceClient()
response = workspace.serving_endpoints.query(
    name="my-chat-endpoint",
    messages=[ChatMessage(role=ChatMessageRole.USER, content="What is OpenTelemetry?")],
)
print(response)
