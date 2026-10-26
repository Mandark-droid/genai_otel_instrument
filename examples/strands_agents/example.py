"""Trace a Strands Agents run using its native OpenTelemetry spans.

Install with: pip install 'genai-otel-instrument[strands-agents]'
Set the credentials required by the configured Strands model provider.
"""

import genai_otel

genai_otel.instrument(
    service_name="strands-agents-example",
    enabled_instrumentors=["strands_agents"],
    exporter_type="console",
    enable_content_capture=False,
)

from strands import Agent

agent = Agent()
result = agent("In one sentence, what does OpenTelemetry do?")
print(result)

genai_otel.flush_telemetry()
