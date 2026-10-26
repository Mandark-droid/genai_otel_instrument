"""Trace an Agno agent run, including model usage and tool activity.

Install with: pip install 'genai-otel-instrument[agno]'
Set OPENAI_API_KEY in the environment before running.
"""

import genai_otel

genai_otel.instrument(
    service_name="agno-example",
    enabled_instrumentors=["agno"],
    exporter_type="console",
    enable_content_capture=False,
)

from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), markdown=False)
agent.print_response("In one sentence, what does OpenTelemetry do?")

genai_otel.flush_telemetry()
