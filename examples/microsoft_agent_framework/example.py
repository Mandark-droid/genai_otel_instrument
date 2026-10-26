"""Trace a Microsoft Agent Framework run with native OpenTelemetry spans.

Install with: pip install 'genai-otel-instrument[microsoft-agent-framework]'
Configure credentials for the model client you choose.
"""

import asyncio

import genai_otel

genai_otel.instrument(
    service_name="microsoft-agent-framework-example",
    enabled_instrumentors=["microsoft_agent_framework"],
    exporter_type="console",
    enable_content_capture=False,
)

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Answer in one concise sentence.",
        name="concise-assistant",
    )
    result = await agent.run("What does OpenTelemetry do?")
    print(result.text)


asyncio.run(main())

genai_otel.flush_telemetry()
