"""Trace a Claude Agent SDK query, including agent and tool spans.

Install with: pip install 'genai-otel-instrument[claude-agent-sdk]'
Set ANTHROPIC_API_KEY in the environment before running.
"""

import genai_otel

genai_otel.instrument(
    service_name="claude-agent-sdk-example",
    enabled_instrumentors=["claude_agent_sdk"],
    exporter_type="console",
    enable_content_capture=False,
)

import asyncio

from claude_agent_sdk import ClaudeAgentOptions, query


async def main():
    async for message in query(
        prompt="List the Python files in the current directory.",
        options=ClaudeAgentOptions(allowed_tools=["Glob"]),
    ):
        if hasattr(message, "result"):
            print(message.result)


asyncio.run(main())
genai_otel.flush_telemetry()
