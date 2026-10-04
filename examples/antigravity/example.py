"""Google Antigravity SDK with genai-otel-instrument tracing.

Install with ``pip install 'genai-otel-instrument[antigravity]'`` and set
``GEMINI_API_KEY`` before running this example.
"""

import asyncio

import genai_otel

genai_otel.instrument(
    service_name="antigravity-example",
    enabled_instrumentors=["antigravity"],
    exporter_type="console",
    enable_gpu_metrics=False,
)

from google.antigravity import Agent, LocalAgentConfig


async def main():
    async with Agent(LocalAgentConfig()) as agent:
        response = await agent.chat("Explain OpenTelemetry in one sentence.")
        print(await response.text())

    genai_otel.flush_telemetry()


if __name__ == "__main__":
    asyncio.run(main())
