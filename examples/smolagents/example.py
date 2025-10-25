"""Smolagents Example - Lightweight Agents Framework

Smolagents is HuggingFace's minimalist agentic framework.
genai-otel-instrument uses OpenInference's SmolagentsInstrumentor for automatic tracing.
"""

import genai_otel

genai_otel.instrument()

from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

model = InferenceClientModel()
agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True)

result = agent.run(
    "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
)

print(f"Response: {result}")
print("[SUCCESS] Smolagents calls instrumented via OpenInference!")
