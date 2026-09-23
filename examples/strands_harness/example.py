"""Minimal Strands Harness + genai-otel-instrument example.

Install the optional integration first:

    pip install 'genai-otel-instrument[strands]'

Set the model provider's credentials before running this example.  The normal
OpenTelemetry exporter configuration is reused; the console exporter below is
only for a quick local smoke test.
"""

import genai_otel

genai_otel.instrument(
    service_name="strands-harness-example",
    enabled_instrumentors=["strands"],
    exporter_type="console",
    enable_gpu_metrics=False,
)

from strands_harness import create_harness

agent = create_harness(
    instructions="Answer concisely and explain your reasoning.",
    session=True,
    memory=False,
)
result = agent("What is OpenTelemetry?")
print(result)

genai_otel.flush_telemetry()
