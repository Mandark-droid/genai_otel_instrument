"""Arize AX Example with genai-otel-instrument

Exports GenAI spans to Arize AX over plain OTLP.

There is no Arize or OpenInference package involved here, and no Arize-specific
code in this file. The entire integration is standard OpenTelemetry environment
variables, because Arize AX ingests the `gen_ai.*` semantic conventions
natively and normalises them on its side.
"""

import os
import sys

import genai_otel

REQUIRED = ("ARIZE_SPACE_ID", "ARIZE_API_KEY", "OPENAI_API_KEY")
missing = [name for name in REQUIRED if not os.getenv(name)]
if missing:
    sys.exit(f"Missing environment variables: {', '.join(missing)}")

PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME", "traceverde-example")

# --- The entire Arize configuration: standard OTel variables only. ---
#
# The OTLP HTTP exporter appends /v1/traces to the base endpoint, so the base is
# the bare host. Use https://otlp.eu-west-1a.arize.com for EU or
# https://otlp.ca-central-1a.arize.com for Canada.
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otlp.arize.com")
os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
    f"space_id={os.environ['ARIZE_SPACE_ID']},api_key={os.environ['ARIZE_API_KEY']}"
)
# Arize needs a project name as a resource attribute. This is the one
# vendor-namespaced value in the setup: service.name alone is rejected with
# HTTP 500. OTEL_RESOURCE_ATTRIBUTES is standard OTel, so no code change is
# needed to supply it.
os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"openinference.project.name={PROJECT_NAME}"
os.environ.setdefault("OTEL_SERVICE_NAME", "traceverde-arize-example")
# ---------------------------------------------------------------------

genai_otel.instrument()

from openai import OpenAI  # noqa: E402

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is OpenTelemetry?"}],
    max_tokens=150,
)

print(f"Response: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}")

# Short-lived scripts must flush before exit, or the batch span processor is
# torn down with spans still queued and nothing reaches Arize.
from opentelemetry import trace  # noqa: E402

provider = trace.get_tracer_provider()
if hasattr(provider, "force_flush"):
    provider.force_flush(30000)

print(f"Spans exported to Arize AX project '{PROJECT_NAME}'.")
