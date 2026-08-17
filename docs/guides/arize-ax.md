# Arize AX

TraceVerde exports to [Arize AX](https://arize.com) over plain OTLP.

There is no Arize integration in this library. There is no adapter, no
`arize-otel`, and no `openinference-instrumentation-*` package. Arize AX
consumes the OpenTelemetry GenAI semantic conventions natively, and TraceVerde
emits them, so the two interoperate through OTLP without a translation layer on
either side.

The entire configuration is four standard OpenTelemetry environment variables.

## Quick Start

```bash
pip install genai-otel-instrument[openai]
```

Get an API key and Space ID from [app.arize.com](https://app.arize.com)
(Settings -> API Keys). `ARIZE_SPACE_ID` needs the base64 Space **ID**, not the
space name.

```python
import os

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://otlp.arize.com"
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
    f"space_id={os.environ['ARIZE_SPACE_ID']},api_key={os.environ['ARIZE_API_KEY']}"
)
os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "openinference.project.name=my-project"

import genai_otel
genai_otel.instrument()

from openai import OpenAI

client = OpenAI()
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is OpenTelemetry?"}],
)
```

The project is created in Arize on first export. A runnable version of this is
in [`examples/arize_ax/`](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/arize_ax).

Set the variables as real environment variables rather than in Python if you
prefer - none of this needs to be in code:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.arize.com
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_HEADERS=space_id=<id>,api_key=<key>
export OTEL_RESOURCE_ATTRIBUTES=openinference.project.name=<project>
```

## Regions

| Region | Endpoint |
|---|---|
| US (default) | `https://otlp.arize.com` |
| EU | `https://otlp.eu-west-1a.arize.com` |
| Canada | `https://otlp.ca-central-1a.arize.com` |

## Two details that are easy to get wrong

**The endpoint has no path.** The OTLP HTTP exporter appends `/v1/traces`
itself, so the base is the bare host. `https://otlp.arize.com/v1` produces a
request to `/v1/v1/traces` and nothing arrives.

**The project name is required.** Arize rejects spans with HTTP 500 when it is
absent, and `service.name` alone is not enough.
`openinference.project.name` is the one vendor-namespaced value in this setup,
and it is supplied through the standard `OTEL_RESOURCE_ATTRIBUTES` variable
rather than any code change.

Short-lived scripts must flush before exiting, or the batch span processor is
torn down with spans still queued:

```python
from opentelemetry import trace

provider = trace.get_tracer_provider()
if hasattr(provider, "force_flush"):
    provider.force_flush(30000)
```

For long-running services, use the SIGTERM flush described in
[Configuration](../getting-started/configuration.md) instead.

## What Arize AX does with the spans

AX normalises the GenAI conventions onto its own OpenInference model at
ingestion. Sending nothing but `gen_ai.*`, a chat completion span arrives with:

| Arize field | Derived from |
|---|---|
| `openinference.span.kind = LLM` | inferred; TraceVerde never sends a span kind |
| `llm.provider`, `llm.system` | `gen_ai.provider.name` |
| `llm.model_name` | `gen_ai.request.model` |
| `llm.finish_reason` | `gen_ai.response.finish_reason` |
| `llm.invocation_parameters` | `gen_ai.request.*` |
| `llm.token_count.prompt` / `.completion` / `.total` | `gen_ai.usage.input_tokens` / `output_tokens` |
| `llm.cost.*` | derived from the token counts |

All original `gen_ai.*` attributes are preserved alongside the normalised
fields, so the same span remains portable to any other OTLP backend. Nothing is
lost in the direction of Arize, and nothing is Arize-shaped on the way out.

## Token counts must use the current conventions

Arize reads `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` - the
names introduced when semantic conventions v1.27.0 renamed `prompt_tokens` and
`completion_tokens`
([semantic-conventions#1200](https://github.com/open-telemetry/semantic-conventions/pull/1200)).

TraceVerde emits both spellings by default
(`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup`), so Arize resolves token counts and
cost correctly while dashboards still querying the superseded names keep
working. Setting `gen_ai` emits only the current names, which Arize also reads
correctly. Only the superseded-names-only configuration would report zero tokens
in AX.

## What a real span carries

A single `gpt-4o-mini` chat completion, exported with no Arize-specific code:

```
gen_ai.operation.name        = chat
gen_ai.provider.name         = openai
gen_ai.system                = openai
gen_ai.request.model         = gpt-4o-mini
gen_ai.response.model        = gpt-4o-mini-2024-07-18
gen_ai.request.max_tokens    = 60
gen_ai.response.finish_reason= length
gen_ai.usage.input_tokens    = 14
gen_ai.usage.output_tokens   = 60
gen_ai.usage.total_tokens    = 74
gen_ai.usage.prompt_tokens   = 14      # dual emission
gen_ai.usage.completion_tokens = 60    # dual emission
gen_ai.usage.cost.prompt     = 2.1e-06
gen_ai.usage.cost.completion = 3.6e-05
gen_ai.usage.cost.total      = 3.81e-05
gen_ai.usage.cost.pricing_source = table
```

Streamed calls add the server-timing conventions, which AX charts alongside
overall latency:

```
gen_ai.server.time_to_first_token  = 2.117
gen_ai.server.time_per_output_token = 0.0115
```

If the [evaluation features](evaluation.md) are enabled, their verdicts ride on
the same span as ordinary attributes and land in Arize with no extra wiring:

```
evaluation.pii.prompt.detected           = False
evaluation.prompt_injection.detected     = False
evaluation.toxicity.prompt.detected      = False
evaluation.hallucination.response.score  = 0.0
```

## Verifying a trace landed

```bash
pip install arize-ax-cli
ax profiles create --api-key $ARIZE_API_KEY --auth-method api-key
ax projects list -l 100 -o json
ax spans export <project-id> --trace-id <trace-id> --output-dir .arize-tmp-traces
```

Trace-ID lookups are immediately consistent once ingested. If a trace is
missing, check that the exporter flushed, that the endpoint region matches your
account, and that the project name resource attribute is present.

## Sending to Arize and somewhere else at once

Because this is ordinary OTLP, an OpenTelemetry Collector can fan the same spans
out to Arize and to any other backend without the application knowing. Point
TraceVerde at the collector and configure the exporters there - no change to the
four variables above beyond the endpoint.
