# Arize AX Example

This example exports GenAI spans to [Arize AX](https://arize.com) over plain
OTLP.

There is no Arize integration in this library. There is no adapter, no
`arize-otel`, and no `openinference-instrumentation-*` package. Arize AX
consumes the OpenTelemetry GenAI semantic conventions natively, and this library
emits them, so the two interoperate through OTLP without a translation layer on
either side.

The entire configuration is four standard OpenTelemetry environment variables.

## Setup

1. Install dependencies:
```bash
pip install genai-otel-instrument[openai]
```

2. Get an API key and Space ID from https://app.arize.com (Settings -> API Keys).
   `ARIZE_SPACE_ID` needs the base64 Space **ID**, not the space name.

3. Export the configuration:
```bash
export ARIZE_SPACE_ID=...
export ARIZE_API_KEY=...
export OPENAI_API_KEY=...
export ARIZE_PROJECT_NAME=traceverde-example   # optional
```

4. Run the example:
```bash
python example.py
```

5. Open the project in Arize AX. It is created on first export.

## RAG example

`rag_example.py` sends a full retrieval pipeline as a single trace:

```
rag.pipeline
  openai.embeddings       index the corpus  (gen_ai.request.input_count = 3)
  openai.embeddings       embed the query   (gen_ai.request.input_count = 1)
  openai.chat.completion  generate the answer
```

Embedding calls are traced as their own spans, so the lookup that selected the
context is visible and its tokens and cost are recorded, rather than the trace
showing only the generation half. The parent span is what groups the three
calls into one trace; without it they arrive as three unrelated traces.

```bash
python rag_example.py
```

The example enables `GENAI_ENABLE_CONTENT_CAPTURE` so the embedded text appears
on the span as `embedding.text`. That is off by default, because retrieval
inputs routinely carry user data. Embedding vectors stay off regardless unless
`capture_embedding_vectors` is set - they would otherwise dominate span size.

## The configuration

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.arize.com
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_HEADERS=space_id=<id>,api_key=<key>
OTEL_RESOURCE_ATTRIBUTES=openinference.project.name=<project>
```

Two details are easy to get wrong:

- **Endpoint has no path.** The OTLP HTTP exporter appends `/v1/traces` itself,
  so the base is the bare host. `https://otlp.arize.com/v1` yields a request to
  `/v1/v1/traces`. For EU use `https://otlp.eu-west-1a.arize.com`, for Canada
  `https://otlp.ca-central-1a.arize.com`.
- **The project name is required.** Arize rejects spans with HTTP 500 when it is
  absent; `service.name` alone is not enough. `openinference.project.name` is
  the one vendor-namespaced value in this setup, and it is supplied through the
  standard `OTEL_RESOURCE_ATTRIBUTES` variable rather than any code change.

Short-lived scripts must `force_flush()` before exiting or the batch span
processor is torn down with spans still queued.

## What Arize AX does with the spans

AX normalises the GenAI conventions onto its own OpenInference model at
ingestion. Sending nothing but `gen_ai.*`, a chat completion span arrives with:

| Arize field | Derived from |
|---|---|
| `openinference.span.kind = LLM` | inferred; we never send a span kind |
| `llm.provider`, `llm.system` | `gen_ai.system` |
| `llm.model_name` | `gen_ai.request.model` |
| `llm.finish_reason` | `gen_ai.response.finish_reason` |
| `llm.invocation_parameters` | `gen_ai.request.*` |
| `llm.token_count.prompt` / `.completion` / `.total` | `gen_ai.usage.input_tokens` / `output_tokens` |
| `llm.cost.*` | derived from the token counts |

All original `gen_ai.*` attributes are preserved alongside the normalised
fields, so the same span remains portable to any other OTLP backend.

## Token counts must use the current conventions

Arize reads `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` - the
names introduced when semantic conventions v1.27.0 renamed `prompt_tokens` and
`completion_tokens`
([semantic-conventions#1200](https://github.com/open-telemetry/semantic-conventions/pull/1200)).

This library emits the current names by default. If you set
`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup` to keep dashboards on the superseded
names working, the current names are still emitted alongside them, so Arize
continues to resolve token counts and cost correctly.

## Verifying a trace landed

```bash
pip install arize-ax-cli
ax profiles create --api-key $ARIZE_API_KEY --auth-method api-key
ax projects list -l 100 -o json
ax spans export <project-id> --trace-id <trace-id> --output-dir .arize-tmp-traces
```

Trace-ID lookups are immediately consistent once ingested. If a trace is
missing, check the exporter flushed, the endpoint region matches your account,
and the project name resource attribute is present.
