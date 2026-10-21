# TypeSafe AI / Jev

This example makes one System One request containing a `Choice`, `Noul`, and
`Score` question. It is useful for manually checking the native TraceVerde
instrumentation against the platform OTLP backend.

## Run it

```bash
pip install 'genai-otel-instrument[typesafe]'
set TYPESAFE_API_KEY=your-key
set OTEL_EXPORTER_OTLP_ENDPOINT=https://otel.example.internal:4318
set OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
set OTEL_EXPORTER_OTLP_CERTIFICATE=C:\path\to\platform-ca.crt
set GENAI_ENABLE_CONTENT_CAPTURE=true
set GENAI_CONTENT_MAX_LENGTH=0
python examples/typesafe/example.py
```

PowerShell uses `$env:NAME = "value"` instead of `set NAME=value`.
`OTEL_EXPORTER_OTLP_CERTIFICATE` must point to the CA certificate that signed
the platform OTLP endpoint. Use a path readable by the Python process.

The script explicitly enables only the `typesafe` instrumentor and disables GPU
metrics, so it is safe to use as a focused smoke test. The emitted span is
`typesafe.system_one`, an OpenInference `DECISION` span (since 1.38; it was an
`LLM` span before). It always carries the model, `decision.provider`, the
question count and types, refusals, token counts and the decision cost; with
content capture enabled it also carries `input.value`, `output.value`, the
question names and, for a one-question call, the answer and its confidence.

The `llm.*` attributes are still emitted until 2.0. Set
`GENAI_DECISION_LEGACY_LLM_KIND=true` to keep `openinference.span.kind=LLM`.
Other decision call paths (OpenAI Decisions, LiquidAI, self-hosted SystemOne,
plain HTTP, Granite Switch, in-process models) are in `examples/decisions/`.
