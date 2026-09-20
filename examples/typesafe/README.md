# TypeSafe AI / Jev

This example makes one System One request containing a `Choice`, `Noul`, and
`Score` question. It is useful for manually checking the native TraceVerde
instrumentation with Jaeger or any OTLP-compatible backend.

## Run it

```bash
pip install 'genai-otel-instrument[typesafe]'
set TYPESAFE_API_KEY=your-key
set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
set GENAI_ENABLE_CONTENT_CAPTURE=true
set GENAI_CONTENT_MAX_LENGTH=0
python examples/typesafe/example.py
```

PowerShell uses `$env:TYPESAFE_API_KEY = "your-key"` instead of `set`.

The script explicitly enables only the `typesafe` instrumentor and disables GPU
metrics, so it is safe to use as a focused smoke test. The emitted span is
`typesafe.system_one`; with content capture enabled it includes `input.value`,
`output.value`, the typed answers, and token counts.
