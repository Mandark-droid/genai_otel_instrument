# Bedrock live validation

Drives real AWS Bedrock traffic through the instrumented client and asserts the
spans that come back. Every span it prints is produced by an actual AWS round
trip — nothing is mocked.

This exists because unit tests can only prove the extraction logic against a
response shape *we* wrote. This proves the shape AWS actually returns matches
it — the gap that let [#27](https://github.com/Mandark-droid/genai_otel_instrument/issues/27)
sit unnoticed.

## Run

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

# keeps exporter retry noise out of the output
export OTEL_EXPORTER_OTLP_ENDPOINT=
export GENAI_SKIP_COLLECTOR_CHECK=true

python examples/bedrock_live_validation/run_validation.py
```

Exits non-zero if any assertion fails, so it works as a release gate rather than
only as a demo. It also exits non-zero — with an explicit `nothing was validated`
— when credentials or model access are missing, so a misconfigured run can never
be mistaken for a passing one.

## Model access

Bedrock grants model access per account **and** per region, so both ids are
overridable:

| Variable | Default | Used for |
|---|---|---|
| `BEDROCK_MODEL_ID` | `amazon.nova-lite-v1:0` | `converse`, `converse_stream` |
| `BEDROCK_INVOKE_MODEL_ID` | `amazon.titan-text-express-v1` | `invoke_model`, `invoke_model_with_response_stream` |

The Converse default is deliberately a **non-Anthropic** model: routing those
through Converse is the whole reason #27 mattered. The `invoke_model` default is
Titan because that call needs a per-model request body, which is precisely the
awkwardness Converse removes.

## What it checks

| Call | Assertions |
|---|---|
| `converse` | one span; model; `system` recorded as instructions without inflating `message_count`; `max_tokens` mapped from `inferenceConfig`; `stopReason` as finish reason; completion captured; input/output tokens > 0; cost > 0 |
| `converse_stream` | span does **not** close while the stream is unread; TTFT > 0; chunks counted; tokens and cost picked up from the trailing `metadata` event |
| `invoke_model` | regression — one span, model, tokens, cost |
| `invoke_model_with_response_stream` | one span, model recorded |

The token and cost assertions are the load-bearing ones. A zero there is the
exact failure #27 described: usage went unread, so the span was priced at zero
rather than flagged as having none.
