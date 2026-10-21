# Decision models

A decision model takes a state and a set of typed questions and returns, for each
question, a probability distribution over answers you supplied. It writes no text.
Every call path below produces one OpenInference `DECISION` span with the same
attributes (see "Decision models" in `docs/guides/llm-providers.md`).

| Example | Call path | Needs | Run by the maintainers |
|---|---|---|---|
| `openai_decisions.py` | `client.decisions.create` (OpenAI Decisions, `gpt-6-luna`) | `OPENAI_API_KEY` | yes, live |
| `openrouter_decisions.py` | OpenAI SDK with OpenRouter's `base_url` | `OPENROUTER_API_KEY` | no key held; tests only |
| `liquid_d1.py` | `typesafe-sdk` with LiquidAI's `base_url` | `LIQUID_API_KEY` | no key held; tests only |
| `systemone_selfhosted.py` | `typesafe-sdk` against any `/v1/systemone` server (vLLM, SGLang + Clef, laya-serve) | a running server | no; tests only |
| `systemone_http.py` | plain `requests`, traced by the HTTP path matcher | a SystemOne endpoint | no; tests only |
| `granite_switch_decision.py` | OpenAI chat completions with Granite Switch's `decision` adapter | vLLM serving Granite Switch | no; tests only |
| `manual_decision_span.py` | `genai_otel.decision_span` for in-process models | nothing | yes |

The TypeSafe Jev example is `examples/typesafe/example.py`.

## Running one

```bash
pip install "genai-otel-instrument" "openai>=3.26"
export OPENAI_API_KEY=...
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=decisions-example
python examples/decisions/openai_decisions.py
```

With no `OTEL_EXPORTER_OTLP_ENDPOINT` the spans are printed to the console.

## What is recorded

With content capture off (the default) a decision span says which model decided,
through which provider, how many questions of which types were asked, how many
were refused, tokens, cost and latency. It records nothing about the answer.

`GENAI_ENABLE_CONTENT_CAPTURE=true` adds the question names, the request and
response JSON (`input.value` / `output.value`, images replaced with
`[image omitted]`), the lowest confidence across the answers and, for a
one-question call, the chosen answer, its confidence and the margin over the
runner-up.
