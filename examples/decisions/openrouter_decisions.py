"""OpenRouter's Decisions API through the OpenAI SDK, traced as a DECISION span.

OpenRouter lists gpt-6-luna as a decision model (``openai/gpt-6-luna-decisions``,
$0.10 per 1M input tokens, output free). The OpenAI SDK's Decisions resource
sends ``POST {base_url}/decisions``; with OpenRouter's base URL that is
``https://openrouter.ai/api/v1/decisions``. The maintainers hold no OpenRouter
key, so this path has been checked against recorded responses only, not run
live. If OpenRouter serves its Decisions API on a different path the SDK call
fails; plain HTTP calls to any ``.../v1/decisions`` path are still traced by the
HTTP path matcher.

Needs:
    pip install "genai-otel-instrument" "openai>=3.26"
    OPENROUTER_API_KEY             an OpenRouter key
    OTEL_EXPORTER_OTLP_ENDPOINT    e.g. http://localhost:4318

The span carries ``decision.provider=openrouter`` (from the host) and is priced
from the ``decision`` section of the pricing table.
"""

import os

import genai_otel


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running this example.")

    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "decisions-openrouter-example"),
        enabled_instrumentors=["openrouter", "openai"],
        enable_gpu_metrics=False,
    )

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    decision = client.decisions.create(
        model="openai/gpt-6-luna-decisions",
        input="Customer: my card was charged twice for one order.",
        questions=[
            {
                "type": "predicate",
                "name": "billing_issue",
                "instructions": "Is this a billing problem?",
            },
            {
                "type": "score",
                "name": "urgency",
                "instructions": "How urgent is this for the customer?",
                "levels": [{"label": "can wait"}, {"label": "this week"}, {"label": "today"}],
            },
        ],
    )
    for answer in decision.answers:
        print(answer)

    genai_otel.flush_telemetry()
    print("Exported an openrouter.decisions DECISION span.")


if __name__ == "__main__":
    main()
