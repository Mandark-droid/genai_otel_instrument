"""LiquidAI D1 through typesafe-sdk with Liquid's base URL, traced as a DECISION span.

LiquidAI's D1 models answer the SystemOne protocol at
``https://api.liquid.ai/decisions/v1/systemone``, so the TypeSafe SDK reaches
them by pointing ``base_url`` at Liquid. The same SystemOne wrapper that traces
TypeSafe's Jev traces them, with ``decision.provider=liquid`` from the host.
D1 bills input only ($0.02 per 1M tokens; D1 Ultra $0.042).

The maintainers hold no Liquid key, so this example has not been run live; the
path is covered by tests against recorded responses.

Needs:
    pip install "genai-otel-instrument[typesafe]"
    LIQUID_API_KEY                 a LiquidAI API key
    OTEL_EXPORTER_OTLP_ENDPOINT    e.g. http://localhost:4318

Optional:
    LIQUID_DECISION_MODEL=d1-ultra
"""

import os

import genai_otel


def main() -> None:
    api_key = os.getenv("LIQUID_API_KEY")
    if not api_key:
        raise SystemExit("Set LIQUID_API_KEY before running this example.")

    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "decisions-liquid-example"),
        enabled_instrumentors=["typesafe"],
        enable_gpu_metrics=False,
    )

    from typesafe_sdk import Choice, Noul, TypeSafeClient

    with TypeSafeClient(api_key=api_key, base_url="https://api.liquid.ai/decisions") as client:
        response = client.system_one(
            state="Hi, the app logs me out every time I switch to another tab.",
            questions={
                "team": Choice(
                    instructions="Which team should take this ticket?",
                    criteria={
                        "auth": "Login, sessions and account access",
                        "payments": "Charges, refunds and payouts",
                        "other": "Anything else",
                    },
                ),
                "bug": Noul(instructions="Is the customer describing a bug?"),
            },
            model=os.getenv("LIQUID_DECISION_MODEL", "d1"),
        )

    for name, answer in response.answers.items():
        print(f"{name}: {answer}")
    print("usage:", response.usage)
    genai_otel.flush_telemetry()
    print("Exported a typesafe.system_one DECISION span (provider liquid).")


if __name__ == "__main__":
    main()
