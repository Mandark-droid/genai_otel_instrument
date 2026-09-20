"""Manual TypeSafe AI / Jev instrumentation example.

Prerequisites:
    pip install -e ".[typesafe]"
    set TYPESAFE_API_KEY=your-key       # PowerShell: $env:TYPESAFE_API_KEY=...
    docker run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 4318:4318 \
        -p 16686:16686 jaegertracing/all-in-one:latest
"""

import os

import genai_otel


def main() -> None:
    if not os.getenv("TYPESAFE_API_KEY"):
        raise SystemExit("Set TYPESAFE_API_KEY before running this example.")

    # Enable TraceVerde before importing the optional SDK. Content capture is
    # deliberately opt-in; enable it in the shell when inspecting the payload.
    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "typesafe-manual-example"),
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
        enabled_instrumentors=["typesafe"],
        enable_gpu_metrics=False,
    )

    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    state = {
        "subject": "The export button crashes the settings page in Safari.",
        "browser": "Safari 18 on macOS",
        "customer": "A business customer with no workaround",
    }
    questions = {
        "category": Choice(
            instructions="Which team should handle this support ticket?",
            criteria={
                "engineering": "Bugs, crashes, or integration failures",
                "support": "How-to questions or account assistance",
                "billing": "Charges, invoices, refunds, or subscriptions",
                "other": "None of the other categories clearly fits",
            },
        ),
        "needs_human": Noul(
            instructions="Does this ticket require a human to follow up?",
        ),
        "severity": Score(
            instructions="How severely is the customer blocked?",
            criteria=[
                "Cosmetic; nothing important is blocked",
                "A degraded feature has a workaround",
                "A critical workflow is blocked with no workaround",
            ],
        ),
    }

    with TypeSafeClient() as client:
        response = client.system_one(
            state=state,
            questions=questions,
            model=os.getenv("TYPESAFE_MODEL", "jev-latest"),
        )

    print("model:", getattr(response, "model", None))
    print("answers:")
    for key, answer in getattr(response, "answers", {}).items():
        print(f"  {key}: {answer}")
    print("usage:", getattr(response, "usage", None))
    genai_otel.flush_telemetry()
    print("Trace exported as typesafe.system_one; inspect Jaeger at http://localhost:16686")


if __name__ == "__main__":
    main()
