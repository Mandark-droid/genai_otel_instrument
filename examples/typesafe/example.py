"""Manual TypeSafe AI / Jev instrumentation example.

Prerequisites:
    pip install -e ".[typesafe]"
    set TYPESAFE_API_KEY=your-key       # PowerShell: $env:TYPESAFE_API_KEY=...
    set OTEL_EXPORTER_OTLP_ENDPOINT=https://otel.example.internal:4318
    set OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
    set OTEL_EXPORTER_OTLP_CERTIFICATE=C:\\path\\to\\platform-ca.crt
"""

import os

import genai_otel


def main() -> None:
    if not os.getenv("TYPESAFE_API_KEY"):
        raise SystemExit("Set TYPESAFE_API_KEY before running this example.")

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        raise SystemExit("Set OTEL_EXPORTER_OTLP_ENDPOINT to the platform OTLP HTTPS endpoint.")
    if not endpoint.lower().startswith("https://"):
        raise SystemExit("OTEL_EXPORTER_OTLP_ENDPOINT must use https:// for the platform.")

    ca_certificate = os.getenv("OTEL_EXPORTER_OTLP_CERTIFICATE")
    if not ca_certificate:
        raise SystemExit("Set OTEL_EXPORTER_OTLP_CERTIFICATE to the platform CA certificate.")
    if not os.path.isfile(ca_certificate):
        raise SystemExit(f"CA certificate does not exist: {ca_certificate}")

    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    # Enable TraceVerde before importing the optional SDK. Content capture is
    # deliberately opt-in; enable it in the shell when inspecting the payload.
    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "typesafe-manual-example"),
        endpoint=endpoint,
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
    print("Trace exported as typesafe.system_one to the configured OTLP backend.")


if __name__ == "__main__":
    main()
