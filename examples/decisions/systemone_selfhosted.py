"""A self-hosted SystemOne server (vLLM, SGLang + Clef, laya-serve) via typesafe-sdk.

Any server that answers ``POST /v1/systemone`` works with the TypeSafe SDK and a
``base_url``: a Hub decision model served by vLLM or SGLang, Cloudflare's Clef
weights under SGLang, or a community server such as laya-serve. Every one of
them is traced by the same SystemOne wrapper as a DECISION span.

A self-hosted model has no list price, so the span carries tokens but no cost.
Price it per input token with a ``decision`` section in
``GENAI_CUSTOM_PRICING_JSON`` if you want one. The host is not one the library
recognises, so ``decision.provider`` is omitted unless you name it with
``GENAI_DECISION_PROVIDER``.

Needs:
    pip install "genai-otel-instrument[typesafe]"
    a running SystemOne server, e.g. at http://localhost:8000
    OTEL_EXPORTER_OTLP_ENDPOINT    e.g. http://localhost:4318

Optional:
    SYSTEMONE_BASE_URL=http://localhost:8000
    SYSTEMONE_MODEL=Cloudflare/clef-flash
    SYSTEMONE_API_KEY=...          if your server checks one
    GENAI_DECISION_PROVIDER=my-gpu-cluster
    GENAI_CUSTOM_PRICING_JSON='{"decision": {"Cloudflare/clef-flash": {"promptPrice": 0.00001}}}'
"""

import os

import genai_otel


def main() -> None:
    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "decisions-selfhosted-example"),
        enabled_instrumentors=["typesafe"],
        enable_gpu_metrics=False,
    )

    from typesafe_sdk import Choice, Score, TypeSafeClient

    base_url = os.getenv("SYSTEMONE_BASE_URL", "http://localhost:8000")
    with TypeSafeClient(
        api_key=os.getenv("SYSTEMONE_API_KEY", "not-checked-by-local-servers"),
        base_url=base_url,
    ) as client:
        response = client.system_one(
            state="The quarterly report export fails with a timeout after 30 seconds.",
            questions={
                "component": Choice(
                    criteria={"exports": None, "dashboards": None, "auth": None, "other": None}
                ),
                "severity": Score(criteria=["cosmetic", "degraded", "blocked"]),
            },
            model=os.getenv("SYSTEMONE_MODEL", "Cloudflare/clef-flash"),
        )

    for name, answer in response.answers.items():
        print(f"{name}: {answer}")
    genai_otel.flush_telemetry()
    print(f"Exported a typesafe.system_one DECISION span for {base_url}.")


if __name__ == "__main__":
    main()
