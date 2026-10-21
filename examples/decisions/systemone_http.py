"""SystemOne over plain ``requests``: no SDK, traced by the HTTP path matcher.

Some applications call a decision endpoint directly, as LiquidAI's image
examples do. A POST whose path ends in ``/v1/systemone``,
``/decisions/v1/systemone`` or ``/v1/decisions`` becomes a DECISION span parsed
from the request and response JSON, the same attributes the SDK wrappers emit.
Turn it off with ``GENAI_DECISION_HTTP_MATCHER=false``.

Images in the state are recorded as ``[image omitted]`` even with content
capture on.

Needs:
    pip install "genai-otel-instrument" requests
    a SystemOne endpoint: a self-hosted server, or LiquidAI with a key
    OTEL_EXPORTER_OTLP_ENDPOINT    e.g. http://localhost:4318

Optional:
    SYSTEMONE_URL=http://localhost:8000/v1/systemone
                  (LiquidAI: https://api.liquid.ai/decisions/v1/systemone)
    SYSTEMONE_API_KEY=...          sent as a bearer token when set
    SYSTEMONE_MODEL=d1
    SYSTEMONE_IMAGE=path/to/photo.png   decide on an image instead of text
"""

import base64
import os

import genai_otel


def main() -> None:
    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "decisions-http-example"),
        enable_gpu_metrics=False,
    )

    import requests

    url = os.getenv("SYSTEMONE_URL", "http://localhost:8000/v1/systemone")
    headers = {"Content-Type": "application/json"}
    if os.getenv("SYSTEMONE_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['SYSTEMONE_API_KEY']}"

    image = os.getenv("SYSTEMONE_IMAGE")
    if image:
        with open(image, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        state = {"type": "image", "data": f"data:image/png;base64,{encoded}"}
    else:
        state = "The parcel arrived with the box crushed and the screen cracked."

    body = {
        "model": os.getenv("SYSTEMONE_MODEL", "d1"),
        "state": state,
        "questions": {
            "damaged": {"type": "noul", "instructions": "Is the item damaged?"},
            "action": {
                "type": "choice",
                "instructions": "What should support offer?",
                "criteria": {"replace": None, "refund": None, "nothing": None},
            },
        },
    }
    response = requests.post(url, json=body, headers=headers, timeout=60)
    print(response.status_code, response.text[:500])

    genai_otel.flush_telemetry()
    print("Exported a decision.http.systemone DECISION span.")


if __name__ == "__main__":
    main()
