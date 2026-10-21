"""An in-process decision model recorded with ``genai_otel.decision_span``.

In-process decision models (a Laya ``Router``, Cloudflare Clef via
``joint_schema_model.systemone``, a ``transformers`` checkpoint) make no network
call for the library to wrap, so the call is recorded by hand. The span is the
same DECISION span the instrumented paths emit (``genai.decision.protocol=
manual``), under the same content-capture rule.

This example runs without any model: ``route`` below is a keyword stand-in that
returns a SystemOne-shaped answer. Replace its body with your model, for
example::

    from laya import Router                      # Laya, ModernBERT
    router = Router.from_pretrained("<laya checkpoint>")
    result = router(message, labels=list(TOOLS))

    # or Cloudflare Clef, in-process
    answers = joint_schema_model.systemone(model, processor, state=message,
                                           questions=QUESTIONS)

Needs:
    pip install "genai-otel-instrument"
    OTEL_EXPORTER_OTLP_ENDPOINT    e.g. http://localhost:4318

Set GENAI_ENABLE_CONTENT_CAPTURE=true to see the chosen tool and its confidence
on the span.
"""

import os

import genai_otel

TOOLS = ["purchaseProduct", "searchCatalog", "trackOrder", "requestRefund", "talkToHuman"]
QUESTIONS = {"next_tool": {"type": "choice", "criteria": {tool: None for tool in TOOLS}}}
KEYWORDS = {
    "buy": "purchaseProduct",
    "price": "searchCatalog",
    "where": "trackOrder",
    "refund": "requestRefund",
}


def route(message: str) -> dict:
    """Stand-in decision model: a probability per tool, SystemOne-shaped."""
    scores = {tool: 1.0 for tool in TOOLS}
    for word, tool in KEYWORDS.items():
        if word in message.lower():
            scores[tool] += 8.0
    total = sum(scores.values())
    probabilities = {tool: score / total for tool, score in scores.items()}
    choice = max(probabilities, key=probabilities.get)
    return {
        "next_tool": {
            "type": "choice",
            "choice": choice,
            "confidence": probabilities[choice],
            "probabilities": probabilities,
        }
    }


def main() -> None:
    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "decisions-manual-example"),
        enable_gpu_metrics=False,
    )
    from genai_otel import decision_span

    for message in ["Where is my order?", "I want a refund", "hello there"]:
        # As a context manager: record the answers (and usage, if your model has one).
        with decision_span(model="keyword-router", questions=QUESTIONS, state=message) as d:
            answers = route(message)
            d.record(answers)
        print(f"{message!r} -> {answers['next_tool']['choice']}")

    # As a decorator: the return value is recorded when it is a set of answers.
    @decision_span(model="keyword-router", questions=QUESTIONS, name="route_ticket")
    def route_ticket(message: str) -> dict:
        return route(message)

    route_ticket("How much is the blue jacket? What's the price?")

    genai_otel.flush_telemetry()
    print("Exported four DECISION spans (protocol manual).")


if __name__ == "__main__":
    main()
