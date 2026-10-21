"""OpenAI Decisions API: a tool router traced as an OpenInference DECISION span.

A decision model picks which tool should handle the user's message and says how
sure it is. Below a confidence threshold the application would hand the choice
to a chat model instead; this example only prints what it would do.

Needs:
    pip install "genai-otel-instrument" "openai>=3.26"
    OPENAI_API_KEY                 an OpenAI key with access to gpt-6-luna
    OTEL_EXPORTER_OTLP_ENDPOINT    e.g. http://localhost:4318 (your collector)

Optional:
    GENAI_ENABLE_CONTENT_CAPTURE=true   also record the chosen tool, its
                                        confidence, question names and the
                                        request / response JSON on the span
    DECISION_CONFIDENCE_GATE=0.7        the threshold the router gates on

With content capture off (the default) the span still records the model, the
provider, how many questions of which types were asked, how many were refused,
tokens, cost and latency, and nothing about the answer.
"""

import os

import genai_otel

TOOLS = {
    "purchaseProduct": "Buy a product for the customer",
    "searchCatalog": "Look up products, sizes or prices",
    "trackOrder": "Find where an existing order is",
    "requestRefund": "Return or refund an order",
    "talkToHuman": "Hand the conversation to a person",
}


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "decisions-openai-example"),
        enabled_instrumentors=["openai"],
        enable_gpu_metrics=False,
    )

    from openai import OpenAI

    client = OpenAI()
    if not hasattr(client, "decisions"):
        raise SystemExit("This openai release has no Decisions API; pip install 'openai>=3.26'.")

    gate = float(os.getenv("DECISION_CONFIDENCE_GATE", "0.7"))
    messages = [
        "I'd like to buy the blue trainers in size 9, please.",
        "My parcel was meant to arrive last Tuesday. Where is it?",
        "Hmm, not sure. Maybe something for my dad?",
    ]
    for message in messages:
        decision = client.decisions.create(
            model=os.getenv("OPENAI_DECISION_MODEL", "gpt-6-luna"),
            input=message,
            questions=[
                {
                    "type": "choice",
                    "name": "next_tool",
                    "instructions": "Which tool should handle the customer's latest message?",
                    "choices": [
                        {"value": name, "description": description}
                        for name, description in TOOLS.items()
                    ],
                }
            ],
        )
        answer = decision.answers[0]
        if answer.type == "refusal":
            print(f"{message!r}\n  -> refused; the chat model decides")
            continue
        route = "use it" if answer.confidence >= gate else "below the gate; the chat model decides"
        print(f"{message!r}\n  -> {answer.choice} (confidence {answer.confidence:.2f}, {route})")
        print(
            f"     usage: {decision.usage.input_tokens} input / {decision.usage.output_tokens} output"
        )

    genai_otel.flush_telemetry()
    print("Exported one openai.decisions DECISION span per call.")


if __name__ == "__main__":
    main()
