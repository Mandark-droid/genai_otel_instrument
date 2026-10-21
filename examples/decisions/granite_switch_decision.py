"""IBM Granite Switch's ``decision`` adapter on vLLM, traced as a DECISION span.

Granite Switch serves chat and decisions from one endpoint. A chat-completions
call that activates the ``decision`` aLoRA
(``extra_body={"chat_template_kwargs": {"adapter_name": "decision"}}``) is a
decision, not a chat turn, and is traced as a DECISION span with
``genai.decision.protocol=granite_adapter``. Its other adapters
(``guardian-core``) generate text and stay chat spans.

When the request asks for the option labels' logprobs (``logprob_token_ids``)
the span records the option count and, with content capture on, the chosen
label and the margin between the top two options. Without them it records
counts only.

Self-hosted: no list price, so no cost attribute.

Needs:
    pip install "genai-otel-instrument" openai
    pip install transformers    (to look up the label token ids; optional)
    vLLM serving ibm-granite/granite-switch-4.1-8b, e.g.
        python -m vllm.entrypoints.openai.api_server \\
            --model ibm-granite/granite-switch-4.1-8b --port 8000
    OTEL_EXPORTER_OTLP_ENDPOINT    e.g. http://localhost:4318

Optional:
    GRANITE_BASE_URL=http://localhost:8000/v1
"""

import math
import os

import genai_otel

MODEL_ID = "ibm-granite/granite-switch-4.1-8b"
LABELS = "ABC"


def label_token_ids():
    """The single-token ids of the option labels, or None without transformers."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return [tokenizer(label, add_special_tokens=False)["input_ids"][0] for label in LABELS]


def main() -> None:
    genai_otel.instrument(
        service_name=os.getenv("OTEL_SERVICE_NAME", "decisions-granite-example"),
        enabled_instrumentors=["openai"],
        enable_gpu_metrics=False,
    )

    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("GRANITE_BASE_URL", "http://localhost:8000/v1"), api_key="unused"
    )
    extra_body = {"chat_template_kwargs": {"adapter_name": "decision"}}
    ids = label_token_ids()
    if ids:
        extra_body["logprob_token_ids"] = ids

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": "State:\nMy payouts have failed three times. Can someone fix this?",
            },
            {
                "role": "user",
                "content": (
                    "### DECISION\nQuestion:\nWhich team should handle this ticket?\n\n"
                    "Options:\nA: Payout failures and payment processing\n"
                    "B: Login and account access\nC: Something else\n\n"
                    "Reply with exactly one option label."
                ),
            },
        ],
        extra_body=extra_body,
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=1,
        temperature=0,
    )

    logprobs = response.choices[0].logprobs
    if logprobs and logprobs.content:
        print({t.token: round(math.exp(t.logprob), 4) for t in logprobs.content[0].top_logprobs})
    else:
        print("answer:", response.choices[0].message.content)
    genai_otel.flush_telemetry()
    print("Exported an openai.chat.decision DECISION span.")


if __name__ == "__main__":
    main()
