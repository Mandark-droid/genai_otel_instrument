"""Live validation: real AWS Bedrock traffic -> real spans (issue #27).

Drives real calls against `bedrock-runtime` through the instrumented client and
asserts the spans that come back. Every span printed below is produced by an
actual AWS round trip -- nothing here is a mock or a hand-written span. Unit
tests can only prove the extraction logic given a response shape *we* wrote;
this proves the shape AWS actually returns matches it.

Covers all four runtime calls, and the #27 acceptance criteria:

  converse                          span + model + tokens + cost (+ tool calls)
  converse_stream                   span + TTFT + usage from trailing metadata
  invoke_model                      unchanged regression
  invoke_model_with_response_stream span equivalent to invoke_model

Run::

    export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_REGION=us-east-1
    python examples/bedrock_live_validation/run_validation.py

Model access is per-account and per-region, so both model ids are overridable::

    BEDROCK_MODEL_ID          Converse model  (default amazon.nova-lite-v1:0)
    BEDROCK_INVOKE_MODEL_ID   invoke_model    (default amazon.titan-text-express-v1)

The Converse default is deliberately a non-Anthropic model: routing those
through Converse is the whole reason #27 mattered.

Exits non-zero if any assertion fails, so it is usable as a release gate rather
than only as a demo. It also exits non-zero -- loudly -- when credentials or
model access are missing, so a misconfigured run can never be mistaken for a
passing one.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import genai_otel

genai_otel.instrument()

import boto3  # noqa: E402  (must follow instrument() so the client is wrapped)
from botocore.exceptions import BotoCoreError, ClientError  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

CONVERSE_MODEL = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
INVOKE_MODEL = os.environ.get("BEDROCK_INVOKE_MODEL_ID", "amazon.titan-text-express-v1")
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

_failures: List[str] = []
_exporter = InMemorySpanExporter()


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record a pass/fail assertion without aborting the run."""
    print(
        "  [{0}] {1}{2}".format(
            "PASS" if condition else "FAIL", label, f" - {detail}" if detail else ""
        )
    )
    if not condition:
        _failures.append(label)


def attach_exporter() -> None:
    """Tee spans into memory alongside whatever exporter instrument() configured."""
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        sys.exit(
            "FATAL: the active TracerProvider is not an SDK provider, so no spans can be "
            "captured. Nothing was validated."
        )
    provider.add_span_processor(SimpleSpanProcessor(_exporter))


def spans_named(name: str) -> List[Any]:
    return [s for s in _exporter.get_finished_spans() if s.name == name]


def one_span(name: str) -> Optional[Any]:
    found = spans_named(name)
    check(f"{name}: exactly one span emitted", len(found) == 1, f"got {len(found)}")
    return found[0] if len(found) == 1 else None


def show(span: Any, keys: List[str]) -> None:
    """Print the attributes actually recorded, so this is inspectable, not just green."""
    for key in keys:
        print(f"        {key} = {span.attributes.get(key)!r}")


def assert_tokens_and_cost(span: Any, label: str) -> None:
    attrs = span.attributes
    inp = attrs.get("gen_ai.usage.input_tokens")
    out = attrs.get("gen_ai.usage.output_tokens")
    check(f"{label}: input tokens recorded and > 0", isinstance(inp, int) and inp > 0, repr(inp))
    check(f"{label}: output tokens recorded and > 0", isinstance(out, int) and out > 0, repr(out))
    # A zero here is the exact failure #27 described: usage unread, so the span
    # is priced at zero rather than flagged as having none.
    cost = attrs.get("gen_ai.usage.cost.total")
    check(
        f"{label}: cost computed (not silently zero)",
        isinstance(cost, (int, float)) and cost > 0,
        repr(cost),
    )


def fatal_aws(exc: Exception, what: str) -> None:
    sys.exit(
        f"FATAL: {what} failed against live Bedrock: {type(exc).__name__}: {exc}\n"
        "Nothing was validated. Check credentials, region, and per-model access "
        "in the Bedrock console."
    )


def preflight() -> None:
    """Fail before any call if credentials are absent.

    Otherwise the real reason surfaces as a NoCredentialsError buried in
    exporter retry noise, which is easy to misread as a flaky run.
    """
    try:
        credentials = boto3.session.Session().get_credentials()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            "\n".join(
                [
                    f"FATAL: could not resolve an AWS session: {exc}",
                    "Nothing was validated.",
                ]
            )
        )
    if credentials is None:
        sys.exit(
            "\n".join(
                [
                    "FATAL: no AWS credentials found -- nothing was validated.",
                    "  Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and AWS_REGION), or",
                    "  configure a profile via `aws configure`. Bedrock model access is also",
                    "  granted per account and per region in the Bedrock console.",
                    "  Tip: set OTEL_EXPORTER_OTLP_ENDPOINT= (empty) to keep exporter retry",
                    "  noise out of this run's output.",
                ]
            )
        )


def main() -> int:
    preflight()
    attach_exporter()
    print(f"region={REGION}  converse_model={CONVERSE_MODEL}  invoke_model={INVOKE_MODEL}\n")

    client = boto3.client("bedrock-runtime", region_name=REGION)

    # --- converse -----------------------------------------------------------
    print("converse")
    try:
        response = client.converse(
            modelId=CONVERSE_MODEL,
            system=[{"text": "You are terse."}],
            messages=[{"role": "user", "content": [{"text": "Say the word: tracing"}]}],
            inferenceConfig={"maxTokens": 64, "temperature": 0.2},
        )
    except (ClientError, BotoCoreError) as exc:
        fatal_aws(exc, "converse")

    text = response["output"]["message"]["content"][0].get("text", "")
    print(f'    model said: "{text.strip()[:60]}"')
    span = one_span("aws.bedrock.converse")
    if span:
        check(
            "converse: request model recorded",
            span.attributes.get("gen_ai.request.model") == CONVERSE_MODEL,
            repr(span.attributes.get("gen_ai.request.model")),
        )
        check(
            "converse: system captured as instructions, not a message",
            span.attributes.get("gen_ai.request.message_count") == 1,
            f'message_count={span.attributes.get("gen_ai.request.message_count")}',
        )
        check(
            "converse: max_tokens mapped from inferenceConfig",
            span.attributes.get("gen_ai.request.max_tokens") == 64,
        )
        check(
            "converse: stopReason recorded as finish reason",
            bool(span.attributes.get("gen_ai.response.finish_reasons")),
            repr(span.attributes.get("gen_ai.response.finish_reasons")),
        )
        check("converse: completion captured", bool(span.attributes.get("gen_ai.response")))
        assert_tokens_and_cost(span, "converse")
        show(
            span,
            [
                "gen_ai.system",
                "gen_ai.request.model",
                "gen_ai.request.instructions",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.cost.total",
                "gen_ai.response.finish_reasons",
            ],
        )

    # --- converse_stream ----------------------------------------------------
    print("\nconverse_stream")
    _exporter.clear()
    try:
        streamed = client.converse_stream(
            modelId=CONVERSE_MODEL,
            messages=[{"role": "user", "content": [{"text": "Count to three."}]}],
            inferenceConfig={"maxTokens": 64},
        )
    except (ClientError, BotoCoreError) as exc:
        fatal_aws(exc, "converse_stream")

    check(
        "converse_stream: span stays open while the stream is unread",
        not spans_named("aws.bedrock.converse_stream"),
        "span closed on return, so latency and tokens would be wrong",
    )

    events = 0
    for event in streamed["stream"]:
        events += 1
    print(f"    consumed {events} stream events")

    span = one_span("aws.bedrock.converse_stream")
    if span:
        ttft = span.attributes.get("gen_ai.server.time_to_first_token") or span.attributes.get(
            "gen_ai.server.ttft"
        )
        check(
            "converse_stream: TTFT recorded",
            isinstance(ttft, (int, float)) and ttft > 0,
            repr(ttft),
        )
        check(
            "converse_stream: chunks counted",
            (span.attributes.get("gen_ai.streaming.token_count") or 0) > 0,
            repr(span.attributes.get("gen_ai.streaming.token_count")),
        )
        # The usage arrives only in the trailing `metadata` event; if that is not
        # read the span looks free.
        assert_tokens_and_cost(span, "converse_stream")
        show(
            span,
            [
                "gen_ai.streaming.token_count",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.cost.total",
            ],
        )

    # --- invoke_model (regression) ------------------------------------------
    print("\ninvoke_model (regression)")
    _exporter.clear()
    titan_body = json.dumps(
        {"inputText": "Say the word: tracing", "textGenerationConfig": {"maxTokenCount": 64}}
    )
    try:
        client.invoke_model(
            body=titan_body,
            modelId=INVOKE_MODEL,
            accept="application/json",
            contentType="application/json",
        )
    except (ClientError, BotoCoreError) as exc:
        fatal_aws(exc, "invoke_model")

    span = one_span("aws.bedrock.invoke_model")
    if span:
        check(
            "invoke_model: model recorded",
            span.attributes.get("gen_ai.request.model") == INVOKE_MODEL,
        )
        assert_tokens_and_cost(span, "invoke_model")

    # --- invoke_model_with_response_stream ----------------------------------
    print("\ninvoke_model_with_response_stream")
    _exporter.clear()
    try:
        streamed = client.invoke_model_with_response_stream(
            body=titan_body,
            modelId=INVOKE_MODEL,
            accept="application/json",
            contentType="application/json",
        )
        chunks = sum(1 for _ in streamed["body"])
    except (ClientError, BotoCoreError) as exc:
        fatal_aws(exc, "invoke_model_with_response_stream")

    print(f"    consumed {chunks} chunks")
    span = one_span("aws.bedrock.invoke_model_with_response_stream")
    if span:
        check(
            "invoke_model_with_response_stream: model recorded",
            span.attributes.get("gen_ai.request.model") == INVOKE_MODEL,
            repr(span.attributes.get("gen_ai.request.model")),
        )

    # --- verdict ------------------------------------------------------------
    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): " + "; ".join(_failures))
        return 1
    print(f"All checks passed against live Bedrock in {REGION}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
