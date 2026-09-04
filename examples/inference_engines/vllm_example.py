"""vLLM in-process inference with OpenTelemetry instrumentation.

Traces `vllm.LLM.generate` / `.chat`, which never cross HTTP and so are
invisible to any OpenAI-SDK-level instrumentation. Each span carries the
queue / prefill / decode breakdown vLLM attaches to every RequestOutput.

Run:
    pip install "genai-otel-instrument[vllm]"
    python examples/inference_engines/vllm_example.py
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import genai_otel

# Instrument BEFORE importing vllm so the LLM class is wrapped on first use.
genai_otel.instrument()

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    llm = LLM(model="facebook/opt-125m")
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64)

    prompts = [
        "Explain what an inference engine scheduler does, briefly.",
        "Name one reason prefill and decode are measured separately.",
    ]

    outputs = llm.generate(prompts, params)

    for output in outputs:
        print("PROMPT:", output.prompt)
        print("OUTPUT:", output.outputs[0].text.strip())
        metrics = getattr(output, "metrics", None)
        if metrics is not None:
            print(
                "  time_in_queue=%s first_token=%s finished=%s"
                % (
                    getattr(metrics, "time_in_queue", None),
                    getattr(metrics, "first_token_time", None),
                    getattr(metrics, "finished_time", None),
                )
            )
        print()


if __name__ == "__main__":
    main()
