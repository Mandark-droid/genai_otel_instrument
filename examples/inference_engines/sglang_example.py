"""SGLang in-process generation with OpenTelemetry instrumentation.

Traces `sglang.Engine.generate`. Spans carry SGLang's per-request meta_info:
token counts, prefix-cache hits (mapped onto the conventions' cache_read
concept) and the latency fields the release exposes.

Run:
    pip install "genai-otel-instrument[sglang]"
    python examples/inference_engines/sglang_example.py
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import genai_otel

genai_otel.instrument()

import sglang  # noqa: E402


def main():
    engine = sglang.Engine(model_path="meta-llama/Llama-3.2-1B-Instruct")

    result = engine.generate(
        "Summarise why prefix caching helps repeated system prompts.",
        sampling_params={"temperature": 0.7, "max_new_tokens": 64},
    )

    print("OUTPUT:", result["text"].strip())
    meta = result.get("meta_info", {})
    print(
        "  prompt_tokens=%s completion_tokens=%s cached_tokens=%s e2e_latency=%s"
        % (
            meta.get("prompt_tokens"),
            meta.get("completion_tokens"),
            meta.get("cached_tokens"),
            meta.get("e2e_latency"),
        )
    )
    engine.shutdown()


if __name__ == "__main__":
    main()
