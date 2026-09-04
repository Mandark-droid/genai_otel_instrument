"""llama.cpp (llama-cpp-python) with OpenTelemetry instrumentation.

Traces the in-process `Llama` completion entry points. Where the build reports
timings, spans carry the prefill and decode phases derived from llama.cpp's
millisecond counters.

Run:
    pip install "genai-otel-instrument[llamacpp]"
    LLAMACPP_MODEL_PATH=/path/to/model.gguf \
        python examples/inference_engines/llamacpp_example.py
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import genai_otel

genai_otel.instrument()

from llama_cpp import Llama  # noqa: E402


def main():
    model_path = os.getenv("LLAMACPP_MODEL_PATH")
    if not model_path:
        print("Set LLAMACPP_MODEL_PATH to a local .gguf file first.")
        return

    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Give one reason to run models locally."}],
        max_tokens=64,
    )

    print("OUTPUT:", result["choices"][0]["message"]["content"].strip())
    usage = result.get("usage", {})
    print(
        "  prompt_tokens=%s completion_tokens=%s"
        % (usage.get("prompt_tokens"), usage.get("completion_tokens"))
    )


if __name__ == "__main__":
    main()
