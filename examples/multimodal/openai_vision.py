"""OpenAI vision example with multimodal content-part instrumentation.

Run:
    export OPENAI_API_KEY=...
    export GENAI_OTEL_MEDIA_CAPTURE_MODE=reference_only
    python examples/multimodal/openai_vision.py
"""

from __future__ import annotations

import os

from genai_otel import instrument


def main() -> None:
    instrument()  # auto-instruments OpenAI

    # Lazy import so the example can show the instrumentor wiring even if openai isn't installed
    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install 'genai-otel-instrument[openai]'")
        return

    client = OpenAI()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
                        },
                    },
                ],
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
