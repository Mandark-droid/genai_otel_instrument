"""Anthropic vision example with base64-encoded image content part.

Run:
    export ANTHROPIC_API_KEY=...
    export GENAI_OTEL_MEDIA_CAPTURE_MODE=full
    export GENAI_OTEL_MEDIA_STORE=filesystem
    export GENAI_OTEL_MEDIA_STORE_ENDPOINT=./.genai-otel-media
    python examples/multimodal/anthropic_vision.py
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from genai_otel import instrument


def _sample_image_b64() -> str:
    sample = Path(__file__).parent / "fixtures" / "sample.png"
    if sample.exists():
        return base64.b64encode(sample.read_bytes()).decode("ascii")
    # 1x1 transparent PNG so the example runs without fixtures
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAW"
        "Z2k+EAAAAASUVORK5CYII="
    )


def main() -> None:
    instrument()
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Install anthropic: pip install 'genai-otel-instrument[anthropic]'")
        return

    client = Anthropic()
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _sample_image_b64(),
                        },
                    },
                    {"type": "text", "text": "What do you see?"},
                ],
            }
        ],
    )
    print(msg.content[0].text if msg.content else "")


if __name__ == "__main__":
    main()
