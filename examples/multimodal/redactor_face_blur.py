"""Demonstrates plugging the `face_blur` redactor into the offload pipeline.

Faces are blurred BEFORE the image bytes are uploaded to the configured store,
so the captured trace can never expose PII faces (DPDP / BFSI compliance use case).

Run:
    pip install 'genai-otel-instrument[multimodal-faces,multimodal-s3,openai]'
    export OPENAI_API_KEY=...
    export GENAI_OTEL_MEDIA_CAPTURE_MODE=full
    export GENAI_OTEL_MEDIA_STORE=minio
    export GENAI_OTEL_MEDIA_STORE_ENDPOINT=http://192.168.206.129:9000
    export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=...
    export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=...
    export GENAI_OTEL_MEDIA_REDACTOR=genai_otel.media.redactors.face_blur
    python examples/multimodal/redactor_face_blur.py path/to/photo.jpg
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from genai_otel import instrument


def main(image_path: str) -> None:
    instrument()
    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install 'genai-otel-instrument[openai]'")
        return

    img = Path(image_path).read_bytes()
    img_b64 = base64.b64encode(img).decode("ascii")

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the scene without identifying any people."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ],
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python redactor_face_blur.py <image-path>")
        sys.exit(1)
    main(sys.argv[1])
