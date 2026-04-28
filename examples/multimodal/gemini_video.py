"""Google Gemini video example with inline_data video bytes.

Run:
    export GOOGLE_API_KEY=...
    export GENAI_OTEL_MEDIA_CAPTURE_MODE=full
    export GENAI_OTEL_MEDIA_STORE=minio
    export GENAI_OTEL_MEDIA_STORE_ENDPOINT=http://localhost:9000
    export GENAI_OTEL_MEDIA_STORE_ACCESS_KEY=...
    export GENAI_OTEL_MEDIA_STORE_SECRET_KEY=...
    python examples/multimodal/gemini_video.py path/to/clip.mp4
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from genai_otel import instrument


def main(video_path: str) -> None:
    instrument()
    try:
        import google.generativeai as genai
    except ImportError:
        print("Install google-generativeai: pip install 'genai-otel-instrument[google]'")
        return

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    video_b64 = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
    resp = model.generate_content(
        [
            {"text": "Summarize the events in this clip in 3 bullets."},
            {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
        ]
    )
    print(resp.text if hasattr(resp, "text") else resp)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gemini_video.py <video-path>")
        sys.exit(1)
    main(sys.argv[1])
