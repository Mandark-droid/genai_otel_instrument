"""Google Gemini multimodal example with inline audio data.

Run:
    export GOOGLE_API_KEY=...
    export GENAI_OTEL_MEDIA_CAPTURE_MODE=reference_only
    python examples/multimodal/gemini_multimodal.py
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from genai_otel import instrument


def _audio_bytes() -> bytes:
    sample = Path(__file__).parent / "fixtures" / "sample.wav"
    if sample.exists():
        return sample.read_bytes()
    # 44-byte minimal WAV header so the example runs without fixtures
    return base64.b64decode("UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=")


def main() -> None:
    instrument()
    try:
        import google.generativeai as genai
    except ImportError:
        print("Install google-generativeai: pip install 'genai-otel-instrument[google]'")
        return

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    audio_b64 = base64.b64encode(_audio_bytes()).decode("ascii")
    resp = model.generate_content(
        [
            {"text": "Transcribe the following audio."},
            {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
        ]
    )
    print(resp.text if hasattr(resp, "text") else resp)


if __name__ == "__main__":
    main()
