"""Anthropic document content block example (PDF KYC-style extraction).

Run:
    export ANTHROPIC_API_KEY=...
    export GENAI_OTEL_MEDIA_CAPTURE_MODE=full
    export GENAI_OTEL_MEDIA_STORE=filesystem
    export GENAI_OTEL_MEDIA_STORE_ENDPOINT=./.genai-otel-media
    export GENAI_OTEL_MEDIA_REDACTOR=genai_otel.media.redactors.pdf_pii_redact
    python examples/multimodal/document_extraction.py path/to/form.pdf
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from genai_otel import instrument


def main(pdf_path: str) -> None:
    instrument()
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Install anthropic: pip install 'genai-otel-instrument[anthropic]'")
        return

    pdf_bytes = Path(pdf_path).read_bytes()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    client = Anthropic()
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract the applicant name, PAN, and date of birth as JSON.",
                    },
                ],
            }
        ],
    )
    print(msg.content[0].text if msg.content else "")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_extraction.py <pdf-path>")
        sys.exit(1)
    main(sys.argv[1])
