"""Provider x modality detection matrix."""

import base64

from genai_otel.media.detector import ContentPart, detect_parts


def test_text_content_passthrough():
    parts = detect_parts("openai", "hello")
    assert len(parts) == 1
    assert parts[0].type == "text"
    assert parts[0].text == "hello"


def test_openai_image_url_part():
    parts = detect_parts(
        "openai",
        [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
        ],
    )
    assert [p.type for p in parts] == ["text", "image"]
    assert parts[1].external_url == "https://example.com/x.png"
    assert parts[1].data is None


def test_openai_data_url_image_decoded():
    png_b64 = base64.b64encode(b"\x89PNG\r\n").decode()
    parts = detect_parts(
        "openai",
        [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}}],
    )
    assert parts[0].type == "image"
    assert parts[0].media_mime_type == "image/png"
    assert parts[0].data is not None
    assert parts[0].media_byte_size == len(parts[0].data)


def test_openai_input_audio():
    audio_b64 = base64.b64encode(b"RIFF....WAVE").decode()
    parts = detect_parts(
        "openai",
        [{"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}],
    )
    assert parts[0].type == "audio"
    assert parts[0].media_mime_type == "audio/wav"


def test_anthropic_base64_image():
    img_b64 = base64.b64encode(b"\x89PNG").decode()
    parts = detect_parts(
        "anthropic",
        [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
            }
        ],
    )
    assert parts[0].type == "image"
    assert parts[0].media_mime_type == "image/png"
    assert parts[0].data is not None


def test_anthropic_url_image_no_data():
    parts = detect_parts(
        "anthropic",
        [{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}],
    )
    assert parts[0].type == "image"
    assert parts[0].external_url == "https://x/y.png"
    assert parts[0].data is None


def test_anthropic_document_pdf():
    pdf_b64 = base64.b64encode(b"%PDF-1.4").decode()
    parts = detect_parts(
        "anthropic",
        [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
            }
        ],
    )
    assert parts[0].type == "document"
    assert parts[0].media_mime_type == "application/pdf"


def test_google_inline_data_image():
    img_b64 = base64.b64encode(b"\x89PNG").decode()
    parts = detect_parts(
        "google",
        [{"inline_data": {"mime_type": "image/png", "data": img_b64}}],
    )
    assert parts[0].type == "image"
    assert parts[0].media_mime_type == "image/png"


def test_google_inline_data_video_classified():
    parts = detect_parts(
        "google",
        [{"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(b"x").decode()}}],
    )
    assert parts[0].type == "video"
    assert parts[0].media_mime_type == "video/mp4"


def test_google_file_data_video_external():
    parts = detect_parts(
        "google",
        [{"file_data": {"mime_type": "video/webm", "file_uri": "gs://bucket/clip.webm"}}],
    )
    assert parts[0].type == "video"
    assert parts[0].external_url == "gs://bucket/clip.webm"


def test_openai_input_video_data_url():
    vid_b64 = base64.b64encode(b"\x00\x00\x00\x18ftyp").decode()
    parts = detect_parts(
        "openai",
        [{"type": "input_video", "input_video": {"url": f"data:video/mp4;base64,{vid_b64}"}}],
    )
    assert parts[0].type == "video"
    assert parts[0].media_mime_type == "video/mp4"
    assert parts[0].data is not None


def test_google_inline_data_audio():
    parts = detect_parts(
        "google",
        [{"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(b"x").decode()}}],
    )
    assert parts[0].type == "audio"


def test_google_file_data_external():
    parts = detect_parts(
        "google",
        [{"file_data": {"mime_type": "application/pdf", "file_uri": "gs://bucket/x.pdf"}}],
    )
    assert parts[0].type == "document"
    assert parts[0].external_url == "gs://bucket/x.pdf"


def test_groq_uses_openai_path():
    parts = detect_parts(
        "groq",
        [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
    )
    assert parts[0].type == "image"


def test_unknown_provider_falls_back_to_openai():
    parts = detect_parts("unknown-provider", [{"type": "text", "text": "hi"}])
    assert parts[0].type == "text"


def test_none_content_returns_empty():
    assert detect_parts("openai", None) == []
