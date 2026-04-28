"""Verify ContentPart -> upstream-canonical OTel JSON mapping."""

import base64

from genai_otel.media.canonical import build_canonical_messages
from genai_otel.media.detector import ContentPart


def test_text_part_becomes_text_part():
    msgs = build_canonical_messages([("user", [ContentPart(type="text", text="hello")])])
    assert msgs == [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]


def test_image_with_uri_becomes_uri_part():
    part = ContentPart(
        type="image",
        media_uri="https://x/y.png",
        media_mime_type="image/png",
        media_byte_size=1234,
        media_source="external_url",
    )
    msgs = build_canonical_messages([("user", [part])])
    p = msgs[0]["parts"][0]
    assert p["type"] == "uri"
    assert p["modality"] == "image"
    assert p["uri"] == "https://x/y.png"
    assert p["mime_type"] == "image/png"
    assert p["byte_size"] == 1234


def test_image_with_inline_bytes_becomes_blob_part():
    part = ContentPart(
        type="image", data=b"\x89PNG", media_mime_type="image/png", media_byte_size=4
    )
    msgs = build_canonical_messages([("user", [part])])
    p = msgs[0]["parts"][0]
    assert p["type"] == "blob"
    assert p["modality"] == "image"
    assert p["mime_type"] == "image/png"
    assert p["content"] == base64.b64encode(b"\x89PNG").decode()
    assert p["byte_size"] == 4


def test_reference_only_becomes_stripped_part():
    part = ContentPart(
        type="image",
        media_mime_type="image/png",
        media_byte_size=2_000_000,
        media_source="reference_only",
        extra={"stripped_reason": "size_exceeded"},
    )
    msgs = build_canonical_messages([("user", [part])])
    p = msgs[0]["parts"][0]
    assert p["type"] == "stripped"
    assert p["modality"] == "image"
    assert p["stripped_reason"] == "size_exceeded"
    assert p["byte_size"] == 2_000_000


def test_document_modality_emitted_as_string():
    """`document` is a proposed addition; current schema accepts free-form strings."""
    part = ContentPart(
        type="document",
        media_uri="s3://bucket/k.pdf",
        media_mime_type="application/pdf",
        media_source="inline_offloaded",
    )
    msgs = build_canonical_messages([("user", [part])])
    p = msgs[0]["parts"][0]
    assert p["type"] == "uri"
    assert p["modality"] == "document"
    assert p["mime_type"] == "application/pdf"


def test_video_modality():
    part = ContentPart(type="video", media_uri="gs://b/clip.mp4", media_mime_type="video/mp4")
    msgs = build_canonical_messages([("user", [part])])
    p = msgs[0]["parts"][0]
    assert p["type"] == "uri"
    assert p["modality"] == "video"


def test_multi_message_multi_part():
    msgs = build_canonical_messages(
        [
            (
                "user",
                [
                    ContentPart(type="text", text="describe"),
                    ContentPart(
                        type="image", media_uri="https://x/y.png", media_mime_type="image/png"
                    ),
                ],
            ),
            ("assistant", [ContentPart(type="text", text="a cat")]),
        ]
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert len(msgs[0]["parts"]) == 2
    assert msgs[0]["parts"][0]["type"] == "text"
    assert msgs[0]["parts"][1]["type"] == "uri"
    assert msgs[1]["role"] == "assistant"
