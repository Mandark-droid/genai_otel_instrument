"""Offload pipeline behaviour: mode gating, size cap, redactor errors."""

from dataclasses import dataclass
from typing import Optional

import pytest

from genai_otel.media.detector import ContentPart
from genai_otel.media.offload import offload_part


@dataclass
class _Cfg:
    media_capture_mode: str = "off"
    media_store_prefix: str = "traces/{date}/{trace_id}/"
    media_max_bytes: int = 10 * 1024 * 1024
    media_allowed_modalities: str = "image,audio,video,document"
    media_redactor: Optional[str] = None


class _MemStore:
    def __init__(self):
        self.calls = []

    def put(self, data, *, key, mime_type):
        self.calls.append((key, mime_type, len(data)))
        return f"mem://{key}"


def test_off_mode_is_passthrough():
    part = ContentPart(type="image", data=b"x" * 10, media_mime_type="image/png")
    out = offload_part(part, config=_Cfg(media_capture_mode="off"), store=None)
    assert out.media_uri is None
    assert out.media_source is None


def test_text_part_never_offloaded():
    part = ContentPart(type="text", text="hello")
    out = offload_part(part, config=_Cfg(media_capture_mode="full"), store=_MemStore())
    assert out.media_uri is None


def test_reference_only_drops_data():
    part = ContentPart(type="image", data=b"abc", media_mime_type="image/png")
    out = offload_part(part, config=_Cfg(media_capture_mode="reference_only"), store=_MemStore())
    assert out.media_source == "reference_only"
    assert out.data is None


def test_external_url_not_rehosted():
    part = ContentPart(type="image", external_url="https://x/y.png", media_mime_type="image/png")
    store = _MemStore()
    out = offload_part(part, config=_Cfg(media_capture_mode="full"), store=store)
    assert out.media_source == "external_url"
    assert out.media_uri == "https://x/y.png"
    assert store.calls == []  # not uploaded


def test_full_mode_uploads():
    part = ContentPart(type="image", data=b"abc", media_mime_type="image/png")
    store = _MemStore()
    out = offload_part(part, config=_Cfg(media_capture_mode="full"), store=store)
    assert out.media_source == "inline_offloaded"
    assert out.media_uri.startswith("mem://")
    assert out.data is None  # bytes dropped after upload
    assert "sha256" in out.extra
    assert len(store.calls) == 1


def test_size_cap_strips_bytes():
    part = ContentPart(type="image", data=b"x" * 100, media_mime_type="image/png")
    cfg = _Cfg(media_capture_mode="full", media_max_bytes=10)
    store = _MemStore()
    out = offload_part(part, config=cfg, store=store)
    assert out.media_source == "reference_only"
    assert out.extra["stripped_reason"] == "size_exceeded"
    assert out.media_byte_size == 100
    assert store.calls == []


def test_disallowed_modality_strips():
    part = ContentPart(type="audio", data=b"abc", media_mime_type="audio/wav")
    cfg = _Cfg(media_capture_mode="full", media_allowed_modalities="image")
    out = offload_part(part, config=cfg, store=_MemStore())
    assert out.media_source == "reference_only"
    assert out.extra["stripped_reason"] == "modality_not_allowed"


def _failing_redactor(modality, mime, data):
    raise RuntimeError("boom")


def test_redactor_failure_drops_bytes(monkeypatch):
    part = ContentPart(type="image", data=b"abc", media_mime_type="image/png")
    cfg = _Cfg(
        media_capture_mode="full",
        media_redactor="tests.media.test_offload._failing_redactor",
    )
    out = offload_part(part, config=cfg, store=_MemStore())
    assert out.media_source == "reference_only"
    assert out.extra["stripped_reason"] == "redactor_error"


def test_no_store_in_full_mode_degrades():
    part = ContentPart(type="image", data=b"abc", media_mime_type="image/png")
    out = offload_part(part, config=_Cfg(media_capture_mode="full"), store=None)
    assert out.media_source == "reference_only"
    assert out.extra["stripped_reason"] == "no_data_or_store"
