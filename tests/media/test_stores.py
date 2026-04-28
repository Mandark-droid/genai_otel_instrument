"""MediaStore backend tests (filesystem + get_store dispatch)."""

import pytest

from genai_otel.media.stores import get_store
from genai_otel.media.stores.filesystem import FilesystemStore
from genai_otel.media.stores.noop import NoopStore


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_get_store_none():
    assert get_store(_Cfg(media_store="none")) is None


def test_get_store_filesystem(tmp_path):
    cfg = _Cfg(
        media_store="filesystem",
        media_store_endpoint=str(tmp_path),
        media_store_bucket="test-bucket",
    )
    store = get_store(cfg)
    assert isinstance(store, FilesystemStore)


def test_get_store_unknown_returns_none(caplog):
    assert get_store(_Cfg(media_store="garbage")) is None


def test_filesystem_store_writes_and_returns_uri(tmp_path):
    store = FilesystemStore(root=str(tmp_path), bucket="b")
    uri = store.put(b"hello", key="a/b.png", mime_type="image/png")
    assert uri.startswith("file:")
    written = (tmp_path / "b" / "a" / "b.png").read_bytes()
    assert written == b"hello"


def test_noop_store_returns_pseudo_uri():
    s = NoopStore()
    assert s.put(b"x", key="k", mime_type="image/png") == "noop://k"
