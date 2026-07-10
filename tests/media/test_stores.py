"""MediaStore backend tests (filesystem + get_store dispatch + hardening)."""

import os

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


# ---------------------------------------------------------------------------
# Filesystem store: path-traversal containment + restrictive perms
# ---------------------------------------------------------------------------
def test_filesystem_store_rejects_parent_traversal(tmp_path):
    store = FilesystemStore(root=str(tmp_path), bucket="b")
    with pytest.raises(ValueError):
        store.put(b"pwned", key="../../../etc/evil", mime_type="text/plain")


def test_filesystem_store_rejects_absolute_key(tmp_path):
    store = FilesystemStore(root=str(tmp_path), bucket="b")
    abs_key = "C:/Windows/evil.txt" if os.name == "nt" else "/etc/evil.txt"
    with pytest.raises(ValueError):
        store.put(b"pwned", key=abs_key, mime_type="text/plain")


def test_filesystem_store_rejects_anchored_key(tmp_path):
    # A leading-slash key is anchored and must be rejected on every platform.
    store = FilesystemStore(root=str(tmp_path), bucket="b")
    with pytest.raises(ValueError):
        store.put(b"pwned", key="/etc/evil.txt", mime_type="text/plain")


def test_filesystem_store_allows_contained_dotdot(tmp_path):
    # `..` that resolves back inside the root is fine.
    store = FilesystemStore(root=str(tmp_path), bucket="b")
    uri = store.put(b"ok", key="a/../c.png", mime_type="image/png")
    assert uri.startswith("file:")
    assert (tmp_path / "b" / "c.png").read_bytes() == b"ok"


def test_filesystem_store_restricts_file_permissions(tmp_path):
    store = FilesystemStore(root=str(tmp_path), bucket="b")
    store.put(b"secret", key="x/y.png", mime_type="image/png")
    target = tmp_path / "b" / "x" / "y.png"
    assert target.exists()
    if os.name != "nt":
        # 0o600: owner rw only, no group/other bits.
        mode = target.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# HTTP store: https enforcement + credential stripping
# ---------------------------------------------------------------------------
def test_http_store_rejects_plaintext_when_require_https():
    from genai_otel.media.stores.http import HttpStore

    with pytest.raises(ValueError):
        HttpStore(endpoint="http://gateway.internal/ingest", require_https=True)


def test_http_store_allows_https_when_require_https():
    from genai_otel.media.stores.http import HttpStore

    store = HttpStore(endpoint="https://gateway.internal/ingest", require_https=True)
    assert store._endpoint == "https://gateway.internal/ingest"


def test_http_store_public_endpoint_strips_credentials():
    from genai_otel.media.stores.http import HttpStore

    store = HttpStore(endpoint="https://user:secret@host:8443/ingest")
    assert "secret" not in store._public_endpoint
    assert store._public_endpoint == "https://host:8443/ingest"


def test_http_strip_userinfo_helper():
    from genai_otel.media.stores.http import _strip_userinfo

    assert _strip_userinfo("https://u:p@h/path") == "https://h/path"
    assert _strip_userinfo("https://h/path") == "https://h/path"


# ---------------------------------------------------------------------------
# S3/MinIO store: https enforcement (pre-boto3) + credential stripping helper
# ---------------------------------------------------------------------------
def test_s3_store_rejects_plaintext_when_require_https():
    from genai_otel.media.stores.s3_minio import S3MinioStore

    with pytest.raises(ValueError):
        S3MinioStore(endpoint="http://minio.internal:9000", bucket="b", require_https=True)


def test_s3_strip_userinfo_helper():
    from genai_otel.media.stores.s3_minio import _strip_userinfo

    assert _strip_userinfo("https://KEY:SECRET@minio:9000") == "https://minio:9000"


# ---------------------------------------------------------------------------
# get_store wiring: strict/no-egress posture forces https on network stores
# ---------------------------------------------------------------------------
def test_get_store_http_requires_https_under_strict_profile():
    cfg = _Cfg(
        media_store="http",
        media_store_endpoint="http://gw.internal/ingest",
        profile="strict",
    )
    with pytest.raises(ValueError):
        get_store(cfg)


def test_get_store_http_requires_https_when_egress_disabled():
    cfg = _Cfg(
        media_store="http",
        media_store_endpoint="http://gw.internal/ingest",
        allow_external_egress=False,
    )
    with pytest.raises(ValueError):
        get_store(cfg)


def test_get_store_http_allows_plaintext_in_default_posture():
    cfg = _Cfg(media_store="http", media_store_endpoint="http://gw.internal/ingest")
    store = get_store(cfg)
    assert store is not None
