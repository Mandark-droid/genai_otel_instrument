"""Built-in redactors are FAIL-CLOSED.

A redactor that is not applicable to a modality is a legitimate no-op and
returns the bytes unchanged. But when a redactor applies to the modality and
cannot complete its job (missing optional dep, undecodable input, detected PII
it cannot truly remove), it MUST raise so ``offload_part`` drops the bytes
rather than uploading unredacted customer content.
"""

import builtins
import io

import pytest

from genai_otel.media.redactors import (
    exif_stripper,
    face_blur,
    pdf_pii_redact,
    pdf_pii_scan_fail_closed,
)


# ---------------------------------------------------------------------------
# Not-applicable passthroughs (NOT fail-open: nothing to redact for this modality)
# ---------------------------------------------------------------------------
def test_exif_stripper_passthrough_for_non_image():
    assert exif_stripper("audio", "audio/wav", b"data") == b"data"


def test_face_blur_passthrough_for_non_image():
    assert face_blur("document", "application/pdf", b"data") == b"data"


def test_pdf_pii_redact_passthrough_for_non_pdf():
    assert pdf_pii_scan_fail_closed("image", "image/png", b"data") == b"data"


def test_pdf_pii_redact_passthrough_for_non_document():
    # Even if mime claims pdf, modality must match
    assert pdf_pii_scan_fail_closed("image", "application/pdf", b"data") == b"data"


# ---------------------------------------------------------------------------
# Fail-closed on undecodable / corrupt input
# ---------------------------------------------------------------------------
def test_exif_stripper_fails_closed_on_garbage_image():
    with pytest.raises(RuntimeError):
        exif_stripper("image", "image/png", b"not-a-png")


def test_face_blur_fails_closed_on_bad_input():
    # cv2 may be absent (fail-closed on missing dep) or present (decode fails);
    # either way it must raise, never return the original bytes.
    with pytest.raises(RuntimeError):
        face_blur("image", "image/png", b"not-an-image")


def test_pdf_pii_redact_fails_closed_on_garbage_pdf():
    with pytest.raises(RuntimeError):
        pdf_pii_scan_fail_closed("document", "application/pdf", b"not-a-pdf")


# ---------------------------------------------------------------------------
# Fail-closed on missing optional dependency
# ---------------------------------------------------------------------------
def test_exif_stripper_fails_closed_when_pillow_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("simulated: Pillow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError):
        exif_stripper("image", "image/png", b"whatever")


def test_face_blur_fails_closed_when_opencv_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("simulated: opencv not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError):
        face_blur("image", "image/png", b"whatever")


# ---------------------------------------------------------------------------
# exif_stripper valid path: strips and does not raise
# ---------------------------------------------------------------------------
def test_exif_stripper_strips_valid_image_without_raising():
    pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    out = exif_stripper("image", "image/png", buf.getvalue())
    assert isinstance(out, bytes) and len(out) > 0
    reopened = Image.open(io.BytesIO(out))
    assert reopened.size == (8, 8)


# ---------------------------------------------------------------------------
# pdf_pii_scan_fail_closed: raise on PII, pass clean docs through unchanged,
# and NEVER stamp /RedactionApplied on an unredacted document.
# ---------------------------------------------------------------------------
def _fake_pdf_reader_factory(text):
    class _Page:
        def extract_text(self):
            return text

    class _Reader:
        def __init__(self, *args, **kwargs):
            self.pages = [_Page()]

    return _Reader


def test_pdf_pii_scan_raises_when_pii_detected(monkeypatch):
    pytest.importorskip("pypdf")
    import pypdf

    # PAN pattern: 5 letters, 4 digits, 1 letter
    monkeypatch.setattr(pypdf, "PdfReader", _fake_pdf_reader_factory("customer PAN ABCDE1234F"))
    with pytest.raises(RuntimeError):
        pdf_pii_scan_fail_closed("document", "application/pdf", b"%PDF-1.4 fake")


def test_pdf_pii_scan_passes_clean_pdf_through_unchanged(monkeypatch):
    pytest.importorskip("pypdf")
    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", _fake_pdf_reader_factory("nothing sensitive here"))
    data = b"%PDF-1.4 clean-body"
    out = pdf_pii_scan_fail_closed("document", "application/pdf", data)
    # Returned unchanged: no re-write, no /RedactionApplied metadata stamped.
    assert out == data
    assert b"RedactionApplied" not in out


def test_pdf_pii_redact_is_alias_for_fail_closed_scan():
    # The legacy name must resolve to the safe, fail-closed implementation so
    # existing GENAI_OTEL_MEDIA_REDACTOR configs get the corrected behavior.
    assert pdf_pii_redact is pdf_pii_scan_fail_closed
