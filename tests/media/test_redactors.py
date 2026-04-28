"""Built-in redactors are best-effort and gracefully degrade without optional deps."""

from genai_otel.media.redactors import exif_stripper, face_blur, pdf_pii_redact


def test_exif_stripper_passthrough_for_non_image():
    assert exif_stripper("audio", "audio/wav", b"data") == b"data"


def test_face_blur_passthrough_for_non_image():
    assert face_blur("document", "application/pdf", b"data") == b"data"


def test_pdf_pii_redact_passthrough_for_non_pdf():
    assert pdf_pii_redact("image", "image/png", b"data") == b"data"


def test_pdf_pii_redact_passthrough_for_non_document():
    # Even if mime claims pdf, modality must match
    assert pdf_pii_redact("image", "application/pdf", b"data") == b"data"


def test_exif_stripper_handles_garbage_image():
    # Not a real image; should return original rather than crash
    out = exif_stripper("image", "image/png", b"not-a-png")
    assert out == b"not-a-png"


def test_face_blur_handles_garbage_image():
    out = face_blur("image", "image/png", b"not-an-image")
    assert out == b"not-an-image"


def test_pdf_pii_redact_handles_garbage_pdf():
    out = pdf_pii_redact("document", "application/pdf", b"not-a-pdf")
    assert out == b"not-a-pdf"
