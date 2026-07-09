"""Built-in redactors for multimodal content.

All redactors share the signature::

    def redactor(modality: str, mime_type: str, data: bytes) -> bytes

A redactor returns the (possibly modified) bytes. Heavy dependencies (Pillow,
opencv, mediapipe, pypdf) are imported lazily so the library stays light when
these features aren't used.

Security posture: these redactors are **fail-closed**. When a redactor is
explicitly configured for a modality it applies to but cannot complete its job
(missing optional dependency, corrupt/undecodable input, or a suspected
decompression/pixel bomb), it raises instead of returning the original bytes.
``offload_part`` catches that exception and drops the payload, so unredacted
customer content is never uploaded while claiming redaction. A redactor that is
simply not applicable to a given modality (e.g. ``exif_stripper`` on an audio
part) is a legitimate no-op and returns the bytes unchanged.
"""

from __future__ import annotations

import io
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Hard ceiling on decoded raster size to bound decompression / pixel bombs
# before any full-image parse. A small compressed file can expand to a huge
# raster; refuse anything above this many pixels rather than allocating it.
# ~64 megapixels (e.g. 8000x8000) is well above legitimate document/photo use.
_MAX_IMAGE_PIXELS = 64_000_000


def exif_stripper(modality: str, mime_type: str, data: bytes) -> bytes:
    """Strip EXIF / metadata from JPEG/PNG images. No-op for non-image parts.

    Fail-closed: if Pillow is unavailable or the image cannot be decoded/parsed,
    raises so the caller drops the bytes rather than uploading them unredacted.
    """
    if modality != "image":
        return data
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "exif_stripper requires Pillow but it is not installed; failing closed"
        ) from e

    # Bound decompression / pixel bombs before decoding the full raster.
    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        img_format = img.format or "PNG"
        # Force decode now so a DecompressionBombError surfaces here (and is
        # translated into a fail-closed RuntimeError below) rather than later.
        img.load()
        # Re-build the image from pixel data only, dropping the info/EXIF dict.
        # Pass the getdata() sequence straight to putdata() to avoid
        # materializing a full Python list of pixel tuples in memory.
        clean = Image.new(img.mode, img.size)
        clean.putdata(img.getdata())
        out = io.BytesIO()
        clean.save(out, format=img_format)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"exif_stripper failed; failing closed: {e}") from e


def face_blur(modality: str, mime_type: str, data: bytes) -> bytes:
    """Detect faces and Gaussian-blur them. Requires opencv-python.

    Fail-closed: if opencv (or its haar cascade) is unavailable, the image is
    undecodable, or it exceeds the pixel-bomb bound, raises so the caller drops
    the bytes rather than uploading unblurred faces. For production-grade
    deployments use a dedicated face-detection model.
    """
    if modality != "image":
        return data
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "face_blur requires opencv-python but it is not installed; failing closed"
        ) from e

    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("face_blur could not decode image bytes")
        # Bound decompression / pixel bombs post-decode: refuse oversized rasters.
        h, w = img.shape[:2]
        if h * w > _MAX_IMAGE_PIXELS:
            raise RuntimeError(
                f"face_blur refusing oversized image {w}x{h} (> {_MAX_IMAGE_PIXELS} px)"
            )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError("face_blur haar cascade unavailable")
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        for x, y, w_box, h_box in faces:
            roi = img[y : y + h_box, x : x + w_box]
            roi = cv2.GaussianBlur(roi, (51, 51), 30)
            img[y : y + h_box, x : x + w_box] = roi
        ext = ".jpg" if (mime_type or "").endswith("jpeg") else ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise RuntimeError("face_blur failed to re-encode image")
        return buf.tobytes()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"face_blur failed; failing closed: {e}") from e


# PII patterns reused across image text-overlay (OCR) and PDF redaction.
# Keep this set small and BFSI-relevant; for full coverage, plug in a custom
# redactor that calls genai_otel.evaluation.PII detectors.
_PII_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),  # Aadhaar
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),  # PAN
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-ish
    re.compile(r"\b\d{16}\b"),  # Card-like
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email
    re.compile(r"\+?\d[\d\s\-]{8,}\d"),  # phone
]


def pdf_pii_scan_fail_closed(modality: str, mime_type: str, data: bytes) -> bytes:
    """PDF PII gate. Does NOT perform visual redaction.

    pypdf cannot reliably strip text from a PDF's visual layer, so this
    function does not pretend to. Instead it acts as a **fail-closed gate**:

    - Non-PDF / non-document parts: no-op passthrough (not applicable).
    - PDF with NO detected PII: returned unchanged (safe to offload as-is).
    - PDF WITH detected PII: raises, so ``offload_part`` drops the bytes. We
      never upload the original document while asserting it was redacted, and
      we never stamp ``/RedactionApplied`` metadata onto an unredacted file.
    - pypdf missing, or the document cannot be parsed: raises (we cannot prove
      the document is PII-free, so we fail closed).

    For true visual redaction, plug in a layout-aware tool (e.g. a
    pdf-redactor / OCR-mask pipeline) via a custom redactor callable.
    """
    if modality != "document":
        return data
    if not (mime_type or "").lower().startswith("application/pdf"):
        return data
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pdf_pii_scan_fail_closed requires pypdf but it is not installed; failing closed"
        ) from e

    try:
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            text = page.extract_text() or ""
            for pat in _PII_PATTERNS:
                if pat.search(text):
                    raise RuntimeError(
                        "PII detected in PDF and true visual redaction is unavailable; "
                        "failing closed (dropping document bytes)"
                    )
    except RuntimeError:
        # PII-detected sentinel (or an explicit fail-closed error): propagate.
        raise
    except Exception as e:  # noqa: BLE001
        # Parse failure: we cannot verify the document is PII-free -> fail closed.
        raise RuntimeError(
            f"pdf_pii_scan_fail_closed could not parse PDF; failing closed: {e}"
        ) from e

    # No PII detected: safe to offload the original document unchanged.
    return data


# Backwards-compatible alias. The old name implied it performed redaction; it
# never did. It now points at the fail-closed gate so any existing
# configuration (GENAI_OTEL_MEDIA_REDACTOR=genai_otel.media.redactors.pdf_pii_redact)
# gets the safe behavior. Prefer the explicit name in new configs.
pdf_pii_redact = pdf_pii_scan_fail_closed
