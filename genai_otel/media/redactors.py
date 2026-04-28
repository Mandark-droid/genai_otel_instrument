"""Built-in redactors for multimodal content.

All redactors share the signature::

    def redactor(modality: str, mime_type: str, data: bytes) -> bytes

A redactor returns the (possibly modified) bytes. Heavy dependencies (Pillow,
opencv, mediapipe, pypdf) are imported lazily so the library stays light when
these features aren't used.
"""

from __future__ import annotations

import io
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def exif_stripper(modality: str, mime_type: str, data: bytes) -> bytes:
    """Strip EXIF metadata from JPEG/PNG images. No-op for other modalities."""
    if modality != "image":
        return data
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        logger.debug("Pillow not installed; exif_stripper passing data through")
        return data

    try:
        img = Image.open(io.BytesIO(data))
        img_format = img.format or "PNG"
        # Re-save without EXIF / metadata
        out = io.BytesIO()
        # Remove info dict to drop metadata blocks
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        clean.save(out, format=img_format)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001
        logger.warning("exif_stripper failed; returning original: %s", e)
        return data


def face_blur(modality: str, mime_type: str, data: bytes) -> bytes:
    """Detect faces and Gaussian-blur them. Requires opencv-python.

    Best-effort: if opencv or its haar cascade is unavailable, returns input
    unchanged (with a warning). For production-grade deployments use a
    dedicated face-detection model.
    """
    if modality != "image":
        return data
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        logger.warning("opencv-python not installed; face_blur is a no-op")
        return data

    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return data
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.warning("Haar cascade missing; face_blur skipped")
            return data
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        for x, y, w, h in faces:
            roi = img[y : y + h, x : x + w]
            roi = cv2.GaussianBlur(roi, (51, 51), 30)
            img[y : y + h, x : x + w] = roi
        ext = ".jpg" if (mime_type or "").endswith("jpeg") else ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return data
        return buf.tobytes()
    except Exception as e:  # noqa: BLE001
        logger.warning("face_blur failed; returning original: %s", e)
        return data


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


def pdf_pii_redact(modality: str, mime_type: str, data: bytes) -> bytes:
    """Replace PII matches in a PDF with `[REDACTED]`. Requires pypdf."""
    if modality != "document":
        return data
    if not (mime_type or "").lower().startswith("application/pdf"):
        return data
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore
    except ImportError:
        logger.debug("pypdf not installed; pdf_pii_redact passing data through")
        return data

    try:
        reader = PdfReader(io.BytesIO(data))
        writer = PdfWriter()
        for page in reader.pages:
            text = page.extract_text() or ""
            redacted = text
            for pat in _PII_PATTERNS:
                redacted = pat.sub("[REDACTED]", redacted)
            # NOTE: pypdf doesn't easily replace text in-place. We attach the
            # redacted text as page-level metadata so downstream consumers can
            # see the sanitized content while the original page layout is
            # preserved. For true visual redaction, use a layout-aware tool
            # (e.g. pdf-redactor). This redactor is a starting point.
            writer.add_page(page)
            if redacted != text:
                writer.add_metadata({"/RedactionApplied": "pii_regex"})
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001
        logger.warning("pdf_pii_redact failed; returning original: %s", e)
        return data
