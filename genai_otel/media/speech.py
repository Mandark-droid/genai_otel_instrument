"""Speech payloads as content parts.

Speech SDKs (Sarvam, ElevenLabs) do not take chat messages: speech-to-text takes an
audio file, text-to-speech returns audio bytes or base64. These helpers turn those
payloads into :class:`ContentPart` objects so they go through the same offload path,
size cap, modality allow-list and redactor as every other media part.

Reading is bounded: at most ``max_bytes + 1`` bytes are read from a file, so an
over-cap upload is detected (and dropped by ``offload_part``) without holding the
whole file in memory. A seekable file is returned to the position it was found at,
because the SDK reads it after we do.
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import Any, Optional, Tuple

from .detector import ContentPart

_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".pcm": "audio/pcm",
}

#: Codec / output-format prefixes as the speech SDKs spell them (`mp3_44100_128`, `wav`, `ulaw_8000`).
_MIME_BY_CODEC = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
    "ulaw": "audio/basic",
    "mulaw": "audio/basic",
    "alaw": "audio/x-alaw-basic",
    "linear16": "audio/pcm",
}


def audio_mime(
    filename: Optional[str] = None, codec: Optional[str] = None, default: str = "audio/wav"
) -> str:
    """The MIME type for an audio payload, from its codec name or file extension."""
    if codec:
        head = str(codec).lower().split("_", 1)[0].split("-", 1)[0]
        if head in _MIME_BY_CODEC:
            return _MIME_BY_CODEC[head]
    if filename:
        ext = os.path.splitext(str(filename))[1].lower()
        if ext in _MIME_BY_EXT:
            return _MIME_BY_EXT[ext]
    return default


def read_audio(audio: Any, max_bytes: int) -> Tuple[Optional[bytes], Optional[str], Optional[int]]:
    """Read up to ``max_bytes + 1`` bytes of an SDK audio argument.

    Accepts raw bytes, a path, a ``(name, bytes_or_file, ...)`` upload tuple or a
    binary file object (its position is restored). Returns ``(data, filename, size)``;
    ``size`` is the full size when it is known without reading everything, else the
    number of bytes read. ``(None, None, None)`` for anything else -- an unreadable
    payload is not captured, and never fails the call being traced.
    """
    name: Optional[str] = None
    try:
        if isinstance(audio, tuple) and len(audio) >= 2:
            name = str(audio[0]) if audio[0] is not None else None
            audio = audio[1]
        if isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
            return data[: max_bytes + 1], name, len(data)
        if isinstance(audio, (str, os.PathLike)):
            path = os.fspath(audio)
            size = os.path.getsize(path)
            with open(path, "rb") as fh:
                return fh.read(max_bytes + 1), name or path, size
        if hasattr(audio, "read") and hasattr(audio, "seek") and hasattr(audio, "tell"):
            name = name or getattr(audio, "name", None)
            pos = audio.tell()
            size: Optional[int] = None
            try:
                # The remaining size, measured without reading it: an over-cap upload
                # is reported at its real size, not at the bounded read.
                size = audio.seek(0, os.SEEK_END) - pos
                audio.seek(pos)
                data = audio.read(max_bytes + 1)
            finally:
                audio.seek(pos)
            if isinstance(data, str):
                return None, None, None
            return bytes(data), name if isinstance(name, str) else None, size
    except Exception:  # noqa: BLE001 - capture is best-effort; the traced call must not fail
        return None, None, None
    return None, None, None


def audio_part(
    data: Optional[bytes], mime: str, size: Optional[int] = None
) -> Optional[ContentPart]:
    """An audio content part for ``offload_part``; None when there are no bytes."""
    if not data:
        return None
    full = size if size is not None else len(data)
    return ContentPart(
        type="audio",
        data=data,
        media_mime_type=mime,
        media_byte_size=full,
        # Kept so an over-cap part reports its real size, not the bounded read.
        extra={"declared_size": full},
    )


def decode_base64_audio(b64: Any, max_bytes: int) -> Tuple[Optional[bytes], Optional[int]]:
    """Decode a base64 audio string, refusing (``(None, estimated_size)``) past the cap."""
    if not isinstance(b64, str) or not b64:
        return None, None
    estimated = (len(b64) * 3) // 4
    if estimated > max_bytes:
        return None, estimated
    try:
        return base64.b64decode(b64, validate=False), None
    except (binascii.Error, ValueError):
        return None, None


def text_part(text: Any) -> Optional[ContentPart]:
    """A text content part, or None for an empty or non-string value."""
    if not isinstance(text, str) or not text:
        return None
    return ContentPart(type="text", text=text)
