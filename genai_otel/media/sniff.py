"""What kind of media a payload is, from its type, MIME type or first bytes.

Decision requests (and other non-chat payloads) can carry raw bytes, data URLs or
SDK asset objects. Before this module, decision serialisation called any bytes an
image, so an audio clip was recorded as ``[image omitted]``: a false statement on
the span. Classification here is conservative: bytes that match no known header are
``binary``, never guessed to be an image or audio.
"""

from __future__ import annotations

import io
import wave
from typing import Any, Optional, Tuple

from .speech import audio_mime

#: (modality, mime). Modality is ``image``, ``audio`` or ``binary`` (not a modality).
Classified = Tuple[str, Optional[str]]

# MPEG audio Layer III bitrates (kbit/s) by bitrate index, and sample rates by index.
_L3_BITRATES = {
    3: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),  # MPEG-1
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),  # MPEG-2
    0: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),  # MPEG-2.5
}
_SAMPLE_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def _l3_frame_length(data: bytes, offset: int) -> int:
    """Length of a valid MPEG Layer III frame starting at ``offset``, or 0 if none."""
    if offset + 4 > len(data) or data[offset] != 0xFF:
        return 0
    b1, b2 = data[offset + 1], data[offset + 2]
    version, layer = (b1 >> 3) & 0x03, (b1 >> 1) & 0x03
    bitrate_index, rate_index = (b2 >> 4) & 0x0F, (b2 >> 2) & 0x03
    if (b1 & 0xE0) != 0xE0 or version == 1 or layer != 1:
        return 0
    if bitrate_index in (0, 15) or rate_index == 3:
        return 0
    bitrate = _L3_BITRATES[version][bitrate_index] * 1000
    rate = _SAMPLE_RATES[version][rate_index]
    samples_factor = 144 if version == 3 else 72
    return samples_factor * bitrate // rate + ((b2 >> 1) & 0x01)


def _is_mp3(data: bytes) -> bool:
    if data[:3] == b"ID3":
        return True
    # No tag: two consecutive valid frame headers, so one stray sync word in
    # unrelated bytes is not read as audio.
    first = _l3_frame_length(data, 0)
    return bool(first) and _l3_frame_length(data, first) > 0


def sniff_bytes(data: Any) -> Optional[Classified]:
    """Classify raw bytes by their header. None for empty or non-bytes input."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return None
    head = bytes(data[:4096])
    if not head:
        return None
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio", "audio/wav"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image", "image/webp"
    if head[:4] == b"OggS":
        return "audio", "audio/ogg"
    if head[:4] == b"fLaC":
        return "audio", "audio/flac"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "audio", "audio/webm"
    if head[:4] == b"\x89PNG":
        return "image", "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image", "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image", "image/gif"
    if _is_mp3(head):
        return "audio", "audio/mpeg"
    return "binary", None


def classify_mime(mime: Any) -> Optional[Classified]:
    """Classify by MIME type: ``audio/*`` or ``image/*``; None for anything else."""
    if not isinstance(mime, str) or "/" not in mime:
        return None
    major = mime.split("/", 1)[0].strip().lower()
    if major in ("audio", "image"):
        return major, mime.strip().lower()
    return None


def classify_data_url(value: Any) -> Optional[Classified]:
    """Classify a ``data:<mime>;base64,...`` string; None if it is not a data URL."""
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    header = value[5:200].split(",", 1)[0]
    if ";base64" not in header:
        return None
    return classify_mime(header.split(";", 1)[0]) or ("binary", None)


def audio_format_mime(fmt: Any) -> str:
    """The MIME type for an ``input_audio`` part's ``format`` (``wav``, ``mp3``...)."""
    return audio_mime(codec=str(fmt) if fmt else None, default="audio/wav")


def wav_seconds(data: Any) -> Optional[float]:
    """Duration of in-memory WAV bytes from the header alone, or None if not a readable WAV."""
    if not isinstance(data, (bytes, bytearray)) or data[:4] != b"RIFF":
        return None
    try:
        with wave.open(io.BytesIO(bytes(data)), "rb") as w:
            rate = w.getframerate()
            return w.getnframes() / float(rate) if rate else None
    except Exception:  # noqa: BLE001 - not a WAV we can read: no duration, never a guess
        return None
