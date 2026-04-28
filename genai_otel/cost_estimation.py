"""Public token / cost estimation helpers for multimodal calls.

Many providers and pipelines do not surface usage data on their responses
(Ollama multimodal, HuggingFace `image-text-to-text`, ASR, audio-to-audio,
text-to-image, etc.). Backends still need a non-zero token count in order
to drive cost calculation and per-modality observability.

These helpers produce conservative estimates suitable for use as a fallback
when the provider response omits exact counts. Callers SHOULD tag the
resulting span with ``gen_ai.usage.token_count_estimated=true`` so consumers
can distinguish exact from estimated counts.

Public API:
    - ``estimate_pipeline_usage(task, args, kwargs, result, pipe=None)``
    - ``estimate_chat_usage(messages, response_text, image_count=0,
                            audio_seconds=0)``
    - ``count_images(args, kwargs)``
    - ``audio_seconds(args, kwargs, sampling_rate=None)``
    - ``coerce_text(value)``, ``result_text(result)``

The estimates use these constants (kept as module-level so callers can
override per-call by passing kwargs):

    CHARS_PER_TOKEN = 4
    IMAGE_TOKEN_ESTIMATE = 256       # Qwen-VL / llava floor
    AUDIO_TOKENS_PER_SECOND = 50     # ~Whisper rate

Designed to be importable from outside `genai_otel_instrument` (e.g. the
traceverse-chaos-lab `tracesense` providers that bypass the standard
`transformers.pipeline` and `ollama` entry points and so do not benefit
from the instrumentors' built-in estimation paths).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

CHARS_PER_TOKEN = 4
IMAGE_TOKEN_ESTIMATE = 256
AUDIO_TOKENS_PER_SECOND = 50


# --- Text / image / audio probes --------------------------------------------


def coerce_text(value: Any) -> str:
    """Pull printable text out of varied pipeline arg shapes.

    Handles strings, lists/tuples (recursively concatenated), and dicts with
    common content keys (``text``, ``content``, ``prompt``, ``question``,
    ``inputs``, ``raw``). Returns empty string when no text is present.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("text", "content", "prompt", "question", "inputs", "raw"):
            v = value.get(k)
            if isinstance(v, str):
                return v
        return ""
    if isinstance(value, (list, tuple)):
        parts = [coerce_text(item) for item in value]
        return " ".join(p for p in parts if p)
    return ""


def result_text(result: Any) -> str:
    """Extract printable text from a pipeline result (dict / list / str)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for k in (
            "generated_text",
            "translation_text",
            "summary_text",
            "text",
            "answer",
            "label",
        ):
            v = result.get(k)
            if isinstance(v, str):
                return v
        return ""
    if isinstance(result, (list, tuple)):
        parts = [result_text(r) for r in result]
        return " ".join(p for p in parts if p)
    return ""


def _count_image_value(v: Any) -> int:
    """Recursively count image-shaped objects in v."""
    if v is None:
        return 0
    # PIL.Image.Image
    if hasattr(v, "size") and hasattr(v, "mode"):
        return 1
    # numpy.ndarray / torch.Tensor with 2D or 3D shape
    if hasattr(v, "shape") and not isinstance(v, (str, bytes)):
        try:
            ndim = len(v.shape)
            if ndim in (2, 3):
                return 1
        except Exception:  # noqa: BLE001
            pass
    if isinstance(v, str):
        low = v.lower()
        if low.startswith(("http://", "https://", "file://", "/")) and any(
            low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
        ):
            return 1
        return 0
    if isinstance(v, dict):
        # Chat-style {type: "image_url", image_url: {...}} = exactly one image.
        if v.get("type") in ("image", "image_url"):
            return 1
        n = 0
        for k in ("image", "images", "image_url", "raw_image", "url"):
            if k in v:
                n += _count_image_value(v[k])
        return n
    if isinstance(v, (list, tuple)):
        return sum(_count_image_value(x) for x in v)
    return 0


def count_images(args: Iterable[Any], kwargs: Dict[str, Any]) -> int:
    """Count image-like inputs across pipeline positional + keyword args."""
    n = 0
    for v in args or ():
        n += _count_image_value(v)
    for key in ("image", "images", "image_url", "raw_image"):
        if key in (kwargs or {}):
            n += _count_image_value(kwargs[key])
    if "inputs" in (kwargs or {}):
        n += _count_image_value(kwargs["inputs"])
    return n


def _audio_seconds_of(v: Any, sampling_rate: Optional[int]) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (list, tuple)):
        return sum(_audio_seconds_of(x, sampling_rate) for x in v)
    if isinstance(v, dict):
        arr = v.get("array")
        sr = v.get("sampling_rate") or sampling_rate
        if arr is not None and sr:
            try:
                return float(len(arr) / sr)
            except Exception:  # noqa: BLE001
                return 0.0
        return 0.0
    if hasattr(v, "shape") and not isinstance(v, (str, bytes)):
        if sampling_rate:
            try:
                return float(v.shape[-1] / sampling_rate)
            except Exception:  # noqa: BLE001
                return 0.0
    return 0.0


def audio_seconds(
    args: Iterable[Any],
    kwargs: Dict[str, Any],
    sampling_rate: Optional[int] = None,
) -> float:
    """Best-effort total audio duration in seconds across args/kwargs.

    Recognizes:
      - dicts with ``array`` + ``sampling_rate``
      - numpy/torch tensors (uses last-dim length / sampling_rate)
      - lists/tuples of the above

    ``sampling_rate`` may be passed in explicitly when only the tensor is
    available (no per-payload sample rate).
    """
    total = 0.0
    for v in args or ():
        total += _audio_seconds_of(v, sampling_rate)
    for k in ("audio", "audios", "raw_audio", "inputs"):
        if k in (kwargs or {}):
            total += _audio_seconds_of(kwargs[k], sampling_rate)
    return total


# --- High-level estimators --------------------------------------------------


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator if numerator > 0 else 0


def estimate_pipeline_usage(
    task: str,
    args: Iterable[Any] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    result: Any = None,
    pipe: Any = None,
    *,
    chars_per_token: int = CHARS_PER_TOKEN,
    image_token_estimate: int = IMAGE_TOKEN_ESTIMATE,
    audio_tokens_per_second: int = AUDIO_TOKENS_PER_SECOND,
) -> Dict[str, Any]:
    """Estimate token usage for a HuggingFace transformers pipeline-style call.

    Returns a dict with:
        - ``prompt_tokens``: int
        - ``completion_tokens``: int
        - ``total_tokens``: int
        - ``image_count``: int (0 when not vision)
        - ``audio_seconds``: float (0.0 when not audio)
        - ``estimated``: True (always — caller should tag the span accordingly)

    All keys are present even when their values are 0 / 0.0, so callers can
    safely set per-modality span attributes only when the value is positive.

    Returns an empty dict (no keys) when there is genuinely nothing to count
    (no text, no images, no audio, no result text).

    Args:
        task: Pipeline task string (e.g. ``"image-text-to-text"``,
            ``"automatic-speech-recognition"``). Determines which input
            modalities are probed.
        args: Pipeline positional args.
        kwargs: Pipeline keyword args.
        result: Pipeline return value.
        pipe: Optional pipeline object (used to read
            ``pipe.feature_extractor.sampling_rate`` for audio when the
            audio payload doesn't carry its own sample rate).
    """
    kwargs = kwargs or {}
    task_l = (task or "").lower()
    prompt_tokens = 0
    completion_tokens = 0
    image_count = 0
    seconds = 0.0

    # Text input
    text_input = coerce_text(args)
    if "inputs" in kwargs:
        more = coerce_text(kwargs["inputs"])
        if more:
            text_input = (text_input + " " + more).strip()
    text_input = text_input.strip()
    if text_input:
        prompt_tokens += _ceil_div(len(text_input), chars_per_token)

    if any(
        kw in task_l
        for kw in (
            "image-text-to-text",
            "image-to-text",
            "visual-question-answering",
            "image-to-image",
            "image-classification",
            "object-detection",
        )
    ):
        image_count = count_images(args, kwargs)
        prompt_tokens += image_count * image_token_estimate

    if any(
        kw in task_l
        for kw in (
            "automatic-speech-recognition",
            "audio-classification",
            "audio-to-audio",
            "speech-to-speech",
            "voice-activity-detection",
        )
    ):
        sr = None
        if pipe is not None:
            try:
                fe = getattr(pipe, "feature_extractor", None)
                if fe is not None:
                    sr = getattr(fe, "sampling_rate", None)
            except Exception:  # noqa: BLE001
                sr = None
        seconds = audio_seconds(args, kwargs, sr)
        if seconds > 0:
            prompt_tokens += int(seconds * audio_tokens_per_second)

    out_text = result_text(result)
    if out_text:
        completion_tokens = _ceil_div(len(out_text), chars_per_token)

    # Output-side modalities that don't yield text: text-to-image / TTS / etc.
    # Apply a 1-token completion floor so cost calc downstream won't drop the
    # row entirely when only input cost is meaningful.
    if not completion_tokens and any(
        kw in task_l for kw in ("text-to-image", "text-to-audio", "text-to-speech")
    ):
        completion_tokens = 1

    if prompt_tokens == 0 and completion_tokens == 0:
        return {}

    total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "image_count": int(image_count),
        "audio_seconds": float(seconds),
        "estimated": True,
    }


def estimate_chat_usage(
    messages: Optional[list] = None,
    response_text: str = "",
    image_count: int = 0,
    audio_seconds_in: float = 0.0,
    *,
    chars_per_token: int = CHARS_PER_TOKEN,
    image_token_estimate: int = IMAGE_TOKEN_ESTIMATE,
    audio_tokens_per_second: int = AUDIO_TOKENS_PER_SECOND,
) -> Dict[str, Any]:
    """Estimate token counts for a chat-style call (Ollama, custom providers).

    ``messages`` is a list of ``{role, content, images?}`` dicts; ``content``
    may be a string OR a multimodal content array (each part being a dict
    with ``text`` / ``image_url`` / etc.). Returns the same dict shape as
    :func:`estimate_pipeline_usage`. Returns ``{}`` when nothing is countable.
    """
    prompt_chars = 0
    img_count = int(image_count or 0)

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            prompt_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text") or part.get("content")
                    if isinstance(t, str):
                        prompt_chars += len(t)
                    if part.get("type") in ("image", "image_url") or part.get("image_url"):
                        img_count += 1
        imgs = msg.get("images")
        if isinstance(imgs, list):
            img_count += len(imgs)

    prompt_tokens = _ceil_div(prompt_chars, chars_per_token)
    prompt_tokens += img_count * image_token_estimate
    if audio_seconds_in and audio_seconds_in > 0:
        prompt_tokens += int(audio_seconds_in * audio_tokens_per_second)

    completion_tokens = _ceil_div(len(response_text or ""), chars_per_token)

    if prompt_tokens == 0 and completion_tokens == 0:
        return {}

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
        "image_count": int(img_count),
        "audio_seconds": float(audio_seconds_in or 0.0),
        "estimated": True,
    }
