"""OpenAI speech endpoints: transcriptions, translations and text-to-speech (issue #46).

``client.audio.transcriptions.create``, ``client.audio.translations.create`` and
``client.audio.speech.create`` produced no span, no cost and no media, so in a
voice application on OpenAI the speech half of every turn was invisible.

One wrapper per resource method, installed on the client instance when the
client is built. The SDK builds ``with_raw_response`` and
``with_streaming_response`` from ``<resource>.create`` the first time they are
accessed, so they pass through the same wrapper; the wrapper recognises what came
back (a parsed model, a raw response, a streamed binary response, an event
stream) and ends the span when that result is done.

Attribute names are the ones the Sarvam and ElevenLabs speech spans already use,
so one fact has one name across providers. Cost comes from what OpenAI bills:
seconds of audio, characters, or input/output tokens. When the unit is not
knowable from the call (``gpt-4o-mini-tts`` without ``stream_format="sse"``
returns audio and no usage), the span carries no cost and says why, rather than
an estimate.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
import weakref
from typing import Any, Dict, List, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..media.speech import audio_mime, audio_part, read_audio, text_part, wav_header_seconds
from ._stream_context import drive_async_in_context, drive_in_context
from .base import _server_attributes, find_base_url_claim

logger = logging.getLogger(__name__)

_WRAPPED_MARK = "_genai_otel_openai_audio"
_RESOURCES = (
    ("transcriptions", "openai.audio.transcriptions", "speech_to_text"),
    ("translations", "openai.audio.translations", "speech_to_text"),
    ("speech", "openai.audio.speech", "text_to_speech"),
)
#: What OpenAI translates into: English only.
_TRANSLATION_TARGET = "en"
_WAV_HEAD_BYTES = 4096


def wrap_audio_resources(owner: Any, client: Any) -> int:
    """Wrap the client's speech methods. Returns how many were wrapped (idempotent)."""
    if find_base_url_claim(getattr(client, "base_url", None)):
        return 0  # an aggregator's own instrumentor traces this client
    audio = getattr(client, "audio", None)
    if audio is None:
        return 0
    wrapped = 0
    for attr, span_name, operation in _RESOURCES:
        resource = getattr(audio, attr, None)
        create = getattr(resource, "create", None) if resource is not None else None
        if create is None or getattr(create, _WRAPPED_MARK, False):
            continue
        try:
            resource.create = _wrapper(owner, client, create, attr, span_name, operation)
            wrapped += 1
        except (AttributeError, TypeError) as e:
            logger.debug("Could not wrap openai audio.%s.create: %s", attr, e)
    return wrapped


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #


class _AudioCall:
    """One speech call's span, from the request to the moment its result is done."""

    def __init__(self, owner, client, attr, span_name, operation, kwargs):
        self.owner = owner
        self.attr = attr
        self.operation = operation
        self.kwargs = kwargs
        self.model = str(kwargs.get("model") or "unknown")
        self.start = time.time()
        self.first_chunk: Optional[float] = None
        self.ended = False
        self.input_audio = None
        self.wav_seconds: Optional[float] = None
        self.text = kwargs.get("input") if operation == "text_to_speech" else None
        attrs = self._request_attributes(client)
        self.span = owner.tracer.start_span(
            span_name, attributes=owner._with_provider_aliases(attrs)
        )
        self.recording = self.span.is_recording()
        if self.recording and operation == "speech_to_text":
            self._read_input_audio()

    # -- request -------------------------------------------------------------

    def _request_attributes(self, client) -> Dict[str, Any]:
        kw = self.kwargs
        attrs: Dict[str, Any] = {
            "gen_ai.system": "openai",
            "gen_ai.operation.name": self.operation,
            "gen_ai.request.type": self.operation,
            "gen_ai.request.model": self.model,
        }
        attrs.update(_server_attributes(client))
        streamed = bool(kw.get("stream")) or kw.get("stream_format") == "sse"
        if self.operation == "speech_to_text":
            if kw.get("language"):
                attrs["gen_ai.request.language_code"] = str(kw["language"])
            if self.attr == "translations":
                attrs["gen_ai.request.target_language_code"] = _TRANSLATION_TARGET
            if kw.get("response_format"):
                attrs["gen_ai.response.output_format"] = str(kw["response_format"])
        else:
            text = self.text if isinstance(self.text, str) else ""
            attrs["gen_ai.usage.characters"] = len(text)
            if kw.get("voice"):
                voice = kw["voice"]
                attrs["gen_ai.request.voice_id"] = str(
                    voice.get("id") if isinstance(voice, dict) else voice
                )
            attrs["gen_ai.response.output_format"] = str(kw.get("response_format") or "mp3")
            if kw.get("speed") is not None:
                attrs["gen_ai.request.speed"] = float(kw["speed"])
            if kw.get("stream_format"):
                attrs["gen_ai.request.stream_format"] = str(kw["stream_format"])
        attrs["gen_ai.request.streamed"] = streamed
        return attrs

    def _read_input_audio(self) -> None:
        """Read the upload BEFORE the SDK does: a file object is at its end afterwards."""
        owner, file = self.owner, self.kwargs.get("file")
        try:
            if owner._speech_media_enabled():
                data, name, size = read_audio(file, owner._media_max_bytes())
                self.input_audio = audio_part(data, audio_mime(name), size)
                self.wav_seconds = wav_header_seconds(data, size)
            elif self._per_second_priced():
                data, _, size = read_audio(file, _WAV_HEAD_BYTES)
                self.wav_seconds = wav_header_seconds(data, size)
        except Exception as e:  # noqa: BLE001 - capture must never fail the call
            logger.debug("Could not read the transcription upload: %s", e)

    def _price_entry(self) -> Optional[Any]:
        calc = getattr(self.owner, "cost_calculator", None)
        if calc is None:
            return None
        try:
            key = calc._normalize_model_name(self.model, "audio")
            return calc.pricing_data["audio"].get(key) if key else None
        except Exception:  # noqa: BLE001
            return None

    def _per_second_priced(self) -> bool:
        cfg = self.owner.config
        if not (cfg and getattr(cfg, "enable_cost_tracking", True)):
            return False
        entry = self._price_entry()
        return isinstance(entry, dict) and "per_second" in entry

    # -- context -------------------------------------------------------------

    def context(self):
        return trace.set_span_in_context(self.span)

    # -- result --------------------------------------------------------------

    def chunk(self) -> None:
        if self.first_chunk is None:
            self.first_chunk = time.time()
            try:
                self.owner._record_time_to_first_token(
                    self.span, self.first_chunk - self.start, self.model
                )
            except Exception:  # noqa: BLE001
                pass

    def finish_stt(self, transcript: Optional[str], usage: Dict[str, Any]) -> None:
        span, owner = self.span, self.owner
        if transcript is not None:
            span.set_attribute("gen_ai.response.transcript_length", len(transcript))
        seconds = usage.get("seconds")
        if seconds is None and self.wav_seconds is not None:
            seconds = self.wav_seconds
            usage = dict(usage, seconds=seconds)
        if isinstance(seconds, (int, float)):
            span.set_attribute("gen_ai.usage.audio_duration_seconds", round(float(seconds), 3))
        self._tokens(usage)
        if owner._speech_media_enabled() and self.input_audio is not None:
            owner._emit_speech_parts(span, "prompt", "user", [self.input_audio])
        if transcript is not None:
            owner._emit_speech_parts(span, "completion", "assistant", [text_part(transcript)])
        self._cost(usage)
        self.end_ok()

    def finish_tts(self, audio: Optional[bytes], total: int, usage: Dict[str, Any]) -> None:
        span, owner = self.span, self.owner
        text = self.text if isinstance(self.text, str) else ""
        if text:
            usage = dict(usage, characters=len(text))
        self._tokens(usage)
        owner._emit_speech_parts(span, "prompt", "user", [text_part(text)])
        if audio is not None and owner._speech_media_enabled():
            mime = audio_mime(codec=self.kwargs.get("response_format") or "mp3")
            owner._emit_speech_parts(
                span, "completion", "assistant", [audio_part(audio, mime, total)]
            )
        self._cost(usage)
        self.end_ok()

    def _tokens(self, usage: Dict[str, Any]) -> None:
        span = self.span
        self.owner._set_token_usage_attributes(
            span, usage.get("input_tokens"), usage.get("output_tokens")
        )
        for key in ("input_audio_tokens", "input_text_tokens", "output_audio_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and value > 0:
                span.set_attribute(f"gen_ai.usage.{key}", int(value))

    def _cost(self, usage: Dict[str, Any]) -> None:
        owner, span = self.owner, self.span
        if owner.latency_histogram:
            try:
                owner.latency_histogram.record(time.time() - self.start, {"operation": span.name})
            except Exception:  # noqa: BLE001
                pass
        cfg = owner.config
        if not (cfg and getattr(cfg, "enable_cost_tracking", True)):
            return
        entry = self._price_entry()
        if entry is None:
            span.set_attribute("gen_ai.usage.cost.unknown_reason", "unpriced_model")
            return
        if isinstance(entry, dict) and not _usage_covers(entry, usage):
            span.set_attribute("gen_ai.usage.cost.unknown_reason", "no_usage_reported")
            return
        try:
            cost = owner.cost_calculator.calculate_cost(self.model, usage, "audio")
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not price %s: %s", self.model, e)
            cost = 0.0
        if cost and cost > 0:
            span.set_attribute("gen_ai.usage.cost.total", cost)
            if owner.cost_counter:
                owner.cost_counter.add(cost, {"model": self.model, "provider": "openai"})
        else:
            span.set_attribute("gen_ai.usage.cost.unknown_reason", "no_usage_reported")

    # -- ends ----------------------------------------------------------------

    def _count(self) -> None:
        if self.owner.request_counter:
            self.owner.request_counter.add(
                1, {"model": self.model, "operation": self.operation, "provider": "openai"}
            )

    def end_ok(self) -> None:
        if self.ended:
            return
        self.ended = True
        self._count()
        self.span.set_status(Status(StatusCode.OK))
        self.span.end()

    def end_closed_early(self) -> None:
        """The caller stopped reading. Billed all the same: characters and cost stay."""
        if self.ended:
            return
        usage: Dict[str, Any] = {}
        text = self.text if isinstance(self.text, str) else ""
        if text:
            usage["characters"] = len(text)
        self.span.set_attribute("genai.stream.closed_early", True)
        if self.operation == "text_to_speech":
            self._cost(usage)
        self.ended = True
        self._count()
        self.span.end()

    def fail(self, error: BaseException) -> None:
        if self.ended:
            return
        self.ended = True
        try:
            if self.owner.error_counter:
                self.owner.error_counter.add(
                    1, {"operation": self.span.name, "error_type": type(error).__name__}
                )
        except Exception:  # noqa: BLE001
            pass
        self.span.set_status(Status(StatusCode.ERROR, str(error)))
        self.span.record_exception(error)
        self.span.end()


def _usage_covers(entry: Dict[str, Any], usage: Dict[str, Any]) -> bool:
    """Whether ``usage`` reports what a declared-unit price entry bills by."""
    if "per_second" in entry:
        return isinstance(usage.get("seconds"), (int, float))
    if "per_1k_chars" in entry:
        return isinstance(usage.get("characters"), (int, float))
    if "per_1k_tokens" in entry:
        return isinstance(usage.get("tokens"), (int, float))
    if any(k.startswith("per_1k_input") for k in entry):
        if not isinstance(usage.get("input_tokens"), (int, float)):
            return False
    if any(k.startswith("per_1k_output") for k in entry):
        if not isinstance(usage.get("output_tokens"), (int, float)):
            return False
    return True


# --------------------------------------------------------------------------- #
# Reading results
# --------------------------------------------------------------------------- #


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stt_usage(result: Any) -> Dict[str, Any]:
    """Usage from a transcription / translation result: tokens, or seconds."""
    usage = _get(result, "usage")
    out: Dict[str, Any] = {}
    kind = _get(usage, "type")
    if kind == "tokens" or _get(usage, "input_tokens") is not None:
        for key in ("input_tokens", "output_tokens"):
            value = _get(usage, key)
            if isinstance(value, (int, float)):
                out[key] = value
        details = _get(usage, "input_token_details")
        for src, dst in (
            ("audio_tokens", "input_audio_tokens"),
            ("text_tokens", "input_text_tokens"),
        ):
            value = _get(details, src)
            if isinstance(value, (int, float)):
                out[dst] = value
    seconds = _get(usage, "seconds")
    if seconds is None:
        seconds = _get(result, "duration")  # verbose_json
    if isinstance(seconds, (int, float)):
        out["seconds"] = float(seconds)
    return out


def _stt_text(result: Any) -> Optional[str]:
    if isinstance(result, str):
        return result
    text = _get(result, "text")
    if isinstance(text, str):
        return text
    segments = _get(result, "segments")
    if isinstance(segments, list):
        return " ".join(str(_get(s, "text", "")) for s in segments).strip() or None
    return None


class _SSEAudio:
    """Speech ``stream_format="sse"``: base64 audio deltas and a usage event, parsed as they pass."""

    def __init__(self, cap: int, keep: bool):
        self.cap, self.keep = cap, keep
        self.pending = b""
        self.audio = bytearray()
        self.total = 0
        self.usage: Dict[str, Any] = {}
        self.saw_audio = False

    def feed(self, chunk: bytes) -> bool:
        """Consume bytes; True once the first audio delta has been seen."""
        self.pending += chunk
        while b"\n\n" in self.pending:
            event, self.pending = self.pending.split(b"\n\n", 1)
            for line in event.splitlines():
                if line.startswith(b"data:"):
                    self._event(line[5:].strip())
        return self.saw_audio

    def _event(self, data: bytes) -> None:
        try:
            event = json.loads(data)
        except (ValueError, TypeError):
            return
        kind = event.get("type")
        if kind == "speech.audio.delta":
            self.saw_audio = True
            b64 = event.get("audio") or ""
            self.total += (len(b64) * 3) // 4
            if self.keep and len(self.audio) <= self.cap:
                try:
                    self.audio.extend(base64.b64decode(b64)[: self.cap + 1 - len(self.audio)])
                except (binascii.Error, ValueError):
                    pass
        elif kind == "speech.audio.done":
            usage = event.get("usage") or {}
            for key in ("input_tokens", "output_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    self.usage[key] = usage[key]


def _binary_body(result: Any) -> Optional[bytes]:
    content = getattr(result, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    return None


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #


def _wrapper(owner, client, original, attr, span_name, operation):
    import inspect

    # The async SDK's resource methods are not always ``async def`` (some return the
    # coroutine from a plain function), so the client decides: an Async* client's call
    # is traced when it is awaited, which is where it runs.
    is_async = inspect.iscoroutinefunction(original) or "Async" in type(client).__name__

    if is_async:

        async def create(*args, **kwargs):
            call = _AudioCall(owner, client, attr, span_name, operation, kwargs)
            token = otel_context.attach(call.context())
            try:
                result = await original(*args, **kwargs)
            except Exception as e:
                call.fail(e)
                raise
            finally:
                otel_context.detach(token)
            return _handle(call, result, is_async=True)

    else:

        def create(*args, **kwargs):
            call = _AudioCall(owner, client, attr, span_name, operation, kwargs)
            token = otel_context.attach(call.context())
            try:
                result = original(*args, **kwargs)
            except Exception as e:
                call.fail(e)
                raise
            finally:
                otel_context.detach(token)
            return _handle(call, result, is_async=False)

    setattr(create, _WRAPPED_MARK, True)
    create.__wrapped__ = original
    return create


def _handle(call: _AudioCall, result: Any, is_async: bool) -> Any:
    """End the span now, or hook the result so it ends when the result is done."""
    if call.ended:
        return result
    try:
        if not call.recording:
            call.end_ok()
            return result
        # Raw responses (with_raw_response / with_streaming_response): the result
        # exists once .parse() is called. A transcription is always read through
        # parse(); streamed speech audio through the httpx response's iterators.
        if callable(getattr(result, "parse", None)) and hasattr(result, "http_response"):
            if call.operation == "text_to_speech" and "Legacy" not in type(result).__name__:
                return _hook_streamed_binary(call, result, is_async)
            return _hook_parse(call, result)
        if call.operation == "speech_to_text":
            if _is_event_stream(result):
                return _TracedEventStream(call, result, is_async)
            call.finish_stt(_stt_text(result), _stt_usage(result))
            return result
        body = _binary_body(result)
        if call.kwargs.get("stream_format") == "sse" and body is not None:
            sse = _SSEAudio(call.owner._media_max_bytes(), call.owner._speech_media_enabled())
            sse.feed(body + b"\n\n")
            call.finish_tts(bytes(sse.audio) if sse.audio else None, sse.total, sse.usage)
        else:
            call.finish_tts(body, len(body) if body else 0, {})
        return result
    except Exception as e:  # noqa: BLE001 - telemetry must never break the call
        logger.debug("Failed to record openai audio result: %s", e)
        if not call.ended:
            call.end_ok()
        return result


def _has_read(http: Any) -> bool:
    try:
        http.content  # noqa: B018 - raises ResponseNotRead on an unread stream
        return True
    except Exception:  # noqa: BLE001
        return False


def _is_event_stream(result: Any) -> bool:
    return (
        hasattr(result, "response")
        and (hasattr(result, "__iter__") or hasattr(result, "__aiter__"))
        and not isinstance(result, (str, bytes, dict, list))
    )


def _hook_parse(call: _AudioCall, raw: Any) -> Any:
    """``with_raw_response``: record when the caller parses; if never, when it is dropped."""
    parse = raw.parse
    import inspect

    if inspect.iscoroutinefunction(parse):

        async def hooked(*a, **k):
            return _handle(call, await parse(*a, **k), is_async=True)

    else:

        def hooked(*a, **k):
            return _handle(call, parse(*a, **k), is_async=False)

    try:
        raw.parse = hooked
        weakref.finalize(raw, call.end_closed_early)
    except (AttributeError, TypeError):
        call.end_ok()
    return raw


def _hook_streamed_binary(call: _AudioCall, response: Any, is_async: bool) -> Any:
    """``with_streaming_response``: every read goes through the httpx response's
    ``iter_bytes`` / ``aiter_bytes``; the span ends when it is exhausted or closed."""
    http = response.http_response
    sse = call.kwargs.get("stream_format") == "sse"
    cap = call.owner._media_max_bytes()
    keep = call.owner._speech_media_enabled()
    state = {"done": False, "buf": bytearray(), "total": 0}
    parser = _SSEAudio(cap, keep) if sse else None

    def observe(chunk: bytes) -> None:
        if not chunk:
            return
        if parser is not None:
            if parser.feed(chunk):
                call.chunk()
            return
        call.chunk()
        state["total"] += len(chunk)
        if keep and len(state["buf"]) <= cap:
            state["buf"].extend(chunk[: cap + 1 - len(state["buf"])])

    def complete() -> None:
        if state["done"]:
            return
        state["done"] = True
        if parser is not None:
            call.finish_tts(
                bytes(parser.audio) if parser.audio else None, parser.total, parser.usage
            )
        else:
            call.finish_tts(bytes(state["buf"]) if state["buf"] else None, state["total"], {})

    if is_async:
        original_iter = http.aiter_bytes
        original_close = http.aclose

        async def aiter_bytes(*a, **k):
            async for chunk in drive_async_in_context(original_iter(*a, **k), call.context()):
                observe(chunk)
                yield chunk
            complete()

        async def aclose():
            try:
                await original_close()
            finally:
                if not state["done"]:
                    call.end_closed_early()

        http.aiter_bytes = aiter_bytes
        http.aclose = aclose
    else:
        original_iter = http.iter_bytes
        original_close = http.close

        def iter_bytes(*a, **k):
            for chunk in drive_in_context(original_iter(*a, **k), call.context()):
                observe(chunk)
                yield chunk
            complete()

        def close():
            try:
                original_close()
            finally:
                if not state["done"]:
                    call.end_closed_early()

        http.iter_bytes = iter_bytes
        http.close = close
    return response


class _TracedEventStream:
    """A transcription ``stream=True`` result, traced while it is read. Behaves as the stream."""

    def __init__(self, call: _AudioCall, stream: Any, is_async: bool):
        self._call = call
        self._stream = stream
        self._is_async = is_async
        self._text: List[str] = []
        self._final: Optional[str] = None
        self._usage: Dict[str, Any] = {}
        self._iter = None

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def _observe(self, event: Any) -> None:
        kind = _get(event, "type")
        if kind == "transcript.text.delta":
            self._call.chunk()
            delta = _get(event, "delta")
            if isinstance(delta, str):
                self._text.append(delta)
        elif kind == "transcript.text.done":
            text = _get(event, "text")
            if isinstance(text, str):
                self._final = text
            self._usage = _stt_usage(event)

    def _end(self, outcome: str, error: Optional[BaseException]) -> None:
        if outcome == "ok":
            text = self._final if self._final is not None else "".join(self._text) or None
            self._call.finish_stt(text, self._usage)
        elif outcome == "error" and error is not None:
            self._call.fail(error)
        else:
            self._call.end_closed_early()

    def __iter__(self):
        return drive_in_context(
            iter(self._stream), self._call.context(), on_item=self._observe, on_end=self._end
        )

    def __aiter__(self):
        return drive_async_in_context(
            self._stream.__aiter__(),
            self._call.context(),
            on_item=self._observe,
            on_end=self._end,
        )

    def __next__(self):
        if self._iter is None:
            self._iter = self.__iter__()
        return next(self._iter)

    async def __anext__(self):
        if self._iter is None:
            self._iter = self.__aiter__()
        return await self._iter.__anext__()

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self._stream.__exit__(*exc)
        finally:
            if not self._call.ended:
                self._call.end_closed_early()

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc):
        try:
            return await self._stream.__aexit__(*exc)
        finally:
            if not self._call.ended:
                self._call.end_closed_early()

    def close(self):
        try:
            return self._stream.close()
        finally:
            if not self._call.ended:
                self._call.end_closed_early()
