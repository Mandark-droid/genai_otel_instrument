"""Drive a generator with an OpenTelemetry context attached only while it runs.

A span wrapped around a generator cannot stay attached in the caller's context
for as long as the generator lives. A generator has no context of its own: it is
resumed, and finalised, by whoever drives it. When the consumer breaks out
early, closes the generator from another task or leaves it to the garbage
collector, a detach held across a ``yield`` runs against a token created in a
different context. OpenTelemetry logs "Failed to detach context" and the span
stays current in the consumer's context for good, so every call traced there
afterwards is parented to a span that has already ended.

The drivers below attach the span's context around each step of the inner
generator and detach it before the item is handed out. Each attach/detach pair
sits inside one stretch of the driver's body that cannot be interrupted, so it
always runs in one context, whoever drives or closes the generator. The
consumer's context is untouched while it handles an item.

Whatever context the inner generator leaves current carries over to its next
step, so a span the inner generator holds open across its own yields stays
current for it. Values and exceptions sent in (``send``/``asend``,
``throw``/``athrow``) are forwarded. The inner generator is closed under the
span's context, because its teardown may trace too.

``on_end(outcome, error)`` is called exactly once. ``outcome`` is ``"ok"``
(exhausted), ``"error"`` (the inner generator raised) or ``"closed"`` (the
consumer stopped early, or the driver was cancelled).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Callable, Generator, Optional

from opentelemetry import context as otel_context

logger = logging.getLogger(__name__)

OnItem = Optional[Callable[[Any], None]]
OnEnd = Optional[Callable[[str, Optional[BaseException]], None]]


def _safe(callback: Optional[Callable[..., None]], *args: Any) -> None:
    if callback is None:
        return
    try:
        callback(*args)
    except Exception as e:  # noqa: BLE001 - telemetry must never break the stream
        logger.debug("Stream telemetry callback failed: %s", e)


async def drive_async_in_context(
    agen: Any, ctx: Any, on_item: OnItem = None, on_end: OnEnd = None
) -> AsyncGenerator[Any, Any]:
    """Yield everything ``agen`` yields, with ``ctx`` attached only while it runs."""
    active = ctx

    async def _drive(step: Any) -> Any:
        nonlocal active
        token = otel_context.attach(active)
        try:
            return await step
        finally:
            active = otel_context.get_current()
            otel_context.detach(token)

    outcome: str = "closed"
    error: Optional[BaseException] = None
    pending: Optional[BaseException] = None
    sent: Any = None
    try:
        while True:
            try:
                if pending is not None:
                    thrown, pending = pending, None
                    item = await _drive(agen.athrow(thrown))
                elif sent is not None:
                    item = await _drive(agen.asend(sent))
                else:
                    item = await _drive(agen.__anext__())
            except StopAsyncIteration:
                outcome = "ok"
                return
            sent = None
            _safe(on_item, item)
            try:
                sent = yield item
            except GeneratorExit:
                raise
            except BaseException as exc:  # noqa: BLE001 - forwarded to agen, not swallowed
                pending = exc
    except GeneratorExit:
        raise
    except Exception as e:
        outcome, error = "error", e
        raise
    finally:
        try:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                await _drive(aclose())
        except Exception as e:  # noqa: BLE001
            logger.debug("Closing a traced async generator failed: %s", e)
        finally:
            _safe(on_end, outcome, error)


def drive_in_context(
    gen: Any, ctx: Any, on_item: OnItem = None, on_end: OnEnd = None
) -> Generator[Any, Any, None]:
    """The synchronous counterpart of :func:`drive_async_in_context`."""
    active = ctx

    def _drive(fn: Callable[..., Any], *args: Any) -> Any:
        nonlocal active
        token = otel_context.attach(active)
        try:
            return fn(*args)
        finally:
            active = otel_context.get_current()
            otel_context.detach(token)

    outcome: str = "closed"
    error: Optional[BaseException] = None
    pending: Optional[BaseException] = None
    sent: Any = None
    try:
        while True:
            try:
                if pending is not None:
                    thrown, pending = pending, None
                    item = _drive(gen.throw, thrown)
                elif sent is not None:
                    item = _drive(gen.send, sent)
                else:
                    item = _drive(next, gen)
            except StopIteration:
                outcome = "ok"
                return
            sent = None
            _safe(on_item, item)
            try:
                sent = yield item
            except GeneratorExit:
                raise
            except BaseException as exc:  # noqa: BLE001 - forwarded to gen, not swallowed
                pending = exc
    except GeneratorExit:
        raise
    except Exception as e:
        outcome, error = "error", e
        raise
    finally:
        try:
            close = getattr(gen, "close", None)
            if close is not None:
                _drive(close)
        except Exception as e:  # noqa: BLE001
            logger.debug("Closing a traced generator failed: %s", e)
        finally:
            _safe(on_end, outcome, error)
