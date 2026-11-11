"""Suppress instrumentation for the SDK's own media-store calls.

Uses OpenTelemetry's standard suppression context key, so any instrumentation that honours
it (including this package's S3 instrumentor) skips the call. The key is a private-named
constant upstream; it is imported once here behind a fallback so a rename degrades to
"spans are not suppressed" instead of an import error.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import context as otel_context

try:
    from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY as SUPPRESS_KEY
except ImportError:  # pragma: no cover - defensive against upstream renames
    SUPPRESS_KEY = "suppress_instrumentation"


def is_suppressed() -> bool:
    """True when instrumentation is currently suppressed in this context."""
    return bool(otel_context.get_value(SUPPRESS_KEY))


@contextmanager
def suppress_instrumentation() -> Iterator[None]:
    """Run the enclosed block with OTel instrumentation suppressed."""
    token = otel_context.attach(otel_context.set_value(SUPPRESS_KEY, True))
    try:
        yield
    finally:
        otel_context.detach(token)
