"""A2A (Agent2Agent) instrumentation for the official ``a2a-sdk`` 1.x.

Client calls become CLIENT spans, server calls enrich the transport's SERVER span (or
open an ``a2a.server`` span when there is none), and W3C trace context travels between
the two agents so one trace crosses them. See :mod:`genai_otel.a2a_semconv.instrumentor`.

It is on by default and does nothing when ``a2a-sdk`` 1.0 or later is not installed.
"""

from .attributes import A2AAttributes
from .instrumentor import A2AInstrumentor, sdk_supported

__all__ = ["A2AAttributes", "A2AInstrumentor", "sdk_supported"]
