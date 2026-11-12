"""A2UI (Agent-to-UI) instrumentation.

Two parts share one reader (:mod:`genai_otel.a2ui_semconv.extract`):

* :class:`A2UIInstrumentor` wraps the Python agent SDK, ``a2ui-agent-sdk`` 0.11+, whatever
  transport carries the UI.
* The A2A instrumentor (:mod:`genai_otel.a2a_semconv`) adds ``a2ui.*`` attributes to the A2A
  span when a message or stream carries A2UI parts.

Both are on by default, and ``GENAI_A2UI_ENABLED=false`` turns the A2UI attributes and spans
off without touching A2A. The instrumentor does nothing when ``a2ui-agent-sdk`` is missing.
"""

from .attributes import A2UIAttributes
from .instrumentor import A2UIInstrumentor, sdk_supported

__all__ = ["A2UIAttributes", "A2UIInstrumentor", "sdk_supported"]
