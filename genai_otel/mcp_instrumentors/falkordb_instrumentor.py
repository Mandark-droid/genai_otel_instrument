"""OpenTelemetry instrumentor for FalkorDB graph database clients.

This module provides the ``FalkorDBInstrumentor`` class, which automatically
instruments FalkorDB graph operations such as Cypher queries, read-only queries,
graph deletion, and graph copying, enabling tracing of graph database
interactions within GenAI applications.

Content bounding
----------------
The captured Cypher (``db.statement``) is intentionally retained for BFSI audit,
but is **bounded** to ``config.content_max_length`` (characters) when that value is
> 0. A cap of 0 (or an unset/unknown config) means unlimited, preserving full
capture for audit. The same bound is applied to error-status descriptions, which
may otherwise echo the full statement.

Double-wrap safety
------------------
``instrument()`` is idempotent: repeated calls (e.g. if
``setup_auto_instrumentation`` runs more than once) do not stack additional
wrappers on ``falkordb.Graph`` / ``falkordb.FalkorDB``. A best-effort
``uninstrument()`` restores the original methods.
"""

import logging

import wrapt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)

# Marker appended when a captured statement/error is truncated, so downstream
# consumers (and auditors) can tell the value was bounded rather than complete.
_TRUNCATION_SUFFIX = "...[truncated]"

# Identity-tracked registry of already-instrumented ``falkordb`` module objects.
# Keyed on object identity (not equality) so that mocked modules in unit tests
# -- a fresh object per test -- never collide, while the real, process-stable
# ``falkordb`` module is wrapped at most once.
_INSTRUMENTED_MODULES = []
# Records (owner, attribute_name) pairs wrapped by this instrumentor so that
# uninstrument() can restore the originals.
_WRAPPED_TARGETS = []


def _bound_text(value, config) -> str:
    """Bound a captured string to ``config.content_max_length`` when > 0.

    A cap of 0 (or an unset/unknown config) means unlimited, preserving full
    capture for audit. ``getattr`` with a default keeps this safe against
    ``MagicMock(spec=OTelConfig)`` in tests and any config missing the field.
    """
    text = value if isinstance(value, str) else str(value)
    max_len = getattr(config, "content_max_length", 0) or 0
    if max_len > 0 and len(text) > max_len:
        return text[:max_len] + _TRUNCATION_SUFFIX
    return text


def _is_module_instrumented(module) -> bool:
    """Return True if ``module`` (by identity) was already instrumented."""
    return any(m is module for m in _INSTRUMENTED_MODULES)


class FalkorDBInstrumentor:  # pylint: disable=R0903
    """Instrument FalkorDB graph database clients"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument FalkorDB graph operations"""
        try:
            import falkordb  # noqa: F401

            # Double-wrap guard: skip if this module object was already wrapped.
            if _is_module_instrumented(falkordb):
                logger.debug("FalkorDB already instrumented; skipping to prevent double-wrap")
                return True

            tracer = self.tracer
            config = self.config

            # Instrument Graph.query
            def wrapped_query(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("falkordb.query", kind=SpanKind.CLIENT) as span:
                    span.set_attribute("db.system", "falkordb")
                    span.set_attribute("db.operation", "query")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("db.name", instance.name)
                    if args:
                        span.set_attribute("db.statement", _bound_text(args[0], config))
                    elif "q" in kwargs:
                        span.set_attribute("db.statement", _bound_text(kwargs["q"], config))
                    try:
                        result = wrapped(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, _bound_text(e, config)))
                        span.record_exception(e)
                        raise

            # Instrument Graph.ro_query (read-only query)
            def wrapped_ro_query(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span(
                    "falkordb.ro_query", kind=SpanKind.CLIENT
                ) as span:
                    span.set_attribute("db.system", "falkordb")
                    span.set_attribute("db.operation", "ro_query")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("db.name", instance.name)
                    if args:
                        span.set_attribute("db.statement", _bound_text(args[0], config))
                    elif "q" in kwargs:
                        span.set_attribute("db.statement", _bound_text(kwargs["q"], config))
                    try:
                        result = wrapped(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, _bound_text(e, config)))
                        span.record_exception(e)
                        raise

            # Instrument Graph.delete
            def wrapped_delete(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("falkordb.delete", kind=SpanKind.CLIENT) as span:
                    span.set_attribute("db.system", "falkordb")
                    span.set_attribute("db.operation", "delete")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("db.name", instance.name)
                    try:
                        result = wrapped(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, _bound_text(e, config)))
                        span.record_exception(e)
                        raise

            # Instrument Graph.copy
            def wrapped_copy(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("falkordb.copy", kind=SpanKind.CLIENT) as span:
                    span.set_attribute("db.system", "falkordb")
                    span.set_attribute("db.operation", "copy")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("db.name", instance.name)
                    # Extract destination graph name
                    if args:
                        span.set_attribute("falkordb.destination_graph", str(args[0]))
                    elif "dest" in kwargs:
                        span.set_attribute("falkordb.destination_graph", str(kwargs["dest"]))
                    try:
                        result = wrapped(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, _bound_text(e, config)))
                        span.record_exception(e)
                        raise

            # Instrument FalkorDB.select_graph
            def wrapped_select_graph(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span(
                    "falkordb.select_graph", kind=SpanKind.CLIENT
                ) as span:
                    span.set_attribute("db.system", "falkordb")
                    span.set_attribute("db.operation", "select_graph")
                    if args:
                        span.set_attribute("db.name", str(args[0]))
                    elif "id" in kwargs:
                        span.set_attribute("db.name", str(kwargs["id"]))
                    try:
                        result = wrapped(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, _bound_text(e, config)))
                        span.record_exception(e)
                        raise

            wrapt.wrap_function_wrapper("falkordb", "Graph.query", wrapped_query)
            wrapt.wrap_function_wrapper("falkordb", "Graph.ro_query", wrapped_ro_query)
            wrapt.wrap_function_wrapper("falkordb", "Graph.delete", wrapped_delete)
            wrapt.wrap_function_wrapper("falkordb", "Graph.copy", wrapped_copy)
            wrapt.wrap_function_wrapper("falkordb", "FalkorDB.select_graph", wrapped_select_graph)

            # Record wrapped targets for best-effort uninstrument, and mark the
            # module instrumented only after all wrapping succeeded.
            _record_wrapped(falkordb, "Graph", "query")
            _record_wrapped(falkordb, "Graph", "ro_query")
            _record_wrapped(falkordb, "Graph", "delete")
            _record_wrapped(falkordb, "Graph", "copy")
            _record_wrapped(falkordb, "FalkorDB", "select_graph")
            _INSTRUMENTED_MODULES.append(falkordb)

            logger.info("FalkorDB instrumentation enabled")
            return True

        except ImportError:
            logger.debug("falkordb not installed, skipping FalkorDB instrumentation")
            return False
        except Exception as e:
            logger.error("Failed to instrument FalkorDB: %s", e, exc_info=True)
            return False

    def uninstrument(self):
        """Best-effort removal of FalkorDB instrumentation.

        Restores any wrapped methods (via wrapt's ``__wrapped__``) and clears the
        double-wrap registry so a later ``instrument()`` can re-apply cleanly.
        Returns True if the FalkorDB module was known to be instrumented.
        """
        try:
            import falkordb  # noqa: F401
        except ImportError:
            return False

        was_instrumented = _is_module_instrumented(falkordb)
        for owner, name in list(_WRAPPED_TARGETS):
            try:
                fn = getattr(owner, name, None)
                original = getattr(fn, "__wrapped__", None)
                if original is not None:
                    setattr(owner, name, original)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Failed to unwrap FalkorDB %s.%s: %s", owner, name, e)
        _WRAPPED_TARGETS.clear()
        _INSTRUMENTED_MODULES[:] = [m for m in _INSTRUMENTED_MODULES if m is not falkordb]
        if was_instrumented:
            logger.info("FalkorDB instrumentation removed")
        return was_instrumented


def _record_wrapped(module, class_name: str, method_name: str):
    """Record a (class, method) pair that was wrapped, for uninstrument()."""
    owner = getattr(module, class_name, None)
    if owner is not None:
        _WRAPPED_TARGETS.append((owner, method_name))
