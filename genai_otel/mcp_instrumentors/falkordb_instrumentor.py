"""OpenTelemetry instrumentor for FalkorDB graph database clients.

This module provides the ``FalkorDBInstrumentor`` class, which automatically
instruments FalkorDB graph operations such as Cypher queries, read-only queries,
graph deletion, and graph copying, enabling tracing of graph database
interactions within GenAI applications.
"""

import logging

import wrapt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class FalkorDBInstrumentor:  # pylint: disable=R0903
    """Instrument FalkorDB graph database clients"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument FalkorDB graph operations"""
        try:
            import falkordb  # noqa: F401

            tracer = self.tracer

            # Instrument Graph.query
            def wrapped_query(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span(
                    "falkordb.query", kind=SpanKind.CLIENT
                ) as span:
                    span.set_attribute("db.system", "falkordb")
                    span.set_attribute("db.operation", "query")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("db.name", instance.name)
                    if args:
                        span.set_attribute("db.statement", str(args[0]))
                    elif "q" in kwargs:
                        span.set_attribute("db.statement", str(kwargs["q"]))
                    try:
                        result = wrapped(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
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
                        span.set_attribute("db.statement", str(args[0]))
                    elif "q" in kwargs:
                        span.set_attribute("db.statement", str(kwargs["q"]))
                    try:
                        result = wrapped(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            # Instrument Graph.delete
            def wrapped_delete(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span(
                    "falkordb.delete", kind=SpanKind.CLIENT
                ) as span:
                    span.set_attribute("db.system", "falkordb")
                    span.set_attribute("db.operation", "delete")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("db.name", instance.name)
                    try:
                        result = wrapped(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            # Instrument Graph.copy
            def wrapped_copy(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span(
                    "falkordb.copy", kind=SpanKind.CLIENT
                ) as span:
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
                        span.set_status(Status(StatusCode.ERROR, str(e)))
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
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            wrapt.wrap_function_wrapper("falkordb", "Graph.query", wrapped_query)
            wrapt.wrap_function_wrapper("falkordb", "Graph.ro_query", wrapped_ro_query)
            wrapt.wrap_function_wrapper("falkordb", "Graph.delete", wrapped_delete)
            wrapt.wrap_function_wrapper("falkordb", "Graph.copy", wrapped_copy)
            wrapt.wrap_function_wrapper(
                "falkordb", "FalkorDB.select_graph", wrapped_select_graph
            )

            logger.info("FalkorDB instrumentation enabled")
            return True

        except ImportError:
            logger.debug("falkordb not installed, skipping FalkorDB instrumentation")
            return False
        except Exception as e:
            logger.error("Failed to instrument FalkorDB: %s", e, exc_info=True)
            return False
