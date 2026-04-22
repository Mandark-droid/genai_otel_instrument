"""OpenTelemetry instrumentor for various vector database clients.

This module provides the `VectorDBInstrumentor` class, which automatically
instruments popular Python vector database libraries such as Pinecone, Weaviate,
Qdrant, ChromaDB, Milvus, FAISS, and LanceDB, enabling tracing of vector search
and related operations within GenAI applications.
"""

import logging
from typing import Any, Dict, Optional

import wrapt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class VectorDBInstrumentor:  # pylint: disable=R0903
    """Instrument vector database clients"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument all detected vector DB libraries"""
        instrumented_count = 0
        if self._instrument_pinecone():
            instrumented_count += 1
        if self._instrument_weaviate():
            instrumented_count += 1
        if self._instrument_qdrant():
            instrumented_count += 1
        if self._instrument_chroma():
            instrumented_count += 1
        if self._instrument_milvus():
            instrumented_count += 1
        if self._instrument_faiss():
            instrumented_count += 1
        if self._instrument_lancedb():
            instrumented_count += 1
        return instrumented_count

    def _instrument_pinecone(self):
        """Instrument Pinecone operations"""
        try:
            import pinecone

            # Check Pinecone version to handle API differences
            pinecone_version = getattr(pinecone, "__version__", "0.0.0")

            # Pinecone 3.0+ uses a different API structure
            if hasattr(pinecone, "Pinecone"):
                # New API (3.0+)
                logger.info("Detected Pinecone 3.0+ API")
                wrapt.wrap_function_wrapper(
                    "pinecone", "Pinecone.__init__", self._wrap_pinecone_init
                )

            elif hasattr(pinecone, "Index"):
                # Old API (2.x)
                logger.info("Detected Pinecone 2.x API")
                wrapt.wrap_function_wrapper(
                    "pinecone", "Index.query", self._wrap_pinecone_wrapt("pinecone.query")
                )
                wrapt.wrap_function_wrapper(
                    "pinecone", "Index.upsert", self._wrap_pinecone_wrapt("pinecone.upsert")
                )
                wrapt.wrap_function_wrapper(
                    "pinecone", "Index.delete", self._wrap_pinecone_wrapt("pinecone.delete")
                )
            else:
                logger.warning("Could not detect Pinecone API version. Skipping instrumentation.")
                return False

            logger.info("Pinecone instrumentation enabled")
            return True

        except ImportError:
            logger.info("Pinecone not installed, skipping instrumentation")
            return False
        except Exception as e:
            if "pinecone-client" in str(e) and "renamed" in str(e):
                logger.error(
                    "Failed to instrument Pinecone: %s. Please ensure only the `pinecone` package is installed (uninstall `pinecone-client` if present).",
                    e,
                )
            else:
                logger.error("Failed to instrument Pinecone: %s", e, exc_info=True)
            return False

    def _wrap_pinecone_init(self, wrapped, instance, args, kwargs):
        """Wrapper for Pinecone.__init__ to instrument index methods."""
        result = wrapped(*args, **kwargs)
        if hasattr(instance, "Index"):
            original_index = instance.Index

            @wrapt.decorator
            def traced_index(wrapped_idx, idx_instance, idx_args, idx_kwargs):
                idx = wrapped_idx(*idx_args, **idx_kwargs)
                if hasattr(idx_instance, "query"):
                    idx_instance.query = self._wrap_pinecone_method(
                        idx_instance.query, "pinecone.index.query"
                    )
                if hasattr(idx_instance, "upsert"):
                    idx_instance.upsert = self._wrap_pinecone_method(
                        idx_instance.upsert, "pinecone.index.upsert"
                    )
                if hasattr(idx_instance, "delete"):
                    idx_instance.delete = self._wrap_pinecone_method(
                        idx_instance.delete, "pinecone.index.delete"
                    )
                return idx

            instance.Index = traced_index(original_index)
        return result

    def _wrap_pinecone_wrapt(self, operation_name):
        """Create a wrapt callback for a Pinecone method."""

        def callback(wrapped, instance, args, kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                operation_name,
                kind=SpanKind.CLIENT,
                attributes={"db.system": "pinecone", "db.operation": operation_name.split(".")[-1]},
            ) as span:
                try:
                    result = wrapped(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return callback

    def _wrap_pinecone_method(self, original_method, operation_name):
        """Wrap a Pinecone method with tracing"""

        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                operation_name,
                kind=SpanKind.CLIENT,
                attributes={"db.system": "pinecone", "db.operation": operation_name.split(".")[-1]},
            ) as span:
                try:
                    result = original_method(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    def _instrument_weaviate(self):
        """Instrument Weaviate"""
        try:
            import weaviate

            @wrapt.decorator
            def wrapped_query(wrapped, instance, args, kwargs):  # pylint: disable=W0613
                with self.tracer.start_as_current_span("weaviate.query") as span:
                    span.set_attribute("db.system", "weaviate")
                    span.set_attribute("db.operation", "query")
                    result = wrapped(*args, **kwargs)
                    return result

            weaviate.Client.query = wrapped_query(weaviate.Client.query)  # pylint: disable=E1120
            logger.info("Weaviate instrumentation enabled")
            return True

        except ImportError:
            return False

    def _instrument_qdrant(self):
        """Instrument Qdrant, supporting both old (search) and new (query_points) APIs."""
        try:
            import qdrant_client

            tracer = self.tracer
            instrumented = False
            has_query_points = hasattr(qdrant_client.QdrantClient, "query_points")

            # New API (qdrant-client 1.10+): query_points replaces search
            if has_query_points:

                def wrapped_query_points(wrapped, instance, args, kwargs):
                    with tracer.start_as_current_span("qdrant.query_points") as span:
                        span.set_attribute("db.system", "qdrant")
                        span.set_attribute("db.operation", "query_points")
                        collection = kwargs.get("collection_name", args[0] if args else "unknown")
                        span.set_attribute("vector.collection", collection)
                        limit = kwargs.get("limit", 10)
                        span.set_attribute("vector.limit", limit)
                        return wrapped(*args, **kwargs)

                try:
                    wrapt.wrap_function_wrapper(
                        "qdrant_client", "QdrantClient.query_points", wrapped_query_points
                    )
                    instrumented = True
                except (AttributeError, ImportError) as e:
                    logger.debug("Failed to wrap QdrantClient.query_points: %s", e)

            # Legacy API: only wrap `search` when `query_points` is unavailable.
            # `search` is deprecated in qdrant-client >=1.10 and removed in 1.16+;
            # wrapping it when the new API exists causes deprecation noise / attribute errors.
            elif hasattr(qdrant_client.QdrantClient, "search"):

                def wrapped_search(wrapped, instance, args, kwargs):
                    with tracer.start_as_current_span("qdrant.search") as span:
                        span.set_attribute("db.system", "qdrant")
                        span.set_attribute("db.operation", "search")
                        collection = kwargs.get("collection_name", args[0] if args else "unknown")
                        span.set_attribute("vector.collection", collection)
                        limit = kwargs.get("limit", 10)
                        span.set_attribute("vector.limit", limit)
                        return wrapped(*args, **kwargs)

                try:
                    wrapt.wrap_function_wrapper(
                        "qdrant_client", "QdrantClient.search", wrapped_search
                    )
                    instrumented = True
                except (AttributeError, ImportError) as e:
                    logger.debug("Failed to wrap QdrantClient.search: %s", e)

            if instrumented:
                logger.info("Qdrant instrumentation enabled")
                return True

            logger.warning(
                "Qdrant client detected but no supported API found (search or query_points)"
            )
            return False

        except ImportError:
            return False

    def _instrument_chroma(self):
        """Instrument ChromaDB"""
        try:
            tracer = self.tracer

            def wrapped_query(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("chroma.query") as span:
                    span.set_attribute("db.system", "chromadb")
                    span.set_attribute("db.operation", "query")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("vector.collection", instance.name)
                    n_results = kwargs.get("n_results", 10)
                    span.set_attribute("vector.n_results", n_results)
                    return wrapped(*args, **kwargs)

            wrapt.wrap_function_wrapper("chromadb", "Collection.query", wrapped_query)
            logger.info("ChromaDB instrumentation enabled")
            return True

        except ImportError:
            return False

    def _instrument_milvus(self):
        """Instrument Milvus"""
        try:
            tracer = self.tracer

            def wrapped_search(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("milvus.search") as span:
                    span.set_attribute("db.system", "milvus")
                    span.set_attribute("db.operation", "search")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("vector.collection", instance.name)
                    limit = kwargs.get("limit", 10)
                    span.set_attribute("vector.limit", limit)
                    return wrapped(*args, **kwargs)

            wrapt.wrap_function_wrapper("pymilvus", "Collection.search", wrapped_search)
            logger.info("Milvus instrumentation enabled")
            return True

        except ImportError:
            return False

    def _instrument_faiss(self):
        """Instrument FAISS"""
        try:
            tracer = self.tracer

            def wrapped_search(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("faiss.search") as span:
                    span.set_attribute("db.system", "faiss")
                    span.set_attribute("db.operation", "search")
                    k = args[1] if len(args) > 1 else kwargs.get("k", 10)
                    span.set_attribute("vector.k", k)
                    return wrapped(*args, **kwargs)

            wrapt.wrap_function_wrapper("faiss", "Index.search", wrapped_search)
            logger.info("FAISS instrumentation enabled")
            return True

        except ImportError:
            return False

    def _instrument_lancedb(self):
        """Instrument LanceDB"""
        try:
            import lancedb  # noqa: F401

            tracer = self.tracer

            def wrapped_search(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("lancedb.search") as span:
                    span.set_attribute("db.system", "lancedb")
                    span.set_attribute("db.operation", "search")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("vector.table", instance.name)
                    return wrapped(*args, **kwargs)

            def wrapped_add(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("lancedb.add") as span:
                    span.set_attribute("db.system", "lancedb")
                    span.set_attribute("db.operation", "add")
                    if instance and hasattr(instance, "name"):
                        span.set_attribute("vector.table", instance.name)
                    return wrapped(*args, **kwargs)

            def wrapped_create_table(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("lancedb.create_table") as span:
                    span.set_attribute("db.system", "lancedb")
                    span.set_attribute("db.operation", "create_table")
                    table_name = args[0] if args else kwargs.get("name", "unknown")
                    span.set_attribute("vector.table", table_name)
                    return wrapped(*args, **kwargs)

            def wrapped_drop_table(wrapped, instance, args, kwargs):
                with tracer.start_as_current_span("lancedb.drop_table") as span:
                    span.set_attribute("db.system", "lancedb")
                    span.set_attribute("db.operation", "drop_table")
                    table_name = args[0] if args else kwargs.get("name", "unknown")
                    span.set_attribute("vector.table", table_name)
                    return wrapped(*args, **kwargs)

            wrapt.wrap_function_wrapper("lancedb.table", "Table.search", wrapped_search)
            wrapt.wrap_function_wrapper("lancedb.table", "Table.add", wrapped_add)
            wrapt.wrap_function_wrapper(
                "lancedb.db", "DBConnection.create_table", wrapped_create_table
            )
            wrapt.wrap_function_wrapper("lancedb.db", "DBConnection.drop_table", wrapped_drop_table)
            logger.info("LanceDB instrumentation enabled")
            return True

        except ImportError:
            return False
        except Exception as e:
            logger.error("Failed to instrument LanceDB: %s", e, exc_info=True)
            return False
