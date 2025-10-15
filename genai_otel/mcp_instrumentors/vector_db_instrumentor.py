"""OpenTelemetry instrumentor for various vector database clients.

This module provides the `VectorDBInstrumentor` class, which automatically
instruments popular Python vector database libraries such as Pinecone, Weaviate,
Qdrant, ChromaDB, Milvus, and FAISS, enabling tracing of vector search and
related operations within GenAI applications.
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
                original_query = pinecone.Index.query
                original_upsert = pinecone.Index.upsert
                original_delete = pinecone.Index.delete

                pinecone.Index.query = self._wrap_pinecone_method(original_query, "pinecone.query")
                pinecone.Index.upsert = self._wrap_pinecone_method(
                    original_upsert, "pinecone.upsert"
                )
                pinecone.Index.delete = self._wrap_pinecone_method(
                    original_delete, "pinecone.delete"
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
                logger.error(f"Failed to instrument Pinecone: {e}", exc_info=True)
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
        """Instrument Qdrant"""
        try:
            from qdrant_client import QdrantClient

            original_search = QdrantClient.search

            def wrapped_search(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("qdrant.search") as span:
                    span.set_attribute("db.system", "qdrant")
                    span.set_attribute("db.operation", "search")

                    collection = kwargs.get("collection_name", args[0] if args else "unknown")
                    span.set_attribute("vector.collection", collection)

                    limit = kwargs.get("limit", 10)
                    span.set_attribute("vector.limit", limit)

                    result = original_search(instance, *args, **kwargs)
                    return result

            QdrantClient.search = wrapped_search
            logger.info("Qdrant instrumentation enabled")
            return True

        except ImportError:
            return False

    def _instrument_chroma(self):
        """Instrument ChromaDB"""
        try:
            import chromadb

            original_query = chromadb.Collection.query

            def wrapped_query(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("chroma.query") as span:
                    span.set_attribute("db.system", "chromadb")
                    span.set_attribute("db.operation", "query")
                    span.set_attribute("vector.collection", instance.name)

                    n_results = kwargs.get("n_results", 10)
                    span.set_attribute("vector.n_results", n_results)

                    result = original_query(instance, *args, **kwargs)
                    return result

            chromadb.Collection.query = wrapped_query
            logger.info("ChromaDB instrumentation enabled")
            return True

        except ImportError:
            return False

    def _instrument_milvus(self):
        """Instrument Milvus"""
        try:
            from pymilvus import Collection

            original_search = Collection.search

            def wrapped_search(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("milvus.search") as span:
                    span.set_attribute("db.system", "milvus")
                    span.set_attribute("db.operation", "search")
                    span.set_attribute("vector.collection", instance.name)

                    limit = kwargs.get("limit", 10)
                    span.set_attribute("vector.limit", limit)

                    result = original_search(instance, *args, **kwargs)
                    return result

            Collection.search = wrapped_search
            logger.info("Milvus instrumentation enabled")
            return True

        except ImportError:
            return False

    def _instrument_faiss(self):
        """Instrument FAISS"""
        try:
            import faiss

            original_search = faiss.Index.search

            def wrapped_search(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("faiss.search") as span:
                    span.set_attribute("db.system", "faiss")
                    span.set_attribute("db.operation", "search")

                    k = args[1] if len(args) > 1 else kwargs.get("k", 10)
                    span.set_attribute("vector.k", k)

                    result = original_search(instance, *args, **kwargs)
                    return result

            faiss.Index.search = wrapped_search
            logger.info("FAISS instrumentation enabled")
            return True

        except ImportError:
            return False
