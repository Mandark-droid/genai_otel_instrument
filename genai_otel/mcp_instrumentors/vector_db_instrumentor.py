"""OpenTelemetry instrumentor for various vector database clients.

This module provides the `VectorDBInstrumentor` class, which automatically
instruments popular Python vector database libraries such as Pinecone, Weaviate,
Qdrant, ChromaDB, Milvus, and FAISS, enabling tracing of vector search and
related operations within GenAI applications.
"""

import logging
from typing import Dict, Optional, Any

import wrapt
from opentelemetry import trace

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

    def _instrument_pinecone(self):  # pylint: disable=W0212
        """Instrument Pinecone"""
        try:
            import pinecone

            original_query = pinecone.Index.query

            def wrapped_query(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("pinecone.query") as span:
                    span.set_attribute("db.system", "pinecone")
                    span.set_attribute("db.operation", "query")
                    span.set_attribute("vector.index", instance._index_name)

                    top_k = kwargs.get("top_k", args[1] if len(args) > 1 else None)
                    if top_k:
                        span.set_attribute("vector.top_k", top_k)

                    result = original_query(instance, *args, **kwargs)
                    return result

            pinecone.Index.query = wrapped_query
            logger.info("Pinecone instrumentation enabled")
            return True

        except ImportError:
            return False

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
