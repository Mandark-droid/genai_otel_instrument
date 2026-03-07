"""OpenTelemetry instrumentor for MinIO object storage clients.

This module provides the `MinIOInstrumentor` class, which automatically
instruments MinIO client operations such as object uploads, downloads,
bucket management, and listing, enabling tracing of object storage
interactions within GenAI applications.
"""

import logging

import wrapt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class MinIOInstrumentor:  # pylint: disable=R0903
    """Instrument MinIO object storage clients"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument MinIO client operations"""
        try:
            import minio  # noqa: F401

            tracer = self.tracer

            methods_to_wrap = [
                "put_object",
                "get_object",
                "remove_object",
                "list_objects",
                "make_bucket",
                "remove_bucket",
                "list_buckets",
                "stat_object",
                "fput_object",
                "fget_object",
            ]

            for method_name in methods_to_wrap:

                def _create_wrapper(op_name):
                    def wrapper(wrapped, instance, args, kwargs):
                        span_name = f"minio.{op_name}"
                        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                            span.set_attribute("db.system", "minio")
                            span.set_attribute("db.operation", op_name)
                            # Extract bucket name from first arg if available
                            if args:
                                span.set_attribute("minio.bucket", str(args[0]))
                            elif "bucket_name" in kwargs:
                                span.set_attribute("minio.bucket", str(kwargs["bucket_name"]))
                            # Extract object name from second arg if available
                            if len(args) > 1:
                                span.set_attribute("minio.object", str(args[1]))
                            elif "object_name" in kwargs:
                                span.set_attribute("minio.object", str(kwargs["object_name"]))
                            try:
                                result = wrapped(*args, **kwargs)
                                span.set_status(Status(StatusCode.OK))
                                return result
                            except Exception as e:
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                                span.record_exception(e)
                                raise

                    return wrapper

                wrapt.wrap_function_wrapper(
                    "minio", f"Minio.{method_name}", _create_wrapper(method_name)
                )

            logger.info("MinIO instrumentation enabled")
            return True

        except ImportError:
            logger.debug("minio not installed, skipping MinIO instrumentation")
            return False
        except Exception as e:
            logger.error("Failed to instrument MinIO: %s", e, exc_info=True)
            return False
