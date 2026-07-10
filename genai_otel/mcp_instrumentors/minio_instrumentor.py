"""OpenTelemetry instrumentor for MinIO object storage clients.

This module provides the `MinIOInstrumentor` class, which automatically
instruments MinIO client operations such as object uploads, downloads,
bucket management, and listing, enabling tracing of object storage
interactions within GenAI applications.

Security note (credentials)
---------------------------
Only object-storage *coordinates* are captured to spans: ``minio.bucket`` (first
positional / ``bucket_name``) and ``minio.object`` (second positional /
``object_name``). Bucket/object names are retained deliberately for BFSI audit.
This instrumentor does **not** wrap ``Minio.__init__`` and never reads the
client's ``access_key`` / ``secret_key`` / session token / endpoint credentials,
so connection secrets are not placed on spans.

Double-wrap safety
------------------
``instrument()`` is idempotent: repeated calls do not stack additional wrappers
on ``minio.Minio``. A best-effort ``uninstrument()`` restores the originals.
"""

import logging

import wrapt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)

# Identity-tracked registry of already-instrumented ``minio`` module objects
# (see falkordb_instrumentor for the rationale on identity vs equality).
_INSTRUMENTED_MODULES = []
# (owner_class, method_name) pairs wrapped, for uninstrument().
_WRAPPED_TARGETS = []


def _is_module_instrumented(module) -> bool:
    """Return True if ``module`` (by identity) was already instrumented."""
    return any(m is module for m in _INSTRUMENTED_MODULES)


class MinIOInstrumentor:  # pylint: disable=R0903
    """Instrument MinIO object storage clients"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument MinIO client operations"""
        try:
            import minio  # noqa: F401

            # Double-wrap guard: skip if this module object was already wrapped.
            if _is_module_instrumented(minio):
                logger.debug("MinIO already instrumented; skipping to prevent double-wrap")
                return True

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
                            # Extract bucket name from first arg if available.
                            # Note: no credentials are read here; only the bucket
                            # and object coordinates are captured (see module docstring).
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

            # Record wrapped targets and mark instrumented only after success.
            minio_client = getattr(minio, "Minio", None)
            if minio_client is not None:
                for method_name in methods_to_wrap:
                    _WRAPPED_TARGETS.append((minio_client, method_name))
            _INSTRUMENTED_MODULES.append(minio)

            logger.info("MinIO instrumentation enabled")
            return True

        except ImportError:
            logger.debug("minio not installed, skipping MinIO instrumentation")
            return False
        except Exception as e:
            logger.error("Failed to instrument MinIO: %s", e, exc_info=True)
            return False

    def uninstrument(self):
        """Best-effort removal of MinIO instrumentation.

        Restores wrapped methods via wrapt's ``__wrapped__`` and clears the
        double-wrap registry. Returns True if MinIO was known to be instrumented.
        """
        try:
            import minio  # noqa: F401
        except ImportError:
            return False

        was_instrumented = _is_module_instrumented(minio)
        for owner, name in list(_WRAPPED_TARGETS):
            try:
                fn = getattr(owner, name, None)
                original = getattr(fn, "__wrapped__", None)
                if original is not None:
                    setattr(owner, name, original)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Failed to unwrap MinIO %s.%s: %s", owner, name, e)
        _WRAPPED_TARGETS.clear()
        _INSTRUMENTED_MODULES[:] = [m for m in _INSTRUMENTED_MODULES if m is not minio]
        if was_instrumented:
            logger.info("MinIO instrumentation removed")
        return was_instrumented
