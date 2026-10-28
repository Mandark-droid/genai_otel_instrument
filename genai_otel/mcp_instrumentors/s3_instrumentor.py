"""OpenTelemetry instrumentation for boto3 S3-compatible object stores.

SeaweedFS exposes an S3 API, so applications using boto3 with SeaweedFS' S3
gateway are covered alongside AWS S3 and other S3-compatible services. The
instrumentor wraps botocore's API dispatch and records operation metadata only;
request bodies, object bytes, credentials, and arbitrary request parameters are
never copied to spans.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Tuple
from urllib.parse import urlsplit

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)

# (BaseClient class, original method, installed wrapper)
_PATCHED_TARGETS: List[Tuple[type, Callable[..., Any], Callable[..., Any]]] = []


def _safe_server_address(endpoint: Any) -> str:
    """Return only the hostname from an endpoint, excluding credentials/path/query."""
    if not isinstance(endpoint, str):
        return ""
    try:
        return urlsplit(endpoint).hostname or ""
    except ValueError:
        return ""


class S3Instrumentor:
    """Trace boto3 S3 API calls, including calls to SeaweedFS' S3 gateway."""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self) -> bool:
        """Patch botocore once; boto3 clients created before or after setup are covered."""
        try:
            from botocore.client import BaseClient
        except ImportError:
            logger.debug("botocore not installed, skipping S3 instrumentation")
            return False

        if any(target is BaseClient for target, _, _ in _PATCHED_TARGETS):
            logger.debug("S3 API calls already instrumented; skipping duplicate wrapper")
            return True

        original = BaseClient._make_api_call
        tracer = self.tracer

        def wrapped_make_api_call(client, operation_name, api_params, *args, **kwargs):
            service_model = getattr(getattr(client, "meta", None), "service_model", None)
            if getattr(service_model, "service_name", None) != "s3":
                return original(client, operation_name, api_params, *args, **kwargs)

            endpoint = getattr(getattr(client, "meta", None), "endpoint_url", None)
            safe_address = _safe_server_address(endpoint)
            span_name = f"s3.{operation_name}"
            # The SDK's automatic exception handling would copy the error message
            # onto the span; the except block below records the type only.
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                span.set_attribute("db.system", "s3")
                span.set_attribute("db.operation", operation_name)
                if safe_address:
                    span.set_attribute("server.address", safe_address)
                if isinstance(api_params, dict):
                    bucket = api_params.get("Bucket")
                    key = api_params.get("Key")
                    if bucket is not None:
                        span.set_attribute("s3.bucket", str(bucket))
                    if key is not None:
                        span.set_attribute("s3.object", str(key))
                try:
                    result = original(client, operation_name, api_params, *args, **kwargs)
                except Exception as exc:
                    # Keep diagnostics useful without copying provider error text,
                    # which can contain sensitive endpoint or request details.
                    span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                    span.set_attribute("error.type", type(exc).__name__)
                    raise
                span.set_status(Status(StatusCode.OK))
                return result

        BaseClient._make_api_call = wrapped_make_api_call
        _PATCHED_TARGETS.append((BaseClient, original, wrapped_make_api_call))
        logger.info("S3-compatible object storage instrumentation enabled")
        return True

    def uninstrument(self) -> bool:
        """Restore botocore methods installed by this instrumentor."""
        try:
            from botocore.client import BaseClient
        except ImportError:
            return False

        restored = False
        for target, original, wrapper in list(_PATCHED_TARGETS):
            if target is BaseClient and target._make_api_call is wrapper:
                target._make_api_call = original
                restored = True
            _PATCHED_TARGETS.remove((target, original, wrapper))
        if restored:
            logger.info("S3-compatible object storage instrumentation removed")
        return restored
