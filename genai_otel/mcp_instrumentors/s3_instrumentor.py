"""OpenTelemetry instrumentation for boto3 / aiobotocore S3-compatible object stores.

SeaweedFS exposes an S3 API, so applications using boto3 with SeaweedFS' S3 gateway are
covered alongside AWS S3 and other S3-compatible services. The instrumentor wraps
botocore's API dispatch (and aiobotocore's async dispatch, which also covers aioboto3 and
s3fs) and records operation metadata only; request bodies, object bytes, credentials, and
arbitrary request parameters are never copied to spans.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Tuple
from urllib.parse import urlsplit

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig
from ..media.suppress import is_suppressed
from . import seaweedfs

logger = logging.getLogger(__name__)

# (BaseClient class, original method, installed wrapper)
_PATCHED_TARGETS: List[Tuple[type, Callable[..., Any], Callable[..., Any]]] = []
# Same shape, for aiobotocore's AioBaseClient (covers aioboto3 and s3fs).
_PATCHED_ASYNC_TARGETS: List[Tuple[type, Callable[..., Any], Callable[..., Any]]] = []


def _safe_server_address(endpoint: Any) -> str:
    """Return only the hostname from an endpoint, excluding credentials/path/query."""
    if not isinstance(endpoint, str):
        return ""
    try:
        return urlsplit(endpoint).hostname or ""
    except ValueError:
        return ""


def _set_request_attrs(
    span: Any, client: Any, operation_name: str, api_params: Any, endpoints: Any
) -> None:
    """Request-side attributes shared by the sync and async wrappers."""
    endpoint = getattr(getattr(client, "meta", None), "endpoint_url", None)
    safe_address = _safe_server_address(endpoint)
    span.set_attribute("db.system", "s3")
    span.set_attribute("db.operation", operation_name)
    if safe_address:
        span.set_attribute("server.address", safe_address)
    port = seaweedfs.endpoint_parts(endpoint)[1] if isinstance(endpoint, str) else None
    if port:
        span.set_attribute("server.port", port)
    if safe_address and seaweedfs.is_declared(endpoints, safe_address, port):
        span.set_attribute("db.system.name", seaweedfs.SYSTEM_NAME)
    if isinstance(api_params, dict):
        bucket = api_params.get("Bucket")
        key = api_params.get("Key")
        if bucket is not None:
            span.set_attribute("s3.bucket", str(bucket))
            span.set_attribute("aws.s3.bucket", str(bucket))
        if key is not None:
            span.set_attribute("s3.object", str(key))
            span.set_attribute("aws.s3.key", str(key))


def _set_response_attrs(span: Any, result: Any) -> None:
    """Record the request id the server returned (x-amz-request-id), if any."""
    try:
        meta = result.get("ResponseMetadata") if isinstance(result, dict) else None
        if not isinstance(meta, dict):
            return
        headers = meta.get("HTTPHeaders") or {}
        request_id = headers.get("x-amz-request-id") or meta.get("RequestId")
        if request_id:
            span.set_attribute("aws.request_id", str(request_id))
    except Exception:  # noqa: BLE001
        pass


def _record_error(span: Any, exc: Exception) -> None:
    # Keep diagnostics useful without copying provider error text,
    # which can contain sensitive endpoint or request details.
    span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
    span.set_attribute("error.type", type(exc).__name__)


def _is_s3(client: Any) -> bool:
    service_model = getattr(getattr(client, "meta", None), "service_model", None)
    return getattr(service_model, "service_name", None) == "s3"


class S3Instrumentor:
    """Trace boto3 and aiobotocore S3 API calls, including SeaweedFS' S3 gateway."""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self) -> bool:
        """Patch botocore (and aiobotocore when present) once."""
        sync_ok = self._instrument_sync()
        async_ok = self._instrument_async()
        return sync_ok or async_ok

    def _instrument_sync(self) -> bool:
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
        endpoints = seaweedfs.declared_endpoints(self.config)

        def wrapped_make_api_call(client, operation_name, api_params, *args, **kwargs):
            if not _is_s3(client) or is_suppressed():
                return original(client, operation_name, api_params, *args, **kwargs)
            # The SDK's automatic exception handling would copy the error message
            # onto the span; the except block below records the type only.
            with tracer.start_as_current_span(
                f"s3.{operation_name}",
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                _set_request_attrs(span, client, operation_name, api_params, endpoints)
                try:
                    result = original(client, operation_name, api_params, *args, **kwargs)
                except Exception as exc:
                    _record_error(span, exc)
                    raise
                _set_response_attrs(span, result)
                span.set_status(Status(StatusCode.OK))
                return result

        BaseClient._make_api_call = wrapped_make_api_call
        _PATCHED_TARGETS.append((BaseClient, original, wrapped_make_api_call))
        logger.info("S3-compatible object storage instrumentation enabled")
        return True

    def _instrument_async(self) -> bool:
        """Patch aiobotocore, whose AioBaseClient overrides _make_api_call as a coroutine.

        The botocore patch never sees that override; aioboto3 and s3fs both build on
        aiobotocore, so this one wrapper covers all three.
        """
        try:
            from aiobotocore.client import AioBaseClient
        except ImportError:
            return False

        if any(target is AioBaseClient for target, _, _ in _PATCHED_ASYNC_TARGETS):
            return True

        original = AioBaseClient._make_api_call
        tracer = self.tracer
        endpoints = seaweedfs.declared_endpoints(self.config)

        async def wrapped_make_api_call(client, operation_name, api_params, *args, **kwargs):
            if not _is_s3(client) or is_suppressed():
                return await original(client, operation_name, api_params, *args, **kwargs)
            with tracer.start_as_current_span(
                f"s3.{operation_name}",
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                _set_request_attrs(span, client, operation_name, api_params, endpoints)
                try:
                    result = await original(client, operation_name, api_params, *args, **kwargs)
                except Exception as exc:
                    _record_error(span, exc)
                    raise
                _set_response_attrs(span, result)
                span.set_status(Status(StatusCode.OK))
                return result

        AioBaseClient._make_api_call = wrapped_make_api_call
        _PATCHED_ASYNC_TARGETS.append((AioBaseClient, original, wrapped_make_api_call))
        logger.info("Async S3 (aiobotocore) instrumentation enabled")
        return True

    def uninstrument(self) -> bool:
        """Restore botocore / aiobotocore methods installed by this instrumentor."""
        restored = False
        for registry, module, name in (
            (_PATCHED_TARGETS, "botocore.client", "BaseClient"),
            (_PATCHED_ASYNC_TARGETS, "aiobotocore.client", "AioBaseClient"),
        ):
            try:
                target_cls = getattr(__import__(module, fromlist=[name]), name)
            except ImportError:
                continue
            for target, original, wrapper in list(registry):
                if target is target_cls and target._make_api_call is wrapper:
                    target._make_api_call = original
                    restored = True
                registry.remove((target, original, wrapper))
        if restored:
            logger.info("S3-compatible object storage instrumentation removed")
        return restored
