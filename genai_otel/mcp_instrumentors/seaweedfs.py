"""SeaweedFS support: endpoint matching and the Filer HTTP API instrumentor.

SeaweedFS is identified by *declared* endpoints only (``GENAI_OTEL_SEAWEEDFS_ENDPOINTS``,
plus the media-store endpoint when ``GENAI_OTEL_MEDIA_STORE=seaweed``). Response headers
cannot identify it and a port number alone is not evidence, so nothing is guessed.

The Filer instrumentor traces plain HTTP calls to declared hosts made through ``httpx`` and
``aiohttp``. ``requests`` is deliberately not patched. Spans carry the operation, status,
sizes and a short path prefix; file contents, query values, headers and error text are never
recorded.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

SYSTEM_NAME = "seaweedfs"

# (host pattern, port or None for any port)
Endpoint = Tuple[str, Optional[int]]

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _parse_one(entry: str) -> Optional[Endpoint]:
    entry = entry.strip().lower()
    if not entry:
        return None
    if "://" in entry:
        try:
            parts = urlsplit(entry)
            host = parts.hostname
            port = parts.port or _DEFAULT_PORTS.get(parts.scheme)
        except ValueError:
            return None
    else:
        try:
            parts = urlsplit("//" + entry)
            host = parts.hostname
            port = parts.port
        except ValueError:
            return None
    if not host:
        return None
    return host, port


def parse_endpoints(*sources: Optional[str]) -> List[Endpoint]:
    """Parse comma-separated ``host[:port]`` / URL lists into endpoint patterns.

    A bare host matches any port; ``*.suffix`` hosts match by suffix. Invalid entries
    are dropped.
    """
    endpoints: List[Endpoint] = []
    for source in sources:
        for entry in (source or "").split(","):
            parsed = _parse_one(entry)
            if parsed and parsed not in endpoints:
                endpoints.append(parsed)
    return endpoints


def declared_endpoints(config: Any) -> List[Endpoint]:
    """Declared SeaweedFS endpoints for this config (explicit list + seaweed media store)."""
    media_endpoint = None
    if (getattr(config, "media_store", "") or "").lower() == "seaweed":
        media_endpoint = getattr(config, "media_store_endpoint", None)
    return parse_endpoints(getattr(config, "seaweedfs_endpoints", None), media_endpoint)


def _host_matches(pattern: str, host: str) -> bool:
    if pattern.startswith("*."):
        return host.endswith(pattern[1:])
    return host == pattern


def is_declared(endpoints: List[Endpoint], host: Optional[str], port: Optional[int]) -> bool:
    """True when ``host``/``port`` match a declared SeaweedFS endpoint."""
    if not host or not endpoints:
        return False
    host = host.lower()
    for pattern, pattern_port in endpoints:
        if _host_matches(pattern, host) and (pattern_port is None or pattern_port == port):
            return True
    return False


def endpoint_parts(url: Any) -> Tuple[str, Optional[int]]:
    """Return ``(hostname, port)`` of a URL with scheme default ports resolved."""
    try:
        parts = urlsplit(str(url))
        host = parts.hostname or ""
        port = parts.port or _DEFAULT_PORTS.get(parts.scheme)
    except ValueError:
        return "", None
    return host, port


def classify_operation(method: str, path: str, query: str) -> str:
    """Map a Filer HTTP request to an operation name (documented Filer API)."""
    method = (method or "").upper()
    params = parse_qs(query or "", keep_blank_values=True)
    if "mv.from" in params:
        return "move"
    if "cp.from" in params:
        return "copy"
    if "tagging" in params:
        return "tag"
    if method == "DELETE":
        return (
            "delete_recursive" if params.get("recursive", [""])[0].lower() == "true" else "delete"
        )
    if method in ("POST", "PUT"):
        if params.get("op", [""])[0] == "append":
            return "append"
        return "mkdir" if (path or "").endswith("/") else "upload"
    if method in ("GET", "HEAD"):
        if "metadata" in params:
            return "stat"
        return "list" if (path or "").endswith("/") else "download"
    return "other"


def path_prefix(path: str, segments: int) -> str:
    """First ``segments`` path segments only; never the full path."""
    if segments <= 0:
        return ""
    parts = [p for p in (path or "").split("/") if p]
    return "/" + "/".join(parts[:segments]) if parts else "/"


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _FilerSpan:
    """Builds one Filer span from request metadata and the eventual response."""

    def __init__(self, tracer: Any, method: str, url: Any, headers: Any, segments: int):
        parts = urlsplit(str(url))
        host, port = endpoint_parts(url)
        self.operation = classify_operation(method, parts.path, parts.query)
        self.span = tracer.start_span(
            f"seaweedfs.filer.{self.operation}",
            kind=SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        )
        span = self.span
        span.set_attribute("db.system.name", SYSTEM_NAME)
        span.set_attribute("db.operation.name", self.operation)
        span.set_attribute("http.request.method", (method or "").upper())
        if host:
            span.set_attribute("server.address", host)
        if port:
            span.set_attribute("server.port", port)
        prefix = path_prefix(parts.path, segments)
        if prefix:
            span.set_attribute("url.path.prefix", prefix)
        size = _int_or_none(_header(headers, "content-length"))
        if size is not None:
            span.set_attribute("seaweedfs.request.size", size)

    def finish(self, status: Optional[int], headers: Any) -> None:
        span = self.span
        if status is not None:
            span.set_attribute("http.response.status_code", status)
        size = _int_or_none(_header(headers, "content-length"))
        if size is not None:
            span.set_attribute("seaweedfs.response.size", size)
        # An absent file is an answer, not a fault.
        missing_ok = status == 404 and self.operation in ("stat", "download")
        if status is not None and status >= 400 and not missing_ok:
            span.set_attribute("error.type", str(status))
            span.set_status(Status(StatusCode.ERROR, str(status)))
        span.end()

    def fail(self, exc: BaseException) -> None:
        self.span.set_attribute("error.type", type(exc).__name__)
        self.span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
        self.span.end()


def _header(headers: Any, name: str) -> Any:
    try:
        return headers.get(name) if headers is not None else None
    except Exception:  # noqa: BLE001
        return None


# (owner object, attribute name, original, wrapper)
_PATCHED: List[Tuple[Any, str, Callable[..., Any], Callable[..., Any]]] = []


class SeaweedFilerInstrumentor:
    """Trace Filer HTTP calls to declared SeaweedFS hosts (httpx and aiohttp)."""

    def __init__(self, config: Any):
        self.config = config
        self.tracer = trace.get_tracer(__name__)
        self.endpoints = declared_endpoints(config)
        self.segments = int(getattr(config, "seaweedfs_path_prefix_segments", 2) or 0)

    def _wanted(self, url: Any) -> bool:
        host, port = endpoint_parts(url)
        return is_declared(self.endpoints, host, port)

    def instrument(self) -> bool:
        """Patch httpx / aiohttp when endpoints are declared. Inert (False) otherwise."""
        if not getattr(self.config, "enable_seaweedfs_instrumentation", True):
            return False
        if not self.endpoints:
            return False
        if _PATCHED:
            return True
        patched = self._patch_httpx()
        patched = self._patch_aiohttp() or patched
        if patched:
            logger.info("SeaweedFS Filer instrumentation enabled")
        return patched

    def uninstrument(self) -> bool:
        restored = False
        for owner, attr, original, wrapper in list(_PATCHED):
            if getattr(owner, attr, None) is wrapper:
                setattr(owner, attr, original)
                restored = True
            _PATCHED.remove((owner, attr, original, wrapper))
        return restored

    # -- httpx ----------------------------------------------------------------------
    def _patch_httpx(self) -> bool:
        try:
            import httpx
        except ImportError:
            return False
        me = self

        def sync_wrapper_factory(owner: Any, attr: str) -> None:
            original = getattr(owner, attr)

            def wrapped(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                if not me._wanted(request.url):
                    return original(client, request, *args, **kwargs)
                fs = _FilerSpan(
                    me.tracer, request.method, request.url, request.headers, me.segments
                )
                try:
                    response = original(client, request, *args, **kwargs)
                except BaseException as exc:
                    fs.fail(exc)
                    raise
                fs.finish(response.status_code, response.headers)
                return response

            setattr(owner, attr, wrapped)
            _PATCHED.append((owner, attr, original, wrapped))

        def async_wrapper_factory(owner: Any, attr: str) -> None:
            original = getattr(owner, attr)

            async def wrapped(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                if not me._wanted(request.url):
                    return await original(client, request, *args, **kwargs)
                fs = _FilerSpan(
                    me.tracer, request.method, request.url, request.headers, me.segments
                )
                try:
                    response = await original(client, request, *args, **kwargs)
                except BaseException as exc:
                    fs.fail(exc)
                    raise
                fs.finish(response.status_code, response.headers)
                return response

            setattr(owner, attr, wrapped)
            _PATCHED.append((owner, attr, original, wrapped))

        # One span per request hop, independent of the generic HTTP instrumentor.
        sync_wrapper_factory(httpx.Client, "_send_single_request")
        async_wrapper_factory(httpx.AsyncClient, "_send_single_request")
        return True

    # -- aiohttp --------------------------------------------------------------------
    def _patch_aiohttp(self) -> bool:
        try:
            import aiohttp
        except ImportError:
            return False
        me = self
        owner = aiohttp.ClientSession
        original = owner._request

        async def wrapped(session: Any, method: str, str_or_url: Any, *args: Any, **kwargs: Any):
            if not me._wanted(str_or_url):
                return await original(session, method, str_or_url, *args, **kwargs)
            fs = _FilerSpan(me.tracer, method, str_or_url, kwargs.get("headers"), me.segments)
            try:
                response = await original(session, method, str_or_url, *args, **kwargs)
            except BaseException as exc:
                fs.fail(exc)
                raise
            fs.finish(response.status, response.headers)
            return response

        owner._request = wrapped
        _PATCHED.append((owner, "_request", original, wrapped))
        return True


__all__ = [
    "SYSTEM_NAME",
    "SeaweedFilerInstrumentor",
    "classify_operation",
    "declared_endpoints",
    "endpoint_parts",
    "is_declared",
    "parse_endpoints",
    "path_prefix",
]
