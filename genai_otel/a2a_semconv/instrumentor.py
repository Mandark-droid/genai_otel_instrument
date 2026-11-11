"""A2A (Agent2Agent) instrumentation for the official ``a2a-sdk`` (1.x).

What it records
---------------
* **Client.** One CLIENT span per logical client operation (``BaseClient.send_message``,
  ``get_task``, ``subscribe`` ...). A streaming call's span ends when the stream does.
* **Server.** The request handlers' ``on_*`` methods. When an HTTP or gRPC server span
  is already current, the A2A attributes go on it (and it is renamed, for HTTP). Only
  when there is none does an ``a2a.server`` span open.
* **Propagation.** The client injects W3C ``traceparent`` / ``tracestate`` (and
  baggage, under the session-baggage policy in :mod:`genai_otel.sessions`) through the
  SDK's own ``ClientCallInterceptor`` hook, so one trace runs across the two agents.
  The server reads it back when no transport instrumentation did.
  ``GENAI_A2A_PROPAGATE=false`` turns both off.

The SDK's own spans
-------------------
``a2a-sdk`` makes spans of its own, named after Python modules and carrying no A2A
attributes; its internal queues are marked SERVER. We switch them off by setting
``OTEL_INSTRUMENTATION_A2A_SDK_ENABLED=false`` unless the user set it. The SDK reads
that variable once, when ``a2a.utils.telemetry`` is first imported, so this only works
if :meth:`A2AInstrumentor.instrument` runs before the application imports ``a2a``. If
the module is already loaded the spans stay on, and one INFO line says so.

Failed tasks
------------
A task that ends ``failed`` or ``rejected`` is not an error span: the call worked and
the agent's answer was no. ``a2a.task.state`` carries it. Only a raised error (a
JSON-RPC error, a gRPC status, an HTTP error) marks the span ERROR.
"""

import contextlib
import importlib
import importlib.util
import inspect
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ..config import OTelConfig
from ..instrumentors.base import BaseInstrumentor
from ..sessions import SessionBaggagePropagator
from .attributes import CONTENT_KEYS, OPERATION_INVOKE_AGENT, A2AAttributes
from .extract import (
    CLIENT_METHODS,
    SERVER_METHODS,
    STREAMING_CLIENT_METHODS,
    STREAMING_SEND_METHOD,
    STREAMING_SERVER_METHODS,
    Observation,
    binding_from_server_state,
    binding_from_transport,
    build_attributes,
    card_attributes,
    error_type,
    pick_interface,
    server_address,
)

logger = logging.getLogger(__name__)

SDK_DISTRIBUTION = "a2a-sdk"
SDK_SCOPE = "a2a-python-sdk"
SDK_TELEMETRY_ENV = "OTEL_INSTRUMENTATION_A2A_SDK_ENABLED"
PROPAGATE_ENV = "GENAI_A2A_PROPAGATE"
MIN_SDK_MAJOR = 1
_INSTRUMENTATION_SCOPE = "genai_otel.a2a"

# Context key marking "an A2A server call is already being observed on this stack", so
# a handler that calls another handler method is not counted twice.
_SERVER_ACTIVE_KEY = "genai_otel.a2a.server_active"

_MARKER = "_genai_otel_a2a_instrumented"
_INTERCEPTOR_MARKER = "_genai_otel_a2a_interceptor"

_propagator = CompositePropagator([TraceContextTextMapPropagator(), SessionBaggagePropagator()])


def _sdk_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(SDK_DISTRIBUTION)
        except PackageNotFoundError:
            return None
    except Exception:  # noqa: BLE001  # pragma: no cover - importlib.metadata always exists
        return None


def sdk_supported() -> bool:
    """True when ``a2a-sdk`` 1.0 or later is installed. Does not import it."""
    if importlib.util.find_spec("a2a") is None:
        return False
    installed = _sdk_version()
    if installed is None:
        # Installed without metadata (vendored, a zipapp): go by what it offers.
        return importlib.util.find_spec("a2a.client.base_client") is not None
    try:
        return int(installed.split(".")[0]) >= MIN_SDK_MAJOR
    except ValueError:
        return False


def _is_sdk_span(span: Any) -> bool:
    """True for a span the SDK's own tracing made (its internal classes are marked SERVER)."""
    return getattr(getattr(span, "instrumentation_scope", None), "name", None) == SDK_SCOPE


def _propagation_enabled(config: Optional[OTelConfig]) -> bool:
    return bool(getattr(config, "a2a_propagate", True)) if config is not None else True


class _CallScope:
    """One observed call: its span, what it saw, and how it ends."""

    def __init__(
        self,
        instrumentor: "A2AInstrumentor",
        span: Any,
        owned: bool,
        request: Observation,
        response: Observation,
        extra_always: Optional[Dict[str, Any]] = None,
        extra_content: Optional[Dict[str, Any]] = None,
        mark_server: bool = False,
    ) -> None:
        self.instrumentor = instrumentor
        self.span = span
        self.owned = owned
        self.request = request
        self.response = response
        self.extra_always = extra_always or {}
        self.extra_content = extra_content or {}
        self.error: Optional[BaseException] = None
        self._mark_server = mark_server
        self._finished = False

    def _context(self) -> Any:
        ctx = trace.set_span_in_context(self.span)
        if self._mark_server:
            ctx = otel_context.set_value(_SERVER_ACTIVE_KEY, True, ctx)
        return ctx

    async def step(self, awaitable: Any) -> Any:
        """Await ``awaitable`` with this call's span current."""
        token = otel_context.attach(self._context())
        try:
            return await awaitable
        finally:
            otel_context.detach(token)

    def call(self, func: Callable[[], Any]) -> Any:
        """Run a plain call with this call's span current."""
        token = otel_context.attach(self._context())
        try:
            return func()
        finally:
            otel_context.detach(token)

    def fail(self, exc: BaseException) -> None:
        # A cancelled task or an abandoned stream is not a failed call.
        if isinstance(exc, Exception):
            self.error = exc

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            always, content = build_attributes(self.request, self.response)
            always.update(self.extra_always)
            span = self.span
            for key, value in always.items():
                span.set_attribute(key, value)
            for key, value in {**content, **self.extra_content}.items():
                self.instrumentor._put_content(span, key, value)
            if self.error is not None:
                span.set_attribute(A2AAttributes.ERROR_TYPE, error_type(self.error))
                etype = error_type(self.error)
                if self.instrumentor._capture():
                    # A2A error messages can echo message or task content, so the text
                    # is kept only with content capture on.
                    span.set_status(Status(StatusCode.ERROR, str(self.error)[:200]))
                    if self.owned:
                        span.record_exception(self.error)
                else:
                    span.set_status(Status(StatusCode.ERROR, etype))
                    if self.owned:
                        span.add_event("exception", {"exception.type": type(self.error).__name__})
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - never break the call
            logger.debug("Could not finish an A2A span: %s", exc)
        finally:
            if self.owned:
                with contextlib.suppress(Exception):
                    self.span.end()


async def _aiter(value: Any) -> Any:
    """An async iterator from a stream method's return value."""
    if inspect.isawaitable(value):
        value = await value
    return value.__aiter__()


class A2AInstrumentor(BaseInstrumentor):
    """Instrumentor for ``a2a-sdk`` clients and request handlers."""

    CONTENT_ATTRIBUTES = CONTENT_KEYS

    def __init__(self) -> None:
        super().__init__()
        self.tracer = trace.get_tracer(_INSTRUMENTATION_SCOPE)
        self._patched: List[Tuple[Any, str, Any]] = []
        self._set_env_default = False
        self._interceptor_class: Optional[type] = None

    # -- availability --------------------------------------------------------------

    @staticmethod
    def available() -> bool:
        """True when a supported ``a2a-sdk`` is installed."""
        return sdk_supported()

    # -- the SDK's own spans -------------------------------------------------------

    def _manage_sdk_telemetry(self) -> None:
        """Default ``OTEL_INSTRUMENTATION_A2A_SDK_ENABLED`` to false, before the SDK loads."""
        if os.environ.get(SDK_TELEMETRY_ENV) is not None:
            return  # the user's choice stands
        loaded = sys.modules.get("a2a.utils.telemetry")
        if loaded is None:
            os.environ[SDK_TELEMETRY_ENV] = "false"
            self._set_env_default = True
            logger.debug("%s defaulted to false", SDK_TELEMETRY_ENV)
            return
        if getattr(loaded, "otel_enabled", False):
            logger.info(
                "a2a-sdk was imported before genai_otel started, and it reads %s once at "
                "import time, so its own spans (named after Python modules, with internal "
                "queues marked SERVER) stay on beside ours. Set %s=false before importing "
                "a2a, or call genai_otel.instrument() before the import, to switch them off.",
                SDK_TELEMETRY_ENV,
                SDK_TELEMETRY_ENV,
            )

    # -- instrument / uninstrument -------------------------------------------------

    def instrument(self, config: OTelConfig) -> None:
        """Wrap the SDK's client and request handlers. Does nothing without ``a2a-sdk`` 1.x."""
        self.config = config
        if not sdk_supported():
            logger.debug("a2a-sdk 1.x is not installed; A2A instrumentation skipped")
            return
        try:
            self._manage_sdk_telemetry()
            wrapped = self._wrap_client() + self._wrap_servers()
            self._instrumented = wrapped > 0
            if wrapped:
                logger.info("A2A instrumentation enabled")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to instrument a2a-sdk: %s", exc, exc_info=True)
            if getattr(config, "fail_on_error", False):
                raise

    def uninstrument(self) -> None:
        """Restore every method we wrapped, and the environment default we set."""
        for owner, name, original in reversed(self._patched):
            try:
                setattr(owner, name, original)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not restore %s.%s: %s", owner, name, exc)
        for owner in {o for o, _, _ in self._patched}:
            with contextlib.suppress(Exception):
                if _MARKER in vars(owner):
                    delattr(owner, _MARKER)
        self._patched = []
        if self._set_env_default and os.environ.get(SDK_TELEMETRY_ENV) == "false":
            del os.environ[SDK_TELEMETRY_ENV]
        self._set_env_default = False
        self._instrumented = False

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        self._patched.append((owner, name, owner.__dict__[name]))
        setattr(owner, name, replacement)

    # -- client --------------------------------------------------------------------

    def _wrap_client(self) -> int:
        try:
            base_client = importlib.import_module("a2a.client.base_client")
            BaseClient = base_client.BaseClient
        except Exception as exc:  # noqa: BLE001
            logger.debug("a2a client not instrumented: %s", exc)
            return 0
        if vars(BaseClient).get(_MARKER):
            return 0
        count = 0
        for attr, method in CLIENT_METHODS.items():
            original = BaseClient.__dict__.get(attr)
            if original is None:
                continue
            if attr in STREAMING_CLIENT_METHODS:
                wrapper = self._client_stream_wrapper(attr, method, original)
            else:
                wrapper = self._client_unary_wrapper(attr, method, original)
            self._patch(BaseClient, attr, wrapper)
            count += 1
        if count:
            setattr(BaseClient, _MARKER, True)
        return count

    def _capture(self) -> bool:
        cfg = self.config
        return cfg is None or bool(getattr(cfg, "enable_content_capture", False))

    def _ensure_interceptor(self, client: Any) -> None:
        """Put the propagation interceptor on a client, once."""
        if not _propagation_enabled(self.config):
            return
        interceptors = getattr(client, "_interceptors", None)
        if interceptors is None or not isinstance(interceptors, list):
            return
        if any(getattr(i, _INTERCEPTOR_MARKER, False) for i in interceptors):
            return
        interceptor = self._make_interceptor()
        if interceptor is not None:
            # Last, so it runs after interceptors that may rebuild the call context, and
            # sees our client span as the current span either way.
            interceptors.append(interceptor)

    def _make_interceptor(self) -> Any:
        if self._interceptor_class is None:
            try:
                from a2a.client.client import ClientCallContext
                from a2a.client.interceptors import AfterArgs, BeforeArgs, ClientCallInterceptor
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not build the A2A propagation interceptor: %s", exc)
                return None

            instrumentor = self

            class TraceContextInterceptor(ClientCallInterceptor):
                """Adds W3C trace context to every outgoing A2A call."""

                _genai_otel_a2a_interceptor = True

                async def before(self, args: BeforeArgs) -> None:
                    if not instrumentor._instrumented or not _propagation_enabled(
                        instrumentor.config
                    ):
                        return
                    carrier: Dict[str, str] = {}
                    _propagator.inject(carrier)
                    if not carrier:
                        return
                    context = args.context
                    merged = dict(getattr(context, "service_parameters", None) or {})
                    merged.update(carrier)
                    # A copy: the caller's own context object is not edited.
                    if context is None:
                        args.context = ClientCallContext(service_parameters=merged)
                    else:
                        args.context = context.model_copy(update={"service_parameters": merged})

                async def after(self, args: AfterArgs) -> None:
                    return None

            self._interceptor_class = TraceContextInterceptor
        return self._interceptor_class()

    def _begin_client(
        self, client: Any, attr: str, method: str, request: Any
    ) -> Optional[_CallScope]:
        self._ensure_interceptor(client)
        capture = self._capture()
        req = Observation(capture)
        req.observe_request(request)
        card = getattr(client, "_card", None)
        transport = getattr(client, "_transport", None)
        binding = binding_from_transport(transport)

        if attr == "send_message":
            config = getattr(client, "_config", None)
            streaming = bool(getattr(config, "streaming", True)) and bool(
                getattr(getattr(card, "capabilities", None), "streaming", False)
            )
            if streaming:
                method = STREAMING_SEND_METHOD

        always, content = card_attributes(card, capture)
        interface = pick_interface(card, binding)
        start: Dict[str, Any] = {
            A2AAttributes.METHOD_NAME: method,
            A2AAttributes.OPERATION_NAME: OPERATION_INVOKE_AGENT,
        }
        if binding:
            start[A2AAttributes.PROTOCOL_BINDING] = binding
        if interface is not None:
            version = getattr(interface, "protocol_version", "")
            if version:
                start[A2AAttributes.PROTOCOL_VERSION] = version
            start.update(server_address(getattr(interface, "url", "")))
        if A2AAttributes.AGENT_NAME in always:
            start[A2AAttributes.AGENT_NAME] = always[A2AAttributes.AGENT_NAME]
        target = always.get(A2AAttributes.AGENT_NAME) or start.get(A2AAttributes.SERVER_ADDRESS)
        name = f"{method} {target}" if target else method
        span = self.tracer.start_span(name, kind=SpanKind.CLIENT, attributes=start)
        return _CallScope(
            self,
            span,
            owned=True,
            request=req,
            response=Observation(capture),
            extra_always={
                k: v for k, v in always.items() if k not in start and k != A2AAttributes.AGENT_NAME
            },
            extra_content=content,
        )

    def _client_unary_wrapper(self, attr: str, method: str, original: Callable[..., Any]) -> Any:
        instrumentor = self

        async def wrapper(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                scope = instrumentor._begin_client(client, attr, method, request)
            except Exception as exc:  # noqa: BLE001
                logger.debug("A2A client span not started: %s", exc)
                scope = None
            if scope is None:
                return await original(client, request, *args, **kwargs)
            try:
                result = await scope.step(original(client, request, *args, **kwargs))
            except BaseException as exc:
                scope.fail(exc)
                scope.finish()
                raise
            scope.response.observe_result(result)
            scope.finish()
            return result

        _copy_metadata(wrapper, original)
        return wrapper

    def _client_stream_wrapper(self, attr: str, method: str, original: Callable[..., Any]) -> Any:
        instrumentor = self

        async def wrapper(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                scope = instrumentor._begin_client(client, attr, method, request)
            except Exception as exc:  # noqa: BLE001
                logger.debug("A2A client span not started: %s", exc)
                scope = None
            if scope is None:
                async for event in _passthrough(original(client, request, *args, **kwargs)):
                    yield event
                return
            iterator = None
            try:
                iterator = scope.call(lambda: original(client, request, *args, **kwargs))
                iterator = await scope.step(_aiter(iterator))
                while True:
                    try:
                        event = await scope.step(iterator.__anext__())
                    except StopAsyncIteration:
                        break
                    scope.response.observe_result(event)
                    yield event
            except BaseException as exc:
                scope.fail(exc)
                raise
            finally:
                if iterator is not None and hasattr(iterator, "aclose"):
                    with contextlib.suppress(Exception):
                        await scope.step(iterator.aclose())
                scope.finish()

        _copy_metadata(wrapper, original)
        return wrapper

    # -- server --------------------------------------------------------------------

    def _wrap_servers(self) -> int:
        count = 0
        for module_name, class_name in (
            ("a2a.server.request_handlers.default_request_handler_v2", "DefaultRequestHandlerV2"),
            ("a2a.server.request_handlers.default_request_handler", "LegacyRequestHandler"),
        ):
            try:
                handler_class = getattr(importlib.import_module(module_name), class_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("a2a handler %s not instrumented: %s", class_name, exc)
                continue
            if vars(handler_class).get(_MARKER):
                continue
            wrapped_here = 0
            for attr, method in SERVER_METHODS.items():
                original = handler_class.__dict__.get(attr)
                if original is None:
                    continue
                if attr in STREAMING_SERVER_METHODS:
                    wrapper = self._server_stream_wrapper(attr, method, original)
                else:
                    wrapper = self._server_unary_wrapper(attr, method, original)
                self._patch(handler_class, attr, wrapper)
                wrapped_here += 1
            if wrapped_here:
                setattr(handler_class, _MARKER, True)
                count += wrapped_here
        return count

    def _server_context(self, call_context: Any) -> Optional[Any]:
        """The caller's trace context from the request's headers or gRPC metadata."""
        state = getattr(call_context, "state", None)
        if state is None:
            return None
        carrier: Dict[str, str] = {}
        headers = state.get("headers") if hasattr(state, "get") else None
        if headers:
            carrier = {str(k).lower(): str(v) for k, v in dict(headers).items()}
        else:
            grpc_context = state.get("grpc_context") if hasattr(state, "get") else None
            if grpc_context is not None:
                for key, value in grpc_context.invocation_metadata() or ():
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", "replace")
                    carrier[str(key).lower()] = str(value)
        if not carrier:
            return None
        extracted = _propagator.extract(carrier)
        if not trace.get_current_span(extracted).get_span_context().is_valid:
            return None
        return extracted

    def _begin_server(
        self, handler: Any, attr: str, method: str, params: Any, call_context: Any
    ) -> Optional[_CallScope]:
        if otel_context.get_value(_SERVER_ACTIVE_KEY):
            return None
        capture = self._capture()
        req = Observation(capture)
        req.observe_request(params)
        card = getattr(handler, "_agent_card", None)
        state = getattr(call_context, "state", None)
        binding = binding_from_server_state(state) if state is not None else None

        always, content = card_attributes(card, capture)
        tenant = getattr(call_context, "tenant", "")
        if tenant and not req.tenant:
            req.tenant = tenant
        attrs: Dict[str, Any] = {A2AAttributes.METHOD_NAME: method}
        if binding:
            attrs[A2AAttributes.PROTOCOL_BINDING] = binding
        attrs.update(always)
        target = always.get(A2AAttributes.AGENT_NAME)
        name = f"{method} {target}" if target else method

        current = trace.get_current_span()
        current_ctx = current.get_span_context()
        transport_span = (
            current
            if current_ctx.is_valid
            and current.is_recording()
            and getattr(current, "kind", None) == SpanKind.SERVER
            and not _is_sdk_span(current)
            else None
        )
        if transport_span is not None:
            if binding in ("jsonrpc", "http_json"):
                transport_span.update_name(name)  # HTTP only; a gRPC span keeps its name
            for key, value in attrs.items():
                transport_span.set_attribute(key, value)
            return _CallScope(
                self,
                transport_span,
                owned=False,
                request=req,
                response=Observation(capture),
                extra_content=content,
                mark_server=True,
            )

        parent = None
        if not current_ctx.is_valid and _propagation_enabled(self.config):
            try:
                parent = self._server_context(call_context)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not read A2A trace context: %s", exc)
        span = self.tracer.start_span(name, context=parent, kind=SpanKind.SERVER, attributes=attrs)
        return _CallScope(
            self,
            span,
            owned=True,
            request=req,
            response=Observation(capture),
            extra_content=content,
            mark_server=True,
        )

    def _server_unary_wrapper(self, attr: str, method: str, original: Callable[..., Any]) -> Any:
        instrumentor = self

        async def wrapper(
            handler: Any, params: Any, context: Any = None, *args: Any, **kwargs: Any
        ) -> Any:
            try:
                scope = instrumentor._begin_server(handler, attr, method, params, context)
            except Exception as exc:  # noqa: BLE001
                logger.debug("A2A server span not started: %s", exc)
                scope = None
            if scope is None:
                return await original(handler, params, context, *args, **kwargs)
            try:
                result = await scope.step(original(handler, params, context, *args, **kwargs))
            except BaseException as exc:
                scope.fail(exc)
                scope.finish()
                raise
            scope.response.observe_result(result)
            scope.finish()
            return result

        _copy_metadata(wrapper, original)
        return wrapper

    def _server_stream_wrapper(self, attr: str, method: str, original: Callable[..., Any]) -> Any:
        instrumentor = self

        async def wrapper(
            handler: Any, params: Any, context: Any = None, *args: Any, **kwargs: Any
        ) -> Any:
            try:
                scope = instrumentor._begin_server(handler, attr, method, params, context)
            except Exception as exc:  # noqa: BLE001
                logger.debug("A2A server span not started: %s", exc)
                scope = None
            if scope is None:
                async for event in _passthrough(
                    original(handler, params, context, *args, **kwargs)
                ):
                    yield event
                return
            iterator = None
            try:
                iterator = scope.call(lambda: original(handler, params, context, *args, **kwargs))
                iterator = await scope.step(_aiter(iterator))
                while True:
                    try:
                        event = await scope.step(iterator.__anext__())
                    except StopAsyncIteration:
                        break
                    scope.response.observe_result(event)
                    yield event
            except BaseException as exc:
                scope.fail(exc)
                raise
            finally:
                if iterator is not None and hasattr(iterator, "aclose"):
                    with contextlib.suppress(Exception):
                        await scope.step(iterator.aclose())
                scope.finish()

        _copy_metadata(wrapper, original)
        return wrapper

    def _put_content(self, span: Any, key: str, value: Any) -> None:
        """Write a content attribute, only when capture is on, capped to the length limit."""
        if not self._capture():
            return
        limit = getattr(self.config, "content_max_length", 0) if self.config else 0
        limit = limit if isinstance(limit, int) and limit > 0 else 0
        try:
            if isinstance(value, (list, tuple)):
                items = [str(v)[:limit] if limit else str(v) for v in value]
                span.set_attribute(key, items)
            else:
                text = str(value)
                span.set_attribute(key, text[:limit] if limit else text)
        except Exception as exc:  # noqa: BLE001 - capture must never fail the call
            logger.debug("Could not set content attribute %s: %s", key, exc)

    # -- BaseInstrumentor ----------------------------------------------------------

    def _extract_usage(self, result: Any) -> Optional[Dict[str, int]]:
        return None


async def _passthrough(value: Any) -> Any:
    """Yield a stream method's events untouched."""
    async for event in await _aiter(value):
        yield event


def _copy_metadata(wrapper: Any, original: Any) -> None:
    for attribute in ("__name__", "__qualname__", "__doc__", "__module__"):
        with contextlib.suppress(Exception):
            setattr(wrapper, attribute, getattr(original, attribute))
    with contextlib.suppress(Exception):
        wrapper.__wrapped__ = original
