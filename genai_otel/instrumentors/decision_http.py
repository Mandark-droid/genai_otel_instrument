"""DECISION spans for decision APIs called over plain HTTP.

Applications often call a decision endpoint without an SDK: LiquidAI's image
examples post to ``/decisions/v1/systemone`` with ``requests``, and any
self-hosted ``/v1/systemone`` server can be reached with ``httpx``. Those calls
never pass through an SDK method this library wraps, so they are recognised by
their path instead:

* ``POST .../v1/decisions``              -> OpenAI Decisions protocol
* ``POST .../v1/systemone`` (including ``/decisions/v1/systemone``) -> SystemOne

``requests``, ``httpx`` and ``httpx2`` are covered.

A request made from inside a decision span this library already opened (the
OpenAI SDK sends ``decisions.create`` through ``httpx``) is left alone, so one
decision is one span.

Everything else passes straight through: the check is the request method and a
path suffix, and nothing else is read. In particular the OTLP exporter's own
``requests`` traffic is never touched, which is what made wrapping ``requests``
unsafe for the generic HTTP instrumentation.

``GENAI_DECISION_HTTP_MATCHER=false`` turns it off.
"""

import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import wrapt

from ..config import OTelConfig
from .decision import (
    PROTOCOL_OPENAI_DECISIONS,
    PROTOCOL_SYSTEMONE,
    SYSTEM_OPENAI,
    SYSTEM_TYPESAFE,
    BaseDecisionInstrumentor,
    DecisionRecorder,
    DecisionRequest,
    _get,
    _inside_decision_span,
    openai_questions,
    provider_for_base_url,
    systemone_questions,
)

logger = logging.getLogger(__name__)

# Longest first, so the LiquidAI path is not read as something shorter.
_DECISION_PATHS = (
    ("/decisions/v1/systemone", PROTOCOL_SYSTEMONE),
    ("/v1/systemone", PROTOCOL_SYSTEMONE),
    ("/v1/decisions", PROTOCOL_OPENAI_DECISIONS),
)

# What has been wrapped, by target, so instrument() is idempotent.
_WRAPPED = set()


def match_decision_path(method: Any, url: Any) -> Optional[str]:
    """The decision protocol a request speaks, or None if it is not a decision call."""
    if not method or str(method).upper() != "POST" or not url:
        return None
    try:
        path = urlparse(str(url)).path.rstrip("/")
    except Exception:  # pragma: no cover - urlparse is near-total
        return None
    for suffix, protocol in _DECISION_PATHS:
        if path.endswith(suffix):
            return protocol
    return None


def _json_or_none(body: Any) -> Any:
    """A request or response body as JSON, or None when it is not JSON (multipart, say)."""
    if body is None:
        return None
    try:
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            return json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return None
    return None


class DecisionHTTPInstrumentor(BaseDecisionInstrumentor):
    """Wrap ``requests`` / ``httpx`` / ``httpx2`` sends to recognise decision endpoints by path."""

    def instrument(self, config: OTelConfig) -> None:
        self.config = config
        if not getattr(config, "decision_http_matcher", True):
            logger.debug("Decision HTTP matcher disabled (GENAI_DECISION_HTTP_MATCHER=false)")
            return
        self._instrumented = True
        self._wrap("requests.sessions", "Session.send", self._wrap_requests_send)
        # httpx2 is what openai >= 3.3 and typesafe-sdk send through; its client
        # API is httpx's, so the same wrappers serve both.
        for module in ("httpx", "httpx2"):
            self._wrap(module, "Client.send", self._wrap_httpx_send)
            self._wrap(module, "AsyncClient.send", self._wrap_httpx_send_async)

    def _wrap(self, module: str, name: str, wrapper) -> None:
        key = f"{module}.{name}"
        if key in _WRAPPED:
            return
        try:
            wrapt.wrap_function_wrapper(module, name, wrapper)
            _WRAPPED.add(key)
            logger.debug("Decision HTTP matcher wrapped %s", key)
        except (ImportError, AttributeError) as e:
            logger.debug("Decision HTTP matcher skipped %s: %s", key, e)

    # -- request ---------------------------------------------------------------

    def _http_request(self, protocol: str, url: Any, body: Any) -> DecisionRequest:
        body = body if isinstance(body, dict) else {}
        questions = body.get("questions")
        model = body.get("model")
        if protocol == PROTOCOL_OPENAI_DECISIONS:
            system, operation = SYSTEM_OPENAI, "openai.decisions"
            parsed = openai_questions(questions)
            payload = {"model": model, "input": body.get("input"), "questions": questions}
        else:
            system, operation = SYSTEM_TYPESAFE, "typesafe.system_one"
            parsed = systemone_questions(questions)
            payload = {"state": body.get("state"), "model": model, "questions": questions}
        return DecisionRequest(
            protocol=protocol,
            system=system,
            operation=operation,
            model=str(model) if model else None,
            provider=provider_for_base_url(url, self._configured_provider()),
            questions=parsed,
            payload=payload if body else None,
            legacy_llm_kind=protocol == PROTOCOL_SYSTEMONE
            and bool(getattr(self.config, "decision_legacy_llm_kind", False)),
        )

    @staticmethod
    def _eval_kwargs(body: Any) -> Dict[str, Any]:
        """The request fields the detectors read: ``state`` / ``questions`` or ``input``."""
        if not isinstance(body, dict):
            return {}
        return {key: body[key] for key in ("state", "questions", "input") if key in body}

    # -- response --------------------------------------------------------------

    @staticmethod
    def _record(recorder: DecisionRecorder, response: Any, body: Any) -> None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and status >= 400:
            recorder.fail(f"HTTP {status}", str(status))
            return
        if isinstance(body, dict):
            recorder.record_response(body)

    @staticmethod
    def _requests_body(response: Any, streamed: bool) -> Any:
        # A streamed response that has not been read is left unread: reading
        # it here would consume what the caller is about to iterate.
        if streamed and getattr(response, "_content", False) is False:
            return None
        try:
            return _json_or_none(response.content)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _httpx_body(response: Any) -> Any:
        try:
            return _json_or_none(response.content)
        except Exception:  # noqa: BLE001 - httpx.ResponseNotRead on an unread stream
            return None

    @staticmethod
    def _httpx_request_body(request: Any) -> Any:
        try:
            return _json_or_none(request.content)
        except Exception:  # noqa: BLE001 - a streaming request body has no content yet
            return None

    @staticmethod
    def _span_name(protocol: str) -> str:
        return (
            "decision.http.openai"
            if protocol == PROTOCOL_OPENAI_DECISIONS
            else "decision.http.systemone"
        )

    # -- wrappers --------------------------------------------------------------

    def _wrap_requests_send(self, wrapped, instance, args, kwargs):
        request = args[0] if args else kwargs.get("request")
        protocol = match_decision_path(_get(request, "method"), _get(request, "url"))
        if protocol is None or _inside_decision_span():
            return wrapped(*args, **kwargs)
        body = _json_or_none(_get(request, "body"))
        decision = self._http_request(protocol, request.url, body)
        with self._run_decision_span(
            self._span_name(protocol), decision, self._eval_kwargs(body)
        ) as recorder:
            response = wrapped(*args, **kwargs)
            self._record(
                recorder, response, self._requests_body(response, bool(kwargs.get("stream")))
            )
            return response

    def _wrap_httpx_send(self, wrapped, instance, args, kwargs):
        request = args[0] if args else kwargs.get("request")
        protocol = match_decision_path(_get(request, "method"), _get(request, "url"))
        if protocol is None or _inside_decision_span():
            return wrapped(*args, **kwargs)
        body = self._httpx_request_body(request)
        decision = self._http_request(protocol, request.url, body)
        with self._run_decision_span(
            self._span_name(protocol), decision, self._eval_kwargs(body)
        ) as recorder:
            response = wrapped(*args, **kwargs)
            self._record(recorder, response, self._httpx_body(response))
            return response

    def _wrap_httpx_send_async(self, wrapped, instance, args, kwargs):
        request = args[0] if args else kwargs.get("request")
        protocol = match_decision_path(_get(request, "method"), _get(request, "url"))
        if protocol is None or _inside_decision_span():
            return wrapped(*args, **kwargs)
        body = self._httpx_request_body(request)
        decision = self._http_request(protocol, request.url, body)

        async def send():
            with self._run_decision_span(
                self._span_name(protocol), decision, self._eval_kwargs(body)
            ) as recorder:
                response = await wrapped(*args, **kwargs)
                self._record(recorder, response, self._httpx_body(response))
                return response

        return send()
