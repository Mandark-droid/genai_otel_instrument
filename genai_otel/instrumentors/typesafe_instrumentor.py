"""Native OpenTelemetry instrumentation for the TypeSafe Python SDK (SystemOne).

TypeSafe's System One API evaluates structured questions against a state and
returns typed decisions rather than chat messages. The same protocol
(``POST /v1/systemone``) is answered by LiquidAI, Cloudflare Workers AI and any
vLLM / SGLang / community server, all reachable with ``typesafe-sdk`` and a
``base_url``, so this one wrapper covers every SystemOne host.

Since 1.38 the spans are OpenInference ``DECISION`` spans. The ``llm.*``
attributes they carried as ``LLM`` spans are still emitted (until 2.0), and
``GENAI_DECISION_LEGACY_LLM_KIND=true`` keeps the old span kind.

This instrumentor wraps the public ``TypeSafeClient.system_one`` and
``AsyncTypeSafeClient.system_one`` methods without depending on OpenInference's
TypeSafe package.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from ..config import OTelConfig
from .decision import (
    PROTOCOL_SYSTEMONE,
    SYSTEM_TYPESAFE,
    BaseDecisionInstrumentor,
    DecisionRequest,
    _get,
    payload_json,
    provider_for_base_url,
    request_attributes,
    request_content_attributes,
    systemone_questions,
    to_jsonable,
)

logger = logging.getLogger(__name__)


class TypeSafeInstrumentor(BaseDecisionInstrumentor):
    """Instrument synchronous and asynchronous TypeSafe System One calls."""

    def __init__(self):
        super().__init__()
        self._typesafe_available = False
        self._check_availability()

    def _check_availability(self) -> None:
        """Detect the optional SDK without importing it at package import time."""
        try:
            import typesafe_sdk

            self._typesafe_available = hasattr(typesafe_sdk, "TypeSafeClient")
        except ImportError:
            self._typesafe_available = False
            logger.debug("typesafe-sdk is not installed; skipping TypeSafe instrumentation")

    def instrument(self, config: OTelConfig) -> None:
        """Wrap the SDK's public sync and async System One methods."""
        if not self._typesafe_available:
            logger.debug("Skipping TypeSafe instrumentation - typesafe-sdk is not installed")
            return
        if self._instrumented:
            logger.debug("TypeSafe instrumentation already enabled")
            return

        self._setup_config(config)
        wrapped = 0
        try:
            import wrapt

            wrapt.wrap_function_wrapper(
                "typesafe_sdk",
                "TypeSafeClient.system_one",
                self._wrap_system_one,
            )
            wrapped += 1
            try:
                wrapt.wrap_function_wrapper(
                    "typesafe_sdk",
                    "AsyncTypeSafeClient.system_one",
                    self._wrap_system_one,
                )
                wrapped += 1
            except (AttributeError, ImportError):
                logger.debug("AsyncTypeSafeClient is not available in this typesafe-sdk version")

            self._instrumented = wrapped > 0
            logger.info("TypeSafe instrumentation enabled")
        except Exception as exc:
            logger.error("Failed to instrument TypeSafe: %s", exc, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_system_one(self, wrapped, instance, args, kwargs):
        # ``state`` and ``questions`` are named parameters. Passed positionally
        # they are moved into kwargs, so block policies and detectors, which
        # read the request from kwargs, see the state that is sent.
        if len(args) <= 2:
            kwargs = dict(kwargs)
            for name, value in zip(("state", "questions"), args):
                kwargs.setdefault(name, value)
            args = ()
        request = self._decision_request(instance, args, kwargs)
        return self._traced_decision(
            "typesafe.system_one", request, wrapped, instance, args, kwargs
        )

    def _json(self, value: Any) -> str:
        return json.dumps(to_jsonable(value), default=str, sort_keys=True)

    @staticmethod
    def _request_values(instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        state = kwargs.get("state", args[0] if len(args) > 0 else None)
        questions = kwargs.get("questions", args[1] if len(args) > 1 else None)
        model = kwargs.get("model", args[2] if len(args) > 2 else None)
        extra_body = kwargs.get("extra_body", args[3] if len(args) > 3 else None)
        return state, questions, model, extra_body

    @staticmethod
    def _client_config_value(instance: Any, name: str) -> Optional[str]:
        for config_name in ("_config", "config"):
            value = _get(getattr(instance, config_name, None), name)
            if value:
                return str(value)
        return None

    @classmethod
    def _client_default_model(cls, instance: Any) -> Optional[str]:
        return cls._client_config_value(instance, "default_model")

    def _model_name(self, instance: Any, model: Any) -> str:
        return str(
            model
            or self._client_default_model(instance)
            or os.getenv("TYPESAFE_DEFAULT_MODEL")
            or "jev-latest"
        )

    def _base_url(self, instance: Any) -> Optional[str]:
        """The host the client sends to: TypeSafe, LiquidAI, Workers AI or a self-hosted server."""
        return self._client_config_value(instance, "base_url") or (
            str(getattr(instance, "base_url", "") or "") or None
        )

    def _decision_request(self, instance: Any, args, kwargs) -> DecisionRequest:
        state, questions, model, extra_body = self._request_values(instance, args, kwargs)
        model_name = self._model_name(instance, model)
        extras = {"extra_body": extra_body} if extra_body is not None else {}
        return DecisionRequest(
            protocol=PROTOCOL_SYSTEMONE,
            system=SYSTEM_TYPESAFE,
            operation="typesafe.system_one",
            model=model_name,
            provider=provider_for_base_url(self._base_url(instance), self._configured_provider()),
            questions=systemone_questions(questions),
            payload={"state": state, "model": model_name, "questions": questions},
            legacy_llm_kind=bool(getattr(self.config, "decision_legacy_llm_kind", False)),
            extras=extras,
        )

    def _extra_request_attributes(self, request: DecisionRequest, capture: bool) -> Dict[str, Any]:
        """The ``llm.*`` request attributes Jev spans carried before 1.38 (kept until 2.0)."""
        invocation: Dict[str, Any] = {"model": request.model}
        invocation.update(request.extras)
        attrs: Dict[str, Any] = {
            "llm.request.model_name": request.model,
            "llm.invocation_parameters": self._json(invocation),
        }
        if capture:
            attrs["gen_ai.request.first_message"] = payload_json(
                {"role": "user", "content": request.payload}, self._max_length()
            )
        return attrs

    def _extract_request_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        """Every scalar request attribute of a System One call, as the span starts with them."""
        request = self._decision_request(instance, args, kwargs)
        capture = self._capture()
        attrs = request_attributes(request)
        if capture:
            attrs.update(request_content_attributes(request, self._max_length()))
        attrs.update(self._extra_request_attributes(request, capture))
        return attrs

    def _content_payload(self, payload: Any) -> Any:
        """``output.value`` keeps the shape Jev spans have always had."""
        return {
            "model": _get(payload, "model"),
            "answers": _get(payload, "answers"),
            "usage": _get(payload, "usage"),
        }

    def _extract_response_attributes(self, result: Any) -> Dict[str, Any]:
        """The ``llm.*`` response attributes Jev spans carried before 1.38 (kept until 2.0).

        The ``decision.*`` and ``genai.decision.*`` attributes are written by
        :meth:`BaseDecisionInstrumentor._record_result_metrics`.
        """
        model = _get(result, "model")
        usage = _get(result, "usage")
        attrs: Dict[str, Any] = {}
        if model:
            attrs["gen_ai.response.model"] = str(model)
            attrs["llm.response.model_name"] = str(model)
        if usage is not None:
            for source, key in (
                ("input_tokens", "llm.token_count.prompt"),
                ("output_tokens", "llm.token_count.completion"),
                ("total_tokens", "llm.token_count.total"),
            ):
                value = _get(usage, source)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    attrs[key] = int(value)
        if self._capture():
            attrs["gen_ai.response"] = payload_json(_get(result, "answers"), self._max_length())
        return attrs

    def _extract_prompt_for_eval(self, kwargs, operation: Optional[str] = None) -> Optional[str]:
        state, questions, _, _ = self._request_values(None, (), kwargs)
        if state is not None or questions is not None:
            return self._json({"state": state, "questions": questions})
        return super()._extract_prompt_for_eval(kwargs, operation)
