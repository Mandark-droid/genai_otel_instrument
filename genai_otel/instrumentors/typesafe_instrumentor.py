"""Native OpenTelemetry instrumentation for the TypeSafe Python SDK.

TypeSafe's System One API evaluates structured questions against a state and
returns typed decisions rather than chat messages. This instrumentor wraps the
public ``TypeSafeClient.system_one`` and ``AsyncTypeSafeClient.system_one``
methods without depending on OpenInference's TypeSafe package.
"""

import dataclasses
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class TypeSafeInstrumentor(BaseInstrumentor):
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
        return self.create_span_wrapper(
            span_name="typesafe.system_one",
            extract_attributes=self._extract_request_attributes,
        )(wrapped)(*args, **kwargs)

    @staticmethod
    def _get_value(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        """Convert SDK structs, Pydantic models, and mappings to JSON values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        try:
            import msgspec

            if isinstance(value, msgspec.Struct):
                return cls._to_jsonable(msgspec.to_builtins(value))
        except (ImportError, TypeError):
            pass
        if isinstance(value, Mapping):
            return {str(key): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [cls._to_jsonable(item) for item in value]
        if dataclasses.is_dataclass(value):
            return cls._to_jsonable(dataclasses.asdict(value))
        for method_name in ("model_dump", "dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    return cls._to_jsonable(method())
                except TypeError:
                    try:
                        return cls._to_jsonable(method(mode="json"))
                    except Exception:  # pragma: no cover - SDK-version defensive path
                        pass
        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, dict):
            return {str(key): cls._to_jsonable(item) for key, item in attrs.items()}
        return str(value)

    def _json(self, value: Any) -> str:
        return json.dumps(self._to_jsonable(value), default=str, sort_keys=True)

    def _capture(self, value: Any) -> str:
        payload = self._json(value)
        config = getattr(self, "config", None)
        maximum = getattr(config, "content_max_length", 200) if config else 200
        return payload[:maximum] if maximum and maximum > 0 else payload

    @staticmethod
    def _request_values(instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        state = kwargs.get("state", args[0] if len(args) > 0 else None)
        questions = kwargs.get("questions", args[1] if len(args) > 1 else None)
        model = kwargs.get("model", args[2] if len(args) > 2 else None)
        extra_body = kwargs.get("extra_body", args[3] if len(args) > 3 else None)
        return state, questions, model, extra_body

    @classmethod
    def _client_default_model(cls, instance: Any) -> Optional[str]:
        for config_name in ("_config", "config"):
            config = getattr(instance, config_name, None)
            model = cls._get_value(config, "default_model")
            if model:
                return str(model)
        return None

    def _model_name(self, instance: Any, model: Any) -> str:
        return str(
            model
            or self._client_default_model(instance)
            or os.getenv("TYPESAFE_DEFAULT_MODEL")
            or "jev-latest"
        )

    def _extract_request_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        state, questions, model, extra_body = self._request_values(instance, args, kwargs)
        model_name = self._model_name(instance, model)
        invocation = {"model": model_name}
        if extra_body is not None:
            invocation["extra_body"] = extra_body

        attrs: Dict[str, Any] = {
            "gen_ai.system": "typesafe",
            "gen_ai.operation.name": "typesafe.system_one",
            "gen_ai.request.model": model_name,
            "gen_ai.request.type": "chat",
            "openinference.span.kind": "LLM",
            "llm.request.model_name": model_name,
            "llm.invocation_parameters": self._json(invocation),
        }
        if self.config and self.config.enable_content_capture:
            request = {"state": state, "model": model_name, "questions": questions}
            attrs["input.value"] = self._capture(request)
            attrs["gen_ai.request.first_message"] = self._capture(
                {"role": "user", "content": request}
            )
        return attrs

    def _extract_response_attributes(self, result: Any) -> Dict[str, Any]:
        model = self._get_value(result, "model")
        answers = self._get_value(result, "answers")
        usage = self._get_value(result, "usage")
        attrs: Dict[str, Any] = {}
        if model:
            attrs["gen_ai.response.model"] = str(model)
            attrs["llm.response.model_name"] = str(model)
        if usage is not None:
            prompt = self._get_value(usage, "input_tokens")
            completion = self._get_value(usage, "output_tokens")
            total = self._get_value(usage, "total_tokens")
            if isinstance(prompt, (int, float)):
                attrs["llm.token_count.prompt"] = int(prompt)
            if isinstance(completion, (int, float)):
                attrs["llm.token_count.completion"] = int(completion)
            if isinstance(total, (int, float)):
                attrs["llm.token_count.total"] = int(total)
        if self.config and self.config.enable_content_capture:
            response = {"model": model, "answers": answers, "usage": usage}
            attrs["output.value"] = self._capture(response)
            attrs["gen_ai.response"] = self._capture(answers)
        return attrs

    def _extract_prompt_for_eval(self, kwargs) -> Optional[str]:
        state, questions, _, _ = self._request_values(None, (), kwargs)
        if state is not None or questions is not None:
            return self._json({"state": state, "questions": questions})
        return super()._extract_prompt_for_eval(kwargs)

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        usage = self._get_value(result, "usage")
        if usage is None:
            return None
        prompt = self._get_value(usage, "input_tokens")
        if prompt is None:
            prompt = self._get_value(usage, "prompt_tokens")
        completion = self._get_value(usage, "output_tokens")
        if completion is None:
            completion = self._get_value(usage, "completion_tokens")
        total = self._get_value(usage, "total_tokens")
        if not any(isinstance(value, (int, float)) for value in (prompt, completion, total)):
            return None
        if (
            not isinstance(total, (int, float))
            and isinstance(prompt, (int, float))
            and isinstance(completion, (int, float))
        ):
            total = prompt + completion
        return {
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(completion or 0),
            "total_tokens": int(total or 0),
        }
