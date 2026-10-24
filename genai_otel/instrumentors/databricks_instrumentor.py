"""OpenTelemetry instrumentation for Databricks Model Serving APIs.

Databricks exposes Model Serving and Foundation Model APIs through an
OpenAI-compatible API. This instrumentor recognizes clients pointed at a
Databricks serving endpoint and records Databricks as the provider rather than
mislabeling the request as OpenAI.
"""

import logging
from typing import Any, Dict
from urllib.parse import urlparse

from ..config import OTelConfig
from .base import BaseInstrumentor, register_base_url_claim
from .openai_instrumentor import OpenAIInstrumentor

logger = logging.getLogger(__name__)


class DatabricksInstrumentor(OpenAIInstrumentor):
    """Instrument OpenAI SDK clients targeting Databricks Model Serving."""

    _CLAIM_BASE_URLS = (
        "databricks.com/serving-endpoints",
        "databricks.com/ai-gateway/",
        "azuredatabricks.net/serving-endpoints",
        "azuredatabricks.net/ai-gateway/",
    )
    _CLAIM_DOMAINS = ("databricks.com", "azuredatabricks.net")
    _SERVING_PATHS = ("/serving-endpoints", "/ai-gateway/")

    def instrument(self, config: OTelConfig):
        """Attach hooks to OpenAI-compatible and Databricks SDK clients."""
        self.config = config
        installed = False
        try:
            import wrapt

            for base_url_fragment in self._CLAIM_BASE_URLS:
                register_base_url_claim(base_url_fragment, "databricks")

            try:
                import openai
            except ImportError:
                openai = None
            if openai is not None:
                marker = "_genai_otel_databricks_instrumented"
                if not getattr(openai, marker, False):
                    for client_name, instrument in (
                        ("OpenAI", self._instrument_client),
                        ("AsyncOpenAI", self._instrument_async_client),
                    ):
                        client_class = getattr(openai, client_name, None)
                        if client_class is None:
                            continue
                        original_init = client_class.__init__

                        def wrapped_init(wrapped, instance, args, kwargs, _instrument=instrument):
                            result = wrapped(*args, **kwargs)
                            if self._is_databricks_client(instance):
                                _instrument(instance)
                            return result

                        client_class.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)
                    # databricks_openai.DatabricksOpenAI subclasses OpenAI and
                    # therefore passes through these constructor hooks.
                    setattr(openai, marker, True)
                installed = True

            try:
                from databricks.sdk import WorkspaceClient
            except ImportError:
                WorkspaceClient = None
            if WorkspaceClient is not None and not getattr(
                WorkspaceClient, "_genai_otel_databricks_instrumented", False
            ):
                original_init = WorkspaceClient.__init__

                def wrapped_workspace_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_serving_endpoints(instance)
                    return result

                WorkspaceClient.__init__ = wrapt.FunctionWrapper(
                    original_init, wrapped_workspace_init
                )
                WorkspaceClient._genai_otel_databricks_instrumented = True
                installed = True

            self._instrumented = installed
            if installed:
                logger.info("Databricks Model Serving instrumentation enabled")
            else:
                logger.debug("Databricks SDK clients are not installed; skipping instrumentation")
        except Exception as e:  # noqa: BLE001 - instrumentation must not break startup
            logger.error("Failed to instrument Databricks clients: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_serving_endpoints(self, client):
        """Wrap the Databricks SDK's native serving endpoint query method."""
        serving_endpoints = getattr(client, "serving_endpoints", None)
        query = getattr(serving_endpoints, "query", None)
        if not callable(query) or getattr(query, "_genai_otel_databricks_wrapped", False):
            return
        wrapped_query = self.create_span_wrapper(
            span_name="databricks.serving.query",
            extract_attributes=self._extract_sdk_query_attributes,
        )(query)
        try:
            wrapped_query._genai_otel_databricks_wrapped = True
        except Exception:  # noqa: BLE001
            pass
        serving_endpoints.query = wrapped_query

    def _extract_sdk_query_attributes(self, instance, args, kwargs):
        """Normalize native SDK query fields to the common GenAI attributes."""
        name = kwargs.get("name")
        attrs: Dict[str, Any] = {
            "gen_ai.system": "databricks",
            "gen_ai.provider.name": "databricks",
            "gen_ai.request.model": str(name or "unknown"),
        }
        if kwargs.get("messages") is not None:
            messages = kwargs.get("messages")
            attrs["gen_ai.operation.name"] = "chat"
            if isinstance(messages, (list, tuple)):
                attrs["gen_ai.request.message_count"] = len(messages)
                first_message = self._build_first_message(messages)
                if first_message:
                    attrs["gen_ai.request.first_message"] = first_message
        elif kwargs.get("input") is not None or kwargs.get("inputs") is not None:
            attrs["gen_ai.operation.name"] = "embeddings"
            attrs["gen_ai.request.type"] = "embedding"
            attrs["gen_ai.request.input_count"] = self._count_embedding_inputs(
                kwargs.get("input", kwargs.get("inputs"))
            )
        else:
            attrs["gen_ai.operation.name"] = "text_completion"
            attrs["gen_ai.request.type"] = "completion"
        for key in ("temperature", "top_p", "max_tokens", "max_output_tokens"):
            value = kwargs.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                attrs[f"gen_ai.request.{key}"] = value
        return attrs

    @classmethod
    def _is_databricks_client(cls, client: Any) -> bool:
        """Recognize the documented Databricks serving and gateway base URLs."""
        base_url = getattr(client, "base_url", None)
        if not base_url:
            return False
        try:
            parsed = urlparse(str(base_url))
            host = (parsed.hostname or "").lower()
            path = (parsed.path or "").lower()
        except Exception:  # noqa: BLE001
            return False
        return any(
            host == domain or host.endswith("." + domain) for domain in cls._CLAIM_DOMAINS
        ) and any(path.startswith(prefix) for prefix in cls._SERVING_PATHS)

    def _instrument_client(self, client):
        """Wrap supported synchronous OpenAI-compatible serving operations."""
        self._instrument_resources(client, asynchronous=False)

    def _instrument_async_client(self, client):
        """Wrap supported asynchronous OpenAI-compatible serving operations."""
        self._instrument_resources(client, asynchronous=True)

    def _instrument_resources(self, client, asynchronous: bool):
        wrap = self._create_async_span_wrapper if asynchronous else self.create_span_wrapper
        resources = (
            (
                "chat",
                "completions",
                "create",
                "chat.completion",
                self._extract_databricks_chat_attributes,
            ),
            (
                "completions",
                None,
                "create",
                "completion",
                self._extract_databricks_completion_attributes,
            ),
            (
                "embeddings",
                None,
                "create",
                "embeddings",
                self._extract_databricks_embedding_attributes,
            ),
            (
                "responses",
                None,
                "create",
                "responses",
                self._extract_databricks_responses_attributes,
            ),
        )
        for resource_name, child_name, method_name, operation, extractor in resources:
            resource = getattr(client, resource_name, None)
            if child_name:
                resource = getattr(resource, child_name, None)
            method = getattr(resource, method_name, None) if resource else None
            if not callable(method):
                continue
            setattr(
                resource,
                method_name,
                wrap(
                    span_name=f"databricks.{operation}",
                    extract_attributes=extractor,
                )(method),
            )

    @staticmethod
    def _provider_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
        attrs["gen_ai.system"] = "databricks"
        attrs["gen_ai.provider.name"] = "databricks"
        return attrs

    def _extract_databricks_chat_attributes(self, instance, args, kwargs):
        attrs = super()._extract_openai_attributes(instance, args, kwargs)
        attrs["gen_ai.operation.name"] = "chat"
        return self._provider_attributes(attrs)

    def _extract_databricks_embedding_attributes(self, instance, args, kwargs):
        attrs = super()._extract_embedding_attributes(instance, args, kwargs)
        return self._provider_attributes(attrs)

    def _extract_databricks_responses_attributes(self, instance, args, kwargs):
        attrs = super()._extract_responses_attributes(instance, args, kwargs)
        return self._provider_attributes(attrs)

    def _extract_databricks_completion_attributes(self, instance, args, kwargs):
        prompt = kwargs.get("prompt", "")
        attrs: Dict[str, Any] = {
            "gen_ai.system": "databricks",
            "gen_ai.provider.name": "databricks",
            "gen_ai.request.model": kwargs.get("model", "unknown"),
            "gen_ai.operation.name": "text_completion",
            "gen_ai.request.type": "completion",
        }
        if isinstance(prompt, (list, tuple)):
            attrs["gen_ai.request.input_count"] = len(prompt)
        elif prompt:
            attrs["gen_ai.request.input_count"] = 1
        for key in ("temperature", "top_p", "max_tokens", "n"):
            value = kwargs.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                attrs[f"gen_ai.request.{key}"] = value
        return attrs


__all__ = ["DatabricksInstrumentor"]
