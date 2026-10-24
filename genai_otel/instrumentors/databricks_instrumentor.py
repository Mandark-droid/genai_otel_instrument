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

            # MLflow's Databricks deployment client is a separate request path
            # from both the OpenAI SDK and WorkspaceClient. Install the factory
            # hook only when MLflow is present; importing this instrumentor must
            # remain safe in environments that do not use MLflow.
            try:
                import mlflow.deployments as mlflow_deployments
            except ImportError:
                mlflow_deployments = None
            if mlflow_deployments is not None and not getattr(
                mlflow_deployments, "_genai_otel_databricks_instrumented", False
            ):
                original_get_deploy_client = mlflow_deployments.get_deploy_client

                def wrapped_get_deploy_client(wrapped, instance, args, kwargs):
                    target = args[0] if args else kwargs.get("target_uri")
                    client = wrapped(*args, **kwargs)
                    if str(target or "").lower() == "databricks":
                        self._instrument_mlflow_client(client)
                    return client

                mlflow_deployments.get_deploy_client = wrapt.FunctionWrapper(
                    original_get_deploy_client, wrapped_get_deploy_client
                )
                mlflow_deployments._genai_otel_databricks_instrumented = True
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

    def _instrument_mlflow_client(self, client):
        """Wrap inference methods on an MLflow Databricks deployment client."""
        for method_name in ("predict", "predict_stream"):
            method = getattr(client, method_name, None)
            if not callable(method) or getattr(method, "_genai_otel_databricks_wrapped", False):
                continue
            if method_name == "predict_stream":
                wrapped_method = self._wrap_mlflow_stream(method)
            else:
                wrapped_method = self.create_span_wrapper(
                    span_name="databricks.mlflow.predict",
                    extract_attributes=self._extract_mlflow_attributes,
                )(method)
            try:
                wrapped_method._genai_otel_databricks_wrapped = True
            except Exception:  # noqa: BLE001
                pass
            setattr(client, method_name, wrapped_method)

    def _wrap_mlflow_stream(self, method):
        """Route MLflow's iterator response through the common stream measurer."""
        import wrapt

        span_wrapper = self.create_span_wrapper(
            span_name="databricks.mlflow.predict_stream",
            extract_attributes=self._extract_mlflow_attributes,
        )

        def call_stream(instance, args, kwargs):
            # The common wrapper uses the stream flag to retain the span and
            # capture final-chunk usage. MLflow exposes this as a separate
            # method, so don't forward the internal flag to its API.
            def invoke(*inner_args, **inner_kwargs):
                inner_kwargs.pop("stream", None)
                return method(*inner_args, **inner_kwargs)

            return span_wrapper(invoke)(*args, **dict(kwargs, stream=True))

        def wrapped_stream(wrapped, instance, args, kwargs):
            return call_stream(instance, args, kwargs)

        return wrapt.FunctionWrapper(method, wrapped_stream)

    def _extract_mlflow_attributes(self, instance, args, kwargs):
        """Extract endpoint, model input, and provider fields from MLflow calls."""
        endpoint = kwargs.get("endpoint") or kwargs.get("deployment_name")
        inputs = kwargs.get("inputs")
        if not endpoint:
            endpoint = (args[2] if len(args) > 2 else None) or (args[0] if args else None)
        if inputs is None and len(args) > 1:
            inputs = args[1]
        attrs: Dict[str, Any] = {
            "gen_ai.system": "databricks",
            "gen_ai.provider.name": "databricks",
            "gen_ai.request.model": str(endpoint or "unknown"),
            "gen_ai.operation.name": "chat",
        }
        if isinstance(inputs, dict):
            messages = inputs.get("messages")
            if isinstance(messages, (list, tuple)):
                attrs["gen_ai.request.message_count"] = len(messages)
                first_message = self._build_first_message(messages)
                if first_message:
                    attrs["gen_ai.request.first_message"] = first_message
            if inputs.get("input") is not None:
                attrs["gen_ai.operation.name"] = "embeddings"
                attrs["gen_ai.request.type"] = "embedding"
                attrs["gen_ai.request.input_count"] = self._count_embedding_inputs(inputs["input"])
        return attrs

    def _extract_usage(self, result):
        """Read MLflow's dictionary-shaped response and standard SDK responses."""
        if isinstance(result, dict):
            usage = result.get("usage")
            if isinstance(usage, dict):
                prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                return {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": usage.get("total_tokens", prompt + completion)
                    or prompt + completion,
                }
        return super()._extract_usage(result)

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
