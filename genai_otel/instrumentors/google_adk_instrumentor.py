"""OpenTelemetry instrumentor for Google Agent Development Kit (ADK).

Wraps ``Runner.run_async()`` and ``InMemoryRunner.run_debug()`` in one span per
run (``google_adk.runner.run`` / ``google_adk.runner.run_debug``).

ADK traces itself through the global tracer provider, so its own spans
(ADK 2.x: ``invoke_workflow``, ``invoke_agent``, ``generate_content``,
``execute_tool``) land under this run span. This span adds what ADK's do not
carry together: the session, the agent the run started at, the agent that
answered, the handoffs between them, and (with content capture on) the input
and the final answer. It carries no token counts: ADK's ``generate_content``
spans and the model instrumentors already do, and a third copy would be
counted again.

Content: with content capture off, this span carries no message text, and
``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`` is set to ``false`` unless the
application set it, so ADK's legacy (schema v1) spans carry none either.

Requirements:
    pip install google-adk>=1.17.0
"""

import logging
import os
from typing import Any, Dict, List, Optional

from ..agent_budget import extract_budget_attributes
from ..config import OTelConfig
from .base import BaseInstrumentor, content_capture_on

logger = logging.getLogger(__name__)

_ADK_CONTENT_ENV = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"
_RUN_SPAN = "google_adk.runner.run"
_LIST_CAP = 20


def _event_text(event: Any) -> Optional[str]:
    """The visible text of an ADK event (thought parts excluded), or None."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    texts = [
        p.text
        for p in parts
        if isinstance(getattr(p, "text", None), str) and p.text and not getattr(p, "thought", False)
    ]
    return "".join(texts) if texts else None


class _RunSummary:
    """What one ADK run did, read from its events as they pass. Never holds the events."""

    def __init__(self) -> None:
        self.agents: List[str] = []
        self.transfers: List[str] = []
        self.transfer_count = 0
        self.event_count = 0
        self.invocation_id: Optional[str] = None
        self.final_agent: Optional[str] = None
        self.final_text: Optional[str] = None

    def add(self, event: Any) -> None:
        self.event_count += 1
        author = getattr(event, "author", None)
        if isinstance(author, str) and author and author != "user":
            if author not in self.agents:
                self.agents.append(author)
        if self.invocation_id is None:
            invocation_id = getattr(event, "invocation_id", None)
            if invocation_id:
                self.invocation_id = str(invocation_id)
        target = getattr(getattr(event, "actions", None), "transfer_to_agent", None)
        if target:
            self.transfer_count += 1
            if len(self.transfers) < _LIST_CAP:
                self.transfers.append(f"{author}>{target}")
        is_final = getattr(event, "is_final_response", None)
        try:
            final = bool(is_final()) if callable(is_final) else False
        except Exception:  # noqa: BLE001 - an exotic event is simply not final
            final = False
        if final and isinstance(author, str) and author != "user":
            self.final_agent = author
            text = _event_text(event)
            if text is not None:
                self.final_text = text

    def attributes(self) -> Dict[str, Any]:
        """Structural attributes only; the final text is content and set separately."""
        attrs: Dict[str, Any] = {"google_adk.event_count": self.event_count}
        if self.agents:
            # ADK starts a later turn at the agent that answered last, not at the root.
            attrs["gen_ai.agent.name"] = self.agents[0]
            attrs["google_adk.agents"] = self.agents[:_LIST_CAP]
        if self.final_agent:
            attrs["google_adk.final_agent.name"] = self.final_agent
        if self.transfers:
            attrs["google_adk.transfers"] = self.transfers
        attrs["google_adk.transfer_count"] = self.transfer_count
        if self.invocation_id:
            attrs["google_adk.invocation_id"] = self.invocation_id
        return attrs


class GoogleADKInstrumentor(BaseInstrumentor):
    """Instrumentor for Google Agent Development Kit (ADK)"""

    # Content keys: dropped with content capture off (BaseInstrumentor.CONTENT_ATTRIBUTES).
    CONTENT_ATTRIBUTES = frozenset(
        {
            "google_adk.input_message",
            "google_adk.output",
        }
    )

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._adk_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Google ADK library is available."""
        try:
            import google.adk  # noqa: F401

            self._adk_available = True
            logger.debug("Google ADK library detected and available for instrumentation")
        except ImportError:
            logger.debug("Google ADK library not installed, instrumentation will be skipped")
            self._adk_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Google ADK framework if available.

        Instruments:
        - Runner.run_async() - Main async execution method
        - InMemoryRunner.run_debug() - Simplified debug execution

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._adk_available:
            logger.debug("Skipping Google ADK instrumentation - library not available")
            return

        # Idempotency guard: repeated instrument() calls must not stack wrappers
        # on the shared Runner / InMemoryRunner classes.
        if self._instrumented:
            logger.debug("Google ADK already instrumented, skipping repeat instrument()")
            return

        self.config = config

        # ADK's legacy span schema records the full model request and response unless
        # this is "false". Follow our content setting, but never override the app.
        if not content_capture_on(config) and _ADK_CONTENT_ENV not in os.environ:
            os.environ[_ADK_CONTENT_ENV] = "false"
            logger.debug(
                "Content capture is off: set %s=false for ADK's own spans", _ADK_CONTENT_ENV
            )

        try:
            # Instrument Runner.run_async()
            try:
                from google.adk.runners import Runner

                run_wrapper = self.create_span_wrapper(
                    span_name=_RUN_SPAN,
                    extract_attributes=self._extract_runner_attributes,
                )
                if hasattr(Runner, "run_async"):
                    Runner.run_async = run_wrapper(Runner.run_async)
                    logger.debug("Instrumented Runner.run_async()")
            except ImportError:
                logger.debug("Could not import google.adk.runners.Runner")

            # Instrument InMemoryRunner.run_debug()
            try:
                from google.adk.runners import InMemoryRunner

                debug_wrapper = self.create_span_wrapper(
                    span_name="google_adk.runner.run_debug",
                    extract_attributes=self._extract_runner_debug_attributes,
                )
                if hasattr(InMemoryRunner, "run_debug"):
                    InMemoryRunner.run_debug = debug_wrapper(InMemoryRunner.run_debug)
                    logger.debug("Instrumented InMemoryRunner.run_debug()")
            except ImportError:
                logger.debug("Could not import google.adk.runners.InMemoryRunner")

            self._instrumented = True
            logger.info("Google ADK instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Google ADK: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _cap_content(self, text: Any, default: int = 200) -> str:
        """Cap captured content per ``config.content_max_length``.

        Only bounds the size. Whether the text reaches a span at all is decided by
        the content-capture setting (``CONTENT_ATTRIBUTES``).
        ``content_max_length == 0`` (or missing) means unlimited. When no usable
        config is present (unit tests) the historical default cap is used.
        """
        s = text if isinstance(text, str) else str(text)
        cfg = getattr(self, "config", None)
        max_len = default
        cfg_len = getattr(cfg, "content_max_length", None) if cfg is not None else None
        if isinstance(cfg_len, int):
            max_len = cfg_len
        if max_len and max_len > 0:
            return s[:max_len]
        return s

    def _extract_runner_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Runner.run_async() call.

        Args:
            instance: The Runner instance.
            args: Positional arguments (user_id, session_id, new_message).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "google_adk"
        attrs["gen_ai.operation.name"] = "runner.run"
        # Google ADK bounds a run with max_llm_calls.
        attrs.update(extract_budget_attributes(kwargs, kwargs.get("run_config"), instance))

        # Extract app_name from runner
        if hasattr(instance, "app_name"):
            attrs["google_adk.app_name"] = str(instance.app_name)

        # Extract agent info
        if hasattr(instance, "agent"):
            agent = instance.agent
            if hasattr(agent, "name"):
                attrs["google_adk.agent.name"] = str(agent.name)
                attrs["google_adk.root_agent.name"] = str(agent.name)
                # Cross-framework agent attribution per upstream proposal
                # semantic-conventions-genai#91. Replaced when the run ends by the
                # agent ADK actually started at, which on a later turn of a
                # handed-off conversation is not the root.
                attrs["gen_ai.agent.name"] = str(agent.name)
            if hasattr(agent, "model"):
                attrs["gen_ai.request.model"] = str(agent.model)
            if hasattr(agent, "description") and agent.description:
                attrs["google_adk.agent.description"] = str(agent.description)[:200]

            # Extract sub_agents count if available
            if hasattr(agent, "sub_agents") and agent.sub_agents:
                attrs["google_adk.sub_agent_count"] = len(agent.sub_agents)
                sub_agent_names = []
                for sub_agent in agent.sub_agents:
                    if hasattr(sub_agent, "name"):
                        sub_agent_names.append(str(sub_agent.name))
                if sub_agent_names:
                    attrs["google_adk.sub_agent_names"] = sub_agent_names[:10]

            # Extract tools
            if hasattr(agent, "tools") and agent.tools:
                tool_names = []
                for tool in agent.tools:
                    name = getattr(tool, "name", None) or type(tool).__name__
                    tool_names.append(str(name))
                if tool_names:
                    attrs["google_adk.tools"] = tool_names[:10]
                    attrs["google_adk.tool_count"] = len(agent.tools)

        # Extract user_id and session_id from kwargs
        user_id = kwargs.get("user_id")
        if user_id:
            attrs["google_adk.user_id"] = str(user_id)

        session_id = kwargs.get("session_id")
        if session_id:
            attrs["google_adk.session_id"] = str(session_id)
            attrs["session.id"] = str(session_id)
            # Cross-framework conversation correlation per upstream proposal
            # semantic-conventions-genai#145.
            attrs["gen_ai.conversation.id"] = str(session_id)

        # Extract message content (from new_message kwarg)
        new_message = kwargs.get("new_message")
        if new_message:
            # new_message is a Content object with parts
            if hasattr(new_message, "parts"):
                for part in new_message.parts:
                    if hasattr(part, "text") and part.text:
                        attrs["google_adk.input_message"] = self._cap_content(part.text)
                        break

        return attrs

    def _extract_runner_debug_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from InMemoryRunner.run_debug() call.

        Args:
            instance: The InMemoryRunner instance.
            args: Positional arguments (message string).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "google_adk"
        attrs["gen_ai.operation.name"] = "runner.run_debug"

        # Extract app_name from runner
        if hasattr(instance, "app_name"):
            attrs["google_adk.app_name"] = str(instance.app_name)

        # Extract agent info
        if hasattr(instance, "agent"):
            agent = instance.agent
            if hasattr(agent, "name"):
                attrs["google_adk.agent.name"] = str(agent.name)
                attrs["google_adk.root_agent.name"] = str(agent.name)
                # Cross-framework agent attribution per upstream proposal
                # semantic-conventions-genai#91. Replaced when the run ends by the
                # agent ADK actually started at, which on a later turn of a
                # handed-off conversation is not the root.
                attrs["gen_ai.agent.name"] = str(agent.name)
            if hasattr(agent, "model"):
                attrs["gen_ai.request.model"] = str(agent.model)

        # Extract the input message (first positional arg)
        if args and len(args) > 0:
            attrs["google_adk.input_message"] = self._cap_content(args[0])

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from ADK execution result.

        Note: Google ADK doesn't directly expose token usage in results.
        Token usage is captured by underlying LLM provider instrumentors.

        Args:
            result: The execution result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Run attributes from a ``run_debug()`` result: a list of ADK events in 2.x.

        ``google_adk.output`` is content: the base wrapper drops it with capture off.
        """
        attrs: Dict[str, Any] = {}
        try:
            if isinstance(result, str):
                attrs["google_adk.output"] = self._cap_content(result)
            elif isinstance(result, (list, tuple)):
                summary = _RunSummary()
                for event in result:
                    summary.add(event)
                attrs.update(summary.attributes())
                if summary.final_text is not None:
                    attrs["google_adk.output"] = self._cap_content(summary.final_text)
            elif hasattr(result, "text"):
                attrs["google_adk.output"] = self._cap_content(result.text)
        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    # ``Runner.run_async()`` is an async generator: the base wrapper hands each event
    # here as it passes, and calls _on_stream_end once the run stops.
    def _on_stream_item(self, span, item: Any) -> None:
        if getattr(span, "name", None) != _RUN_SPAN:
            return
        summary = getattr(span, "_genai_adk_run", None)
        if summary is None:
            summary = _RunSummary()
            span._genai_adk_run = summary
        summary.add(item)

    def _on_stream_end(self, span, outcome: str) -> None:
        summary = getattr(span, "_genai_adk_run", None)
        if summary is None:
            return
        for key, value in summary.attributes().items():
            span.set_attribute(key, value)
        if summary.final_text is not None:
            self._set_content_attribute(
                span, "google_adk.output", self._cap_content(summary.final_text)
            )

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from ADK execution result.

        Args:
            result: The execution result.

        Returns:
            Optional[str]: The finish reason string or None.
        """
        if result:
            return "completed"
        return None
