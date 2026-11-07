"""Voice-agent frameworks: Pipecat and LiveKit Agents (issue #48).

Both trace themselves through the global tracer provider, so once ``instrument()``
has run their spans reach the same exporter as ours. These instrumentors make those
spans follow this library's content setting and add the ``voice.*`` turn shape:

* **Pipecat** (``pipecat``): every pipeline gets an observer that records who spoke
  when (``voice.turn`` spans under ``voice.conversation``). With content capture off,
  Pipecat's own spans lose their message, transcript and spoken-text attributes.
* **LiveKit Agents** (``livekit_agents``): LiveKit's ``user_speaking`` /
  ``agent_speaking`` spans get the ``voice.turn.*`` attributes and ``agent_session``
  the conversation totals. With content capture off,
  ``LIVEKIT_TELEMETRY_ALLOW_PII=false`` is set unless the application set it, so
  LiveKit withholds transcripts itself.

Version floors are the versions measured (Pipecat 1.15, LiveKit Agents 1.8). Older
versions get the content rule but no turn shape.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Tuple

import wrapt
from opentelemetry import trace

from ..config import OTelConfig
from .base import BaseInstrumentor, content_capture_on

logger = logging.getLogger(__name__)

_LIVEKIT_PII_ENV = "LIVEKIT_TELEMETRY_ALLOW_PII"


def _version(dist: str) -> Optional[Tuple[int, int]]:
    try:
        parts = version(dist).split(".")
        return int(parts[0]), int(parts[1])
    except (PackageNotFoundError, ValueError, IndexError):
        return None


def _install_processor(config: OTelConfig, **kwargs) -> None:
    from ..voice import processor

    processor.install(trace.get_tracer_provider(), content_capture_on(config), **kwargs)


class PipecatInstrumentor(BaseInstrumentor):
    """Voice turn telemetry and the content rule for Pipecat pipelines."""

    MIN_VERSION = (1, 15)

    def __init__(self):
        super().__init__()
        self._pipecat_version = _version("pipecat-ai")

    def instrument(self, config: OTelConfig):
        self.config = config
        if self._pipecat_version is None or self._instrumented:
            return
        try:
            _install_processor(config, pipecat=True)
            if self._pipecat_version < self.MIN_VERSION:
                logger.debug(
                    "Pipecat %s is older than %s: content rule only, no voice turns",
                    self._pipecat_version,
                    self.MIN_VERSION,
                )
                self._instrumented = True
                return
            from pipecat.pipeline import worker as worker_module

            from ..voice.pipecat_observer import make_observer

            tracer = trace.get_tracer("genai_otel.voice")

            def after_init(wrapped, instance, args, kwargs):
                result = wrapped(*args, **kwargs)
                try:
                    instance.add_observer(make_observer(tracer, kwargs.get("conversation_id")))
                except Exception as e:  # noqa: BLE001 - never break the pipeline
                    logger.debug("Could not add the voice turn observer: %s", e)
                return result

            wrapt.wrap_function_wrapper(worker_module, "PipelineWorker.__init__", after_init)
            self._instrumented = True
            logger.info("Pipecat voice turn instrumentation enabled")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to instrument Pipecat: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_usage(self, result):
        return None


class LiveKitAgentsInstrumentor(BaseInstrumentor):
    """Voice turn shape on LiveKit Agents' spans, and its PII switch from ours."""

    MIN_VERSION = (1, 8)

    def __init__(self):
        super().__init__()
        self._livekit_version = _version("livekit-agents")

    def instrument(self, config: OTelConfig):
        self.config = config
        if self._livekit_version is None or self._instrumented:
            return
        try:
            if not content_capture_on(config) and _LIVEKIT_PII_ENV not in os.environ:
                os.environ[_LIVEKIT_PII_ENV] = "false"
                logger.debug("Content capture is off: set %s=false", _LIVEKIT_PII_ENV)
            _install_processor(config, livekit=self._livekit_version >= self.MIN_VERSION)
            self._instrumented = True
            logger.info("LiveKit Agents voice turn instrumentation enabled")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to instrument LiveKit Agents: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_usage(self, result):
        return None
