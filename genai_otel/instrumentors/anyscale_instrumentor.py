"""OpenTelemetry instrumentor for Anyscale Endpoints.

This instrumentor integrates with Anyscale Endpoints, which often leverage
OpenAI-compatible APIs. It ensures that calls made to Anyscale services are
properly traced and attributed within the OpenTelemetry ecosystem.
"""

from typing import Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor


class AnyscaleInstrumentor(BaseInstrumentor):
    """Instrumentor for Anyscale Endpoints"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            # Anyscale uses OpenAI SDK, already instrumented
            pass

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
