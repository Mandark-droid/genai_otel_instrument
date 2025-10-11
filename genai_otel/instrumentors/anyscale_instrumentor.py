from .base import BaseInstrumentor
from ..config import OTelConfig
from typing import Dict, Optional


class AnyscaleInstrumentor(BaseInstrumentor):
    """Instrumentor for Anyscale Endpoints"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import openai
            # Anyscale uses OpenAI SDK, already instrumented

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
