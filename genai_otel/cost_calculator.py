"""Module for calculating estimated costs of LLM API calls."""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CostCalculator:
    """Calculate estimated costs for LLM API calls based on loaded pricing data."""

    DEFAULT_PRICING_FILE = "llm_pricing.json"

    def __init__(self):
        """Initializes the CostCalculator by loading pricing data from a JSON file."""
        self.pricing_data: Dict[str, Any] = {}
        self._load_pricing()

    def _load_pricing(self):
        """Load pricing data from the JSON configuration file."""
        try:
            try:
                from importlib.resources import files

                pricing_file = files("genai_otel").joinpath(self.DEFAULT_PRICING_FILE)
                data = json.loads(pricing_file.read_text(encoding="utf-8"))
            except (ImportError, AttributeError):
                try:
                    import importlib_resources

                    pricing_file = importlib_resources.files("genai_otel").joinpath(
                        self.DEFAULT_PRICING_FILE
                    )
                    data = json.loads(pricing_file.read_text(encoding="utf-8"))
                except ImportError:
                    import pkg_resources

                    pricing_file_path = pkg_resources.resource_filename(
                        "genai_otel", self.DEFAULT_PRICING_FILE
                    )
                    with open(pricing_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

            if isinstance(data, dict):
                self.pricing_data = data
                logger.info("Successfully loaded pricing data.")
            else:
                logger.error("Invalid format in pricing file. Root element is not a dictionary.")
        except FileNotFoundError:
            logger.warning(
                "Pricing file '%s' not found. Cost tracking will be disabled.",
                self.DEFAULT_PRICING_FILE,
            )
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode JSON from pricing file: %s. Cost tracking will be disabled.", e
            )
        except Exception as e:
            logger.error("An unexpected error occurred while loading pricing: %s", e, exc_info=True)

    def calculate_cost(
        self,
        model: str,
        usage: Dict[str, Any],
        call_type: str,
    ) -> float:
        """Calculate cost in USD for a request based on model, usage, and call type."""
        if not self.pricing_data:
            return 0.0

        if call_type == "chat":
            return self._calculate_chat_cost(model, usage)
        if call_type == "embedding":
            return self._calculate_embedding_cost(model, usage)
        if call_type == "image":
            return self._calculate_image_cost(model, usage)
        if call_type == "audio":
            return self._calculate_audio_cost(model, usage)

        logger.warning("Unknown call type '%s' for cost calculation.", call_type)
        return 0.0

    def _calculate_chat_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost for chat models."""
        model_key = self._normalize_model_name(model, "chat")
        if not model_key:
            logger.debug("Pricing not found for chat model: %s", model)
            return 0.0

        pricing = self.pricing_data["chat"][model_key]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        prompt_cost = (prompt_tokens / 1000) * pricing.get("promptPrice", 0.0)
        completion_cost = (completion_tokens / 1000) * pricing.get("completionPrice", 0.0)

        return prompt_cost + completion_cost

    def _calculate_embedding_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost for embedding models."""
        model_key = self._normalize_model_name(model, "embeddings")
        if not model_key:
            logger.debug("Pricing not found for embedding model: %s", model)
            return 0.0

        price_per_1k_tokens = self.pricing_data["embeddings"][model_key]
        total_tokens = usage.get("prompt_tokens", 0) or usage.get("total_tokens", 0)
        return (total_tokens / 1000) * price_per_1k_tokens

    def _calculate_image_cost(self, model: str, usage: Dict[str, Any]) -> float:
        """Calculate cost for image generation models."""
        model_key = self._normalize_model_name(model, "images")
        if not model_key:
            logger.debug("Pricing not found for image model: %s", model)
            return 0.0

        pricing_info = self.pricing_data["images"][model_key]
        quality = usage.get("quality", "standard")
        size = usage.get("size")
        n = usage.get("n", 1)

        if quality not in pricing_info:
            logger.warning("Quality '%s' not found for image model %s", quality, model_key)
            return 0.0

        # Handle pricing per million pixels
        if "1000000" in pricing_info[quality]:
            price_per_million_pixels = pricing_info[quality]["1000000"]
            height = usage.get("height", 0)
            width = usage.get("width", 0)
            return (height * width / 1_000_000) * price_per_million_pixels * n

        if not size:
            logger.warning("Image size not provided for model %s", model_key)
            return 0.0

        if size not in pricing_info[quality]:
            logger.warning(
                "Size '%s' not found for image model %s with quality '%s'", size, model_key, quality
            )
            return 0.0

        price_per_image = pricing_info[quality][size]
        return price_per_image * n

    def _calculate_audio_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost for audio models."""
        model_key = self._normalize_model_name(model, "audio")
        if not model_key:
            logger.debug("Pricing not found for audio model: %s", model)
            return 0.0

        pricing = self.pricing_data["audio"][model_key]

        if "characters" in usage:
            # Price is per 1000 characters
            return (usage["characters"] / 1000) * pricing
        if "seconds" in usage:
            # Price is per second
            return usage["seconds"] * pricing

        logger.warning(
            "Could not determine usage unit for audio model %s. Expected 'characters' or 'seconds'.",
            model_key,
        )
        return 0.0

    def _normalize_model_name(self, model: str, category: str) -> Optional[str]:
        """Normalize model name to match pricing keys for a specific category."""
        if category not in self.pricing_data:
            return None

        normalized_model = model.lower()

        # Exact match (case-insensitive)
        for key in self.pricing_data[category]:
            if normalized_model == key.lower():
                return key

        # Substring match (case-insensitive)
        sorted_keys = sorted(self.pricing_data[category].keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key.lower() in normalized_model:
                return key
        return None
