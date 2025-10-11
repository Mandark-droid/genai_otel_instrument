"""Module for calculating estimated costs of LLM API calls.

This module provides the `CostCalculator` class, which loads pricing data
from a JSON file and uses it to estimate the cost of LLM requests based on
model name and token usage. It includes logic for normalizing model names
to match pricing keys.
"""

import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class CostCalculator:
    """Calculate estimated costs for LLM API calls based on loaded pricing data."""

    DEFAULT_PRICING_FILE = "llm_pricing.json"

    def __init__(self):
        """Initializes the CostCalculator by loading pricing data from a JSON file."""
        self.pricing_data: Dict[str, Dict[str, float]] = {}
        self._load_pricing()

    def _load_pricing(self):
        """Load pricing data from the JSON configuration file.

        Uses importlib.resources for Python 3.9+ or importlib_resources for older versions.
        Falls back to pkg_resources if neither is available.
        """
        try:
            # Try Python 3.9+ importlib.resources
            try:
                from importlib.resources import files

                pricing_file = files("genai_otel").joinpath(self.DEFAULT_PRICING_FILE)
                data = json.loads(pricing_file.read_text(encoding="utf-8"))
            except (ImportError, AttributeError):
                # Fallback for Python 3.8
                try:
                    import importlib_resources

                    pricing_file = importlib_resources.files("genai_otel").joinpath(
                        self.DEFAULT_PRICING_FILE
                    )
                    data = json.loads(pricing_file.read_text(encoding="utf-8"))
                except ImportError:
                    # Final fallback to pkg_resources
                    import pkg_resources

                    pricing_file_path = pkg_resources.resource_filename(
                        "genai_otel", self.DEFAULT_PRICING_FILE
                    )
                    with open(pricing_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

            if "models" in data and isinstance(data["models"], dict):
                self.pricing_data = data["models"]
                logger.info(
                    "Successfully loaded pricing data for %d models", len(self.pricing_data)
                )
            else:
                logger.error(
                    "Invalid format in pricing file. 'models' key not found or not a dictionary."
                )
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

    def calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost in USD for a request based on model and token usage.

        Args:
            model (str): The name of the LLM model used.
            usage (Dict[str, int]): A dictionary containing token usage, expected keys are
                                    'prompt_tokens' and 'completion_tokens'.

        Returns:
            float: The estimated cost in USD, or 0.0 if pricing is unavailable or calculation fails.
        """
        if not self.pricing_data:
            return 0.0

        model_key = self._normalize_model_name(model)

        if model_key not in self.pricing_data:
            logger.debug("Pricing not found for model: %s (normalized: %s)", model, model_key)
            return 0.0

        pricing = self.pricing_data[model_key]
        prompt_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * pricing.get("prompt", 0.0)
        completion_cost = (usage.get("completion_tokens", 0) / 1_000_000) * pricing.get(
            "completion", 0.0
        )

        return prompt_cost + completion_cost

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name to match pricing keys.

        Attempts to find a key in the pricing data that is a substring of the model name.
        Prioritizes longer matches to avoid incorrect partial matches.

        Args:
            model (str): The input model name.

        Returns:
            str: The normalized model name that matches a key in the pricing data,
                 or the original model name if no match is found.
        """
        model = model.lower()
        sorted_keys = sorted(self.pricing_data.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in model:
                return key
        return model
