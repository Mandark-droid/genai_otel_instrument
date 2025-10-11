import os
import json
import logging
from typing import Dict, Optional

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

        The pricing file is expected to be in the project root directory.
        It should contain a 'models' key with a dictionary of model names to their pricing.
        """
        # Construct path relative to the project root
        # Assumes this file is in genai_otel/cost_calculator.py, so go up two levels to project root.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        pricing_file_path = os.path.join(project_root, self.DEFAULT_PRICING_FILE)
        
        try:
            with open(pricing_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "models" in data and isinstance(data["models"], dict):
                    self.pricing_data = data["models"]
                    logger.info(f"Successfully loaded pricing from {pricing_file_path}")
                else:
                    logger.error(f"Invalid format in pricing file {pricing_file_path}. 'models' key not found or not a dictionary.")
        except FileNotFoundError:
            logger.warning(f"Pricing file not found at {pricing_file_path}. Cost tracking will be disabled.")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from pricing file {pricing_file_path}. Cost tracking will be disabled.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading pricing: {e}", exc_info=True)

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
            # Changed from DEBUG to INFO for better visibility in production if a model is not found
            logger.info(f"Pricing not found for model: {model} (normalized: {model_key})")
            return 0.0

        pricing = self.pricing_data[model_key]
        prompt_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * pricing.get("prompt", 0.0)
        completion_cost = (usage.get("completion_tokens", 0) / 1_000_000) * pricing.get("completion", 0.0)

        return prompt_cost + completion_cost

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name to match pricing keys.

        Attempts to find a key in the pricing data that is a substring of the model name.
        Prioritizes longer matches to avoid incorrect partial matches.

        Args:
            model (str): The input model name.

        Returns:
            str: The normalized model name that matches a key in the pricing data, or the original model name if no match is found.
        """
        model = model.lower()
        # Prioritize longer matches first to avoid partial matches like 'gpt-4' matching 'gpt-4-turbo'
        sorted_keys = sorted(self.pricing_data.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in model:
                return key
        return model
