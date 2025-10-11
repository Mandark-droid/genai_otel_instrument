from typing import Dict


class CostCalculator:
    """Calculate estimated costs for LLM API calls"""

    PRICING = {
        # OpenAI
        "gpt-4": {"prompt": 30.0, "completion": 60.0},
        "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
        "gpt-4o": {"prompt": 5.0, "completion": 15.0},
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},

        # Anthropic Claude
        "claude-3-opus": {"prompt": 15.0, "completion": 75.0},
        "claude-3-sonnet": {"prompt": 3.0, "completion": 15.0},
        "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
        "claude-3.5-sonnet": {"prompt": 3.0, "completion": 15.0},

        # Google AI
        "gemini-pro": {"prompt": 0.5, "completion": 1.5},
        "gemini-ultra": {"prompt": 10.0, "completion": 30.0},
        "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.0},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.3},

        # Cohere
        "command": {"prompt": 1.0, "completion": 2.0},
        "command-light": {"prompt": 0.3, "completion": 0.6},
        "command-r": {"prompt": 0.5, "completion": 1.5},
        "command-r-plus": {"prompt": 3.0, "completion": 15.0},

        # Mistral AI
        "mistral-tiny": {"prompt": 0.14, "completion": 0.42},
        "mistral-small": {"prompt": 0.6, "completion": 1.8},
        "mistral-medium": {"prompt": 2.5, "completion": 7.5},
        "mistral-large": {"prompt": 4.0, "completion": 12.0},

        # AWS Bedrock (approximations)
        "claude-v2": {"prompt": 8.0, "completion": 24.0},
        "titan-text": {"prompt": 0.8, "completion": 2.4},

        # Azure OpenAI (same as OpenAI generally)
    }

    def calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost in USD for a request"""
        model_key = self._normalize_model_name(model)

        if model_key not in self.PRICING:
            return 0.0

        pricing = self.PRICING[model_key]
        prompt_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * pricing["prompt"]
        completion_cost = (usage.get("completion_tokens", 0) / 1_000_000) * pricing["completion"]

        return prompt_cost + completion_cost

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name to match pricing keys"""
        model = model.lower()
        for key in self.PRICING.keys():
            if key in model:
                return key
        return model
