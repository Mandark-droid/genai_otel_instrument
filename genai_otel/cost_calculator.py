"""Module for calculating estimated costs of LLM API calls."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sentinel distinguishing "not cached yet" from a cached None result, so that
# negative lookups (unknown/internal models) are memoized too.
_MISS = object()


class CostCalculator:
    """Calculate estimated costs for LLM API calls based on loaded pricing data."""

    DEFAULT_PRICING_FILE = "llm_pricing.json"

    def __init__(self, custom_pricing_json: Optional[str] = None):
        """Initializes the CostCalculator by loading pricing data from a JSON file.

        Args:
            custom_pricing_json: Optional JSON string with custom model pricing.
                Format: {"chat": {"model-name": {"promptPrice": 0.001, "completionPrice": 0.002}}}
                Custom prices will be merged with default pricing, with custom taking precedence.
        """
        # Precomputed lookup structures (rebuilt whenever pricing_data is set):
        #   _exact_index[category]  = {lowercased_key: original_key}
        #   _substr_index[category] = [(lowercased_key, original_key), ...] longest-first
        # plus per-model memoization caches for name resolution and chat pricing.
        self._exact_index: Dict[str, Dict[str, str]] = {}
        self._substr_index: Dict[str, List[Tuple[str, str]]] = {}
        self._norm_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self._chat_pricing_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._pricing_data: Dict[str, Any] = {}
        self._load_pricing()
        self._build_indices()
        if custom_pricing_json:
            self._merge_custom_pricing(custom_pricing_json)

    @property
    def pricing_data(self) -> Dict[str, Any]:
        """The pricing table.

        Reassigning it (``calculator.pricing_data = {...}``) rebuilds the lookup
        indices and clears the memo caches, so code and tests that swap in a
        different pricing dict stay consistent. In-place mutation of the returned
        dict does NOT rebuild indices; callers doing that should call
        :meth:`_build_indices` themselves (as :meth:`_merge_custom_pricing` does).
        """
        return self._pricing_data

    @pricing_data.setter
    def pricing_data(self, value: Dict[str, Any]) -> None:
        self._pricing_data = value if isinstance(value, dict) else {}
        # During __init__ the index attributes are created before this setter is
        # first driven by _load_pricing, so rebuild eagerly on every reassignment.
        if hasattr(self, "_exact_index"):
            self._build_indices()

    def _build_indices(self) -> None:
        """Precompute lowercase exact-match and length-sorted substring indices
        for each pricing category and clear the per-model memo caches.

        Called once after loading and again whenever custom pricing is merged.
        This turns per-call model resolution from an O(n) scan plus a per-call
        ``sorted()`` over all keys into an O(1) dict hit (exact) or a single
        precomputed longest-first pass (substring), memoized per model name.
        """
        self._exact_index = {}
        self._substr_index = {}
        self._norm_cache = {}
        self._chat_pricing_cache = {}
        for category, models in self.pricing_data.items():
            if not isinstance(models, dict):
                continue
            lowered = [(str(k).lower(), k) for k in models]
            self._exact_index[category] = {lk: k for lk, k in lowered}
            # Longest keys first so a substring match prefers the most specific model.
            self._substr_index[category] = sorted(lowered, key=lambda t: len(t[0]), reverse=True)

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

    def _merge_custom_pricing(self, custom_pricing_json: str):
        """Merge custom pricing from JSON string into existing pricing data.

        Args:
            custom_pricing_json: JSON string with custom model pricing.
                Format: {"chat": {"model-name": {"promptPrice": 0.001, "completionPrice": 0.002}}}
        """
        try:
            custom_pricing = json.loads(custom_pricing_json)

            if not isinstance(custom_pricing, dict):
                logger.error(
                    "Custom pricing must be a JSON object/dict. Got: %s",
                    type(custom_pricing).__name__,
                )
                return

            # Merge custom pricing into each category (chat, embeddings, images, audio)
            for category, models in custom_pricing.items():
                if category not in ["chat", "embeddings", "images", "audio"]:
                    logger.warning(
                        "Unknown pricing category '%s' in custom pricing. Valid categories: "
                        "chat, embeddings, images, audio",
                        category,
                    )
                    continue

                if not isinstance(models, dict):
                    logger.error(
                        "Custom pricing for category '%s' must be a dict. Got: %s",
                        category,
                        type(models).__name__,
                    )
                    continue

                # Initialize category if it doesn't exist
                if category not in self.pricing_data:
                    self.pricing_data[category] = {}

                # Merge models into the category
                for model_name, pricing in models.items():
                    self.pricing_data[category][model_name] = pricing
                    logger.info(
                        "Added custom pricing for %s model '%s': %s",
                        category,
                        model_name,
                        pricing,
                    )

            # Rebuild lookup indices and drop memo caches so merged pricing is visible.
            self._build_indices()

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode custom pricing JSON: %s. Custom pricing will be ignored.", e
            )
        except Exception as e:
            logger.error(
                "An unexpected error occurred while merging custom pricing: %s", e, exc_info=True
            )

    def calculate_cost(
        self,
        model: str,
        usage: Dict[str, Any],
        call_type: str,
    ) -> float:
        """Calculate cost in USD for a request based on model, usage, and call type.

        Note: For chat requests, use calculate_granular_cost() to get prompt/completion/reasoning/cache breakdown.
        This method returns total cost for backwards compatibility.
        """
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

    def calculate_granular_cost(
        self,
        model: str,
        usage: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, float]:
        """Calculate granular cost breakdown for a request.

        Returns a dictionary with:
        - total: Total cost
        - prompt: Prompt tokens cost
        - completion: Completion tokens cost
        - reasoning: Reasoning tokens cost (OpenAI o1 models)
        - cache_read: Cache read cost (Anthropic)
        - cache_write: Cache write cost (Anthropic)
        """
        if not self.pricing_data:
            return {
                "total": 0.0,
                "prompt": 0.0,
                "completion": 0.0,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            }

        if call_type == "chat":
            return self._calculate_chat_cost_granular(model, usage)

        # For non-chat requests, only return total cost
        total_cost = self.calculate_cost(model, usage, call_type)
        return {
            "total": total_cost,
            "prompt": 0.0,
            "completion": 0.0,
            "reasoning": 0.0,
            "cache_read": 0.0,
            "cache_write": 0.0,
        }

    def _calculate_chat_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost for chat models."""
        granular = self._calculate_chat_cost_granular(model, usage)
        return granular["total"]

    def _resolve_chat_pricing(self, model: str) -> Optional[Dict[str, Any]]:
        """Resolve the pricing dict for a chat model, memoized per model name.

        Returns the model's pricing entry, a parameter-count-based local-model
        price tier for unknown Ollama/HuggingFace models, or None when no pricing
        can be determined. Negative results are cached too, so unknown/internal
        model names do not repeatedly scan the pricing table on the hot path.
        """
        cached = self._chat_pricing_cache.get(model, _MISS)
        if cached is not _MISS:
            return cached

        model_key = self._normalize_model_name(model, "chat")
        if model_key:
            pricing: Optional[Dict[str, Any]] = self.pricing_data["chat"][model_key]
        else:
            # Fallback for unknown local models (Ollama, HuggingFace): estimate
            # pricing based on parameter count parsed from the model name.
            param_count = self._extract_param_count_from_model_name(model)
            if param_count is not None:
                pricing = self._get_local_model_price_tier(param_count)
                logger.info(
                    "Using fallback pricing for unknown local model '%s' with %.2fB parameters: "
                    "$%.4f prompt / $%.4f completion per 1k tokens",
                    model,
                    param_count,
                    pricing["promptPrice"],
                    pricing["completionPrice"],
                )
            else:
                logger.debug("Pricing not found for chat model: %s", model)
                pricing = None

        self._chat_pricing_cache[model] = pricing
        return pricing

    def _calculate_chat_cost_granular(self, model: str, usage: Dict[str, int]) -> Dict[str, float]:
        """Calculate granular cost breakdown for chat models.

        Returns:
            Dict with keys: total, prompt, completion, reasoning, cache_read, cache_write
        """
        pricing = self._resolve_chat_pricing(model)
        if pricing is None:
            return {
                "total": 0.0,
                "prompt": 0.0,
                "completion": 0.0,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            }

        # Standard prompt and completion tokens
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        prompt_cost = (prompt_tokens / 1000) * pricing.get("promptPrice", 0.0)
        completion_cost = (completion_tokens / 1000) * pricing.get("completionPrice", 0.0)

        # Reasoning tokens (OpenAI o1 models)
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        reasoning_cost = 0.0
        if reasoning_tokens > 0 and "reasoningPrice" in pricing:
            reasoning_cost = (reasoning_tokens / 1000) * pricing.get("reasoningPrice", 0.0)

        # Cache costs (Anthropic models)
        cache_read_tokens = usage.get("cache_read_input_tokens", 0)
        cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
        cache_read_cost = 0.0
        cache_write_cost = 0.0

        if cache_read_tokens > 0 and "cacheReadPrice" in pricing:
            cache_read_cost = (cache_read_tokens / 1000) * pricing.get("cacheReadPrice", 0.0)
        if cache_write_tokens > 0 and "cacheWritePrice" in pricing:
            cache_write_cost = (cache_write_tokens / 1000) * pricing.get("cacheWritePrice", 0.0)

        total_cost = (
            prompt_cost + completion_cost + reasoning_cost + cache_read_cost + cache_write_cost
        )

        return {
            "total": total_cost,
            "prompt": prompt_cost,
            "completion": completion_cost,
            "reasoning": reasoning_cost,
            "cache_read": cache_read_cost,
            "cache_write": cache_write_cost,
        }

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
        """Normalize model name to match pricing keys for a specific category.

        Memoized per ``(model, category)`` including negative results, backed by
        the precomputed exact/substring indices from :meth:`_build_indices`.
        """
        cache_key = (model, category)
        cached = self._norm_cache.get(cache_key, _MISS)
        if cached is not _MISS:
            return cached
        result = self._resolve_model_key(model, category)
        self._norm_cache[cache_key] = result
        return result

    def _resolve_model_key(self, model: str, category: str) -> Optional[str]:
        """Resolve a model name to its pricing key using precomputed indices.

        Preserves the original semantics: case-insensitive exact match first,
        then the longest matching substring key.
        """
        cat_data = self._pricing_data.get(category)
        if not isinstance(cat_data, dict) or not cat_data:
            return None
        # Detect out-of-band in-place additions/removals to pricing_data[category]
        # (callers that mutate the nested dict instead of reassigning pricing_data)
        # and rebuild the indices so lookups stay correct. This runs only on a memo
        # miss, so the warm hot path (all cache hits) never pays for it.
        if len(self._exact_index.get(category, ())) != len(cat_data):
            self._build_indices()
        exact = self._exact_index.get(category)
        if not exact:
            return None
        normalized_model = model.lower()
        hit = exact.get(normalized_model)
        if hit is not None:
            return hit
        for lower_key, original_key in self._substr_index.get(category, ()):  # longest-first
            if lower_key in normalized_model:
                return original_key
        return None

    def _extract_param_count_from_model_name(self, model: str) -> Optional[float]:
        """Extract parameter count from Ollama or HuggingFace model name.

        Supports both explicit size indicators and common model size names.

        Examples:
            Ollama models:
            "smollm2:360m" -> 0.36
            "llama3:7b" -> 7.0
            "llama3.1:70b" -> 70.0
            "deepseek-r1:32b" -> 32.0

            HuggingFace models:
            "gpt2" -> 0.124 (base)
            "gpt2-xl" -> 1.5
            "bert-base-uncased" -> 0.11
            "bert-large-uncased" -> 0.34
            "t5-small" -> 0.06
            "t5-xxl" -> 11.0
            "llama-2-7b" -> 7.0
            "mistral-7b-v0.1" -> 7.0

        Returns:
            Parameter count in billions, or None if not parseable.
        """
        model_lower = model.lower()

        # First try explicit parameter count patterns (e.g., 135m, 7b, 70b)
        # Matches: digits followed by optional decimal, then 'm' or 'b'
        pattern = r"(\d+(?:\.\d+)?)(m|b)(?:\s|:|$|-)"
        match = re.search(pattern, model_lower)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "m":
                return value / 1000  # Convert millions to billions
            elif unit == "b":
                return value

        # Fallback to common model size indicators for HuggingFace models
        # These are approximate values based on typical model sizes
        size_map = {
            # T5 family
            "t5-small": 0.06,
            "t5-base": 0.22,
            "t5-large": 0.77,
            "t5-xl": 3.0,
            "t5-xxl": 11.0,
            # GPT-2 family
            "gpt2-small": 0.124,
            "gpt2-medium": 0.355,
            "gpt2-large": 0.774,
            "gpt2-xl": 1.5,
            "gpt2": 0.124,  # default GPT-2 is small
            # BERT family
            "bert-tiny": 0.004,
            "bert-mini": 0.011,
            "bert-small": 0.029,
            "bert-medium": 0.041,
            "bert-base": 0.11,
            "bert-large": 0.34,
            # Generic size indicators (fallback)
            "tiny": 0.01,
            "mini": 0.02,
            "small": 0.06,
            "base": 0.11,
            "medium": 0.35,
            "large": 0.77,
            "xl": 1.5,
            "xxl": 11.0,
        }

        # Check for size indicators in the model name
        for size_key, param_count in size_map.items():
            if size_key in model_lower:
                return param_count

        return None

    def _get_local_model_price_tier(self, param_count_billions: float) -> Dict[str, float]:
        """Get pricing tier based on parameter count for local models (Ollama, HuggingFace).

        Local models (Ollama, HuggingFace Transformers) are free but consume GPU power
        and electricity. We estimate costs based on parameter count and comparable
        cloud API pricing.

        Price Tiers (based on parameter count):
        - Tiny (< 1B params): $0.0001 / $0.0002 (prompt/completion)
        - Small (1-10B): $0.0003 / $0.0006
        - Medium (10-20B): $0.0005 / $0.001
        - Large (20-80B): $0.0008 / $0.0008
        - XLarge (80B+): $0.0012 / $0.0012

        Args:
            param_count_billions: Model parameter count in billions

        Returns:
            Dict with promptPrice and completionPrice
        """
        if param_count_billions < 1.0:
            return {"promptPrice": 0.0001, "completionPrice": 0.0002}
        elif param_count_billions < 10.0:
            return {"promptPrice": 0.0003, "completionPrice": 0.0006}
        elif param_count_billions < 20.0:
            return {"promptPrice": 0.0005, "completionPrice": 0.001}
        elif param_count_billions < 80.0:
            return {"promptPrice": 0.0008, "completionPrice": 0.0008}
        else:
            return {"promptPrice": 0.0012, "completionPrice": 0.0012}
