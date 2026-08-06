import json
import unittest
from unittest.mock import MagicMock, patch

from genai_otel.cost_calculator import CostCalculator


class TestCostCalculator(unittest.TestCase):

    def setUp(self):
        self.pricing_data = {
            "embeddings": {"text-embedding-ada-002": 0.0001, "mistral-embed": 0.0001},
            "images": {
                "dall-e-3": {"standard": {"1024x1024": 0.040}, "hd": {"1024x1024": 0.080}},
                "black-forest-labs/FLUX.1-schnell": {"standard": {"1000000": 0.0027}},
            },
            "audio": {"tts-1": 0.015, "best": 0.00010277777},
            "chat": {
                "gpt-4o": {"promptPrice": 0.0005, "completionPrice": 0.0015},
                "gpt-3.5-turbo": {"promptPrice": 0.0005, "completionPrice": 0.0015},
            },
        }

        # To mock the file loading, we'll patch the json.loads and the file reading part.
        # A simple way is to patch the CostCalculator's _load_pricing method directly.
        patcher = patch.object(CostCalculator, "_load_pricing", MagicMock())
        self.addCleanup(patcher.stop)
        self.mock_load_pricing = patcher.start()

        self.calculator = CostCalculator()
        self.calculator.pricing_data = self.pricing_data

    def test_calculate_chat_cost(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
        cost = self.calculator.calculate_cost("gpt-4o", usage, "chat")
        expected_cost = (1000 / 1000 * 0.0005) + (2000 / 1000 * 0.0015)
        self.assertAlmostEqual(cost, expected_cost)

    def test_calculate_embedding_cost(self):
        usage = {"prompt_tokens": 5000}
        cost = self.calculator.calculate_cost("text-embedding-ada-002", usage, "embedding")
        expected_cost = (5000 / 1000) * 0.0001
        self.assertAlmostEqual(cost, expected_cost)

    def test_calculate_image_cost_per_image(self):
        usage = {"size": "1024x1024", "quality": "standard", "n": 2}
        cost = self.calculator.calculate_cost("dall-e-3", usage, "image")
        expected_cost = 0.040 * 2
        self.assertAlmostEqual(cost, expected_cost)

    def test_calculate_image_cost_per_pixel(self):
        usage = {"height": 1024, "width": 1024, "n": 1}
        cost = self.calculator.calculate_cost("black-forest-labs/FLUX.1-schnell", usage, "image")
        expected_cost = (1024 * 1024 / 1000000) * 0.0027
        self.assertAlmostEqual(cost, expected_cost)

    def test_calculate_audio_cost_tts(self):
        usage = {"characters": 2000}
        cost = self.calculator.calculate_cost("tts-1", usage, "audio")
        expected_cost = (2000 / 1000) * 0.015
        self.assertAlmostEqual(cost, expected_cost)

    def test_calculate_audio_cost_stt(self):
        usage = {"seconds": 60}
        cost = self.calculator.calculate_cost("best", usage, "audio")
        expected_cost = 60 * 0.00010277777
        self.assertAlmostEqual(cost, expected_cost)

    def test_unknown_model(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
        cost = self.calculator.calculate_cost("unknown-model", usage, "chat")
        self.assertEqual(cost, 0.0)

    def test_unknown_call_type(self):
        usage = {"prompt_tokens": 1000}
        cost = self.calculator.calculate_cost("gpt-4o", usage, "unknown")
        self.assertEqual(cost, 0.0)

    def test_normalize_model_name(self):
        # Test exact match
        self.assertEqual(self.calculator._normalize_model_name("gpt-4o", "chat"), "gpt-4o")
        # Test substring match
        self.assertEqual(
            self.calculator._normalize_model_name("some-prefix-gpt-3.5-turbo-some-suffix", "chat"),
            "gpt-3.5-turbo",
        )
        # Test no match
        self.assertIsNone(self.calculator._normalize_model_name("no-match", "chat"))

    def test_normalize_model_name_category_not_found(self):
        # Test when category doesn't exist
        self.assertIsNone(self.calculator._normalize_model_name("gpt-4o", "nonexistent"))

    def test_calculate_cost_with_empty_pricing_data(self):
        # Test when pricing data is empty
        self.calculator.pricing_data = {}
        usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
        cost = self.calculator.calculate_cost("gpt-4o", usage, "chat")
        self.assertEqual(cost, 0.0)

    def test_calculate_embedding_cost_with_total_tokens(self):
        """Test embedding cost calculation using total_tokens instead of prompt_tokens"""
        usage = {"total_tokens": 5000}
        cost = self.calculator.calculate_cost("text-embedding-ada-002", usage, "embedding")
        expected_cost = (5000 / 1000) * 0.0001
        self.assertAlmostEqual(cost, expected_cost)

    def test_calculate_embedding_cost_unknown_model(self):
        """Test embedding cost with unknown model"""
        usage = {"prompt_tokens": 5000}
        cost = self.calculator._calculate_embedding_cost("unknown-embedding-model", usage)
        self.assertEqual(cost, 0.0)

    def test_calculate_image_cost_quality_not_found(self):
        """Test image cost when quality is not in pricing"""
        usage = {"size": "1024x1024", "quality": "ultra_hd", "n": 1}
        cost = self.calculator._calculate_image_cost("dall-e-3", usage)
        self.assertEqual(cost, 0.0)

    def test_calculate_image_cost_no_size_provided(self):
        """Test image cost when size is not provided and no per-pixel pricing"""
        usage = {"quality": "standard", "n": 1}
        cost = self.calculator._calculate_image_cost("dall-e-3", usage)
        self.assertEqual(cost, 0.0)

    def test_calculate_image_cost_size_not_found(self):
        """Test image cost when size is not in pricing"""
        usage = {"size": "2048x2048", "quality": "standard", "n": 1}
        cost = self.calculator._calculate_image_cost("dall-e-3", usage)
        self.assertEqual(cost, 0.0)

    def test_calculate_image_cost_unknown_model(self):
        """Test image cost with unknown model"""
        usage = {"size": "1024x1024", "quality": "standard", "n": 1}
        cost = self.calculator._calculate_image_cost("unknown-image-model", usage)
        self.assertEqual(cost, 0.0)

    def test_calculate_audio_cost_unknown_model(self):
        """Test audio cost with unknown model"""
        usage = {"characters": 2000}
        cost = self.calculator._calculate_audio_cost("unknown-audio-model", usage)
        self.assertEqual(cost, 0.0)

    def test_calculate_audio_cost_no_usage_unit(self):
        """Test audio cost when usage doesn't have characters or seconds"""
        usage = {"tokens": 1000}
        cost = self.calculator._calculate_audio_cost("tts-1", usage)
        self.assertEqual(cost, 0.0)

    def test_calculate_chat_cost_unknown_model(self):
        """Test chat cost with unknown model"""
        usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
        cost = self.calculator._calculate_chat_cost("unknown-chat-model", usage)
        self.assertEqual(cost, 0.0)

    def test_calculate_granular_cost_basic(self):
        """Test granular cost calculation for basic chat request"""
        usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
        costs = self.calculator.calculate_granular_cost("gpt-4o", usage, "chat")

        expected_prompt = (1000 / 1000) * 0.0005
        expected_completion = (2000 / 1000) * 0.0015
        expected_total = expected_prompt + expected_completion

        self.assertAlmostEqual(costs["prompt"], expected_prompt)
        self.assertAlmostEqual(costs["completion"], expected_completion)
        self.assertAlmostEqual(costs["total"], expected_total)
        self.assertEqual(costs["reasoning"], 0.0)
        self.assertEqual(costs["cache_read"], 0.0)
        self.assertEqual(costs["cache_write"], 0.0)

    def test_calculate_granular_cost_with_reasoning(self):
        """Test granular cost calculation with reasoning tokens (o1 models)"""
        # Add o1 model pricing with reasoning
        self.calculator.pricing_data["chat"]["o1-preview"] = {
            "promptPrice": 0.015,
            "completionPrice": 0.060,
            "reasoningPrice": 0.030,
        }

        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 2000,
            "completion_tokens_details": {"reasoning_tokens": 500},
        }

        costs = self.calculator.calculate_granular_cost("o1-preview", usage, "chat")

        expected_prompt = (1000 / 1000) * 0.015
        expected_completion = (2000 / 1000) * 0.060
        expected_reasoning = (500 / 1000) * 0.030
        expected_total = expected_prompt + expected_completion + expected_reasoning

        self.assertAlmostEqual(costs["prompt"], expected_prompt)
        self.assertAlmostEqual(costs["completion"], expected_completion)
        self.assertAlmostEqual(costs["reasoning"], expected_reasoning)
        self.assertAlmostEqual(costs["total"], expected_total)
        self.assertEqual(costs["cache_read"], 0.0)
        self.assertEqual(costs["cache_write"], 0.0)

    def test_calculate_granular_cost_with_cache(self):
        """Test granular cost calculation with cache tokens (Anthropic models)"""
        # Add Anthropic pricing with cache costs
        self.calculator.pricing_data["chat"]["claude-3-5-sonnet-20241022"] = {
            "promptPrice": 0.003,
            "completionPrice": 0.015,
            "cacheReadPrice": 0.0003,
            "cacheWritePrice": 0.00375,
        }

        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 2000,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 200,
        }

        costs = self.calculator.calculate_granular_cost("claude-3-5-sonnet-20241022", usage, "chat")

        expected_prompt = (1000 / 1000) * 0.003
        expected_completion = (2000 / 1000) * 0.015
        expected_cache_read = (500 / 1000) * 0.0003
        expected_cache_write = (200 / 1000) * 0.00375
        expected_total = (
            expected_prompt + expected_completion + expected_cache_read + expected_cache_write
        )

        self.assertAlmostEqual(costs["prompt"], expected_prompt)
        self.assertAlmostEqual(costs["completion"], expected_completion)
        self.assertAlmostEqual(costs["cache_read"], expected_cache_read)
        self.assertAlmostEqual(costs["cache_write"], expected_cache_write)
        self.assertAlmostEqual(costs["total"], expected_total)
        self.assertEqual(costs["reasoning"], 0.0)

    def test_calculate_granular_cost_non_chat(self):
        """Test granular cost calculation for non-chat requests returns zeros for granular costs"""
        usage = {"prompt_tokens": 5000}
        costs = self.calculator.calculate_granular_cost(
            "text-embedding-ada-002", usage, "embedding"
        )

        # For non-chat, only total should be set
        self.assertGreater(costs["total"], 0)
        self.assertEqual(costs["prompt"], 0.0)
        self.assertEqual(costs["completion"], 0.0)
        self.assertEqual(costs["reasoning"], 0.0)
        self.assertEqual(costs["cache_read"], 0.0)
        self.assertEqual(costs["cache_write"], 0.0)


class TestCustomPricing(unittest.TestCase):
    """Test custom pricing functionality via JSON string."""

    def test_custom_pricing_chat_model(self):
        """Test adding custom pricing for a chat model."""
        custom_pricing_json = json.dumps(
            {"chat": {"my-custom-model": {"promptPrice": 0.001, "completionPrice": 0.002}}}
        )

        with patch.object(CostCalculator, "_load_pricing", MagicMock()):
            calculator = CostCalculator(custom_pricing_json=custom_pricing_json)
            calculator.pricing_data["chat"] = {}  # Start with empty pricing

            # Re-merge to test
            calculator._merge_custom_pricing(custom_pricing_json)

            # Verify custom pricing was added
            self.assertIn("my-custom-model", calculator.pricing_data["chat"])
            self.assertEqual(
                calculator.pricing_data["chat"]["my-custom-model"]["promptPrice"], 0.001
            )
            self.assertEqual(
                calculator.pricing_data["chat"]["my-custom-model"]["completionPrice"], 0.002
            )

            # Calculate cost with custom model
            usage = {"prompt_tokens": 1000, "completion_tokens": 500}
            cost = calculator.calculate_cost("my-custom-model", usage, "chat")
            expected_cost = (1000 / 1000 * 0.001) + (500 / 1000 * 0.002)
            self.assertAlmostEqual(cost, expected_cost)

    def test_custom_pricing_override_existing(self):
        """Test that custom pricing overrides default pricing."""
        custom_pricing_json = json.dumps(
            {"chat": {"gpt-4o": {"promptPrice": 0.999, "completionPrice": 0.999}}}
        )

        with patch.object(CostCalculator, "_load_pricing", MagicMock()):
            calculator = CostCalculator(custom_pricing_json=custom_pricing_json)
            calculator.pricing_data = {
                "chat": {"gpt-4o": {"promptPrice": 0.0005, "completionPrice": 0.0015}}
            }

            # Merge custom pricing (should override)
            calculator._merge_custom_pricing(custom_pricing_json)

            # Verify override
            self.assertEqual(calculator.pricing_data["chat"]["gpt-4o"]["promptPrice"], 0.999)
            self.assertEqual(calculator.pricing_data["chat"]["gpt-4o"]["completionPrice"], 0.999)

    def test_custom_pricing_multiple_categories(self):
        """Test custom pricing across multiple categories."""
        custom_pricing_json = json.dumps(
            {
                "chat": {"custom-chat-model": {"promptPrice": 0.001, "completionPrice": 0.002}},
                "embeddings": {"custom-embedding-model": 0.0001},
                "audio": {"custom-tts-model": 0.02},
            }
        )

        with patch.object(CostCalculator, "_load_pricing", MagicMock()):
            calculator = CostCalculator(custom_pricing_json=custom_pricing_json)
            calculator.pricing_data = {"chat": {}, "embeddings": {}, "audio": {}}
            calculator._merge_custom_pricing(custom_pricing_json)

            # Verify all categories were added
            self.assertIn("custom-chat-model", calculator.pricing_data["chat"])
            self.assertIn("custom-embedding-model", calculator.pricing_data["embeddings"])
            self.assertIn("custom-tts-model", calculator.pricing_data["audio"])

    def test_custom_pricing_invalid_json(self):
        """Test that invalid JSON is handled gracefully."""
        custom_pricing_json = "this is not valid json {"

        with patch.object(CostCalculator, "_load_pricing", MagicMock()):
            calculator = CostCalculator()
            calculator.pricing_data = {"chat": {}}

            # Should not raise exception, just log error
            calculator._merge_custom_pricing(custom_pricing_json)

            # Pricing data should remain unchanged
            self.assertEqual(calculator.pricing_data, {"chat": {}})

    def test_custom_pricing_invalid_format(self):
        """Test that non-dict custom pricing is handled gracefully."""
        custom_pricing_json = json.dumps(["not", "a", "dict"])

        with patch.object(CostCalculator, "_load_pricing", MagicMock()):
            calculator = CostCalculator()
            calculator.pricing_data = {"chat": {}}

            calculator._merge_custom_pricing(custom_pricing_json)

            # Pricing data should remain unchanged
            self.assertEqual(calculator.pricing_data, {"chat": {}})

    def test_custom_pricing_invalid_category(self):
        """Test that invalid categories are ignored with warning."""
        custom_pricing_json = json.dumps({"invalid_category": {"some-model": {"price": 0.001}}})

        with patch.object(CostCalculator, "_load_pricing", MagicMock()):
            calculator = CostCalculator()
            calculator.pricing_data = {}

            calculator._merge_custom_pricing(custom_pricing_json)

            # Invalid category should not be added
            self.assertNotIn("invalid_category", calculator.pricing_data)

    def test_custom_pricing_with_embedding_model(self):
        """Test custom pricing for embedding models."""
        custom_pricing_json = json.dumps({"embeddings": {"my-custom-embeddings": 0.00005}})

        with patch.object(CostCalculator, "_load_pricing", MagicMock()):
            calculator = CostCalculator(custom_pricing_json=custom_pricing_json)
            calculator.pricing_data = {"embeddings": {}}
            calculator._merge_custom_pricing(custom_pricing_json)

            # Calculate cost
            usage = {"total_tokens": 10000}
            cost = calculator.calculate_cost("my-custom-embeddings", usage, "embedding")
            expected_cost = (10000 / 1000) * 0.00005
            self.assertAlmostEqual(cost, expected_cost)


if __name__ == "__main__":
    unittest.main()


class TestPricingSource(unittest.TestCase):
    """A cost of 0.0 is ambiguous. pricing_source() says which kind of zero it is."""

    def setUp(self):
        patcher = patch.object(CostCalculator, "_load_pricing", MagicMock())
        self.addCleanup(patcher.stop)
        patcher.start()
        self.calc = CostCalculator()
        self.calc.pricing_data = {
            "chat": {"gpt-4o": {"promptPrice": 0.0005, "completionPrice": 0.0015}},
            "embeddings": {"text-embedding-ada-002": 0.0001},
        }

    def test_known_model_reports_table(self):
        self.assertEqual(self.calc.pricing_source("gpt-4o"), "table")

    def test_substring_match_still_reports_table(self):
        self.assertEqual(self.calc.pricing_source("gpt-4o-2024-08-06"), "table")

    def test_unknown_model_with_a_size_token_reports_estimated(self):
        # Priced off the parameter-count tier, so indicative rather than billable.
        self.assertEqual(self.calc.pricing_source("some-random-finetune-7b"), "estimated")

    def test_unknown_model_without_a_size_token_reports_unpriced(self):
        self.assertEqual(self.calc.pricing_source("Dhanishtha"), "unpriced")

    def test_unpriced_model_costs_zero_but_is_not_reported_as_free(self):
        costs = self.calc.calculate_granular_cost(
            "Dhanishtha", {"prompt_tokens": 1000, "completion_tokens": 1000}, "chat"
        )
        self.assertEqual(costs["total"], 0.0)
        self.assertEqual(self.calc.pricing_source("Dhanishtha"), "unpriced")


class TestParamCountDelimiters(unittest.TestCase):
    """Size tokens may be closed by '_' or '.', not only space/colon/dash.

    Fine-tune and checkpoint names use those separators routinely; treating them
    as non-delimiters dropped the model to an unpriced $0.00 instead of the
    local-size tier.
    """

    def setUp(self):
        patcher = patch.object(CostCalculator, "_load_pricing", MagicMock())
        self.addCleanup(patcher.stop)
        patcher.start()
        self.calc = CostCalculator()

    def test_underscore_delimited_size(self):
        self.assertEqual(
            self.calc._extract_param_count_from_model_name("MMPO_Gemma_7b_gamma1.1"), 7.0
        )

    def test_underscore_delimited_millions(self):
        self.assertEqual(
            self.calc._extract_param_count_from_model_name("smollm2-135M_pretrained_400k"), 0.135
        )

    def test_existing_delimiters_still_work(self):
        self.assertEqual(self.calc._extract_param_count_from_model_name("llama3:70b"), 70.0)
        self.assertEqual(self.calc._extract_param_count_from_model_name("llama-2-7b"), 7.0)
        self.assertEqual(self.calc._extract_param_count_from_model_name("smollm2:360m"), 0.36)


class TestIndexRebuildGuard(unittest.TestCase):
    """Keys differing only by case collapse in the exact index.

    The staleness guard therefore cannot compare the index size against the
    pricing dict size: for a table containing both 'MiniMax-M3' and
    'minimax-m3' those counts never match, so the guard fired on every lookup,
    rebuilt all indices and cleared the memo cache each time - defeating
    memoisation entirely.
    """

    def setUp(self):
        patcher = patch.object(CostCalculator, "_load_pricing", MagicMock())
        self.addCleanup(patcher.stop)
        patcher.start()
        self.calc = CostCalculator()
        self.calc.pricing_data = {
            "chat": {
                "MiniMax-M3": {"promptPrice": 0.0003, "completionPrice": 0.0012},
                "minimax-m3": {"promptPrice": 0.0003, "completionPrice": 0.0012},
                "gpt-4o": {"promptPrice": 0.0005, "completionPrice": 0.0015},
            }
        }

    def test_case_colliding_keys_do_not_defeat_memoisation(self):
        for name in ("gpt-4o", "unknown-a", "unknown-b"):
            self.calc._normalize_model_name(name, "chat")
        self.assertEqual(len(self.calc._norm_cache), 3)

    def test_in_place_addition_is_still_detected(self):
        self.assertIsNone(self.calc._normalize_model_name("brand-new-model", "chat"))
        self.calc.pricing_data["chat"]["brand-new-model"] = {
            "promptPrice": 1.0,
            "completionPrice": 2.0,
        }
        self.assertEqual(
            self.calc._normalize_model_name("brand-new-model", "chat"), "brand-new-model"
        )
