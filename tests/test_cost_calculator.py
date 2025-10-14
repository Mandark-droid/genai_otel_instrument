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


if __name__ == "__main__":
    unittest.main()
