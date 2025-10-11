"""Tests for cost calculator module."""

import pytest
from unittest.mock import patch, mock_open
import json
from genai_otel.cost_calculator import CostCalculator


class TestCostCalculator:
    """Test cases for CostCalculator class."""

    @pytest.fixture
    def mock_pricing_data(self):
        """Fixture providing mock pricing data."""
        return {
            "models": {
                "gpt-4": {"prompt": 30.0, "completion": 60.0},
                "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
                "claude-3-opus": {"prompt": 15.0, "completion": 75.0},
            }
        }

    @patch("genai_otel.cost_calculator.logger")
    def test_init_with_missing_file(self, mock_logger):
        """Test initialization when pricing file is missing."""
        with patch("genai_otel.cost_calculator.json.loads", side_effect=FileNotFoundError):
            calculator = CostCalculator()
            assert calculator.pricing_data == {}

    def test_calculate_cost_gpt4(self):
        """Test cost calculation for GPT-4."""
        calculator = CostCalculator()
        calculator.pricing_data = {"gpt-4": {"prompt": 30.0, "completion": 60.0}}

        usage = {"prompt_tokens": 1000, "completion_tokens": 500}

        cost = calculator.calculate_cost("gpt-4", usage)

        # (1000 / 1M * 30) + (500 / 1M * 60) = 0.03 + 0.03 = 0.06
        assert cost == pytest.approx(0.06, rel=1e-6)

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model."""
        calculator = CostCalculator()
        calculator.pricing_data = {"gpt-4": {"prompt": 30.0, "completion": 60.0}}

        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = calculator.calculate_cost("unknown-model", usage)

        assert cost == 0.0

    def test_calculate_cost_no_pricing_data(self):
        """Test cost calculation when no pricing data is loaded."""
        calculator = CostCalculator()
        calculator.pricing_data = {}

        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = calculator.calculate_cost("gpt-4", usage)

        assert cost == 0.0

    def test_normalize_model_name(self):
        """Test model name normalization."""
        calculator = CostCalculator()
        calculator.pricing_data = {
            "gpt-4": {"prompt": 30.0, "completion": 60.0},
            "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        }

        # Should match the longer key first
        assert calculator._normalize_model_name("gpt-3.5-turbo-16k") == "gpt-3.5-turbo"
        assert calculator._normalize_model_name("GPT-4-0613") == "gpt-4"
        assert calculator._normalize_model_name("unknown") == "unknown"

    def test_normalize_model_name_prioritizes_longer_match(self):
        """Test that normalization prioritizes longer matches."""
        calculator = CostCalculator()
        calculator.pricing_data = {
            "gpt": {"prompt": 1.0, "completion": 2.0},
            "gpt-4": {"prompt": 30.0, "completion": 60.0},
        }

        # Should match "gpt-4" not "gpt"
        assert calculator._normalize_model_name("gpt-4-turbo") == "gpt-4"

    def test_calculate_cost_missing_tokens(self):
        """Test cost calculation with missing token counts."""
        calculator = CostCalculator()
        calculator.pricing_data = {"gpt-4": {"prompt": 30.0, "completion": 60.0}}

        # Missing completion_tokens
        usage = {"prompt_tokens": 1000}
        cost = calculator.calculate_cost("gpt-4", usage)
        assert cost == pytest.approx(0.03, rel=1e-6)

        # Missing prompt_tokens
        usage = {"completion_tokens": 500}
        cost = calculator.calculate_cost("gpt-4", usage)
        assert cost == pytest.approx(0.03, rel=1e-6)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        calculator = CostCalculator()
        calculator.pricing_data = {"gpt-4": {"prompt": 30.0, "completion": 60.0}}

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        cost = calculator.calculate_cost("gpt-4", usage)

        assert cost == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
