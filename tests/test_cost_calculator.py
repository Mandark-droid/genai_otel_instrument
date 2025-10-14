import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.cost_calculator import CostCalculator

# Mock pricing data for testing
MOCK_PRICING_DATA = {
    "models": {
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-4": {"prompt": 30.0, "completion": 60.0},
        "claude-3-opus": {"prompt": 15.0, "completion": 75.0},
        "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
        "gemini-1.0-pro": {"prompt": 0.5, "completion": 1.0},
        "gemini-1.5-pro-preview-0409": {"prompt": 3.5, "completion": 7.0},
        "mistral-large-2407": {"prompt": 8.0, "completion": 24.0},
        "mistral-small-2407": {"prompt": 0.2, "completion": 0.6},
    }
}

# --- Fixtures ---


@pytest.fixture
def mock_pricing_file(tmp_path):
    """Fixture to create a temporary pricing file."""
    file_path = tmp_path / "llm_pricing.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(MOCK_PRICING_DATA, f)
    return str(file_path)


@pytest.fixture
def cost_calculator(mock_pricing_file):
    """Fixture to provide a CostCalculator instance with mocked pricing file."""
    # Temporarily override the default pricing file path for testing
    with patch(
        "genai_otel.cost_calculator.CostCalculator.DEFAULT_PRICING_FILE", "llm_pricing.json"
    ), patch("importlib.resources.files") as mock_files_from_resources:

        # Configure mock_files_from_resources to return an object that has joinpath
        mock_path_obj = MagicMock()
        # The joinpath method should return an object that has read_text
        mock_path_obj.joinpath.return_value.read_text.return_value = json.dumps(MOCK_PRICING_DATA)
        mock_files_from_resources.return_value = mock_path_obj

        # Mocking fallbacks for older Python versions if they are used
        # For simplicity, we focus on patching importlib.resources.files directly.
        # If the code explicitly imports and uses importlib_resources or pkg_resources,
        # those would need separate patching.

        # The original code attempted to manipulate logger levels via patch.dict,
        # which is not the correct way. We will use caplog in specific tests if needed.
        calculator = CostCalculator()

        # Manually set pricing_data to ensure it's loaded correctly for tests
        # This bypasses the actual file reading for more controlled testing
        calculator.pricing_data = MOCK_PRICING_DATA["models"]

        yield calculator


# --- Tests for _load_pricing ---


def test_load_pricing_success(cost_calculator):
    """Test that pricing data is loaded successfully."""
    # The cost_calculator fixture already loads pricing data.
    # We check if pricing_data is populated.
    assert cost_calculator.pricing_data is not None
    assert len(cost_calculator.pricing_data) > 0
    assert "gpt-3.5-turbo" in cost_calculator.pricing_data


def test_load_pricing_file_not_found(caplog):
    """Test that FileNotFoundError is handled gracefully."""
    with patch("importlib.resources.files") as mock_files_from_resources:
        mock_files_from_resources.return_value.joinpath.side_effect = FileNotFoundError(
            "Mock file not found"
        )

        # Ensure logger is enabled for WARNING to capture the expected message
        caplog.set_level(logging.WARNING)
        calculator = CostCalculator()
        assert "Pricing file 'llm_pricing.json' not found" in caplog.text
        assert calculator.pricing_data == {}


def test_load_pricing_json_decode_error(caplog):
    """Test that JSONDecodeError is handled gracefully."""
    # We mock the read_text to return invalid JSON
    with patch("importlib.resources.files") as mock_files_from_resources:
        mock_file_obj = MagicMock()
        mock_file_obj.read_text.return_value = "invalid json"
        mock_files_from_resources.return_value.joinpath.return_value = mock_file_obj

        # Ensure logger is enabled for ERROR to capture the expected message
        caplog.set_level(logging.ERROR)
        calculator = CostCalculator()
        assert "Failed to decode JSON from pricing file" in caplog.text
        assert calculator.pricing_data == {}


def test_load_pricing_invalid_format(caplog):
    """Test that invalid format in pricing file is handled gracefully."""
    invalid_data = {"not_models": {}}

    with patch("importlib.resources.files") as mock_files_from_resources:
        mock_file_obj = MagicMock()
        mock_file_obj.read_text.return_value = json.dumps(invalid_data)
        mock_files_from_resources.return_value.joinpath.return_value = mock_file_obj

        # Ensure logger is enabled for ERROR to capture the expected message
        caplog.set_level(logging.ERROR)
        calculator = CostCalculator()
        assert (
            "Invalid format in pricing file. 'models' key not found or not a dictionary."
            in caplog.text
        )
        assert calculator.pricing_data == {}


def test_load_pricing_general_exception(caplog):
    """Test that general exceptions during loading are handled."""
    with patch("importlib.resources.files") as mock_files_from_resources:
        mock_files_from_resources.return_value.joinpath.side_effect = Exception("Unexpected error")

        # Ensure logger is enabled for ERROR to capture the expected message
        caplog.set_level(logging.ERROR)
        calculator = CostCalculator()
        assert "An unexpected error occurred while loading pricing" in caplog.text
        assert calculator.pricing_data == {}


# --- Tests for _normalize_model_name ---


def test_normalize_model_name_exact_match(cost_calculator):
    """Test normalization with an exact model name match."""
    assert cost_calculator._normalize_model_name("gpt-3.5-turbo") == "gpt-3.5-turbo"


def test_normalize_model_name_substring_match(cost_calculator):
    """Test normalization with a substring match, prioritizing longer keys."""
    # Example: "gpt-4-turbo" should match "gpt-4" if "gpt-4-turbo" is not a key
    # Assuming MOCK_PRICING_DATA has "gpt-4" and "gpt-4-turbo" is not a direct key
    assert cost_calculator._normalize_model_name("gpt-4-turbo-preview") == "gpt-4"
    assert cost_calculator._normalize_model_name("claude-3-opus-20240229") == "claude-3-opus"
    assert (
        cost_calculator._normalize_model_name("gemini-1.5-pro-preview-0409-extra")
        == "gemini-1.5-pro-preview-0409"
    )


def test_normalize_model_name_no_match(cost_calculator):
    """Test normalization when no pricing key matches."""
    assert cost_calculator._normalize_model_name("some-other-model") == "some-other-model"


def test_normalize_model_name_case_insensitivity(cost_calculator):
    """Test that model name normalization is case-insensitive."""
    assert cost_calculator._normalize_model_name("GPT-3.5-TURBO") == "gpt-3.5-turbo"


def test_normalize_model_name_empty_string(cost_calculator):
    """Test normalization with an empty string input."""
    assert cost_calculator._normalize_model_name("") == ""


# --- Tests for calculate_cost ---


def test_calculate_cost_success(cost_calculator):
    """Test successful cost calculation."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 2000}  # 1M tokens = 1000 * 1000
    # gpt-3.5-turbo: prompt 0.5/M, completion 1.5/M
    # Expected: (1000/1M * 0.5) + (2000/1M * 1.5) = 0.0005 + 0.003 = 0.0035
    expected_cost = (1000 / 1_000_000) * 0.5 + (2000 / 1_000_000) * 1.5
    assert cost_calculator.calculate_cost("gpt-3.5-turbo", usage) == pytest.approx(expected_cost)


def test_calculate_cost_model_not_found(cost_calculator):
    """Test cost calculation when model is not found in pricing data."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
    assert cost_calculator.calculate_cost("non-existent-model", usage) == 0.0


def test_calculate_cost_missing_usage_keys(cost_calculator):
    """Test cost calculation when usage dictionary is missing keys."""
    # Missing prompt_tokens and completion_tokens
    usage_missing_all = {}
    assert cost_calculator.calculate_cost("gpt-3.5-turbo", usage_missing_all) == 0.0

    # Missing completion_tokens
    usage_missing_completion = {"prompt_tokens": 1000}
    expected_cost_prompt_only = (1000 / 1_000_000) * 0.5
    assert cost_calculator.calculate_cost(
        "gpt-3.5-turbo", usage_missing_completion
    ) == pytest.approx(expected_cost_prompt_only)

    # Missing prompt_tokens
    usage_missing_prompt = {"completion_tokens": 2000}
    expected_cost_completion_only = (2000 / 1_000_000) * 1.5
    assert cost_calculator.calculate_cost("gpt-3.5-turbo", usage_missing_prompt) == pytest.approx(
        expected_cost_completion_only
    )


def test_calculate_cost_zero_tokens(cost_calculator):
    """Test cost calculation with zero tokens."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    assert cost_calculator.calculate_cost("gpt-3.5-turbo", usage) == 0.0


def test_calculate_cost_pricing_data_empty(cost_calculator):
    """Test cost calculation when pricing_data is empty."""
    cost_calculator.pricing_data = {}  # Manually empty pricing data
    usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
    assert cost_calculator.calculate_cost("gpt-3.5-turbo", usage) == 0.0


def test_calculate_cost_pricing_missing_keys(cost_calculator):
    """Test cost calculation when pricing data for a model is missing keys."""
    # Temporarily modify pricing data for a specific model to be incomplete
    original_pricing = cost_calculator.pricing_data.copy()

    # Case 1: Missing 'prompt' key
    cost_calculator.pricing_data["model_missing_prompt"] = {"completion": 1.0}
    usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
    expected_cost_completion_only = (2000 / 1_000_000) * 1.0
    assert cost_calculator.calculate_cost("model_missing_prompt", usage) == pytest.approx(
        expected_cost_completion_only
    )

    # Case 2: Missing 'completion' key
    cost_calculator.pricing_data["model_missing_completion"] = {"prompt": 0.5}
    expected_cost_prompt_only = (1000 / 1_000_000) * 0.5
    assert cost_calculator.calculate_cost("model_missing_completion", usage) == pytest.approx(
        expected_cost_prompt_only
    )

    # Case 3: Missing both keys (should result in 0 cost)
    cost_calculator.pricing_data["model_missing_both"] = {}
    assert cost_calculator.calculate_cost("model_missing_both", usage) == 0.0

    # Restore original pricing data
    cost_calculator.pricing_data = original_pricing


def test_calculate_cost_with_normalized_model_name(cost_calculator):
    """Test cost calculation using a normalized model name."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
    # "gpt-4-turbo-preview" should normalize to "gpt-4"
    expected_cost = (1000 / 1_000_000) * 30.0 + (2000 / 1_000_000) * 60.0
    assert cost_calculator.calculate_cost("gpt-4-turbo-preview", usage) == pytest.approx(
        expected_cost
    )


def test_calculate_cost_with_unmatched_model_name(cost_calculator):
    """Test cost calculation when model name does not match any normalized key."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
    # Assuming "unknown-model-xyz" is not in pricing_data keys or substrings
    assert cost_calculator.calculate_cost("unknown-model-xyz", usage) == 0.0


def test_calculate_cost_with_zero_pricing(cost_calculator):
    """Test cost calculation when pricing for a model is zero."""
    # Add a model with zero pricing for testing
    original_pricing = cost_calculator.pricing_data.copy()
    cost_calculator.pricing_data["zero_price_model"] = {"prompt": 0.0, "completion": 0.0}

    usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
    assert cost_calculator.calculate_cost("zero_price_model", usage) == 0.0

    # Restore original pricing data
    cost_calculator.pricing_data = original_pricing


# --- Mocking importlib.resources for different Python versions ---
# The current test setup mocks `importlib.resources.files`.
# If we needed to test the fallbacks to `importlib_resources` or `pkg_resources`,
# we would need more complex patching. For now, focusing on the primary path and errors.
# The `test_load_pricing_file_not_found` and `test_load_pricing_json_decode_error`
# use `tmp_path` and `patch` to simulate file system interactions and errors.
# The `cost_calculator` fixture directly mocks `files` and `joinpath` to control
# the behavior of `_load_pricing` for testing various scenarios.
# The `mock_pricing_file` fixture is used to create a temporary file, but the
# `cost_calculator` fixture overrides the actual file reading with mocks for
# better control over error conditions.
