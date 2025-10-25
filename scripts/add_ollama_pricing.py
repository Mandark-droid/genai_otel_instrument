"""Script to add Ollama model pricing to llm_pricing.json

Ollama models are local and free, but we estimate costs based on:
1. Model parameter count
2. Comparable cloud API model pricing
3. GPU power consumption and electricity costs

Price Tiers (based on parameter count):
- Tiny (< 1B params): $0.0001 / $0.0002 (prompt/completion)
- Small (1-10B): $0.0003 / $0.0006
- Medium (10-20B): $0.0005 / $0.001
- Large (20-80B): $0.0008 / $0.0008
- XLarge (80B+): $0.0012 / $0.0012

Note: CostCalculator also has fallback logic to estimate costs for:
- Unknown Ollama models based on parameter count parsing from model name
- HuggingFace Transformers models (also typically run locally)
"""

import json
from pathlib import Path

# Common Ollama models with their parameter counts
OLLAMA_MODELS = {
    # Tiny models (< 1B)
    "smollm2:135m": 0.135,
    "smollm2:360m": 0.360,
    "tinyllama": 1.1,
    # Small models (1-10B)
    "llama3.2:1b": 1.0,
    "llama3.2:3b": 3.0,
    "gemma:2b": 2.0,
    "gemma2:2b": 2.0,
    "phi3:3.8b": 3.8,
    "phi4:14b": 14.0,
    "qwen2.5:0.5b": 0.5,
    "qwen2.5:1.5b": 1.5,
    "qwen2.5:3b": 3.0,
    "qwen2.5:7b": 7.0,
    "qwen3:3b": 3.0,
    "qwen3:8b": 8.0,
    "llama2:7b": 7.0,
    "llama3:8b": 8.0,
    "llama3.1:8b": 8.0,
    "mistral:7b": 7.0,
    "gemma:7b": 7.0,
    "gemma2:9b": 9.0,
    "codellama:7b": 7.0,
    # Medium models (10-20B)
    "llama2:13b": 13.0,
    "codellama:13b": 13.0,
    # Large models (20-80B)
    "llama2:70b": 70.0,
    "llama3:70b": 70.0,
    "llama3.1:70b": 70.0,
    "qwen2.5:72b": 72.0,
    "codellama:34b": 34.0,
    "mixtral:8x7b": 47.0,  # Mixture of Experts, effective ~47B params
    "deepseek-r1:7b": 7.0,
    "deepseek-r1:14b": 14.0,
    "deepseek-r1:32b": 32.0,
    "deepseek-r1:70b": 70.0,
    "deepseek-v3.1": 671.0,
    # XLarge models (80B+)
    "llama3.1:405b": 405.0,
}


def get_price_tier(param_count_billions):
    """Get pricing tier based on parameter count."""
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


def main():
    # Load existing pricing file
    pricing_file = Path(__file__).parent.parent / "genai_otel" / "llm_pricing.json"
    with open(pricing_file, "r", encoding="utf-8") as f:
        pricing_data = json.load(f)

    # Add Ollama models to chat pricing
    ollama_pricing = {}
    for model_name, param_count in OLLAMA_MODELS.items():
        ollama_pricing[model_name] = get_price_tier(param_count)

    # Add all Ollama models
    pricing_data["chat"].update(ollama_pricing)

    # Write back to file
    with open(pricing_file, "w", encoding="utf-8") as f:
        json.dump(pricing_data, f, indent=2)

    print(f"Added {len(ollama_pricing)} Ollama models to llm_pricing.json")
    print("\nSample pricing:")
    for model, price in list(ollama_pricing.items())[:5]:
        print(f"  {model}: ${price['promptPrice']:.4f} / ${price['completionPrice']:.4f}")


if __name__ == "__main__":
    main()
