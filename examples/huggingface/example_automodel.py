"""HuggingFace AutoModelForCausalLM Example with Token Counting and Cost Tracking.

This example demonstrates:
1. Auto-instrumentation of AutoModelForCausalLM.generate()
2. Automatic token counting (prompt + completion tokens)
3. Cost calculation for local model inference
4. Full observability with traces and metrics
5. Manual server metrics (KV cache, request queue)

Requirements:
    pip install transformers torch
"""

import genai_otel

# Auto-instrument HuggingFace Transformers
genai_otel.instrument()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Get server metrics collector for manual instrumentation
server_metrics = genai_otel.get_server_metrics()

print("\n" + "=" * 80)
print("Loading model and tokenizer...")
print("=" * 80 + "\n")

# Load a small model for testing (117M parameters)
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Model loaded: {model_name}")
print(f"Model config: {model.config._name_or_path}\n")

# Set server metrics (simulating server-side metrics)
# In production, these would be populated from your serving framework (vLLM, TGI, etc.)
print("=" * 80)
print("Setting server metrics...")
print("=" * 80 + "\n")

# Simulate KV cache usage (would come from serving framework in production)
server_metrics.set_kv_cache_usage(model_name, 45.5)  # 45.5% cache usage
print(f"KV cache usage set: 45.5% for {model_name}")

# Set request queue metrics
server_metrics.set_requests_max(10)  # Max 10 concurrent requests
server_metrics.set_requests_running(1)  # 1 request currently running (this one)
server_metrics.set_requests_waiting(0)  # No requests waiting
print("Request queue metrics set: running=1, waiting=0, max=10\n")

# Prepare input
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"Prompt: '{prompt}'")
print(f"Input tokens: {inputs['input_ids'].shape[-1]}\n")

print("=" * 80)
print("Generating text (instrumented)...")
print("=" * 80 + "\n")

# Generate text - This is automatically instrumented!
# The wrapper will:
# - Create a span with model info
# - Count input tokens (from input_ids.shape)
# - Count output tokens (from generated sequence length)
# - Calculate cost based on GPT-2's parameter count (117M -> tier pricing)
# - Record metrics for tokens and cost
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated text: {generated_text}\n")
print(f"Total output tokens: {outputs.shape[-1]}")
print(f"Input tokens: {inputs['input_ids'].shape[-1]}")
print(f"Generated (new) tokens: {outputs.shape[-1] - inputs['input_ids'].shape[-1]}\n")

# Update server metrics after generation (simulate cache usage increase)
server_metrics.set_kv_cache_usage(model_name, 62.3)  # Cache usage increased
server_metrics.set_requests_running(0)  # Request completed
print("Server metrics updated: KV cache now at 62.3%, request completed\n")

print("=" * 80)
print("Telemetry captured:")
print("=" * 80)
print("✓ Span created: huggingface.model.generate")
print("✓ Attributes set:")
print(f"  - gen_ai.system: huggingface")
print(f"  - gen_ai.request.model: {model_name}")
print(f"  - gen_ai.operation.name: text_generation")
print(f"  - gen_ai.usage.prompt_tokens: {inputs['input_ids'].shape[-1]}")
print(f"  - gen_ai.usage.completion_tokens: {outputs.shape[-1] - inputs['input_ids'].shape[-1]}")
print(f"  - gen_ai.usage.total_tokens: {outputs.shape[-1]}")
print("  - gen_ai.usage.cost.total: $X.XXXXXX (estimated)")
print("  - gen_ai.usage.cost.prompt: $X.XXXXXX")
print("  - gen_ai.usage.cost.completion: $X.XXXXXX")
print("\n✓ Metrics recorded:")
print("  - gen_ai.requests counter")
print("  - gen_ai.client.token.usage (prompt + completion)")
print("  - gen_ai.client.token.usage.prompt histogram")
print("  - gen_ai.client.token.usage.completion histogram")
print("  - gen_ai.client.operation.duration histogram")
print("  - gen_ai.usage.cost counter")
print("\n✓ Server metrics (manual):")
print(f"  - gen_ai.server.kv_cache.usage: 62.3% (model={model_name})")
print("  - gen_ai.server.requests.running: 0")
print("  - gen_ai.server.requests.waiting: 0")
print("  - gen_ai.server.requests.max: 10")
print("\n✓ Traces and metrics exported to OTLP endpoint!")
print("=" * 80)

print("\nNote: Cost is estimated based on model size (GPT-2 = 117M params)")
print("Local models are free to run, but costs reflect GPU/compute resources.")
print("\nServer metrics are manually set for demonstration.")
print("In production, integrate with your serving framework (vLLM, TGI, etc.) to populate these.")
