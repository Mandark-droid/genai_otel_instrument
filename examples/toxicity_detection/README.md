# Toxicity Detection Examples

This folder contains examples demonstrating toxicity detection capabilities in GenAI applications using OpenTelemetry instrumentation.

## Prerequisites

### Local Detection (Detoxify)
```bash
pip install genai-otel-instrument openai
pip install detoxify
```

### Cloud Detection (Perspective API)
```bash
pip install genai-otel-instrument openai
pip install google-api-python-client
```

Get your Perspective API key from: https://developers.perspectiveapi.com/

## Environment Setup

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OPENAI_API_KEY=your_api_key_here

# Optional: For Perspective API
export GENAI_TOXICITY_PERSPECTIVE_API_KEY=your_perspective_api_key
```

## Examples

### Detection Methods

- **`basic_detoxify.py`** - Local ML-based detection with Detoxify
  - Offline detection, no external API calls
  - Privacy-friendly for on-premise deployments
  - Good accuracy for most use cases
  - Supports 6 toxicity categories

- **`perspective_api.py`** - Cloud-based detection with Google Perspective API
  - Higher accuracy with continuously updated models
  - Industry-standard toxicity classification
  - Requires API key and internet connection
  - Better handling of context and nuance

### Detection Modes

- **`blocking_mode.py`** - Block toxic content
  - Sets span status to ERROR when toxicity detected
  - Records blocked metrics
  - Enables filtering/alerting in observability platform

### Advanced Features

- **`category_detection.py`** - Category-specific detection
  - Demonstrates all 6 toxicity categories
  - Shows how different content types are classified
  - Multiple categories can be triggered simultaneously

- **`custom_threshold.py`** - Custom confidence threshold
  - Demonstrates threshold tuning (0.5-0.9)
  - Balance between sensitivity and false positives
  - Guidelines for different use cases

- **`response_detection.py`** - Detection in LLM responses
  - Checks both prompts AND responses
  - Catches accidentally generated toxic content
  - Comprehensive safety monitoring

- **`env_var_config.py`** - Environment variable configuration
  - Zero-code configuration
  - Ideal for containerized deployments

- **`combined_with_pii.py`** - Multiple safety features
  - PII + Toxicity detection together
  - Comprehensive protection
  - Example of full safety stack

## Toxicity Categories

All examples detect these categories:

1. **toxicity**: General toxic language
2. **severe_toxicity**: Extremely harmful content
3. **identity_attack**: Discrimination, hate speech
4. **insult**: Insulting or demeaning language
5. **profanity**: Swearing and obscene content
6. **threat**: Threatening or violent language

## Telemetry Attributes

All examples record these span attributes:

```
evaluation.toxicity.prompt.detected = true/false
evaluation.toxicity.prompt.max_score = <score 0-1>
evaluation.toxicity.prompt.categories = [list of categories]
evaluation.toxicity.prompt.toxicity_score = <score>
evaluation.toxicity.prompt.severe_toxicity_score = <score>
evaluation.toxicity.prompt.identity_attack_score = <score>
evaluation.toxicity.prompt.insult_score = <score>
evaluation.toxicity.prompt.profanity_score = <score>
evaluation.toxicity.prompt.threat_score = <score>
evaluation.toxicity.prompt.blocked = true (blocking mode)

# Same for responses
evaluation.toxicity.response.*
```

## Metrics

```
genai.evaluation.toxicity.detections (counter)
genai.evaluation.toxicity.categories (counter, by category)
genai.evaluation.toxicity.score (histogram)
genai.evaluation.toxicity.blocked (counter)
```

## Running Examples

Each example is standalone and can be run directly:

```bash
python basic_detoxify.py
python perspective_api.py  # Requires API key
python blocking_mode.py
# ... etc
```

## Detection Method Comparison

| Feature | Detoxify | Perspective API |
|---------|----------|-----------------|
| **Location** | Local | Cloud |
| **Privacy** | Offline, private | Sends data to Google |
| **Accuracy** | Good | Excellent |
| **Latency** | Fast (local) | Slower (API call) |
| **Cost** | Free | Free tier available |
| **Setup** | `pip install detoxify` | API key required |
| **Use Case** | On-premise, privacy-critical | Higher accuracy needed |

## Threshold Selection Guidelines

- **0.5-0.6**: Strict moderation
  - Children's platforms
  - Educational apps
  - High sensitivity, more false positives

- **0.7-0.8**: Balanced (recommended)
  - General consumer apps
  - Social platforms
  - Good balance of accuracy

- **0.9+**: Permissive
  - Internal tools
  - Developer forums
  - Only severe toxicity

## Important Notes

1. **One instrument() call per process**: Each example should be run separately. Do not call `instrument()` multiple times in the same process.

2. **Automatic fallback**: If Perspective API is configured but fails, the library automatically falls back to Detoxify.

3. **Both prompts and responses**: Toxicity detection is applied to both user inputs and LLM outputs.

4. **Blocking behavior**: Block mode sets span status to ERROR but does not actually prevent the request. Your application should check span status and handle accordingly.

5. **Context limitations**: Current models may not fully understand sarcasm, irony, or context-specific language use.

## Technology Stack

- **Detoxify**: Unitary's open-source ML toxicity model
- **Perspective API**: Google's toxicity detection service
- **OpenTelemetry**: Instrumentation and telemetry collection
- **Automatic fallback**: Graceful degradation between methods

## Use Cases

- **Content moderation** for user-facing chatbots
- **Workplace collaboration** tools (prevent harassment)
- **Educational platforms** (safe learning environments)
- **Social media** and community forums
- **Customer support** systems (professional interactions)
- **Gaming communities** (reduce toxic behavior)

## Further Reading

- [Detoxify GitHub](https://github.com/unitaryai/detoxify)
- [Perspective API Documentation](https://perspectiveapi.com/)
- [Google Perspective API Developer Guide](https://developers.perspectiveapi.com/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
