# Sarvam AI

TraceVerde provides full instrumentation for [Sarvam AI](https://www.sarvam.ai/), India's sovereign AI platform for multi-modal Indian language processing.

## Installation

```bash
pip install genai-otel-instrument[sarvamai]
pip install sarvamai
```

## Quick Start

```python
import genai_otel
genai_otel.instrument(service_name="sarvam-app")

from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key="your_key")

# Chat completion with sarvam-m
response = client.chat.completions.create(
    model="sarvam-m",
    messages=[{"role": "user", "content": "What is AI?"}],
)
print(response.choices[0].message.content)
# Traces, tokens, and costs captured automatically
```

## Supported Models and APIs

### Chat (sarvam-m)

Sarvam's multilingual chat model with support for Indian languages.

```python
response = client.chat.completions.create(
    model="sarvam-m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant who speaks Hindi."},
        {"role": "user", "content": "Tell me about India's space program"},
    ],
)
```

### Translation (Mayura v1)

Translation across 22+ Indian languages with speaker gender adaptation.

```python
translation = client.text.translate(
    input="Hello, how are you?",
    source_language_code="en-IN",
    target_language_code="hi-IN",
    model="mayura:v1",
    mode="classic-formal",
    speaker_gender="female",
)
print(translation.translated_text)
```

Features:

- 22+ Indian languages
- `mayura:v1` (latest) and `sarvam-translate:v1` (legacy)
- Translation modes: `classic-formal`, `classic-colloquial`
- Speaker gender adaptation
- Numerals format control (international vs native)

See [Mayura translation example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/mayura_translate_example.py).

### Text-to-Speech (Bulbul v3)

High-quality TTS with 48 speakers across 11 Indian languages.

```python
tts_response = client.text.speech(
    input="Namaste, aap kaise hain?",
    target_language_code="hi-IN",
    model="bulbul:v2",
    speaker="meera",
    pace=1.0,
    pitch=0,
    loudness=1.5,
)
# tts_response.audios contains base64-encoded WAV audio
```

Bulbul v3 features:

- 48 speakers (male and female voices)
- 11 Indian languages
- Pace control (0.5 to 2.0)
- Temperature control for expressiveness
- 24000 Hz sample rate
- Max 2500 characters per request

See [Bulbul v3 TTS example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/bulbul_v3_tts_example.py).

### Speech-to-Text (Saarika)

Automatic speech recognition for Indian languages.

```python
with open("audio.wav", "rb") as f:
    transcript = client.speech.text(
        file=f,
        model="saarika:v2",
        language_code="hi-IN",
    )
print(transcript.transcript)
```

### Language Detection

Identify the language of input text.

```python
detection = client.text.detect_language(
    input="Namaste, kaise ho?",
)
print(detection.language_code)  # "hi-IN"
```

## Multilingual Pipeline Example

A full production-style pipeline combining multiple Sarvam APIs:

```
Customer Query (any Indian language)
  +-- Language Detection
  +-- Translate to English
  +-- sarvam-m Chat (generate response)
  +-- Translate response back to customer's language
  +-- Bulbul v3 TTS (voice response)
```

All steps are automatically traced with cost breakdown per API call.

See [multilingual pipeline example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/multilingual_pipeline_example.py).

## Arya Agent Orchestration

Simulates a Sarvam Arya-style agentic workflow with:

- Voice/text input in any Indian language
- Multi-step reasoning pipeline
- Tool usage (translate, detect, search, synthesize)
- Immutable conversation state
- Localized voice+text output

See [Arya agent example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/arya_agent_example.py).

## Cost Tracking

TraceVerde includes pricing for all Sarvam AI models:

- **sarvam-m**: Chat model with per-token pricing
- **Saarika/Saaras**: STT models with per-second pricing
- **Bulbul**: TTS with per-character pricing
- **Mayura/Sarvam Translate**: Translation with per-token pricing
- **Vision**: Image understanding models

12+ models tracked in the pricing database.

## Environment Setup

```bash
export SARVAM_API_KEY=your_sarvam_api_key
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=sarvam-app
```

## All Examples

| Example | Description |
|---------|-------------|
| [Basic usage](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/example.py) | Chat, translate, TTS, STT, language detection |
| [Arya agent](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/arya_agent_example.py) | Agent orchestration pipeline |
| [Bulbul v3 TTS](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/bulbul_v3_tts_example.py) | Text-to-speech with 48 speakers |
| [Mayura translate](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/mayura_translate_example.py) | Translation with modes and gender adaptation |
| [Multilingual pipeline](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/multilingual_pipeline_example.py) | Full customer support pipeline |
