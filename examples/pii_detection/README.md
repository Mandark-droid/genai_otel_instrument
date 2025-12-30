# PII Detection Examples

This folder contains examples demonstrating PII (Personally Identifiable Information) detection capabilities in GenAI applications using OpenTelemetry instrumentation.

## Prerequisites

```bash
pip install genai-otel-instrument openai
pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_lg
```

## Environment Setup

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OPENAI_API_KEY=your_api_key_here
```

## Examples

### Basic Detection Modes

- **`basic_detect_mode.py`** - Basic PII detection without modification
  - Detects: Email, phone numbers, SSN, etc.
  - Records detection in telemetry attributes
  - Does not modify prompts or block requests

- **`redaction_mode.py`** - PII detection with redaction
  - Detects PII entities
  - Redacts PII in telemetry data (replaces with asterisks)
  - Protects sensitive data in monitoring systems

- **`blocking_mode.py`** - PII detection with blocking
  - Detects PII entities
  - Sets span status to ERROR when PII detected
  - Enables filtering/alerting on PII-containing requests

### Compliance Modes

- **`gdpr_compliance.py`** - GDPR compliance for EU data protection
  - Detects EU-specific entities: IBAN, NHS numbers
  - Ensures GDPR regulatory compliance

- **`hipaa_compliance.py`** - HIPAA compliance for healthcare
  - Detects healthcare PHI: medical licenses, dates
  - Ensures HIPAA regulatory compliance

- **`pci_dss_compliance.py`** - PCI-DSS compliance for payment data
  - Detects payment card data: credit cards, bank accounts
  - Ensures PCI-DSS regulatory compliance

- **`combined_compliance.py`** - All compliance modes enabled
  - Comprehensive protection across all regulations
  - Recommended for multi-jurisdiction applications

### Advanced Features

- **`response_detection.py`** - PII detection in LLM responses
  - Checks both prompts AND responses
  - Catches accidentally generated PII

- **`custom_threshold.py`** - Custom confidence threshold
  - Demonstrates threshold tuning (0.5-0.9)
  - Balance between sensitivity and false positives

- **`env_var_config.py`** - Environment variable configuration
  - Zero-code configuration via environment variables
  - Ideal for containerized deployments

## Detected Entity Types

### Standard PII
- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- IP addresses
- Physical addresses
- Names

### GDPR-specific
- IBAN codes (European bank accounts)
- UK NHS numbers
- Named Recognized Persons (NRP)

### HIPAA-specific
- Medical license numbers
- US Passport numbers
- Protected Health Information (PHI)
- Date/time fields

### PCI-DSS-specific
- Credit card numbers (all major types)
- US bank account numbers

## Telemetry Attributes

All examples record the following span attributes:

```
evaluation.pii.prompt.detected = true/false
evaluation.pii.prompt.entity_count = <number>
evaluation.pii.prompt.entity_types = [list of types]
evaluation.pii.prompt.<ENTITY_TYPE>_count = <count>
evaluation.pii.prompt.score = <confidence 0-1>
evaluation.pii.prompt.redacted = <redacted text> (redact mode)
evaluation.pii.prompt.blocked = true (block mode)

# Same for responses
evaluation.pii.response.*
```

## Metrics

```
genai.evaluation.pii.detections (counter)
genai.evaluation.pii.entities (counter, by entity_type)
genai.evaluation.pii.blocked (counter)
```

## Running Examples

Each example is standalone and can be run directly:

```bash
python basic_detect_mode.py
python redaction_mode.py
python gdpr_compliance.py
# ... etc
```

## Important Notes

1. **One instrument() call per process**: Each example should be run separately. Do not call `instrument()` multiple times in the same process.

2. **Dependencies required**: Make sure Presidio and spacy are installed with the English model.

3. **Fallback behavior**: If Presidio is unavailable, the library falls back to regex-based detection (less accurate).

4. **Privacy**: In redaction mode, PII is only redacted in telemetry - the actual LLM request is not modified.

5. **Blocking**: Block mode sets span status to ERROR but does not actually prevent the request. Your application should check span status and handle accordingly.

## Technology Stack

- **Microsoft Presidio**: ML-based PII detection and anonymization
- **Spacy**: Natural language processing for entity recognition
- **Regex Fallback**: Pattern-based detection when Presidio unavailable
- **OpenTelemetry**: Instrumentation and telemetry collection

## Further Reading

- [Microsoft Presidio Documentation](https://microsoft.github.io/presidio/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [GDPR Compliance Guide](https://gdpr.eu/)
- [HIPAA Requirements](https://www.hhs.gov/hipaa/)
- [PCI-DSS Standards](https://www.pcisecuritystandards.org/)
