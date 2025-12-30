## Critical Fixes

### PII Evaluation Attributes Export to Jaeger
Fixed a critical issue where PII and Toxicity evaluation attributes were not appearing in Jaeger traces.

**Root Cause:** Attributes were being set AFTER `span.end()` when the span becomes immutable (ReadableSpan).

**Solution:** Added `_run_evaluation_checks()` method in `BaseInstrumentor` that runs BEFORE `span.end()`.

**Result:** Evaluation attributes now successfully appear in Jaeger:
- `evaluation.pii.prompt.detected`
- `evaluation.pii.prompt.entity_count`
- `evaluation.pii.prompt.entity_types`
- `evaluation.pii.prompt.score`
- And more...

### Editable Installation for Development
Fixed issue where examples were using old code from site-packages. Developers must now use:
```bash
pip install -e .
```

## New Features

### Reorganized Examples
- **examples/pii_detection/** - 10 dedicated PII detection examples
  - Basic detect, redaction, and blocking modes
  - GDPR, HIPAA, PCI-DSS compliance examples
  - Response detection, custom thresholds, combined compliance

- **examples/toxicity_detection/** - 8 dedicated toxicity detection examples
  - Detoxify (local model) and Perspective API (Google)
  - Blocking mode, category detection, custom thresholds
  - Combined PII + Toxicity detection

- **examples/bias_detection/** - Placeholder for future development

### Validation Scripts
- **scripts/validate_examples.sh** (Linux/Mac)
- **scripts/validate_examples.bat** (Windows)

Features:
- `--dry-run` - List all examples without running
- `--verbose` - Show detailed output
- `--timeout N` - Configurable timeout (default: 90s)
- Color-coded output with comprehensive summary

### Documentation
- New **scripts/README.md** with comprehensive usage guide
- Updated **CHANGELOG.md** with full v0.1.27 details
- Moved debug files to scripts/ folder

## Verification

All changes have been tested and validated:
- ✅ PII attributes appear in Jaeger (6 attributes per detection)
- ✅ Toxicity attributes appear in Jaeger (9 attributes per detection)
- ✅ All 19 examples execute successfully
- ✅ Metrics correctly exported to Prometheus

## Installation

```bash
pip install genai-otel-instrument==0.1.27
```

For development:
```bash
git clone https://github.com/Mandark-droid/genai_otel_instrument.git
cd genai_otel_instrument
pip install -e .
```

## Running Examples

```bash
# Set environment variables
export OTEL_EXPORTER_OTLP_ENDPOINT=http://your-collector:4318
export OPENAI_API_KEY=your_api_key_here

# Run a PII detection example
python examples/pii_detection/basic_detect_mode.py

# Run a toxicity detection example
python examples/toxicity_detection/basic_detoxify.py

# Validate all examples
bash scripts/validate_examples.sh --dry-run
```

## Breaking Changes

None - this is a backward-compatible bugfix release.

## Full Changelog

See [CHANGELOG.md](https://github.com/Mandark-droid/genai_otel_instrument/blob/main/CHANGELOG.md) for complete details.
