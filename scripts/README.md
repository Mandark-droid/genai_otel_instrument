# Scripts

This directory contains utility scripts for development, testing, and validation.

## Validation Scripts

### validate_examples.sh / validate_examples.bat

Validates all PII, Toxicity, and Bias detection examples.

**Usage (Linux/Mac):**
```bash
# Dry run - list all examples without running
./scripts/validate_examples.sh --dry-run

# Run all examples with default settings
./scripts/validate_examples.sh

# Run with custom endpoint
./scripts/validate_examples.sh http://192.168.206.128:55681

# Run with verbose output
./scripts/validate_examples.sh --verbose http://192.168.206.128:55681

# Run with custom timeout (120 seconds)
./scripts/validate_examples.sh --timeout 120 http://192.168.206.128:55681

# Show help
./scripts/validate_examples.sh --help
```

**Usage (Windows):**
```bat
REM Run all examples
scripts\validate_examples.bat

REM Run with custom endpoint
scripts\validate_examples.bat http://192.168.206.128:55681
```

**Environment Variables:**
- `OTEL_EXPORTER_OTLP_ENDPOINT` - OTLP endpoint (can be overridden by script argument)
- `OPENAI_API_KEY` - OpenAI API key (required for examples to work)
- `PERSPECTIVE_API_KEY` - Google Perspective API key (optional, for perspective_api.py)

**Options:**
- `-v, --verbose` - Show detailed output from examples
- `-d, --dry-run` - List examples without running them
- `-t, --timeout N` - Set timeout in seconds (default: 90)
- `-h, --help` - Show help message

**Exit Codes:**
- `0` - All examples passed
- `1` - One or more examples failed

## Test Scripts

Temporary test scripts created during debugging are stored here for reference:

- `test_*.py` - Various debugging and validation scripts
- `SOLUTION_SUMMARY.md` - Documentation of PII/Toxicity detection solution
- `VALIDATION_REPORT.md` - Comprehensive validation report for all examples

## Release Scripts

### test_release.sh

Script for testing the release process (if applicable).

## Usage Examples

### Validate All Examples Before Release

```bash
# 1. List all examples
bash scripts/validate_examples.sh --dry-run

# 2. Run validation with custom endpoint
export OPENAI_API_KEY="your-key-here"
bash scripts/validate_examples.sh http://192.168.206.128:55681

# 3. If failures occur, run with verbose to debug
bash scripts/validate_examples.sh --verbose --timeout 120 http://192.168.206.128:55681
```

### Manual Example Testing

```bash
# Test a specific PII example
export OTEL_EXPORTER_OTLP_ENDPOINT="http://192.168.206.128:55681"
export OPENAI_API_KEY="your-key-here"
python examples/pii_detection/basic_detect_mode.py

# Test a specific Toxicity example
python examples/toxicity_detection/basic_detoxify.py

# Test combined PII + Toxicity
python examples/toxicity_detection/combined_with_pii.py
```

## Notes

- Examples require an OpenAI API key to run successfully
- The default timeout is 90 seconds to account for API latency
- Some examples (env_var_config.py) are skipped automatically as they require specific environment variable configurations
- Perspective API examples require `PERSPECTIVE_API_KEY` environment variable
