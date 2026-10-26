# Python Version Compatibility

## Supported Versions

| Python Version | Status | Notes |
|---------------|--------|-------|
| 3.9 | Supported | Minimum required version |
| 3.10 | Supported | |
| 3.11 | Supported | |
| 3.12 | Supported | Primary development/testing version |
| 3.13 | Supported | Added to the regular CI and release-validation matrices. |
| 3.14 | Candidate | Full CI and release-validation matrices are configured; support is pending green hosted runs. |
| 3.15 | Prerelease canary | Non-blocking CI probe only; not a supported Python version. |
| < 3.9 | Not supported | OpenTelemetry SDK requires >= 3.9 |

## Key Dependencies and Their Python Requirements

| Dependency | Min Python | Notes |
|-----------|-----------|-------|
| opentelemetry-api | 3.9 | Core dependency |
| opentelemetry-sdk | 3.9 | Core dependency |
| wrapt | 3.9 | Core dependency |
| requests | 3.8 | Core dependency |
| httpx | 3.9 | Core dependency |
| protobuf | 3.9 | Transitive (via OTel exporters) |
| grpcio | 3.9 | Transitive (via OTel gRPC exporter) |

## Version-Specific Notes

### Python 3.9
- `typing.get_type_hints` may behave differently with some type annotations
- `importlib.resources` API differs from 3.10+; the library does not rely on this

### Python 3.10
- No known issues

### Python 3.11
- `tomllib` is available in stdlib (used by some dev tools)

### Python 3.12
- Primary development and testing version
- All tests pass consistently

### Python 3.13
- Supported; included in the regular CI and release-validation matrices
- Some optional dependencies may not yet have wheels for 3.13

### Python 3.14
- Candidate version under evaluation; regular CI and release validation run the full suite
- Do not treat it as supported until those hosted runs pass

### Python 3.15
- Prerelease canary only; the non-blocking probe helps identify upstream compatibility issues
- Not included in the supported-version claim until Python reaches a stable release and the suite passes

## pyproject.toml Configuration

```toml
[project]
requires-python = ">=3.9"
```

## Testing Across Versions

To test against multiple Python versions, use `tox` or run directly:

```bash
# Using specific Python version
python3.9 -m pytest tests/ -v
python3.12 -m pytest tests/ -v
python3.13 -m pytest tests/ -v
python3.14 -m pytest tests/ -v  # Candidate version; support pending hosted CI results
```
