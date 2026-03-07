# Security Assessment Report

Last updated: 2026-03-07

## Static Analysis (Bandit)

**Tool**: `bandit -r genai_otel/`
**Result**: 11 findings, all LOW severity. No MEDIUM or HIGH severity issues.

| Severity | Confidence | Count | Details |
|----------|-----------|-------|---------|
| LOW | HIGH | 10 | `try/except/pass` patterns in instrumentors (intentional graceful degradation) |
| LOW | MEDIUM | 1 | False positive: `gen_ai.client.token.usage` constant flagged as possible hardcoded password |

**Assessment**: All findings are false positives or intentional design patterns. The `try/except/pass` blocks are used in instrumentors for graceful degradation when extracting optional fields from provider responses. No action needed.

## Dependency Vulnerability Scan (pip-audit)

**Tool**: `pip-audit` (last run: 2026-03-07)
**Result**: 481 packages scanned, 24 packages with 53 known vulnerabilities.

### Core Dependency Status
- opentelemetry-api/sdk/instrumentation: **No vulnerabilities**
- wrapt: **No vulnerabilities**
- requests: **No vulnerabilities**
- httpx: **No vulnerabilities**

### Notable Findings in Transitive Dependencies
| Package | Severity | Fix Available | Notes |
|---------|----------|--------------|-------|
| protobuf 5.29.5 | DoS via ParseDict | Yes (5.29.6) | Transitive dep via OTel gRPC exporter |
| urllib3 2.3.0 | 5 vulns (redirect, WASM) | Yes (2.5.0) | Transitive dep via requests |

### Optional/Dev Dependencies with Vulnerabilities
aiohttp, gradio, langchain-core, pydantic-ai, pypdf, and others have known vulnerabilities.
These are optional dependencies not shipped with the core package. Users should update
them independently if they install the corresponding extras.

**Recommendation**: Run `pip-audit --fix` periodically to keep dependencies updated.

## PII and Data Security

### Content Capture Controls
- `enable_content_capture` (default: `true`) - Controls whether prompt/response content is included in spans
- `content_max_length` (default: `1000`) - Truncates captured content to limit data exposure
- Users can disable content capture entirely for compliance-sensitive deployments

### PII Detection (Evaluation Module)
- **DETECT mode**: Returns PII findings with original text for review
- **REDACT mode**: Replaces PII with placeholders; `original_text` is set to `None` (P1-7 fix)
- **BLOCK mode**: Blocks content with PII; `original_text` is set to `None` (P1-7 fix)
- PII detection uses Presidio (local processing, no data leaves the machine)

### API Key Security
- No API keys or secrets are included in span attributes
- `OTEL_EXPORTER_OTLP_HEADERS` values are not logged or traced
- Provider API keys are only used by their respective SDKs, never captured by instrumentors

## Network Security

### Timeout Configuration
- OTLP exporter timeout: configurable via `OTEL_EXPORTER_OTLP_TIMEOUT` (default: 10s)
- Perspective API timeout: configurable via `ToxicityConfig.api_timeout` (default: 30s)
- All external API calls have timeout protection

### Export Endpoints
- Telemetry data is only sent to user-configured OTLP endpoints
- Default endpoint is `localhost:4318` (local collector)
- No data is sent to third-party services by the library itself

## Thread Safety

- Shared metrics use `threading.Lock` during initialization (P1-6 fix)
- GPU metrics collector runs in a daemon thread with proper shutdown
- OpenTelemetry SDK components are thread-safe by design
- Re-entrancy guard prevents double-wrapping on repeated `instrument()` calls (P0-3 fix)

## License Compliance

### Core Dependencies (all compatible with AGPL-3.0)
- opentelemetry-api/sdk: Apache-2.0
- wrapt: BSD-2-Clause
- requests: Apache-2.0
- httpx: BSD-3-Clause

### Notable Findings
- `mysql-connector-python`: GPL-2.0 (optional `[databases]` extra only)
- `pylint`: GPL-2.0 (dev dependency only, not shipped)
- All other dependencies: MIT, Apache-2.0, or BSD variants

### UNKNOWN License Packages
The following packages have unresolved license metadata (but are all optional/dev deps):
crewai, google-crc32c, mistralai, namex, sarvamai, sentencepiece, smolagents, ujson

## Recommendations

1. **For production deployments**: Set `enable_content_capture=false` to prevent prompt/response logging
2. **For compliance-sensitive environments**: Use REDACT or BLOCK mode for PII detection
3. **For network-restricted environments**: Configure OTLP endpoint to point to a local collector
4. **MySQL users**: Note that `mysql-connector-python` is GPL-2.0 licensed
