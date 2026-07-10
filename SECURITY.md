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
- `enable_content_capture` (env `GENAI_ENABLE_CONTENT_CAPTURE`, default: `false`) -
  Controls whether prompt/response content is included in spans. It is **off by
  default**; prompt/response text is only captured on spans when an operator
  explicitly opts in.
- `content_max_length` (env `GENAI_CONTENT_MAX_LENGTH`, default: `200`) -
  Truncates captured content to limit data exposure. Set to `0` for no limit.
- Content capture is configurable rather than always-on, and is intended for
  **audit / explainability** use cases where the deployment needs prompt/response
  visibility (e.g. regulated tracing mandates). Deployments that must not place
  customer content in telemetry should leave it disabled.
- Under the bank / BFSI hardening profile (`GENAI_PROFILE=strict|bfsi|bank`),
  content capture is turned **on for audit** unless the operator has explicitly
  pinned `GENAI_ENABLE_CONTENT_CAPTURE`; that same profile forces all third-party
  egress paths off (see below).

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
- Telemetry itself goes only to the operator's OTLP endpoint. With all
  third-party features left at their (off) defaults, the library performs no
  outbound calls of its own beyond that OTLP export. See the caveats below for
  the optional features that can reach external services.

### Third-Party Network Paths and Air-Gap Controls
When **opt-in** evaluation features are enabled, two code paths can reach hosts
outside the deployment; these are the only third-party network paths in the
library:

1. **Toxicity via Google Perspective API** (`GENAI_TOXICITY_USE_PERSPECTIVE_API=true`
   plus an API key) - sends prompt/response text to `commentanalyzer.googleapis.com`.
   Disabled by default.
2. **Detoxify model download** - the local Detoxify toxicity model fetches model
   weights from the internet (torch-hub / Hugging Face) on first use. Only
   reachable if toxicity detection is explicitly enabled with the local model.

Egress controls (config, env-overridable):

- `GENAI_PROFILE=strict|bfsi|bank` - deployment hardening profile for
  air-gapped / bank use. It forces `allow_external_egress=false`,
  `air_gapped=true`, offline CO2 mode, and **disables the Perspective API path**
  (`toxicity_use_perspective_api` is forced off), so no prompt/response text is
  sent to Google under this profile.
- `GENAI_ALLOW_EXTERNAL_EGRESS` (default `true`) and `GENAI_AIR_GAPPED`
  (default `false`) express the intended no-external-egress / air-gapped posture
  and are forced to the safe values by the strict profile.

Caveat for air-gapped deployments: the Detoxify **model download** requires
network access on first use. In an air-gapped or strict-profile deployment, do
not enable the Detoxify toxicity path unless the model weights have been
pre-provisioned locally; treat any toxicity/Detoxify usage as requiring
operator verification that no runtime download occurs.

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
