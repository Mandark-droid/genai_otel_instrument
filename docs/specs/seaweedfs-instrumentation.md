# SeaweedFS instrumentation (spec)

Status: **APPROVED 2026-11-11** (implemented in the Unreleased changes).

Approved decisions:

- D1: keep `db.system="s3"`; add `db.system.name="seaweedfs"` only on declared endpoints; always
  emit `server.port`; add `aws.s3.bucket` / `aws.s3.key` alongside the legacy `s3.bucket` /
  `s3.object` (legacy names are deprecated but not removed in a patch).
- D2: use the custom `seaweedfs` value now. Follow-up: propose an upstream well-known value.
- D3: declared endpoints only (`GENAI_OTEL_SEAWEEDFS_ENDPOINTS`, auto-filled from the
  `seaweed` media-store endpoint); no port heuristic.
- D4: the SDK's own media-store spans are suppressed by default via OpenTelemetry's standard
  suppression context; `GENAI_OTEL_MEDIA_STORE_TRACE_CALLS=true` opts back in.
- D5: conservative path. `requests` is NOT patched. The Filer is covered through `httpx` and
  `aiohttp` only, for declared Filer hosts.

Implementation notes and deviations from the draft below:

- Filer spans are made by wrapping `httpx.Client` / `AsyncClient._send_single_request` and
  `aiohttp.ClientSession._request`. If the generic httpx instrumentation is also enabled, the
  generic client span and the Filer span are siblings; the Filer span is the one that carries
  the SeaweedFS meaning. De-duplication is a follow-up.
- Open follow-ups: `requests` support for the Filer (needs a proof that OTLP exporters are
  unaffected); upstream semconv proposal for a SeaweedFS / object-store `db.system.name`;
  confirm response headers on a live gateway and Filer (Q3); an aiohttp trace-config variant.

## 1. Problem

Today SeaweedFS is observable only through its S3 gateway, via the botocore patch in
`genai_otel/mcp_instrumentors/s3_instrumentor.py`:

- every span carries `db.system="s3"`, the same as AWS S3 and MinIO;
- `server.address` is the hostname only (by design, to keep credentials and paths out),
  so there is no port and SeaweedFS cannot be told apart from AWS;
- the Filer HTTP API (the path SeaweedFS-native clients use) has no instrumentation;
- async S3 clients are not covered (section 4);
- the SDK's own media-store uploads produce `s3.*` spans, which is noise in user traces
  (section 5).

## 2. Identifying SeaweedFS on S3-gateway spans

### 2.1 What was verified (upstream docs and source, master branch)

| Fact | Evidence |
| --- | --- |
| S3 gateway default port is 8333 | `weed/command/s3.go`: `-port` default `8333` |
| Filer default port is 8888 | `weed/command/filer.go`: `-port` default `8888` |
| The gateway sets `x-amz-request-id` only when the response lacks one | `weed/util/request_id` middleware |
| No `Server` header or `X-Seaweedfs-*` response header was found in `s3api_server.go` or the filer command file | source read; not an exhaustive audit of every handler |
| The Filer sets `Seaweed-*` response headers only for user tags | Filer Server API wiki |

Conclusion: **response headers cannot identify SeaweedFS.** `x-amz-request-id` is also what
AWS sends, and no vendor header was found. Detection therefore rests on configuration, with
the port as a weak hint. Before implementation, re-check a live gateway with `curl -i`
(open question Q3), because docs and source reading are not proof of runtime behaviour.

### 2.2 Semantic-convention position

- `server.port` is a standard attribute. It is always emitted (when the endpoint states or
  implies one), regardless of vendor.
- `db.system` is deprecated upstream in favour of `db.system.name`. The registry has no
  well-known value for S3, object stores or SeaweedFS (only `aws.dynamodb`, `aws.redshift`
  among AWS ones), and says a custom value MAY be used.
- The AWS S3 attributes are `aws.s3.bucket`, `aws.s3.key` (Development status), alongside
  `rpc.method` and `aws.request_id`. The current instrumentor emits non-standard
  `s3.bucket` / `s3.object` instead.

### 2.3 Proposal

1. Always add `server.port` to S3 spans.
2. Add `aws.s3.bucket` and `aws.s3.key` next to the existing `s3.bucket` / `s3.object`
   (keep the old ones for one minor version, mark deprecated in the changelog), and
   `aws.request_id` from the `x-amz-request-id` response header when present.
3. When the endpoint matches a declared SeaweedFS endpoint (2.4), add
   `db.system.name="seaweedfs"`. On all other S3 spans add nothing new, so existing
   dashboards keyed on `db.system="s3"` keep working.
4. Keep `db.system="s3"` unchanged on gateway spans (it names the API dialect in use);
   see decision D1.

### 2.4 Detection by configuration

New setting `GENAI_OTEL_SEAWEEDFS_ENDPOINTS`: comma-separated `host:port` entries (the
gateway and the Filer may both be listed). Matching is exact on lowercased host and port,
after resolving the default port of the scheme. An optional wildcard host (`*.internal`)
is allowed; no regexes.

Convenience: if `GENAI_OTEL_MEDIA_STORE=seaweed`, its endpoint is added to the declared set
automatically, so the SDK's own uploads are labelled with no extra setup.

No automatic classification by port number alone. A port heuristic (8333 / 8888) is
rejected as a default because 8888 is also Jupyter's port and 8333 is Bitcoin's; it may be
offered as an explicit opt-in (D3).

## 3. Filer HTTP API instrumentor

### 3.1 Scope

A lightweight instrumentor named `SeaweedFilerInstrumentor`, enabled by
`GENAI_ENABLE_SEAWEEDFS_INSTRUMENTATION` (default `true`, but inert unless at least one
endpoint is declared; with none declared it does nothing and costs nothing).

It must work with the clients applications really use. Filer clients are plain HTTP, so the
real surface is:

| Client | Status today | Plan |
| --- | --- | --- |
| `httpx` | Patched by `HTTPXClientInstrumentor` only when `enable_http_instrumentation` is on (default off) | Hook at the httpx transport level, independent of that flag |
| `requests` | Deliberately NOT instrumented (`manager.py`: patching `requests.Session` breaks OTLP HTTP exporters) | Wrap `Session.send` narrowly, applying only to declared Filer hosts, so the exporter (different host) is never touched. Needs a test proving an OTLP exporter on another host is unaffected. See Q2 |
| `aiohttp` | Not covered | Same pattern, via a trace config on `ClientSession` creation; optional, phase 2 |
| `pyseaweed` (PyPI, 2.0.0) | Not verified; it appears to target the master/volume servers, not the Filer, and its HTTP library and maintenance status were not confirmed | Out of scope unless Q1 shows it is used |

No other Python Filer SDK was found on PyPI under the obvious names (`seaweedfs`,
`seaweed-client`, `seaweedfs-client`, `weed-python` all absent). Because everything is plain
HTTP, a transport-level hook with host matching covers any wrapper library for free.

### 3.2 Span shape

- Name: `seaweedfs.filer.<operation>`.
- Kind: CLIENT.
- Operations, derived from method plus query string (documented Filer API): `upload`
  (POST/PUT), `append` (`op=append`), `download` (GET/HEAD on a path), `list` (GET with a
  trailing slash or JSON accept), `stat` (`?metadata`), `delete`, `delete_recursive`,
  `move` (`mv.from`), `copy` (`cp.from`), `mkdir`, `tag` (`?tagging`), else `other`.
- Attributes:
  - `db.system.name="seaweedfs"`, `db.operation.name=<operation>`
  - `server.address`, `server.port`, `http.request.method`, `http.response.status_code`
  - `url.path.prefix`: first N path segments only (default 2), never the full path; query
    strings are never recorded
  - `seaweedfs.request.size` from `Content-Length` of the request, and
    `seaweedfs.response.size` from the response `Content-Length`, when stated
  - `error.type` on failure: exception class name or the status code, never error text
- Never recorded: file contents, filenames beyond the prefix, query values (they can carry
  `Seaweed-*` tag values), headers, credentials.
- Status: 5xx and transport errors are ERROR; 4xx are ERROR except 404 on stat/download,
  which is unset (an absent file is an answer, not a fault).
- Streaming bodies are never read or buffered by the instrumentor.

### 3.3 Interaction with the HTTP instrumentor

If httpx instrumentation is also on, a Filer call would produce two spans. The Filer
instrumentor must suppress or parent itself to avoid duplicates (decision: it replaces the
generic HTTP client span for declared Filer hosts). Test required.

## 4. Async S3 clients

| Client | Call path | Covered today | Plan |
| --- | --- | --- | --- |
| boto3 / botocore (sync) | `botocore.client.BaseClient._make_api_call` | Yes | keep |
| aiobotocore | `AioBaseClient(BaseClient)` overrides `_make_api_call` as **`async def`** (verified in upstream source) | **No**: the override replaces the patched method, so no span is produced | Add an async wrapper on `AioBaseClient._make_api_call`, sharing attribute logic with the sync path |
| aioboto3 | Thin wrapper over aiobotocore | No (same reason) | Covered by the aiobotocore wrapper |
| s3fs | Uses aiobotocore; a filesystem call fans out to several S3 operations | No | Covered by the aiobotocore wrapper; add a parent span only if Q4 says users want one `s3fs.*` span per file operation |

Requirements: the wrapper must be a coroutine that awaits the original, record the same
attributes and the same error handling (type only, no message), be removable by
`uninstrument()`, and be skipped silently when aiobotocore is not installed. Tests need a
fake `AioBaseClient`, not a network.

## 5. Suppressing the SDK's own media-store calls (optional)

The media-store fix (1.46.8 candidate) bounds the cost of an unreachable store, but a
failed or successful upload still emits `s3.*` spans into the user's trace.

Proposal: wrap the S3 media store's boto calls (`create_bucket`, `put_object`) in OpenTelemetry's
standard suppression context (`opentelemetry.context` `_SUPPRESS_INSTRUMENTATION_KEY`
set via `attach` / `detach`), and make the S3 instrumentor honour it by returning the
original call when the key is set. The key is OTel's own mechanism but its constant is in a
private-named module attribute; pin the import behind a small shim with a test that fails
if it moves.

Setting: `GENAI_OTEL_MEDIA_STORE_TRACE_CALLS` (default `false`, i.e. suppress). Setting it
to `true` keeps today's nested spans, which the multimodal guide currently documents, so
that page needs updating either way. This is a behaviour change, hence D4.

## 6. Configuration

| Name | Default | Meaning |
| --- | --- | --- |
| `GENAI_ENABLE_SEAWEEDFS_INSTRUMENTATION` | `true` | Filer instrumentor; inert without endpoints |
| `GENAI_OTEL_SEAWEEDFS_ENDPOINTS` | empty | Declared gateway/Filer `host:port` list |
| `GENAI_OTEL_SEAWEEDFS_PATH_PREFIX_SEGMENTS` | `2` | Path segments kept in `url.path.prefix` |
| `GENAI_OTEL_MEDIA_STORE_TRACE_CALLS` | `false` | Keep spans for the SDK's own uploads |

Naming follows the existing split: `GENAI_ENABLE_*` for feature toggles,
`GENAI_OTEL_*` for the media/storage settings. Each gets a field on `OTelConfig`.

## 7. Tests

- Unit, no network: operation derivation table (every Filer verb and query form); path
  prefix truncation and query stripping; 404 handling; endpoint matching (exact, default
  ports, wildcard, non-match); `server.port` present on S3 spans.
- Privacy: assert no body bytes, query values, full paths, credentials or error text appear
  on any span (extend the existing S3 privacy tests).
- Duplicate-span test with httpx instrumentation on.
- OTLP-exporter non-interference test for the `requests` wrapper.
- aiobotocore: fake async client, success, error, uninstrument, aiobotocore absent.
- Suppression: media upload produces no `s3.*` span by default and does with
  `TRACE_CALLS=true`; unrelated user S3 calls are still traced in the same context.
- Back-compat: `db.system="s3"` still present; old `s3.bucket` / `s3.object` still present.

## 8. Documentation plan

- New guide `docs/guides/seaweedfs.md` (add to `mkdocs.yml` nav) with endpoints setup,
  span reference and a Jaeger/Grafana filter example.
- Update `docs/guides/multimodal.md` (nested-span sentence), `docs/reference/semantic-conventions.md`
  (new attributes, deprecations), `docs/getting-started/configuration.md` (env table),
  `README.md` storage line.
- CHANGELOG entries under Added / Changed / Deprecated.

## 9. Decisions for the founder

- D1. Keep `db.system="s3"` and add `db.system.name="seaweedfs"` only on matched endpoints
  (recommended, non-breaking), or switch matched spans to `db.system="seaweedfs"`?
- D2. Is a custom `seaweedfs` value acceptable given the registry has none, or should we
  propose one upstream?
- D3. Offer an explicit opt-in port heuristic (8333/8888), or require declared endpoints only?
- D4. Suppress the SDK's own media-store spans by default (recommended), accepting a visible
  behaviour change from the documented nested spans?
- D5. Approve wrapping `requests.Session.send` for declared Filer hosts only, given the
  history of exporter breakage (Q2)?

## 10. Open questions

- Q1. Which Filer clients do target users actually run (`requests`, `httpx`, `aiohttp`, a
  FUSE mount, `pyseaweed`)? FUSE mounts and the Go/Java clients are invisible to this SDK.
- Q2. Can the `requests` wrapper be proven safe for OTLP exporters, or should Filer support
  ship httpx/aiohttp only first?
- Q3. Confirm on a running gateway and Filer which response headers are actually set;
  source reading found none useful.
- Q4. Should s3fs get a parent span per filesystem operation, or are per-S3-call spans enough?
- Q5. Does the gateway honour `x-amz-request-id` per request, so `aws.request_id` is usable
  for correlation with gateway logs?
