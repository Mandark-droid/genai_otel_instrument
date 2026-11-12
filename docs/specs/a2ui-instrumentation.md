# A2UI (Agent-to-UI) instrumentation (spec)

Status: **APPROVED 2026-11-12**. A2UI-D1 to A2UI-D11 are approved as recommended. The Python part ships in
**1.48.0**. The Node part covers Node, browser and React Native applications and ships as the next npm release of
`@traceverse/genai-otel-instrument-node` (after 0.1.0-alpha.10). Correction 2026-11-12: the Node port is published
on npm, so it gets a release too; its contract is pinned to v1.46.0, not v1.30.0.

## 1. Problem

A2UI is Google's open protocol (Apache 2.0) for an agent to send a user interface as declarative JSON instead of
code. The agent describes surfaces and components from a catalog the client trusts. The client draws them with its
own native components (React, Lit, Angular, Flutter, SwiftUI, Jetpack Compose). The client sends user actions back
to the agent. v1.0 is the current protocol version on a2ui.org; v0.9.1 and v0.9 are previous, and v0.8 is
deprecated.

This library records nothing A2UI-specific today. A2UI usually travels inside A2A messages, so 1.47.0's A2A spans
see it, but only as an anonymous `data` part: the span cannot say that a UI was sent, which surface or catalog it
used, what the user did with it, or that the client rejected it. The protocol defines no telemetry of its own.

What matters for observability, in order:

1. **The agent produced UI the client rejected.** Renderer `error` messages (`VALIDATION_FAILED`,
   `UNALLOWED_PARENT`, `UNALLOWED_CHILD`, function errors) mean the user saw a broken or missing interface while
   every server-side span looked healthy.
2. **The model produced UI the agent SDK could not parse or validate.** The agent SDK drops the response, so the
   user sees nothing. This is a model-quality signal: invalid output from the LLM.
3. **What the user did.** An `action` message is the user's click or submit. It starts the agent's next turn and
   belongs in the same trace as the UI that offered it.
4. **What was shown.** The surface, the catalog, the protocol version and the size of the update.

## 2. The protocol, as read from the sources

Sources: the v1.0 specification and the A2A extension specification on a2ui.org, and the Python packages
`a2ui-agent-sdk` 0.11.0, `a2ui-a2a` 0.3.0 and `a2ui-core` 0.6.0 (all read from their published wheels).

- **Agent-to-renderer messages.** Each message has exactly one of `createSurface`, `updateComponents`,
  `updateDataModel`, `deleteSurface`, `callRendererFunction` or `agentFunctionResponse`, and a top-level
  `"version": "1.0"`. v0.9 messages have the same shape with their own version; v0.8 messages have no `version`.
- **Renderer-to-agent messages.** Each has exactly one of `action`, `callAgentFunction`,
  `rendererFunctionResponse` or `error`.
- **Identity.** `surfaceId` on surface messages. `catalogId` on `createSurface` (the surface default), `@catalogId`
  on a component that overrides it, and `catalogId` on function calls. These are string identifiers, not URLs.
- **Errors.** A validation error is `{"error": {"code", "surfaceId", "path", "message"}}`. Function responses carry
  `error.code` and `error.message` (`INVALID_FUNCTION_CALL`, `EXECUTION_ERROR`, `EXPRESSION_ERROR`,
  `INTEGRITY_ERROR`).
- **Over A2A.**
  - The extension URI is `https://a2ui.org/a2a-extension/a2ui/v1.0`. It is activated through the
    `X-A2A-Extensions` header (JSON-RPC and HTTP) or the gRPC `metadata["X-A2A-Extensions"]`. Activation is
    optional.
  - An A2UI part is a `DataPart` whose `metadata["mimeType"]` is `application/a2ui+json`. v0.8 and v0.9 use the
    deprecated `application/json+a2ui`, and receivers must accept both.
  - **The spec and the SDK disagree on the part payload.** The spec says the `data` field MUST be an array of
    messages, processed in order and not as a transaction. The Python SDK writes one message per `DataPart`
    (`create_a2ui_part`). Both shapes must be read.
  - The renderer advertises its catalogs in `message.metadata["a2uiRendererCapabilities"]["1.0"]["supportedCatalogIds"]`.
    The Python SDK's constant is `a2uiClientCapabilities`, and both keys must be read.
- **Other transports.** The agent SDK also integrates with Google ADK and MCP, and leaves the transport to the
  application.
- **A version trap.** `a2ui-a2a` 0.3.0 requires `a2a-sdk>=0.3.0,<0.4.0`. 1.47.0's A2A instrumentor covers
  `a2a-sdk` 1.x only and is a no-op on 0.3. An agent built on the official A2UI-over-A2A bridge therefore gets
  **no** A2A spans today, so A2UI coverage cannot depend on the A2A instrumentor alone.

### 2.1 The SDKs a2ui.org lists, and what covers each

| Listed SDK | Side | Covered by |
|---|---|---|
| `a2ui-agent-sdk` (PyPI, 0.11.0; protocol v0.8, v0.9.1, v1.0) | Agent | 3.2 |
| `a2ui-a2a` (PyPI, 0.3.0; pins `a2a-sdk` 0.3) | Agent, A2A bridge | 3.2 (its parts come from the agent SDK). 3.1 does not apply, because that needs A2A 1.x (A2UI-D2) |
| `a2ui-adk` (PyPI, 0.3.0; Google ADK) | Agent | 3.2 (it calls the agent SDK's processor) |
| `@a2ui/agent` (npm, 0.1.0, 2026-11-02; the TypeScript port of the agent SDK, same `A2uiRequestProcessor` / `Parser` surface) | Agent, Node | 4.0 |
| `@a2ui/web_core` (npm, 0.15.0) with `@a2ui/react`, `@a2ui/lit`, `@a2ui/angular` | Renderer, web | 4.3, plus 4.1 and 4.2 for the transport |
| `a2ui_agent` (Dart) and Flutter | Agent and renderer | Out of scope: no Dart port |
| SwiftUI, Jetpack Compose renderers, legacy Kotlin agent SDK | Renderer / agent | Out of scope: no Swift port. The Java port has no A2A or A2UI support (a later spec) |
| Go community libraries (`satorunooshie/a2ui-go`, `tmc/a2ui`) and terminal renderers | Agent / renderer | A2UI-D9 |

## 3. Proposal: Python (1.48.0)

### 3.1 Recognise A2UI inside A2A spans (a2a-sdk 1.x)

Extend `genai_otel/a2a_semconv/extract.py`. When a data part carries either A2UI media type (in
`metadata["mimeType"]` or in the part's `media_type`), parse its payload as an A2UI message or list of messages and
add the `a2ui.*` attributes of section 5 to the existing A2A span:

- the request side for what the renderer sent;
- the response side, including stream events and artifacts, for what the agent sent.

This needs no new span. A2UI parts still count in the A2A `data` part counts, unchanged. The extension URI in
`X-A2A-Extensions`, when present, sets `a2ui.extension.active=true`.

### 3.2 Instrument the agent SDK (`a2ui-agent-sdk`, any transport)

This works whether the agent speaks A2A 0.3, A2A 1.x, ADK, MCP or something else.

- **`A2uiRequestProcessor.parse_response`** gets one INTERNAL span, `a2ui.parse_response`. It records the
  messages produced and their attributes.
  - If parsing or validation raises, the span is ERROR with `error.type` set to the exception class.
  - `a2ui.validation.outcome` is `valid` or `invalid`.
  - Under the active LLM span when there is one, this is where "the model wrote broken UI" becomes visible.
- **`A2uiRequestProcessor.open_stream`** gets one span, `a2ui.parse_stream`, which covers the whole stream. Its
  attributes are the totals at the end.
- **`RendererToAgentMessageProcessor.process_message`** gets one INTERNAL span,
  `a2ui.process {message type}` (for example `a2ui.process action`). It records:
  - the incoming message's attributes;
  - the reply count;
  - renderer `error` messages, by code.
- **`parse_content_to_parts`** catches every exception and logs only its class. Our wrapper records it as
  `a2ui.validation.outcome=invalid` on the current span. It changes neither the return value nor the logging.

No-op when `a2ui-agent-sdk` is not installed or is older than 0.11. Protocol versions v1.0 and v0.9 are read; v0.8
is recognised by media type and counted, but its message shapes are not parsed (it is deprecated).

### 3.3 What is not instrumented in Python

- No renderer. Python has no A2UI renderer.
- The prompt-building parts of the agent SDK (schema prompts, catalog pruning, examples). They run before the LLM
  call and the LLM span already records the prompt under content capture.

## 4. Proposal: Node, browser and React Native

The Node SDK is the port for Node services, browsers and React Native apps. In a mobile or web app the app is the
**renderer**: it receives the agent's UI and sends the user's actions. Three pieces are needed.

### 4.0 The TypeScript agent SDK (`@a2ui/agent`)

`@a2ui/agent` 0.1.0 is the TypeScript port of the Python agent SDK. It exports the same `A2uiRequestProcessor`,
`Parser` and response-stream classes, and its error classes (`A2uiValidationError`, `ParseError`, the `Express*`
errors). An agent written in Node gets the same spans as section 3.2:

- `a2ui.parse_response`;
- `a2ui.parse_stream`, which is ERROR on a parse or validation failure, with `error.type` set to the error class.

The exact method names are checked against the 0.1.0 sources at implementation time. 0.1.0 exports no
renderer-to-agent processor, so the `a2ui.process` span is Python-only until the TypeScript SDK adds one. The Node
agent's A2A **server** spans remain A2UI-D6.

### 4.1 A2A client instrumentation for `@a2a-js/sdk` (prerequisite)

The Node SDK has no A2A support today. A renderer talks to its agent over A2A, so without this there is no span for
the call, no A2UI attributes on it, and no trace from the user's tap to the agent's work.

- Port the client half of the 1.47.0 Python design to `@a2a-js/sdk` 1.x: `a2a.client` spans, the same
  attributes, W3C trace-context injection through the SDK's interceptor or fetch hook, and the same content rule.
- **The server half (a Node agent) is in scope only if A2UI-D6 says so.**

### 4.2 Wire-level A2UI recognition, independent of the renderer

The renderer landscape is fragmented. The official ones are React, Lit and Angular (sharing `@a2ui/web_core`),
plus Flutter, SwiftUI and Compose. Community ones include two React Native renderers
(`a2ui-react-native-renderer` 1.0.0 on npm for v1.0, and `sivamrudram-eng/a2ui-react-native` for v0.8), Vue,
Svelte, Lynx and terminal renderers. Patching each renderer does not scale. Instead:

- **A2A client spans** parse A2UI parts the same way as section 3.1. On the request, they record the user action
  being sent and the renderer capabilities. On the response, they record what the agent sent. This alone covers any
  renderer that talks to its agent through `@a2a-js/sdk`.
- **A small public API** for apps whose transport we do not see (AG-UI, a custom WebSocket, MCP Apps):
  `recordA2uiReceived(messages)`, `recordA2uiSent(message)` and `recordA2uiRendererError(error)`. Each adds the
  attributes of section 5 to the current span, or opens a short INTERNAL span when none is active. All three are
  exported from the Node, browser and React Native entry points.

### 4.3 Renderer adapter for `@a2ui/web_core` (official web renderers)

`@a2ui/react`, `@a2ui/lit` and `@a2ui/angular` share `@a2ui/web_core` (0.15). An opt-in adapter wraps its message
processor:

- it records one `a2ui.apply` span per batch of agent messages applied, with the duration until the batch is
  processed;
- it records renderer-side validation errors as span events with their code, even when the app never sends the
  `error` message back to the agent.

The community React Native renderers do not depend on `web_core`. For them the A2A path in 4.1 and 4.2 and the API
in 4.2 apply. A dedicated adapter for `a2ui-react-native-renderer` is decision A2UI-D7.

### 4.4 Porting rule

The Node port vendors the Python contract (semantic conventions, configuration, pricing) from one pinned Python
tag, with SHA checks, and its parity matrix tracks every difference. The A2UI and A2A attribute names are added to
the vendored contract **from the 1.48.0 tag**, not typed in by hand. Re-pinning the whole contract from v1.46.0 to
v1.48.0 also brings in 1.46.x-1.47.0 changes unrelated to A2UI. A2UI-D8 decides between a full re-pin and a scoped addition recorded in
`SDK_ADDITIONS.md`.

## 5. Attributes

All names are `a2ui.*` additions of this library and are marked provisional: there is no OpenTelemetry convention
for A2UI. Multi-valued rules follow 1.47.0's A2A rules: a single-valued identifier is omitted when one operation saw
several distinct values, and lists are sorted and de-duplicated.

**Always recorded.** These are identifiers, counts and codes, not content:

| Attribute | Meaning |
|---|---|
| `a2ui.protocol.version` | `1.0`, `0.9`, `0.8` (single value; omitted if mixed) |
| `a2ui.agent_message.count` | Agent-to-renderer messages in the operation |
| `a2ui.renderer_message.count` | Renderer-to-agent messages in the operation |
| `a2ui.message.types` | Sorted distinct message types, snake_case (`create_surface`, `update_components`, `action`, `error`, ...) |
| `a2ui.surface.id` / `a2ui.surface.count` | The surface, when single; the distinct surface count |
| `a2ui.catalog.id` / `a2ui.catalog.ids` | The catalog when single; all catalogs referenced |
| `a2ui.component.count` | Components in `updateComponents` messages |
| `a2ui.renderer.supported_catalog_ids` | From the renderer capabilities metadata |
| `a2ui.action.name` | The user action's name, when single (see A2UI-D3) |
| `a2ui.function.name` | For `callAgentFunction` / `callRendererFunction`, when single |
| `a2ui.error.codes` / `a2ui.error.count` | Renderer and function error codes; their count |
| `a2ui.validation.outcome` | `valid` / `invalid` (agent-side parsing, section 3.2) |
| `a2ui.extension.active` | The A2A extension URI was negotiated |

**Recorded only with content capture on:**

- `a2ui.messages`, the JSON of the messages, truncated at the existing content limit;
- the data-model values in `updateDataModel`;
- the action's `context`;
- the error `path` and `message`, because a validation message can quote user data;
- the component type names (see A2UI-D3).

Never recorded: the content of image or file URLs inside components.

**Span status.**

- A renderer `error` message or a function error **does not** by itself make the A2A span ERROR. The call carrying
  it succeeded; this follows A2A-D5. It sets `a2ui.error.codes`.
- The agent-side `a2ui.parse_response` span **is** ERROR on a parse or validation failure, because that operation
  failed.
- The Node `a2ui.apply` span is ERROR when the renderer rejected the batch.

**Span events.** One event `a2ui.error` per renderer error, with `a2ui.error.code` and `a2ui.surface.id`, so each
error is visible even when one operation carries several.

## 6. Configuration

- `GENAI_A2UI_ENABLED` (default `true`) turns the A2UI attributes and spans off without touching A2A.
- The instrumentor `a2ui` is added to `DEFAULT_INSTRUMENTORS`. It is a no-op without `a2ui-agent-sdk`. The A2A
  enrichment of section 3.1 lives in the `a2a` instrumentor and follows `GENAI_A2UI_ENABLED`.
- The existing content-capture setting governs everything in section 5's second list. There is no separate A2UI
  capture switch.

## 7. Tests

No network and no real servers are used. The optional libraries are imported with `importorskip`, since CI installs
only `.[dev]`; `a2ui-agent-sdk` is added to the `dev` extra.

- **Python.**
  - A2A 1.x parts carrying A2UI in both shapes (one message per part, list per part), under both media types.
  - Request and response sides, and streaming.
  - `parse_response` valid and invalid, including an LLM span as parent.
  - `open_stream` totals.
  - `process_message` for `action`, `callAgentFunction` and `error`.
  - Content capture off and on.
  - Multi-value omission.
  - `GENAI_A2UI_ENABLED=false`.
  - No-op without the package.
  - v0.8 counted but not parsed.
- **Node.**
  - `@a2a-js/sdk` client spans and trace injection.
  - A2UI parsing on request and response.
  - The public API under each entry point (Node, browser bundle, React Native bundle).
  - The `web_core` adapter against `@a2ui/web_core` 0.15.
  - Content rules.

## 8. Decisions for approval

| | Question | Recommendation |
|---|---|---|
| A2UI-D1 | Python: enrich A2A 1.x spans (3.1) **and** instrument `a2ui-agent-sdk` directly (3.2), so agents on A2A 0.3, ADK or MCP are covered | **Yes**. The official A2UI-over-A2A bridge pins `a2a-sdk` 0.3, which our A2A instrumentor does not cover |
| A2UI-D2 | Do **not** extend the 1.47.0 A2A instrumentor to `a2a-sdk` 0.3 in this release | **Yes**. The agent-SDK spans cover the A2UI signal; A2A 0.3 support is its own change if wanted |
| A2UI-D3 | Content rule: ids, counts, types of message, catalog ids, action and function names, error codes always; payloads, data-model values, action context, error path and message, and component type names only under content capture | **Yes**. Action names are developer-defined identifiers, like span names. Component type names in a custom catalog can describe a business process, as skill names do (A2A-D4) |
| A2UI-D4 | A renderer error does not by itself make the carrying A2A span ERROR; an agent-side parse or validation failure does make its own span ERROR | **Yes** |
| A2UI-D5 | Node: add A2A **client** instrumentation for `@a2a-js/sdk` as part of this work | **Yes**. Without it a mobile or web app gets no A2UI telemetry and no trace from tap to agent |
| A2UI-D6 | Node: A2A **server** instrumentation (agents written in Node) | **Later**. Apps are clients; Node agents can follow |
| A2UI-D11 | Node: instrument the TypeScript agent SDK `@a2ui/agent` (4.0), not only renderers | **Yes**. It is the official agent SDK for Node and costs one wrapper on the same two methods as Python |
| A2UI-D7 | Node: renderer adapters. The `@a2ui/web_core` adapter now; React Native through the A2A path and the public API, with no adapter for a community renderer | **Yes**. Revisit an `a2ui-react-native-renderer` adapter if a customer standardises on it |
| A2UI-D8 | Node contract: add the A2A and A2UI names as a scoped addition from the v1.48.0 tag (recorded in `SDK_ADDITIONS.md` and the parity matrix), not a full re-pin from v1.46.0 | **Scoped addition**. A full re-pin pulls in unrelated 1.46.x-1.47.0 changes |
| A2UI-D9 | Go: not in this release. Go's A2UI libraries are community agent libraries at v0.9 and two terminal renderers, and the Go port has no A2A support yet | **Later**, as a separate spec covering A2A and A2UI for Go together |
| A2UI-D10 | Versions: Python ships as **1.48.0**, so the parked price-identity release moves to the next minor. The Node work ships as the next npm alpha of the Node port (corrected 2026-11-12: it is published on npm) | **Yes** |

## 9. Out of scope

- Flutter, SwiftUI and Jetpack Compose renderers. They need Dart, Swift and Kotlin instrumentation, and this
  library has no Dart or Swift port.
- Metrics. Spans only, as with A2A.
- Rendering performance beyond the `web_core` adapter's batch duration (frame timing, layout).
- Proposing A2UI conventions upstream to OpenTelemetry. This is worth doing once the attributes have been used.

## 10. Implementation notes and deviations (1.48.0)

The Python part (sections 2, 3, 5, 6 and 7-Python; A2UI-D1 to D4 and D10) as built. Checked against
`a2ui-agent-sdk` 0.11.0 and `a2ui-core` 0.6.0 as installed, and `a2a-sdk` 1.3.0. The Node part is not
in this change.

Layout: `genai_otel/a2ui_semconv/` holds `attributes.py` (names, media types, constants),
`extract.py` (the shared reader, an `A2uiObservation` over wire-format dicts; it imports neither `a2a`
nor `a2ui`) and `instrumentor.py` (`A2UIInstrumentor`, the agent-SDK wrappers). The A2A path adds an
optional `a2ui` accumulator to `a2a_semconv.extract.Observation`, shared by a call's request and
response, so a single-valued key seen as two values across the two sides is omitted.

Signatures checked against the installed package, and wrapped as follows:

- `A2uiRequestProcessor.parse_response(self, content, wrapped=True, *, state=None)` returns
  `list[ResponsePart]`; `A2uiPart.a2ui` is a list of pydantic schema models, read through
  `model_dump(mode="json", by_alias=True, exclude_unset=True)` with the model's `version` put back (a
  v1.0 model leaves `version` unset, so a plain dump drops it).
- `A2uiRequestProcessor.open_stream(self, *, wrapped=True, state=None, after_rejected_block="end")` is
  **synchronous** and returns a `ResponseStream` whose `feed(chunk)` and `close()` are synchronous. The
  span starts in `open_stream` (a child of the current span, but **not made current**, because it outlives
  the call) and ends when `close()` returns or `feed` / `close` raises. The two methods are replaced on the
  stream instance. A stream that is never closed ends its span when it is garbage collected.
- `RendererToAgentMessageProcessor.process_message(self, message)` is async and returns `ProcessResult(handled,
  replies)`. Its replies go back to the renderer, so they are counted as **agent** messages.

Deviations and choices the spec left open:

1. **`parse_content_to_parts` lives in `a2ui-a2a`, not in `a2ui-agent-sdk`.** `a2ui/a2a/parts.py` imports
   `a2a.types.DataPart`, which `a2a-sdk` 1.x does not have, so it cannot be imported beside the 1.x the A2A
   tests need. It is wrapped only when `a2ui.a2a.parts` imports. Because the function swallows the
   exception inside itself, the wrapper cannot see it; it notices the `Failed to parse A2UI response: <class>`
   warning the function logs (a log filter that only reads, installed on the `a2ui.a2a.parts` logger and
   keyed to a context variable that is set only while the wrapper runs). Nothing is suppressed or changed.
   Consequence: it is missed if that logger is set above WARNING. The failure is recorded as
   `a2ui.validation.outcome=invalid` on the current span plus an event `a2ui.validation_failed` carrying
   `error.type`; `error.type` is not set as an attribute and the span is not made ERROR, because the
   operation did not fail. **Not verified against the real module**, only against a stand-in with the same
   name, logger and message, taken from the 0.3.0 source.
2. **Rejected block in a continuing stream.** A stream opened with `after_rejected_block="continue"` returns
   an `ErrorPart` instead of raising. `a2ui.parse_stream` is ERROR for it too (a block failed validation),
   with `error.type` set to the error's class and the outcome `invalid`.
3. **`a2ui.process` only sets `invalid` on a validation exception.** The SDK does not validate a message
   that has no callback, so a successful call does not claim `valid`. An unrecognised message names its
   span `a2ui.process unknown`.
4. **`a2ui.component.count` also counts `createSurface.components`.** The v1.0 schema lets `createSurface`
   carry components, so counting only `updateComponents` would undercount.
5. **Extension and header names.** `a2a-sdk` 1.x renamed the header to `A2A-Extensions`; the spec text says
   `X-A2A-Extensions`. Both names are read, from the client's `service_parameters`, the server's
   `requested_extensions` and request headers, and `Message.extensions` / `Artifact.extensions`. Any URI
   under `https://a2ui.org/a2a-extension/a2ui/` matches, so a v0.8 or v0.9 extension URI also sets
   `a2ui.extension.active`. The attribute is written only when true.
6. **Content attribute names the spec left open** (provisional, `a2ui.*`): `a2ui.messages`,
   `a2ui.data_model`, `a2ui.action.context` (JSON strings), `a2ui.error.paths`, `a2ui.error.messages`,
   `a2ui.component.types` (string lists). Each is dropped with content capture off, and nothing of the
   payload is held in memory then. The event `a2ui.validation_failed` is the one event name beyond
   `a2ui.error`.
7. **Media URLs are scrubbed.** Inside the components of `createSurface` / `updateComponents`, the value of a
   key named `url`, `src`, `uri` or ending in `Url` / `Uri` becomes `[redacted]` (a `{"@path": ...}` binding is
   kept, since it names where a value lives); a string starting `data:` is redacted anywhere. The existing
   `a2a.*.data` content attribute is also written from the scrubbed copy for A2UI parts, but only when the
   A2UI reader is on; with `GENAI_A2UI_ENABLED=false` that attribute is exactly what 1.47.0 wrote.
8. **v0.8 is detected by the missing `version` field**, counted, and its content is not recorded.
   Versions are normalised to `major.minor` (`v0.9.1` is `0.9`).
9. **`a2ui.renderer.supported_catalog_ids`** is the union over the per-version map, and over either metadata
   key, and over `message.metadata` and `SendMessageRequest.metadata`.
10. **Packaging and tests.** `genai_otel.a2ui_semconv` is in `[tool.setuptools] packages`. `a2ui-agent-sdk>=0.11,<1`
    is in the `dev` extra; `a2ui-a2a` is deliberately not (it would downgrade `a2a-sdk`). There is no
    optional extra for A2A, so none was added for A2UI. The tests `importorskip` the SDK and use the real
    processors; the A2A path runs over the same in-process ASGI rig as the A2A tests.

Not verified: the real `a2ui.a2a.parts` module (see 1); `a2ui-agent-sdk` versions other than 0.11.0; the
A2UI extension header on gRPC; Google ADK and MCP transports (the wrappers are transport-independent, and
are tested only through the processors).
