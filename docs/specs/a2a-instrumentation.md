# A2A (Agent2Agent) instrumentation (spec)

Status: **APPROVED 2026-11-12**. A2A-D1 to A2A-D7 are approved as recommended. It ships in **1.47.0**, together with
SeaweedFS and the Gemini Decisions API.

## 1. Problem

A2A reached 1.0 and moved to the Linux Foundation. It is the cross-vendor protocol agents use to discover one
another (the Agent Card), negotiate modalities, run tasks together and exchange messages, without exposing their
internals. This library records nothing A2A-specific today: no module, test or document mentions it.

The official Python SDK, `a2a-sdk` (1.3.0, 2026-10-27; spec 1.0 with a 0.3 compatibility mode), already makes
spans of its own through `a2a.utils.telemetry.trace_class`. These are on by default whenever OpenTelemetry is
installed, and `OTEL_INSTRUMENTATION_A2A_SDK_ENABLED=false` turns them off. Reading the 1.3.0 source:

- **Generic spans.** Spans are named `<module>.<Class>.<method>` and carry no A2A attributes: no method, task,
  context, message or agent.
- **Wrong span kinds on internal plumbing.** Internal classes are marked `SpanKind.SERVER`: the event queues, the
  event consumer and the in-memory queue manager. The client transports (JSON-RPC, REST, gRPC) are `CLIENT`, and
  the request handlers and dispatchers are `SERVER`. One request therefore produces many SERVER spans, so a
  backend that counts SERVER spans as requests over-counts.
- **No trace propagation between agents.** No module calls `opentelemetry.propagate.inject` or `extract`, or
  mentions `traceparent`. A call from agent A to agent B therefore starts a new trace in B unless an HTTP or gRPC
  instrumentation happens to carry the context. This is the gap described in A2A issue #2103.

OpenTelemetry merged A2A semantic conventions on 2026-10-27 (`semantic-conventions-genai` PR #195, development
status). Still open:

- metrics (PR #651);
- how to record which binding (JSON-RPC, gRPC or HTTP+JSON) carried a call (issue #612).

## 2. The conventions we follow (PR #195)

- **`a2a.client`** (CLIENT) is one logical client operation, including consuming a response stream.
- **`a2a.server`** (SERVER) is used only when no transport server span exists. Otherwise the A2A attributes
  **enrich the existing HTTP or gRPC server span** instead of nesting a new one.
- **Span name:** `{a2a.method.name} {target}`.
- **A2A attributes:**
  - `a2a.method.name`;
  - `a2a.protocol.version`;
  - `a2a.tenant`;
  - `a2a.message.id`;
  - `a2a.message.reference_task_ids`;
  - `a2a.task.id`;
  - `a2a.task.state`.
- **Reused attributes:**
  - `gen_ai.conversation.id` holds the A2A `contextId`;
  - `gen_ai.agent.name`, `gen_ai.agent.description` and `gen_ai.agent.version` come from the Agent Card;
  - `error.type`, `server.address` and `server.port`.
- **Multi-valued attributes:** a task or conversation attribute is omitted when one operation saw several
  distinct values, for example task states across stream events.

## 3. Proposal

1. **New instrumentor `a2a`**, which follows the `mcp_semconv` layout. It is a no-op when `a2a-sdk` is not
   installed or is older than 1.0. The 0.3 compatibility transports are covered only where they share the 1.x
   client and handler classes.
2. **Client side.** Wrap the `a2a.client.Client` / `BaseClient` operations:
   - `send_message`, including streaming;
   - `get_task`, `list_tasks` and `cancel_task`;
   - `subscribe`;
   - the push-notification config calls;
   - `get_extended_agent_card`.

   Each call emits one `a2a.client` span, and a streaming call ends the span when the stream ends. Attributes
   come from the request, the response and the resolved Agent Card. `server.address` and `server.port` come from
   the card URL.
3. **Server side.** Wrap the request handler (`DefaultRequestHandler`, v1 and v2, and the gRPC handler).
   - When a transport server span is current (the FastAPI, Starlette, ASGI or gRPC instrumentation), add the A2A
     attributes to it and rename it to `{a2a.method.name} {target}` per the convention, for HTTP only.
   - Otherwise open one `a2a.server` span.
4. **Trace propagation between agents.**
   - The client injects W3C `traceparent` / `tracestate` and baggage through the SDK's own interceptor hook
     (`ClientCallInterceptor`). For HTTP it uses headers; for gRPC, metadata.
   - The server extracts them when no transport instrumentation has already done so.
   - The result: agent A's call and agent B's work are one trace. `GENAI_A2A_PROPAGATE=false` turns this off.
5. **The SDK's own spans.** When our instrumentor is active and the user has **not** set
   `OTEL_INSTRUMENTATION_A2A_SDK_ENABLED`, we set it to `false` before `a2a` is imported. This stops the generic
   and mis-kinded spans from doubling every operation. A user who sets the variable keeps their choice. This is
   decision A2A-D2.
6. **Binding (issue #612).** Record the binding under a provisional name, `a2a.protocol.binding` (`jsonrpc`,
   `grpc` or `http_json`), marked provisional in the docs. If the convention settles on another name or on
   derivation, we follow it in a later release. This is decision A2A-D3.
7. **Content.** The existing content-capture rule applies.
   - **Always recorded** (not content): the method; task, message and context ids; the task state; the
     counts of message parts by kind (text, file, data); the artifact count; and the Agent Card name,
     description and version.
   - **Recorded only with content capture on:** message and artifact text, and data parts. Files are never
     inlined; they are counted and typed only.
   - **Card skills:** recorded by count, with names under content capture, because skill names can describe
     business processes. This is decision A2A-D4.
8. **Errors.** A JSON-RPC error, gRPC status or HTTP error sets the span to ERROR with `error.type`. The type is
   the A2A error name where one exists (for example `TaskNotFoundError`), otherwise the status code. A task that
   ends in `failed` or `rejected` sets `a2a.task.state` but does not on its own mark the span ERROR: the call
   succeeded and the agent's answer was a refusal or a failure. This is decision A2A-D5.
9. **Agent graph.** The `a2a.client` span gets `gen_ai.operation.name = invoke_agent` and the target agent's
   `gen_ai.agent.name`. Backends can then draw an edge from the calling agent to the called agent. That makes A2A
   calls visible in agent graphs, not only in waterfalls.
10. **Metrics.** None in this release. They follow PR #651 when it merges, so we don't publish names that
    change.
11. **Tests.** No network and no real servers are needed; fakes and the `a2a-sdk` test helpers are used where
    installed (importorskip). Cover:
    - client send, including streaming, get and cancel;
    - server enrichment of an existing span, and a standalone `a2a.server` span;
    - propagation round trip, with one trace across client and server;
    - `OTEL_INSTRUMENTATION_A2A_SDK_ENABLED` respected when the user has set it, and defaulted otherwise;
    - content capture off and on;
    - error mapping;
    - failed-task semantics;
    - no-op without `a2a-sdk`.

## 4. Decisions for approval

| | Question | Recommendation |
|---|---|---|
| A2A-D1 | Instrument the official Python `a2a-sdk` (1.x) only in this release | **Yes**. The Java, Go, JavaScript and .NET ports follow their own pins |
| A2A-D2 | Turn off `a2a-sdk`'s own generic spans by default when ours are active, unless the user set the variable | **Yes**. They double every operation and mark queue internals as SERVER |
| A2A-D3 | Record the binding as provisional `a2a.protocol.binding` while issue #612 is open | **Yes, marked provisional** |
| A2A-D4 | Content rule: ids, states, counts and card identity always; text, data and skill names only under content capture; files never inlined | **Yes** |
| A2A-D5 | A failed or rejected task is not by itself an ERROR span | **Yes**. The call worked; the agent said no |
| A2A-D6 | Propagate trace context between agents by default, with `GENAI_A2A_PROPAGATE=false` to opt out | **Yes**. This is the main value: one trace across agents |
| A2A-D7 | Ship in 1.47.0 with SeaweedFS and the Gemini Decisions API | **Yes**, per the founder |

## 5. Out of scope

- A2A metrics (PR #651) and an upstream proposal for the binding attribute, which belongs to the convention.
- Non-Python A2A SDKs.
- Agent Card signature or authentication details, which are not recorded.

## 6. Implementation notes and deviations (1.47.0)

Code: `genai_otel/a2a_semconv/` (`attributes.py`, `extract.py`, `instrumentor.py`); registered as
`a2a` in `INSTRUMENTORS` and in `DEFAULT_INSTRUMENTORS`; setting `a2a_propagate`
(`GENAI_A2A_PROPAGATE`). Tests: `tests/a2a_semconv/`, against the real `a2a-sdk` 1.3.0 over an
in-process ASGI app (the client's httpx transport runs the server in an empty context, so only the
propagated header can join the two sides). The SDK is in the `dev` extra and the tests
`importorskip` it.

What is wrapped, checked against the 1.3.0 source:

- Client: `a2a.client.base_client.BaseClient`: `send_message`, `subscribe` (async generators),
  `get_task`, `list_tasks`, `cancel_task`, `create_task_push_notification_config`,
  `get_task_push_notification_config`, `list_task_push_notification_configs`,
  `delete_task_push_notification_config`, `get_extended_agent_card`. `Client` itself is abstract.
  Propagation is an `a2a.client.interceptors.ClientCallInterceptor` whose `before()` merges the
  W3C headers into `ClientCallContext.service_parameters`, which the SDK turns into HTTP headers
  (`http_helpers.get_http_args`) and gRPC metadata (`GrpcTransport._get_grpc_metadata`).
- Server: `DefaultRequestHandlerV2` (`DefaultRequestHandler` is an alias of it) and
  `LegacyRequestHandler`: `on_message_send`, `on_message_send_stream`, `on_get_task`,
  `on_list_tasks`, `on_cancel_task`, the three push-config methods that exist plus
  `on_delete_task_push_notification_config`, `on_subscribe_to_task`, `on_get_extended_agent_card`.
  `GrpcHandler` and both HTTP dispatchers call these, so all three bindings are covered by one
  wrapper. The binding on a server span comes from `ServerCallContext.state`
  (`grpc_context` / `method` / `headers`).

Deviations and choices the spec left open:

1. **Span names.** Spans are named `{a2a.method.name} {target}` per section 2, for both client and
   server. `a2a.client` / `a2a.server` are the convention's names for the two kinds of span, not
   literal span names. The instrumentation scope is `genai_otel.a2a`.
2. **Baggage goes through the existing session-baggage policy** (`SessionBaggagePropagator`), not
   a blanket W3C baggage inject: baggage crosses a trust boundary here, and `session.id` /
   `user.id` leave only when named in `GENAI_SESSION_BAGGAGE_KEYS` and are accepted only when
   named in `GENAI_SESSION_BAGGAGE_ACCEPT`. Other baggage passes as the stock propagator would.
   Trace context uses the explicit W3C propagators, not whatever the global propagator is.
3. **Attribute names the spec left open** (all `a2a.*`, marked as genai_otel additions in the
   docs): `a2a.message.{text,file,data}_part_count`, `a2a.response.{text,file,data}_part_count`,
   `a2a.artifact.count`, `a2a.agent.skill_count`, `a2a.message.file_media_types`,
   `a2a.response.file_media_types`; content keys `a2a.message.text`, `a2a.message.data`,
   `a2a.response.text`, `a2a.response.data`, `a2a.agent.skill_names`. File media types are
   recorded without content capture (they are type, not content); file bytes, URLs and names never.
4. **Task state values** are lower-case without the proto prefix (`input_required`).
5. **Multi-valued rule applies to streams literally**: a normal stream passes through
   `submitted`, `working`, `completed`, so `a2a.task.state` is omitted on it, as the spec says.
   Task id and context id stay when single. Unary calls and single-state streams carry the state.
6. **Enabled by default** (in `DEFAULT_INSTRUMENTORS`), since it is a no-op without `a2a-sdk` 1.x.
7. **The propagation interceptor is attached on the first call of each client** (so clients built
   before `instrument()` are covered), not by wrapping `BaseClient.__init__`. It edits a copy of the
   caller's `ClientCallContext`, never the caller's object.
8. **The SDK's own spans.** `a2a.utils.telemetry` reads `OTEL_INSTRUMENTATION_A2A_SDK_ENABLED` once
   at import and swaps its tracer for a no-op. We set the default only while that module is not yet
   in `sys.modules`; when it is and its spans are on, one INFO line says so and nothing is hacked
   (the module's `trace` global could be patched, and is deliberately not). In that mode the SDK's
   SERVER spans are recognised by their scope (`a2a-python-sdk`) and are never mistaken for the
   transport span; trace-context extraction on the server then needs transport instrumentation,
   because the SDK's span is already current.
9. **Packaging.** `genai_otel.a2a_semconv` was added to `[tool.setuptools] packages`.
   `genai_otel.mcp_semconv` (and `genai_otel.voice`) are not in that list; that predates this change
   and was left alone, but it should be checked before the release is built.

Not verified: gRPC end to end (no server is started; the metadata read, the binding and the
no-rename rule are tested against a stand-in `ServicerContext`, and the interceptor's output is the
same `service_parameters` the SDK's gRPC transport turns into metadata); `a2a-sdk` versions other
than 1.3.0; the 0.3 compatibility transports beyond the class-name binding match; a client built on
`Client` subclasses other than `BaseClient`.
