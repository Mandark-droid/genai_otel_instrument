# Gemini Decisions API instrumentation (spec)

Status: **APPROVED 2026-11-12**. GD-D1 to GD-D5 are approved as recommended. It ships in **1.47.0** together with the
SeaweedFS change.

## 1. Problem

Google released the Gemini Decisions API on 2026-11-11, with the model `gemini-decision-1`. A
decision call takes a shared `input` plus up to 1,000 named, typed questions, and returns a
probability or a distribution for each one. It writes no text.

The library already emits `DECISION` spans (`genai_otel/instrumentors/decision.py`) for two wire
protocols: OpenAI Decisions (`POST /v1/decisions`) and SystemOne (`POST /v1/systemone`). A
Gemini decision call is not covered today:

- `google-genai`'s `client.decisions.create` is not wrapped. The Google instrumentor wraps
  only `models.generate_content` and the embeddings calls.
- The plain-HTTP matcher (`decision_http.py`) recognises `.../v1/decisions` and `.../v1/systemone`.
  Gemini's REST path is `POST https://generativelanguage.googleapis.com/v1beta/decisions`. It
  does not match `/v1/decisions`, so a REST call gets no span, or only a generic HTTP span.
- `llm_pricing.json` has no `gemini-decision-1` entry, so cost is unpriced.

## 2. The API, as documented

Source: https://aistudio.google.com/docs/decisions-api (read 2026-11-11).

- **Request:**
  - `model`;
  - `input`, which is a string, a `Content`, or a `Content[]`. Content blocks are `text`,
    `image`, `audio`, `video` and `document`, each carried as `data` or `uri`;
  - `questions`, a map from a name to `{type, instructions, criteria?}`.
- **Question types:**
  - `noul` (yes/no, with optional `true_description` / `false_description`);
  - `choice` (`criteria` maps option keys to descriptions, 2-255 options);
  - `score` (`criteria` is an ordered list of levels, 2-255).
- **Response:** `answers` is a map keyed by question name.
  - `noul` answers carry `noul` (0.01-0.99) and `confidence`.
  - `choice` answers carry `choice`, `probabilities` and `confidence`.
  - `score` answers carry `score` (the probability-weighted level index), `legend`,
    `probabilities` and `confidence`.
- **Usage:**
  - `total_input_tokens`;
  - `input_tokens_by_modality` (`[{modality, tokens}]`);
  - `total_output_tokens`, which is always 0;
  - `total_tokens`.
- **Errors:** HTTP 400 with `error.code`, which is one of `safety`, `prohibited_content`,
  `blocklist`, `image_safety` or `content_blocked`, and no `answers`. Limit violations also
  return 400. The API has no per-question refusal.
- **SDKs:**
  - Python `google-genai` 3.2 or later: `genai.Client(http_options={"api_version": "v1beta"}).decisions.create(...)`.
    An async variant is not documented.
  - JavaScript: `ai.decisions.create(...)`.
- **Pricing:** $0.10 per 1M input tokens and $0 for output. The shared `input` is billed once
  per request.

The shape is SystemOne's: a map of named questions (`noul` / `choice` / `score`) and a map of
named answers. Only the usage block and the error shape differ.

## 3. Proposal

1. **Protocol.** Add `PROTOCOL_GEMINI_DECISIONS = "gemini_decisions"`.
   - Questions and answers are parsed by the existing SystemOne parsers (`systemone_questions`,
     `systemone_answers`), extended only where a field name differs. `noul` maps to the existing
     `noul` handling.
   - A new `gemini_usage` reads `total_input_tokens` as input and `total_output_tokens` as output.
     When the response carries `input_tokens_by_modality`, the per-modality input counts are
     recorded as `decision.token_count.input.<modality>`. That is a new attribute; see D3.
2. **SDK path.** The Google instrumentor wraps `client.decisions.create`, and the async variant
   when one exists, through the existing `BaseDecisionInstrumentor` and `DecisionRecorder`. One
   decision is one span, of kind `DECISION`.
   - `decision.system = "google"`, matching the instrumentor's `gen_ai.system`.
   - `decision.provider = "google"`.
   - The operation is `google.genai.decisions`.
   - The instrumentor stays a no-op when the installed `google-genai` has no `decisions`
     attribute (older than 3.2).
   - `GENAI_ENABLED_INSTRUMENTORS` keeps its meaning: the Google instrumentor owns these calls,
     as the OpenAI instrumentor owns OpenAI decisions.
3. **REST path.** `decision_http.py` matches `POST` to a path ending in `/v1beta/decisions` or
   `/v1/decisions`, only on host `generativelanguage.googleapis.com`. Those requests are parsed as
   `gemini_decisions`. An OpenAI-compatible server on any other host keeps today's behaviour.
   - A request made from inside a decision span this library already opened is left alone, as
     today.
   - `generativelanguage.googleapis.com` is added to the provider host map as `google`.
4. **Errors.** A 400 with an `error.code` sets the span status to ERROR, with
   `error.type = <error.code>`. The span records no answers and no refusals, because the API has
   no per-question refusal. Question counts are still recorded from the request.
5. **Content.** The existing rule is unchanged. With content capture off, the span records:
   - which model, through which provider;
   - how many questions, of which types;
   - tokens and cost.

   Chosen answers, probabilities, confidence and question names are recorded only with content
   capture on. Media in `input` is replaced by the existing placeholders.
6. **Pricing.** Add `decision.gemini-decision-1` at $0.10/1M input and $0 output, citing the
   Google source. The chat table gets no entry, so the model is never priced as chat.
7. **Tests.**
   - SDK path: sync, plus async if it exists, using a fake client, so that no network and no
     `google-genai` install is needed (importorskip if real types are used).
   - REST path: requests and httpx.
   - Each of the three question types, with content capture both off and on.
   - The 400 safety error.
   - The `inside_decision_span` de-duplication.
   - The host gate: `/v1beta/decisions` on another host is not treated as Gemini.
   - The pricing lookup.

## 4. Decisions for approval

| | Question | Recommendation |
|---|---|---|
| GD-D1 | Reuse the SystemOne parsers under a new protocol id, rather than writing a separate parser | **Yes**. The shape is the same |
| GD-D2 | REST matching only on `generativelanguage.googleapis.com` | **Yes**. A path-only match would misread other servers' `/v1beta/decisions` |
| GD-D3 | Record the per-modality input tokens as a new `decision.token_count.input.<modality>` attribute | **Yes, behind content capture off**. Counts are not content, and they explain cost on multimodal inputs |
| GD-D4 | A safety block is an ERROR span, not refusals | **Yes**. The API returns no answers |
| GD-D5 | Ship together with the SeaweedFS change in one minor release | **Yes**, per the founder's request |

## 5. Out of scope

- Vertex AI (Gemini in Google Cloud). Google says support is "coming soon"; the gap is
  recorded here as a follow-up.
- The JavaScript SDK. The Python library cannot see it.
- Upstream semantic-convention proposals for the `noul` type.

## 6. Implementation notes

Implemented as specified. Details and deviations:

- **Usage reader.** `gemini_decision_usage` (in `decision.py`) reads the Gemini block. `usage_of`
  delegates to it whenever the block carries `total_input_tokens` or `total_output_tokens`, so the
  base wrapper's cost and token metrics need no Gemini-specific hook. `DecisionUsage` gained
  `input_tokens_by_modality` (modality names lower-cased, enum values unwrapped).
- **Parsers.** Answers go through `systemone_answers` unchanged (`noul` already maps to the
  predicate type); no parser change was needed.
- **Error code on the SDK path.** The API's `error.code` is read from the exception's `details` (or
  `response_json`) when it is a name, not an HTTP number. `error.type` is set only then. An
  exception with no named code keeps the base wrapper's ERROR status without `error.type` (the
  spec did not say what to record, and inventing a value would misreport the cause).
- **Error code on the REST path.** Same rule; when the 400 body carries no named code, `error.type`
  falls back to the HTTP status string, as for every other decision protocol.
- **Host gate.** On `generativelanguage.googleapis.com` only `/v1beta/decisions` and `/v1/decisions`
  match, and both are Gemini. Any other path on that host matches nothing.
- **Async.** `client.aio.decisions.create` is wrapped only when present, through the same tracer
  (span name `google.genai.aio.decisions`). Verified with a fake client; not against a real SDK,
  because an async variant is undocumented.
- **Provider.** `decision.provider` is fixed to `google` on the SDK path (the SDK's base URL is not
  read); the REST path derives it from the host map.
- **Not verified against a real `google-genai` install.** The tests use fakes shaped like the
  documented response (`answers` as a map of attribute-access objects).
