# Upstream PR Draft — Narrow Multimodal Additions

> **Scope corrected (2026-04-28):** OTel's existing `gen-ai-input-messages.json` /
> `gen-ai-output-messages.json` schemas already define `BlobPart`, `FilePart`, `UriPart` and a
> `Modality` enum (`image`, `video`, `audio`). Our originally drafted "flat attribute namespace"
> proposal duplicated work that was already done. This redraft narrows the upstream PR to the
> three pieces that are genuinely missing.

## What this PR adds

1. **`document` value added to the `Modality` enum.** Needed for PDF / DOCX workloads (BFSI KYC
   extraction is a real high-volume example).
2. **Optional `byte_size` field added to `BlobPart`, `FilePart`, and `UriPart`.** Useful for
   cost-of-capture telemetry and storage planning. Optional, additive.
3. **New `StrippedPart` type** for fail-closed observability. Records that the instrumentor
   detected a content part but intentionally did not capture its bytes (size cap exceeded,
   modality disallowed by config, redactor failed, store unavailable). Without this, consumers
   can't distinguish *"no media in this turn"* from *"media was deliberately stripped."*

## Files in this folder

| File | Maps to in OTel repo | Purpose |
|---|---|---|
| `input-messages.json.diff` | `docs/gen-ai/gen-ai-input-messages.json` | Add `document` to Modality enum, `byte_size` field, `StrippedPart` definition |
| `output-messages.json.diff` | `docs/gen-ai/gen-ai-output-messages.json` | Mirror of the above |
| `chloggen-entry.yaml` | `.chloggen/gen-ai-multimodal-narrow.yaml` | Changelog entry |
| `gen-ai-spans.md.fragment` | `docs/gen-ai/gen-ai-spans.md` (small note) | Brief mention of fail-closed semantics |

## Filing checklist (per `CONTRIBUTING.md`)

- [x] Issue filed and refined: [#3672](https://github.com/open-telemetry/semantic-conventions/issues/3672)
- [ ] CLA signed
- [ ] Forked repo
- [ ] Apply JSON-schema diffs
- [ ] Add `.chloggen/*.yaml`
- [ ] `make check` passing
- [ ] PR description references #3672
- [ ] Two code-owner approvals from `@open-telemetry/specs-semconv-approvers` or `@open-telemetry/semconv-genai-approvers`

## Why this is the right size

The original PR plan (~480 lines of new attribute registry + spans references + markdown spec)
duplicated existing OTel structure. The narrow PR is ~20–30 lines of JSON schema changes plus
~10 lines of chloggen + a paragraph of markdown. Smaller diff, mirrors existing pattern, much
more likely to merge cleanly without bouncing between revisions.
