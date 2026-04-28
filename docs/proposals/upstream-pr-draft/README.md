# Upstream PR Draft — Multimodal Content-Part Attributes

This folder contains the draft artifacts to be committed to a fork of
[`open-telemetry/semantic-conventions`](https://github.com/open-telemetry/semantic-conventions)
once the working group signals the proposal shape is acceptable
(see issue [#3672](https://github.com/open-telemetry/semantic-conventions/issues/3672)).

## Files

| File in this folder | Maps to in OTel repo | Purpose |
|---|---|---|
| `registry.yaml.fragment` | `model/gen-ai/registry.yaml` (append) | New attribute definitions for the registry |
| `spans.yaml.fragment` | `model/gen-ai/spans.yaml` (extend group) | Reference the new attributes from gen_ai spans |
| `gen-ai-spans.md.fragment` | `docs/gen-ai/gen-ai-spans.md` (insert section) | Human-readable spec of the attribute namespace |
| `chloggen-entry.yaml` | `.chloggen/<feature-slug>.yaml` | Changelog entry per repo policy |

## Filing checklist (per `CONTRIBUTING.md`)

- [ ] CLA signed
- [ ] Forked `open-telemetry/semantic-conventions` to your account
- [ ] Created a feature branch (`gen-ai/multimodal-content-parts`)
- [ ] Applied YAML fragments to `model/gen-ai/`
- [ ] Updated `docs/gen-ai/gen-ai-spans.md`
- [ ] Added `.chloggen/gen-ai-multimodal-content-parts.yaml`
- [ ] Ran `make check` locally and it passes
- [ ] PR description references issue #3672 with `Resolves #3672` (or `Refs #3672` if the WG wants the issue to stay open for follow-ups like tool-call multimodal)
- [ ] Two code-owner approvals from `@open-telemetry/specs-semconv-approvers` or `@open-telemetry/semconv-genai-approvers`
- [ ] No requested changes / no open discussions
- [ ] Two working days elapsed since last modification

## Stability tier

All new attributes are proposed under `stability: development` (the OTel semconv equivalent of
"experimental"). Promotion to `stable` would happen in a follow-up after real-world validation
across multiple instrumentations.

## Why these attributes are not redundant with `gen_ai.input.messages`

`gen_ai.input.messages` is an opt-in serialized blob; the flat per-part attributes proposed here
are queryable in any OTel backend without JSON parsing — see the issue for the full rationale.
Both can be emitted together via the existing dual-emission pattern.
