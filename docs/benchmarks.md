# Benchmarks

How much time TraceVerde adds to an LLM call, how that compares with other tracing
SDKs, and what the evaluation detectors cost on top.

## How it is measured

- **A real SDK call, no network.** Each library instruments the real `openai` Python SDK,
  which talks to an in-process fake server. A bare call costs about 425 microseconds; the
  tables give what each library adds to that.
- **Same conditions for everyone.** One environment per library with the same Python, the
  same `openai` version and the same OpenTelemetry SDK. Nothing is exported: figures are
  for the instrumentation itself unless a table says a batch processor was attached.
- **Noise is shown, not hidden.** Warm-up, several repetitions, at least two rounds in
  alternating order, and a check that the rounds agree. An uninstrumented control runs
  alongside; it moved by up to 7 microseconds, so smaller differences mean nothing.
- **Defaults are not equivalent.** Libraries record different things out of the box.
  TraceVerde's default records token usage, cost and metrics but no prompt or response
  text; several others record message text by default. Results are given both ways.

## Reference results

One full run, as measured. These numbers belong to one machine on one day and to the
versions listed. A real LLM call takes hundreds of milliseconds, so every tracing figure
here is well under 0.1 % of a request.

**Environment.** 2026-10-12. AMD Ryzen 9 9955HX3D (16 cores), Windows 11, Python 3.12.13,
`openai` 2.54.0, OpenTelemetry SDK 1.45.1. `genai-otel-instrument` 1.35.0, `openlit`
1.45.0, `traceloop-sdk` 0.62.4, `openinference-instrumentation-openai` 0.1.65, `langfuse`
4.17.0, `opik` 2.2.96, `galileo` 2.6.0.

Figures are microseconds added per call unless stated.

### Non-streaming calls, default settings

| Library | Sync | Async | Async, 10 at once |
|---|---:|---:|---:|
| TraceVerde | 152 | 149 | 153 |
| OpenInference | 183 | 185 | 187 |
| Langfuse | 191 | 192 | 194 |
| Opik | 214 | 221 | 218 |
| Galileo | 223 | not instrumented | not instrumented |
| OpenLIT | 238 | 252 | 246 |
| Traceloop | 1,219 | 186 | 189 |

Traceloop's sync figure was not stable between rounds on Windows and should not be
quoted. Galileo ran with its own pipeline attached and covers sync calls only.

### With content capture matched

Sync / async.

| Library | Content on | Content off |
|---|---:|---:|
| TraceVerde | 180 / 191 | 152 / 149 |
| OpenInference | 183 / 185 | 207 / 200 |
| Langfuse | 191 / 192 | no switch |
| OpenLIT | 238 / 252 | 216 / 224 |

With content on, TraceVerde, OpenInference and Langfuse are within about 10 microseconds
of each other. With everything optional turned off, OpenLIT's leanest configuration adds
117 microseconds and TraceVerde's 131 to 134.

### Larger payloads

Sync, content on.

| Library | 50 tokens | 500 tokens | 4,000 tokens | 20-turn conversation |
|---|---:|---:|---:|---:|
| TraceVerde | 176 | 172 | 178 | 254 |
| OpenInference | 180 | 185 | 243 | 333 |
| Langfuse | 189 | 195 | 237 | 322 |
| Opik | 226 | 220 | 221 | 267 |
| OpenLIT | 232 | 240 | 288 | 329 |

TraceVerde's content cost depends on the number of messages, not the amount of text. In
async on the 20-turn conversation it is 342 to 349 microseconds, behind Traceloop (247),
Opik (283) and Langfuse (303).

### Streaming

Overhead per chunk on a 50-chunk stream.

| Library | Sync | Async |
|---|---:|---:|
| Opik | 6.1 | 6.6 |
| OpenLIT, leanest configuration | 8.2 | 8.2 |
| Langfuse | 12.0 | 21.3 |
| TraceVerde | 16.1 | 16.0 |
| OpenLIT, default | 17.2 | 17.4 |
| OpenInference | 19.1 | 19.7 |
| Traceloop | 56.0 | 18.8 |

Streaming is where TraceVerde has the most room to improve. About half of its per-chunk
cost is one metric sample taken on every chunk.

### Start-up and memory

| Library | Import and setup | Memory after setup |
|---|---:|---:|
| Uninstrumented SDK | 1 ms | 54 MB |
| OpenInference | 68 ms | 59 MB |
| TraceVerde | 320 ms | 79 MB |
| Langfuse | 490 ms | 87 MB |
| Traceloop | 811 ms | 110 MB |
| Opik | 883 ms | 109 MB |
| OpenLIT | 1,174 ms | 116 MB |
| Galileo | 1,437 ms | 132 MB |

No library's memory grew over 20,000 calls.

## Evaluation detectors

The detectors are optional and off by default. They do real work on your text, so they
cost far more than tracing does. Measured on sync calls with CPU-only models; a call with
detectors off takes about 0.6 ms.

Milliseconds added per call:

| Detector | About 50 tokens | About 500 tokens | About 4,000 tokens |
|---|---:|---:|---:|
| Prompt injection | 0.1 | 0.5 | 3.1 |
| Restricted topics | 0.2 | 1.0 | 6.7 |
| Hallucination | 0.3 | 1.8 | 12 |
| Bias | 1.0 | 5.1 | 35 |
| PII | 11 | 58 | 427 |
| Toxicity | 30 | 105 | 131 |
| All six | 47 | 183 | 657 |

| Configuration | Memory after `instrument()` | `instrument()` time |
|---|---:|---:|
| Detectors off | 84 MB | 0.3 s |
| One pattern-based detector | 276 MB | 2.4 s |
| Toxicity | 802 MB | 4.0 s |
| PII | 1,016 MB | 5.2 s |
| All six | 1,450 MB | 5.4 s |

- PII detection uses a named-entity model and toxicity a transformer model. That is where
  the time and memory go, and why they find things a list of patterns cannot.
- Toxicity uses every CPU core it is given: one 500-token call takes about 1.8
  CPU-seconds.
- In block mode a clean prompt is checked before the call and again after it.

These figures describe 1.35.0. Plan capacity for the detectors you enable, and prefer the
pattern-based ones on a latency-critical path.

## Reading these numbers

- **One machine, one SDK.** Results on Linux, on fewer cores or with another provider's
  SDK will differ. Measure where your application runs.
- **A fake server.** Nothing here measures export, network or a collector.
- **Configuration decides more than the library does.** What an SDK records by default
  changes its cost more than which SDK it is. Compare like with like.
- **Detector figures are costs, not quality.** Nothing on this page says how well a
  detector detects.
