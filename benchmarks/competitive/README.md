# Tracing-SDK overhead benchmark

Measures how much time LLM tracing SDKs add to a call made with the OpenAI Python
SDK, and, as a second command, how much TraceVerde's evaluation detectors add on
top of that.

Every number comes from running the tool, and the tables it writes carry the
environment they were measured in. The [reference results](#reference-results) at
the end of this file are one such run, kept here so that a new run has something
to be compared with. They describe one machine on one day; run the tool to get
yours.

## Requirements

- Python 3.9 or newer to launch the tool.
- `uv` on PATH (recommended): the environments are then created with Python 3.12,
  the version the pins were resolved on, whatever interpreter launched the tool
  (`--python` changes it). Without `uv` the tool falls back to the standard
  library's `venv` plus `pip` and uses the launching interpreter, which should
  then be 3.12.
- Network access to a package index while the environments are created. Nothing
  else: no API key, no account, no LLM provider, no collector.

Works on Linux, macOS and Windows.

## Quick start

```
python run_benchmark.py --quick
```

This creates one virtual environment per library under `.venvs/` from the pinned
files in `requirements/`, runs every library in its default configuration at 10 %
volume, and writes `results/<timestamp>/tables.md` plus one JSON file and one log
per run. It takes a couple of minutes once the environments exist.

The full comparison:

```
python run_benchmark.py                      # two rounds, every configuration, both exporter modes
python run_benchmark.py --libraries traceverde,openlit --rounds 3
python run_benchmark.py --payloads S,M,L,conv20 --scenarios sync,async_seq --modes P0
python run_benchmark.py --inspect            # only list what each configuration records
python run_benchmark.py --help
```

| flag | meaning |
|---|---|
| `--libraries` | `traceverde,openlit,traceloop,openinference,langfuse,opik` by default; `galileo` only when named (see limitations) |
| `--configs` | configuration names, see `benchlib/adapters.py`; default is every configuration of each library |
| `--modes` | `P0` no exporting span processor, `P1` batch processor feeding a discarding exporter |
| `--scenarios` | `sync`, `async_seq`, `async_gather10`, `sync_stream`, `async_stream` |
| `--payloads` | `default`, `S` (about 50 tokens each side), `M` (500), `L` (4,000), `conv20` (20-turn conversation) |
| `--rounds`, `--reps`, `--scale` | passes over the matrix, timed repetitions per scenario, multiplier on calls per repetition |
| `--pin-cpus` | logical CPUs to pin every run to, for example `4,5,6,7` |
| `--quick` | default configuration only, mode `P1`, one round, three repetitions at 10 % volume |
| `--env-dir` | where the environments are created (default `.venvs` in this directory) |

The evaluation-detector benchmark is separate because it needs torch, spaCy,
Presidio and Detoxify (several GB):

```
python run_eval_benchmark.py --quick
python run_eval_benchmark.py                           # every detector, alone and combined, all modes
python run_eval_benchmark.py --with-openlit-guards     # adds OpenLIT's local guards for comparison
```

Its first run downloads the spaCy model as a pinned package and, in a preparation
step, the Detoxify weights and tokenizer files into the usual model caches. That
is the only network access besides package installation. `--skip-prepare` skips
it once the files are cached.

## What is measured

Every run is one fresh process for one (library, configuration, exporter mode):

1. The bare OpenAI SDK is timed first (the baseline).
2. The library is imported and set up.
3. The same scenarios are timed again on new, instrumented clients.

Overhead is instrumented minus baseline, in microseconds per call. Minimum and
median over the repetitions are both reported. The baseline is measured in the
same process and before the library is imported, because several libraries patch
the OpenAI SDK at class level and an untouched client cannot exist next to them.

The OpenAI SDK is real; the network is not. Every client is given an in-process
`httpx.MockTransport` that returns a canned chat completion with a usage block,
or a canned stream of 50 server-sent events ending with a usage chunk. No time is
spent on I/O, so the numbers are the libraries' CPU cost and nothing else.

Scenarios: sync non-streaming; async non-streaming awaited one at a time; async
with `asyncio.gather` of 10; sync streaming and async streaming, each fully
consumed (reported per call and per chunk).

Exporter modes, identical for every OpenTelemetry-based library:

- `P0`: the tracer provider has no exporting span processor. Processors a library
  installs for its own bookkeeping stay.
- `P1`: a `BatchSpanProcessor` with SDK defaults feeding an exporter that discards
  what it receives. Metrics go to a periodic reader with a discarding exporter.

Libraries with their own sender are mapped onto the closest equivalent, and each
result file says exactly how in its `notes` field:

- Langfuse: dummy keys, an unreachable local host, its own span processor kept,
  the exporter replaced by the discarding one (`P1`) or its `on_end` disabled (`P0`).
- Opik: its REST client is given an in-process mock transport through Opik's
  documented httpx client hook, so batching, JSON encoding and compression run and
  the send is answered locally (`P1`); in `P0` messages are dropped before the
  background queue. Error reporting and product analytics are disabled.
- Time spent flushing (`flush per call` in the tables) is measured separately from
  the calls.

### What each configuration records

Libraries differ in what they record by default, and that changes their cost.
The tables therefore always name the configuration, `--inspect` lists the
attributes and events each one writes, and every library is run both with its
own defaults and, where it has a switch, with message content off. TraceVerde,
whose default is content off, is also run with content on (`content_on`, which
keeps its default 200-character cap, and `content_on_full`, which removes it).
Compare like with like: content-on rows with content-on rows.

Langfuse and Opik have no switch that turns content capture off for their OpenAI
integrations, so they only appear in the content-on comparison.

### Evaluation detectors

`run_eval_benchmark.py` measures TraceVerde with detectors off (the baseline),
each detector alone, all six together, and the redact and block modes. Prompts
and responses come in three sizes and in a `clean` and a `dirty` variant; the
dirty one contains fabricated PII-like strings, insults, prompt-injection phrases
and similar, so the detection branches run. For each cell it reports latency
(min, median, p95, p99) for sync and async calls, how long one call holds the
event loop, process CPU time per call and the CPU/wall ratio, resident memory and
native thread counts, setup and first-call time, throughput with 8 threads and
with `asyncio.gather` of 50, and the attributes added to the span.

## How noise is controlled

- Each scenario is warmed up, then timed for several repetitions of a few thousand
  calls (a few hundred for streaming). The garbage collector runs between
  repetitions and stays enabled during them.
- The matrix is run in rounds, and the order is reversed every other round, so a
  slow drift of the machine does not always land on the same library.
- The tables show each round's overhead and flag any row whose rounds differ by
  more than 10 % (`rounds agree`).
- Two `none` control runs execute both phases on the bare SDK. Their "overhead" is
  the measurement floor of that session; differences smaller than it mean nothing.
- `--pin-cpus` restricts every run to the same cores. Close other programs; a busy
  machine moves every number.

## What is not measured, and other limitations

- Export. The exporter discards, so serialising spans and sending them is excluded
  for every OpenTelemetry-based library. Libraries that record more will pay more
  there. Opik in `P1` is the exception: its encoding runs, on background threads
  that compete with the calling thread, so its `P1` rows are not comparable with
  the `P1` rows of the others.
- HTTP-level auto-instrumentation. Instrumentations that patch the HTTP transport
  (OpenLIT enables one by default) do not see a mock transport, so that part of
  their default cost is absent.
- Real latency. With no I/O between calls, background work has no idle time to run
  in. Per-call overhead on a real network is the same CPU time, but a smaller
  share of the call.
- One provider SDK, one model name, CPython only.
- Traceloop's sync wrapper calls `asyncio.run()` once per call. On Windows every
  such call creates a loopback socket pair, and a tight loop can exhaust the
  ephemeral port range; the tool therefore runs Traceloop's two sync scenarios at
  15 % volume on Windows. Its sync numbers are platform-dependent.
- Galileo's OpenAI wrapper cannot run without an account through public API. It is
  available as an experimental library (`--libraries galileo`) by placing a logger
  built with its documented ingestion hook into a private cache of the SDK; only
  the sync non-streaming scenario is measured. Treat it as indicative.
- OpenLIT's default downloads a pricing table at start-up; the tool passes a
  bundled one-model table in the same format instead.
- Windows path length. The evaluation environment contains deeply nested native
  libraries (spaCy, torch). If this directory already sits under a long path and
  long-path support is not enabled in Windows, importing them fails with "The
  filename or extension is too long". Put the tool in a short path or pass
  `--env-dir` with a short one.
- Requirement files pin the library under test, the OpenAI SDK, OpenTelemetry and
  the packages on the measured call path. Other transitive dependencies resolve at
  install time; every result file records the full resolved set.

## Safety properties

- A network guard inside every measurement process refuses any connection or name
  lookup that is not loopback, and records the attempt in the result file
  (`blocked egress` in the tables).
- The environment handed to measurement processes is scrubbed of variables that
  configure tracing libraries, point at endpoints or carry credentials.
- No file outside this directory is written, apart from package and model caches.

## Reference results

One full run, recorded as measured. Nothing here is a general claim about any
library: the numbers belong to the machine, versions and configurations listed, and
the limitations in [What is not measured](#what-is-not-measured-and-other-limitations)
apply to all of them.

### Environment of this run

- Date: 2026-10-11 to 2026-10-12.
- Machine: AMD Ryzen 9 9955HX3D (16 cores, 32 threads), 93.6 GB RAM, Windows 11,
  on AC power.
- Python 3.12.13, one virtual environment per library.
- Identical in every environment: `openai` 2.54.0, `httpx` 0.28.1, `pydantic`
  2.14.0, `opentelemetry-api` and `-sdk` 1.45.1 where used.
- Libraries: `genai-otel-instrument` (TraceVerde) 1.35.0, `openlit` 1.45.0,
  `traceloop-sdk` 0.62.4, `openinference-instrumentation-openai` 0.1.65,
  `langfuse` 4.17.0, `opik` 2.2.96, `galileo` 2.6.0.
- A bare SDK call through the in-process fake costs 423 to 443 us. The bare-SDK
  control moved by up to 7 us between runs on non-streaming calls and 28 us per
  streamed call, so smaller differences than that mean nothing.

Unless a table says otherwise, a figure is the overhead added to one call, in
microseconds: the minimum over repetitions, averaged over two rounds, with no
exporting span processor attached.

### Non-streaming calls, each library's defaults

| library | sync | async | async, `gather` of 10 |
|---|---:|---:|---:|
| TraceVerde | 152 | 149 | 153 |
| OpenInference | 183 | 185 | 187 |
| Langfuse | 191 | 192 | 194 |
| Opik | 214 | 221 | 218 |
| Galileo | 223 | not instrumented | not instrumented |
| OpenLIT | 238 | 252 | 246 |
| Traceloop | 1,219 | 186 | 189 |

Defaults do not record the same things. TraceVerde's default records token usage,
cost and metrics but no prompt or response text; OpenInference, Langfuse, OpenLIT
and Traceloop record message text by default. The next table puts them on the
same footing.

- Traceloop's sync figure is not stable (its two rounds disagreed by 24 %). Its
  sync wrapper starts an event loop per call, which is slow on Windows.
- Galileo ran only with its own pipeline attached and only for sync calls; its
  OpenAI wrapper does not cover the async client.

### Content capture, like for like

Sync / async, microseconds per call.

| library | content on | content off |
|---|---:|---:|
| TraceVerde | 180 / 191 | 152 / 149 |
| OpenInference | 183 / 185 | 207 / 200 |
| Langfuse | 191 / 192 | no switch |
| OpenLIT | 238 / 252 | 216 / 224 |
| Traceloop | 1,219 / 186 | 1,028 / 165 |

- TraceVerde's content capture costs 28 us per sync call and 42 us per async call.
- Leanest configurations: OpenLIT with content, metrics and events off adds 117 us
  sync; TraceVerde with cost tracking and concurrency metrics off adds 131 to 134
  us, and has no switch that turns metric recording off.
- With a batch span processor attached, async with content on: TraceVerde 209 to
  216 us, Traceloop 188, OpenInference 194, Langfuse 216.

### Content capture as the payload grows

Sync, content on, microseconds per call.

| library | 50 tokens | 500 tokens | 4,000 tokens | 20-turn conversation |
|---|---:|---:|---:|---:|
| TraceVerde (no length cap) | 176 | 172 | 178 | 254 |
| OpenInference | 180 | 185 | 243 | 333 |
| Langfuse | 189 | 195 | 237 | 322 |
| Opik (messages dropped) | 226 | 220 | 221 | 267 |
| OpenLIT | 232 | 240 | 288 | 329 |

TraceVerde's content cost does not grow with the amount of text but does grow with
the number of messages. On the 20-turn conversation in async it is 342 to 349 us,
against 247 for Traceloop, 283 for Opik, 303 for Langfuse and 340 for
OpenInference.

### Streaming, 50 chunks

Overhead per chunk, microseconds.

| library | sync | async |
|---|---:|---:|
| Opik | 6.1 | 6.6 |
| OpenLIT, lean configuration | 8.2 | 8.2 |
| Langfuse | 12.0 | 21.3 |
| TraceVerde | 16.1 | 16.0 |
| OpenLIT, default | 17.2 | 17.4 |
| OpenInference | 19.1 | 19.7 |
| Traceloop | 56.0 | 18.8 |

With content capture on, TraceVerde records no completion text for a stream; the
libraries that are faster per chunk rebuild it at the end.

### Start-up and memory

| library | import + setup, ms | resident memory after setup, MB |
|---|---:|---:|
| bare SDK | 1 | 54 |
| OpenInference | 68 | 59 |
| TraceVerde | 320 | 79 |
| Langfuse | 490 | 87 |
| Traceloop | 811 | 110 |
| Opik | 883 | 109 |
| OpenLIT | 1,174 | 116 |
| Galileo | 1,437 | 132 |

No library's memory grew over 20,000 sync calls. Async streaming grew by about
30 MB over 4,000 calls with and without TraceVerde (the bare SDK does the same).

### Where TraceVerde's own overhead goes

Sync, non-streaming, default configuration (152 us).

| part | share |
|---|---:|
| metric recording, 5 measurements | 20 % |
| span start with 11 attributes | 14 % |
| result bookkeeping | up to 11 % |
| 13 separate span attribute writes | 7 % |
| usage extraction | 6 % |
| span end | 4 % |
| request attribute extraction | 4 % |
| server address, re-parsed on every call | 4 % |
| token attribute helpers | 4 % |
| cost calculation | 3 % |
| response attribute extraction | 3 % |
| context attach and detach, budget, gauges, disabled evaluation hooks | about 5 % |
| wrapper dispatch and remainder | 14 % |

- Streaming: the time-between-tokens histogram sample taken on every chunk is 41 %
  of the streaming overhead by timer. With only that sample removed, a 50-chunk
  stream went from 847 to 426 us sync and from 833 to 390 us async.
- The async OpenAI wrapper spends about 26 us per call that the sync path does
  not: it looks up the tracer on every request, writes request attributes one at a
  time and extracts the response twice.
- A wrapped no-op function in a tight loop measures 43 to 54 us for the same
  wrapper. A real call costs 152 us: about 37 % of the gap is the tight loop
  keeping the wrapper in the CPU caches, and about 60 % is work a no-op skips.

### Evaluation detectors

`run_eval_benchmark.py`, TraceVerde only, sync calls, CPU-only torch, 16 cores free.
A call with detectors off takes about 0.6 ms. Sizes are nominal: by the detectors'
own tokenisers "500 tokens" is about 400 and "4,000" about 3,000.

Latency added per call, median, milliseconds:

| detector | 50 tokens | 500 tokens | 4,000 tokens |
|---|---:|---:|---:|
| prompt injection | 0.1 | 0.5 | 3.1 |
| restricted topics | 0.2 | 1.0 | 6.7 |
| hallucination | 0.3 | 1.8 | 12 |
| bias | 1.0 | 5.1 | 35 |
| PII (Presidio, spaCy `en_core_web_lg`) | 11 | 58 | 427 |
| toxicity (Detoxify `original`) | 30 | 105 | 131 |
| all six | 47 | 183 | 657 |

- Block mode on a clean prompt runs each detector before and after the call: all
  six cost 365 ms at 500 tokens and 1.33 s at 4,000.
- CPU: PII and the pattern detectors use one core. Toxicity uses about 16; one
  500-token call takes 1.78 CPU-seconds. All six take 3.1 CPU-seconds per call at
  500 tokens and 5.4 at 4,000.
- Throughput at 500 tokens, single thread: 1,576 calls/s with detectors off, 17
  with PII, 9.5 with toxicity, 5.4 with all six. Eight threads reach 7.9 calls/s
  with all six while using 26 cores.

Memory and start-up:

| configuration | resident memory after `instrument()`, MB | `instrument()`, s |
|---|---:|---:|
| detectors off | 84 | 0.33 |
| any one pattern detector | 276 | 2.4 |
| toxicity | 802 | 4.0 |
| PII | 1,016 | 5.2 |
| all six | 1,450 | 5.4 |

A single pattern detector costs 192 MB and 2 s because the toxicity module imports
torch at module level whether or not toxicity is enabled.

Inside the detectors:

- PII: the spaCy pipeline is 95 to 99 % of the time. Within it, entity recognition
  is 40 to 42 %, the embedding layer 37 to 38 %, and the dependency parser, whose
  output nothing uses, 13 to 15 %.
- Toxicity: the model's forward pass is 95 to 99 %. It reads at most 512 word
  pieces and ignores the rest of the text.
- Pattern detectors: all of the time is regular-expression scanning. Precompiling
  the patterns saves 5 %.

The only other library with an inline, in-SDK equivalent is OpenLIT. Its four
guards together add 0.5 ms at 50 tokens, 1.8 to 2.1 ms at 500 and 12 to 14 ms at
4,000, with no change in memory. They are regular expressions with no
named-entity or learned toxicity model, so this compares cost, not detection.

### What not to quote from this run

- Traceloop's sync figures (unstable between rounds).
- Galileo against the others (different pipeline, sync only).
- Percentiles for detector cells where a call takes over 50 ms: they rest on 20 to
  60 samples.
- Any single figure without its configuration: what a library records by default
  changes its cost more than the library does.

## Layout

```
run_benchmark.py          entry point: competitor comparison
run_eval_benchmark.py     entry point: evaluation-detector overhead
requirements/             one pinned requirement file per environment
benchlib/common.py        fake backend, discarding exporters, network guard, timing loops
benchlib/adapters.py      per-library setup and configurations
benchlib/worker.py        one run (baseline, then instrumented); writes one JSON file
benchlib/report.py        JSON -> tables.md
benchlib/envs.py          environment creation (uv or venv + pip)
benchlib/eval_*.py        the evaluation-detector benchmark
```

## Adding a library or a scenario

Library: add `requirements/<name>.txt`, write `setup_<name>(adapter)` in
`benchlib/adapters.py` (set up the library for modes `P0`, `P1` and `INSPECT`,
set `adapter.drain` to its flush, describe what you did in `adapter.notes`),
register it in `SETUPS`, and add its configurations to `CONFIGS` in
`run_benchmark.py`. If the library cannot run without an account or a network
endpoint, do not approximate it silently: skip it, or mark it experimental and say
how it was made to run.

Scenario: add an entry to `SCENARIOS` and a branch to `run_once` in
`benchlib/common.py`, and a title in `benchlib/report.py`.

Payload: add a name to `PAYLOADS` and a branch to `build_payload`.
