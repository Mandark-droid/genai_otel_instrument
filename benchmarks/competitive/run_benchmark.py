#!/usr/bin/env python3
"""Per-call overhead of LLM tracing SDKs on the OpenAI Python SDK.

    python run_benchmark.py --quick
    python run_benchmark.py --libraries traceverde,openlit --rounds 2

Creates one virtual environment per library from requirements/*.txt, measures
each library against the bare SDK in a fresh process per (library, configuration,
mode), and writes raw JSON plus markdown tables under results/<timestamp>/.

No API key, no account and no network access are needed after the environments
have been installed. See README.md for what is and is not measured.
"""

import argparse
import datetime
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from benchlib import envs  # noqa: E402

# library -> configurations run by default (first one is the library's own default)
CONFIGS = {
    "traceverde": ["default", "content_on", "content_on_full", "cost_off"],
    "openlit": ["default", "content_off", "lean"],
    "traceloop": ["default", "content_off"],
    "openinference": ["default", "content_off"],
    "langfuse": ["default"],
    "opik": ["default"],
    "galileo": ["default"],
}
DEFAULT_LIBRARIES = ["traceverde", "openlit", "traceloop", "openinference", "langfuse", "opik"]
SCENARIOS = ["sync", "async_seq", "async_gather10", "sync_stream", "async_stream"]
PAYLOADS = ["default", "S", "M", "L", "conv20"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--libraries",
        default=",".join(DEFAULT_LIBRARIES),
        help="comma-separated; known: %s (galileo is experimental and off by default)"
        % ",".join(sorted(CONFIGS)),
    )
    p.add_argument(
        "--configs",
        default="",
        help="comma-separated configuration names; default: all known per library",
    )
    p.add_argument(
        "--modes",
        default="P0,P1",
        help="P0 = no exporting span processor, P1 = batch processor + discarding exporter",
    )
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument(
        "--payloads", default="default", help="comma-separated from: %s" % ",".join(PAYLOADS)
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="full passes over the matrix; order is reversed every other round",
    )
    p.add_argument("--reps", type=int, default=5, help="timed repetitions per scenario")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiplier on calls per repetition (2000 non-streaming, 400 streaming)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="a couple of minutes: default configuration only, mode P1, one round, 3 reps at 10%% volume",
    )
    p.add_argument(
        "--python",
        default="3.12",
        help="Python version uv creates the environments with (default 3.12, the version the pins "
        "were resolved on); ignored by the venv + pip fallback, which uses this interpreter",
    )
    p.add_argument(
        "--pin-cpus", default="", help="comma-separated logical CPU numbers to pin every run to"
    )
    p.add_argument("--out", default=None, help="results directory (default: results/<timestamp>)")
    p.add_argument(
        "--env-dir",
        default=None,
        help="where to create the environments (default: .venvs next to this file)",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="only record what each configuration writes, then exit",
    )
    p.add_argument("--no-control", action="store_true", help="skip the bare-SDK control runs")
    return p.parse_args()


def main():
    args = parse_args()
    libraries = [x for x in args.libraries.split(",") if x]
    unknown = [x for x in libraries if x not in CONFIGS]
    if unknown:
        raise SystemExit("unknown libraries: %s" % unknown)
    modes = [m for m in args.modes.split(",") if m]
    wanted_configs = [c for c in args.configs.split(",") if c]
    scenarios = [s for s in args.scenarios.split(",") if s]
    payloads = [p for p in args.payloads.split(",") if p]
    rounds, reps, scale = args.rounds, args.reps, args.scale
    if args.quick:
        rounds, reps, scale, modes = 1, 3, 0.1, ["P1"]
        if not wanted_configs:
            wanted_configs = ["default"]

    out_dir = args.out or os.path.join(
        ROOT, "results", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, "work")
    os.makedirs(work_dir, exist_ok=True)

    envs.set_env_dir(args.env_dir)
    interpreters = {}
    for lib in libraries:
        try:
            interpreters[lib] = envs.ensure(lib, args.python)
        except (subprocess.CalledProcessError, OSError) as e:
            print("SKIPPED %s: its environment could not be created (%s)" % (lib, e))
    libraries = [lib for lib in libraries if lib in interpreters]
    if not libraries:
        raise SystemExit("no library environment available")
    control_py = interpreters[libraries[0]]

    matrix = []
    if not args.no_control:
        matrix.append(("none", "control", "P0"))
    for lib in libraries:
        configs = [c for c in CONFIGS[lib] if not wanted_configs or c in wanted_configs]
        lib_modes = ["P1"] if lib == "galileo" else modes
        for cfg in configs:
            for mode in lib_modes:
                matrix.append((lib, cfg, mode))
    if not args.no_control:
        matrix.append(("none", "control2", "P0"))

    env = envs.clean_env()
    env.update(
        BENCH_REPS=str(reps),
        BENCH_SCALE=str(scale),
        BENCH_SCENARIOS=",".join(scenarios),
        BENCH_PIN_CPUS=args.pin_cpus,
        BENCH_WORKDIR=work_dir,
    )

    if args.inspect:
        for lib, cfg, _mode in matrix:
            if lib == "none":
                continue
            out = os.path.join(out_dir, "inspect_%s_%s.json" % (lib, cfg))
            rc = subprocess.call(
                [interpreters[lib], "-m", "benchlib.worker", lib, cfg, "INSPECT", out],
                cwd=ROOT,
                env=env,
            )
            print("inspect %s/%s -> %s (rc=%d)" % (lib, cfg, out, rc))
        return

    total = len(matrix) * rounds * len(payloads)
    done = 0
    for payload in payloads:
        for rnd in range(1, rounds + 1):
            order = matrix if rnd % 2 == 1 else list(reversed(matrix))
            for lib, cfg, mode in order:
                done += 1
                name = "run_r%d_%s_%s_%s_%s" % (rnd, payload, lib, cfg, mode)
                out = os.path.join(out_dir, name + ".json")
                if os.path.exists(out) and os.path.getsize(out) > 0:
                    continue
                run_env = dict(env, BENCH_PAYLOAD=payload)
                if lib == "traceloop" and sys.platform == "win32":
                    # See README, "Known limitations": asyncio.run() per sync call.
                    run_env["BENCH_SYNC_SCALE"] = "0.15"
                py = control_py if lib == "none" else interpreters[lib]
                with open(os.path.join(out_dir, name + ".log"), "w", encoding="utf-8") as log:
                    rc = subprocess.call(
                        [py, "-m", "benchlib.worker", lib, cfg, mode, out],
                        cwd=ROOT,
                        env=run_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                print(
                    "[%d/%d] %-52s %s"
                    % (
                        done,
                        total,
                        name,
                        "ok" if rc == 0 else "FAILED rc=%d, see %s.log" % (rc, name),
                    )
                )
                sys.stdout.flush()

    subprocess.call([sys.executable, "-m", "benchlib.report", out_dir], cwd=ROOT)
    print("results: %s" % out_dir)


if __name__ == "__main__":
    main()
