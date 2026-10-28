#!/usr/bin/env python3
"""Overhead of TraceVerde's evaluation detectors (PII, toxicity, bias, prompt
injection, restricted topics, hallucination), per detector and combined.

    python run_eval_benchmark.py --quick
    python run_eval_benchmark.py --configs ev_base,ev_pii,ev_all6 --sizes S,M,L

This is a separate command because it needs the heavy evaluation dependencies
(torch, spaCy, Presidio, Detoxify; several GB on disk). The first run downloads
the spaCy model as a package and, in a one-off preparation step, the Detoxify
weights and its tokenizer files. After that no network access is used: the
measurement processes refuse any non-loopback connection.

The Perspective API is never used. No API key or account is needed.
"""

import argparse
import datetime
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from benchlib import envs  # noqa: E402

TRACEVERDE = [
    "ev_base",
    "ev_pii",
    "ev_tox",
    "ev_bias",
    "ev_inj",
    "ev_topics",
    "ev_halluc",
    "ev_all6",
    "ev_pii_redact",
    "ev_pii_block",
    "ev_tox_block",
    "ev_bias_block",
    "ev_inj_block",
    "ev_topics_block",
    "ev_all_block",
    "ev_base_cap",
    "ev_all6_cap",
    "ev_all6_cap200",
]
OPENLIT = ["og_base", "og_pii", "og_inj", "og_mod", "og_topic", "og_all4", "og_all4_warn"]
QUICK = ["ev_base", "ev_inj", "ev_pii", "ev_tox", "ev_all6"]

PREPARE = (
    "from detoxify import Detoxify; Detoxify('original'); "
    "import spacy; spacy.load('en_core_web_lg'); print('model files ready')"
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--configs", default="", help="comma-separated; default: every TraceVerde configuration"
    )
    p.add_argument(
        "--with-openlit-guards",
        action="store_true",
        help="also measure OpenLIT's local guards (needs the openlit environment)",
    )
    p.add_argument(
        "--sizes",
        default="S,M,L",
        help="prompt/response sizes: S about 50, M about 500, L about 4,000 tokens",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=2.0,
        help="seconds of calls per cell; slow cells use at least 20 calls",
    )
    p.add_argument(
        "--quick", action="store_true", help="five configurations, sizes S and M, short cells"
    )
    p.add_argument(
        "--python",
        default="3.12",
        help="Python version uv creates the environments with (default 3.12, the version the pins "
        "were resolved on); ignored by the venv + pip fallback, which uses this interpreter",
    )
    p.add_argument(
        "--skip-prepare", action="store_true", help="model files are already in the local caches"
    )
    p.add_argument("--out", default=None)
    p.add_argument(
        "--env-dir",
        default=None,
        help="where to create the environments (default: .venvs next to this file)",
    )
    args = p.parse_args()

    configs = [c for c in args.configs.split(",") if c] or list(TRACEVERDE)
    sizes, budget = args.sizes, args.budget
    if args.quick:
        configs = [c for c in args.configs.split(",") if c] or list(QUICK)
        sizes, budget = "S,M", 0.5
    if args.with_openlit_guards:
        configs += OPENLIT

    out_dir = args.out or os.path.join(
        ROOT, "results", "eval-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    work_dir = os.path.join(out_dir, "work")
    os.makedirs(work_dir, exist_ok=True)

    envs.set_env_dir(args.env_dir)
    interpreters = {"ev": envs.ensure("traceverde-eval", args.python)}
    if any(c.startswith("og_") for c in configs):
        interpreters["og"] = envs.ensure("openlit", args.python)

    # Model caches (HF_HOME and friends) are honoured; everything else is scrubbed.
    env = envs.clean_env(keep_prefixes=("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME"))
    if not args.skip_prepare:
        print("preparing model files (one-off download if they are not cached yet)", flush=True)
        rc = subprocess.call([interpreters["ev"], "-c", PREPARE], cwd=ROOT, env=env)
        if rc != 0:
            raise SystemExit("model preparation failed; see the output above")
    env.update(
        BENCH_EVAL_SIZES=sizes,
        BENCH_EVAL_BUDGET_S=str(budget),
        BENCH_WORKDIR=work_dir,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )

    for i, cfg in enumerate(configs, 1):
        out = os.path.join(out_dir, "eval_%s.json" % cfg)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        py = interpreters["og" if cfg.startswith("og_") else "ev"]
        with open(os.path.join(out_dir, "eval_%s.log" % cfg), "w", encoding="utf-8") as log:
            rc = subprocess.call(
                [py, "-m", "benchlib.eval_worker", cfg, out],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(
            "[%d/%d] %-18s %s"
            % (
                i,
                len(configs),
                cfg,
                "ok" if rc == 0 else "FAILED rc=%d, see eval_%s.log" % (rc, cfg),
            )
        )
        sys.stdout.flush()

    subprocess.call([sys.executable, "-m", "benchlib.eval_report", out_dir], cwd=ROOT)
    print("results: %s" % out_dir)


if __name__ == "__main__":
    main()
