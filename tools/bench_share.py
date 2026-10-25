#!/usr/bin/env python3
"""The compute-sharing benchmark on the local GPU.

Two jobs (tests/cuda/share_load: batches of spin kernels, each batch waited
for) run under cuscope in these configurations:

  solo plain       job A alone, no sharing             baseline throughput
  solo fail-open   job A alone, --share, no daemon     must match the baseline
  solo scheduled   job A alone, --share, daemon        must match the baseline
  pair none        A and B together, no sharing        the driver's own split
  pair 50/50       A and B, weights 50 and 50          50% +- 5 each
  pair 80/20       A and B, weights 80 and 20          80% +- 5 and 20% +- 5

A job's work is its part of the kernels the pair completed (both run the same
kernel, calibrated once). Its share is its part of the pair's GPU time as
cuscope measured it, which is what the scheduler divides; the two differ when
time-sliced kernels' measured times include the other job's slices.
Each job's share is its part of the pair's GPU time, as cuscope measured it
(the trace's sched.gpu_ns_charged when sharing, kernel GPU time otherwise).
A job's busy fraction is its GPU time over the run's wall time: unlike
kernels per second it does not depend on the laptop GPU's clock, which moves
with power and temperature, so it is what the solo checks compare.
Prints a table, writes docs/sharing.json, exits 1 if a criterion fails.

  python3 tools/bench_share.py [--seconds 20] [--kernel-ms 2] [--batch 10] [--warmup 15]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("CUSCOPE_BUILD", REPO / "build"))
LOAD = BUILD / "tests" / "cuda" / "share_load"
SCHED = BUILD / "cuscope-sched"


def base_env(name: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    env["CUSCOPE_SCHED_NAME"] = name
    return env


def run_jobs(weights: list, args, tmp: Path, name: str) -> list:
    """Runs one share_load per weight (None = no --share) at the same time."""
    procs = []
    for i, weight in enumerate(weights):
        env = {**base_env(name), "LD_PRELOAD": str(BUILD / "libcuscope.so"),
               "CUSCOPE_OUT": str(tmp / f"job{i}.json"), "CUSCOPE_LOG": str(tmp / f"job{i}.log")}
        if weight is not None:
            env["CUSCOPE_SHARE"] = str(weight)
        cmd = [str(LOAD), str(args.seconds), str(args.kernel_ms), str(args.batch), str(args.iters)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True))
    results = []
    for i, proc in enumerate(procs):
        out, _ = proc.communicate(timeout=args.seconds * 10 + 60)
        if proc.returncode != 0:
            raise SystemExit(f"share_load failed ({proc.returncode}): {(tmp / f'job{i}.log').read_text()}")
        load = json.loads(out.splitlines()[-1])
        report = json.loads((tmp / f"job{i}.json").read_text())["cuscope"]
        sched = report.get("sched")
        gpu_ns = sched["gpu_ns_charged"] if sched and sched["registered"] else sum(
            k["gpu_ns_total"] for k in report["kernels"])
        results.append({"weight": weights[i], "kernels_per_s": load["kernels"] / args.seconds,
                        "busy": gpu_ns / (args.seconds * 1e9),
                        "p50_ms": load["p50_ms"], "p99_ms": load["p99_ms"], "gpu_ns": gpu_ns,
                        "waits": sched["waits"] if sched else 0, "wait_ns": sched["wait_ns"] if sched else 0,
                        "registered": bool(sched and sched["registered"])})
    total = sum(r["gpu_ns"] for r in results) or 1
    kernels = sum(r["kernels_per_s"] for r in results) or 1
    for r in results:
        r["share"] = r["gpu_ns"] / total
        r["work"] = r["kernels_per_s"] / kernels  # every job runs the same kernel
    return results


def configuration(label: str, weights: list, daemon: bool, args, tmp: Path) -> dict:
    name = f"/cuscope-bench-{uuid.uuid4().hex[:12]}"
    proc = None
    if daemon:
        proc = subprocess.Popen([str(SCHED)], env=base_env(name), stderr=subprocess.PIPE, text=True)
        time.sleep(0.3)
    try:
        print(f"running {label} ...", flush=True)
        jobs = run_jobs(weights, args, tmp, name)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=5)
        Path("/dev/shm" + name).unlink(missing_ok=True)
    return {"label": label, "daemon": daemon, "jobs": jobs}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--kernel-ms", type=int, default=2)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--warmup", type=float, default=15,
                   help="seconds of load first: a laptop GPU drops from its boost clock under sustained load")
    p.add_argument("--out", default=str(REPO / "docs" / "sharing.json"))
    args = p.parse_args()

    # Warm up to the clock the GPU will hold for the whole benchmark, then
    # calibrate once, alone: every job then runs the same kernel, so their
    # kernels per second compare.
    print(f"warming up for {args.warmup:g} s ...", flush=True)
    subprocess.run([str(LOAD), str(args.warmup), str(args.kernel_ms), str(args.batch)], capture_output=True,
                   check=True, env=base_env("/unused"))
    calib = subprocess.run([str(LOAD), "2", str(args.kernel_ms), str(args.batch)], capture_output=True, text=True,
                           check=True, env=base_env("/unused"))
    args.iters = json.loads(calib.stdout.splitlines()[-1])["iters"]
    print(f"calibrated: {args.iters} iterations per kernel for {args.kernel_ms} ms", flush=True)

    with tempfile.TemporaryDirectory(prefix="cuscope-share-") as d:
        tmp = Path(d)
        configs = [
            configuration("solo plain", [None], False, args, tmp),
            configuration("solo fail-open", [50], False, args, tmp),
            configuration("solo scheduled", [50], True, args, tmp),
            configuration("pair none", [None, None], False, args, tmp),
            configuration("pair 50/50", [50, 50], True, args, tmp),
            configuration("pair 80/20", [80, 20], True, args, tmp),
        ]

    print(f"\n{'configuration':16} {'job':3} {'weight':>6} {'kernels/s':>9} {'busy':>6} {'share':>6} {'work':>6} {'p50 ms':>7} "
          f"{'p99 ms':>7} {'waits':>6}")
    for c in configs:
        for i, j in enumerate(c["jobs"]):
            print(f"{c['label']:16} {'AB'[i]:3} {str(j['weight'] or '-'):>6} {j['kernels_per_s']:9.1f} "
                  f"{100 * j['busy']:5.1f}% "
                  f"{100 * j['share']:5.1f}% {100 * j['work']:5.1f}% {j['p50_ms']:7.2f} {j['p99_ms']:7.2f} {j['waits']:6}")

    by = {c["label"]: c["jobs"] for c in configs}
    base = by["solo plain"][0]["busy"]
    checks = [
        ("fail-open keeps the GPU 95% as busy as solo", by["solo fail-open"][0]["busy"] >= 0.95 * base),
        ("fail-open did not register", not by["solo fail-open"][0]["registered"]),
        ("scheduled solo keeps the GPU 95% as busy as solo", by["solo scheduled"][0]["busy"] >= 0.95 * base),
        ("50/50 measured GPU time within 5 points", all(abs(j["share"] - 0.5) <= 0.05 for j in by["pair 50/50"])),
        ("80/20 measured GPU time within 5 points", abs(by["pair 80/20"][0]["share"] - 0.8) <= 0.05
         and abs(by["pair 80/20"][1]["share"] - 0.2) <= 0.05),
    ]
    print()
    for text, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    # Not a check: under time-slicing the work split is wider than the GPU-time
    # split (docs/sharing.md says why).
    for label in ("pair 50/50", "pair 80/20"):
        print(f"info  {label} work split: " + "/".join(f"{100 * j['work']:.1f}" for j in by[label]))
    Path(args.out).write_text(json.dumps({"seconds": args.seconds, "kernel_ms": args.kernel_ms, "batch": args.batch,
                                          "iters": args.iters, "warmup_s": args.warmup,
                                          "configurations": configs,
                                          "checks": [{"check": t, "ok": ok} for t, ok in checks]}, indent=1) + "\n")
    print(f"\nwritten to {args.out}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
