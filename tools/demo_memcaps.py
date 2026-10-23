#!/usr/bin/env python3
"""The memory-cap demo: two PyTorch training jobs sharing one GPU.

Runs the pair twice under cuscope (allocation-only mode), first uncapped, then
with job A capped at --cap-a and job B at --cap-b. Job B starts --offset
seconds after job A, once A's cache has grown. Both runs use
expandable_segments:False, because expandable segments allocate through the
virtual memory APIs, which cuscope does not hook.

Uncapped, PyTorch's allocator keeps every block it ever reserved and gives
memory back only when one of its own allocations fails, so each job holds far
more than it uses. Together they reserve more than the GPU has. On native
Linux the later job is expected to run out of memory; under WSL2 the Windows driver
pages the excess to system RAM instead, so both survive on oversubscribed
memory. Capped, each job trims its cache at its cap and the pair stays within
the budgets it was given.

  python3 tools/demo_memcaps.py --cap-a 2.25G --cap-b 3G \
      --a-args="--batch 32 --min-res 128 --max-res 320" --b-args="--batch 48 --min-res 128 --max-res 352"

Prints a table and writes docs/memcaps.json.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "cli" / "cuscope.py"
JOB = REPO / "tools" / "memcaps_job.py"
GiB = 1 << 30


def run_pair(caps: dict, offset: float, steps: int, job_args: dict, tmp: Path) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    procs = []
    for name in ("A", "B"):
        cap = caps[name]
        out = tmp / f"{name}-{cap or 'none'}.json"
        cmd = [sys.executable, str(CLI), "run", "--mode", "alloc", "--out", str(tmp / f"trace-{name}-%p.json"),
               "--log", str(tmp / f"log-{name}.txt")]
        if cap:
            cmd += ["--mem-limit", cap]
        cmd += ["--", sys.executable, str(JOB), "--steps", str(steps), "--seed", str(ord(name)), "--json", str(out),
                *job_args[name]]
        procs.append((name, out, subprocess.Popen(cmd, env=env)))
        if name == "A":
            time.sleep(offset)
    results = []
    for name, out, proc in procs:
        proc.wait()
        result = json.loads(out.read_text()) if out.exists() else {"exit": f"crashed ({proc.returncode})"}
        traces = sorted(tmp.glob(f"trace-{name}-*.json"))
        if traces:
            report = json.loads(traces[-1].read_text())["cuscope"]
            result["cuscope_peak_device"] = report["peak_bytes"]["device_total"]
            result["mem_limit"] = report.get("mem_limit")
            for t in traces:
                t.unlink()
        results.append({"job": name, "cap": caps[name], **result})
    return results


def gib(n) -> str:
    return f"{n / GiB:.2f}" if isinstance(n, (int, float)) else "-"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cap-a", required=True, help="job A's cap in the capped run, e.g. 2.25G")
    p.add_argument("--cap-b", required=True, help="job B's cap in the capped run, e.g. 3G")
    p.add_argument("--a-args", default="", help="arguments for job A's tools/memcaps_job.py")
    p.add_argument("--b-args", default="", help="arguments for job B's tools/memcaps_job.py")
    p.add_argument("--offset", type=float, default=10.0, help="seconds between starting job A and job B")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--out", default=str(REPO / "docs" / "memcaps.json"))
    args = p.parse_args()
    job_args = {"A": args.a_args.split(), "B": args.b_args.split()}

    rows = []
    with tempfile.TemporaryDirectory(prefix="cuscope-memcaps-") as tmp:
        for caps in ({"A": None, "B": None}, {"A": args.cap_a, "B": args.cap_b}):
            label = f"capped at {caps['A']} / {caps['B']}" if caps["A"] else "uncapped"
            print(f"running the pair {label} ...", flush=True)
            rows += run_pair(caps, args.offset, args.steps, job_args, Path(tmp))

    print(f"\n{'run':10} {'job':3} {'exit':6} {'steps':>5} {'alloc GiB':>9} {'reserved GiB':>12} "
          f"{'step ms':>8} {'denied':>6}")
    for r in rows:
        cap = r.get("mem_limit") or {}
        step = f"{r['step_ms_median']:.1f}" if r.get("step_ms_median") else "-"
        print(f"{(r['cap'] or 'uncapped'):10} {r['job']:3} {r['exit']:6} {r.get('steps_done', 0):>5} "
              f"{gib(r.get('max_allocated')):>9} {gib(r.get('max_reserved')):>12} {step:>8} "
              f"{cap.get('denied_count', '-'):>6}")
    import torch  # only for the GPU's size, after the jobs have exited
    gpu_total = torch.cuda.mem_get_info()[1]
    uncapped = [r for r in rows if r["cap"] is None]
    capped = [r for r in rows if r["cap"] is not None]
    reserved = lambda runs: sum(r.get("max_reserved") or 0 for r in runs)  # noqa: E731
    within = all(r["exit"] == "ok" and (r.get("mem_limit") or {}).get("peak_charged_bytes", 0)
                 <= r["mem_limit"]["limit_bytes"] for r in capped if r.get("mem_limit"))
    Path(args.out).write_text(json.dumps({"job_args": job_args, "steps": args.steps, "offset_s": args.offset,
                                          "gpu_total_bytes": gpu_total, "runs": rows}, indent=1) + "\n")
    print(f"\nwritten to {args.out}")
    print(f"GPU {gib(gpu_total)} GiB. Uncapped: the pair reserved {gib(reserved(uncapped))} GiB "
          f"({'more than the GPU has: oversubscribed' if reserved(uncapped) > gpu_total else 'within the GPU'}), "
          f"exits {[r['exit'] for r in uncapped]}. Capped: {gib(reserved(capped))} GiB, "
          f"{'both finished within their caps' if within and len(capped) == 2 else 'a job failed or passed its cap'}.")
    return 0 if within and len(capped) == 2 and reserved(capped) <= gpu_total else 1


if __name__ == "__main__":
    sys.exit(main())
