#!/usr/bin/env python3
"""Cross-checks cuscope against PyTorch's own numbers on ResNet-18 training.

Runs bench/pytorch_resnet.py:
  A  under cuscope, 50 steps          peak device memory vs torch max_memory_reserved()
  B  under cuscope, 10 steps          launches per training step = (A - B) / 40
  C  under cuscope, caching disabled  PYTORCH_NO_CUDA_MEMORY_CACHING=1: per-tensor allocations
  D  without cuscope, profiled        torch.profiler kernels per step, if it works here

and writes the measured tables into docs/validation.md. Each device allocation
is attributed to the library that made it (the first frame of its stack
outside the CUDA runtime), which is what explains any gap: PyTorch's caching
allocator lives in libc10_cuda.so, so bytes allocated anywhere else are not in
max_memory_reserved().

usage: validate_pytorch.py [--batch N] [--out docs/validation.md]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "cli" / "cuscope.py"
RESNET = REPO / "bench" / "pytorch_resnet.py"
MiB = 1 << 20


def run_resnet(tmp: Path, name: str, steps: int, batch: int, under_cuscope: bool, env: dict | None = None,
               profile: int = 0) -> tuple[dict, dict | None]:
    result_json, trace_json = tmp / f"{name}.result.json", tmp / f"{name}.trace.json"
    cmd = [sys.executable, str(RESNET), "--steps", str(steps), "--batch", str(batch), "--json", str(result_json)]
    if profile:
        cmd += ["--profile", str(profile)]
    if under_cuscope:
        cmd = [sys.executable, str(CLI), "run", "--mode", "full", "--out", str(trace_json),
               "--log", str(tmp / f"{name}.log"), "--", *cmd]
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_")}
    run_env.update(env or {})
    proc = subprocess.run(cmd, env=run_env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    trace = json.loads(trace_json.read_text())["cuscope"] if under_cuscope else None
    return json.loads(result_json.read_text()), trace


def owner(frames: list[dict]) -> str:
    """The library that made an allocation: the first frame outside the CUDA runtime and driver."""
    for f in frames:
        module = Path(f["module"]).name
        if module.startswith(("libcudart", "libcuda.", "libcuscope")) or f["symbol"].startswith("cuda"):
            continue
        return module.split(".so")[0] + ".so" if ".so" in module else module
    return "unknown"


def device_bytes_by_owner(report: dict) -> dict[str, tuple[int, int]]:
    """Device-resident allocations still live at exit, by owner: (count, bytes)."""
    out = defaultdict(lambda: [0, 0])
    for leak in report["leaks"]:
        if leak["kind"] == "host_pinned":
            continue
        who = owner(report["stacks"].get(str(leak["stack"]), []))
        out[who][0] += 1
        out[who][1] += leak["bytes"]
    return {k: tuple(v) for k, v in sorted(out.items(), key=lambda kv: -kv[1][1])}


def mib(n: float) -> str:
    return f"{n / MiB:.1f} MiB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default=str(REPO / "docs" / "validation.md"))
    args = p.parse_args()

    with tempfile.TemporaryDirectory(prefix="cuscope-validate-") as d:
        tmp = Path(d)
        print("A: 50 steps under cuscope ...", flush=True)
        res_a, trace_a = run_resnet(tmp, "A", 50, args.batch, True)
        print("B: 10 steps under cuscope ...", flush=True)
        res_b, trace_b = run_resnet(tmp, "B", 10, args.batch, True)
        print("C: 10 steps, caching allocator disabled ...", flush=True)
        res_c, trace_c = run_resnet(tmp, "C", 10, args.batch, True, env={"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"})
        print("D: torch.profiler, without cuscope ...", flush=True)
        res_d, _ = run_resnet(tmp, "D", 10, args.batch, False, profile=5)

    peaks = trace_a["peak_bytes"]
    reserved = res_a["max_memory_reserved"]
    owners_a = device_bytes_by_owner(trace_a)
    caching = owners_a.get("libc10_cuda.so", (0, 0))[1]
    outside = sum(b for who, (_, b) in owners_a.items() if who != "libc10_cuda.so")
    launches = lambda t: sum(k["count"] for k in t["kernels"])  # noqa: E731
    per_step = (launches(trace_a) - launches(trace_b)) / (res_a["steps"] - res_b["steps"])
    kernel_ns = lambda t: sum(k["gpu_ns_total"] for k in t["kernels"])  # noqa: E731
    gpu_us_per_step = (kernel_ns(trace_a) - kernel_ns(trace_b)) / (res_a["steps"] - res_b["steps"]) / 1e3

    lines = [
        "# Validation: cuscope vs PyTorch",
        "",
        "Produced by `python3 tools/validate_pytorch.py` (ResNet-18, random 3x224x224 data, SGD with momentum,",
        f"batch {args.batch}, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`). RTX 3060 Laptop GPU, WSL2,",
        "CUDA 13.0, PyTorch 2.14.0+cu130.",
        "",
        "## Peak device memory (run A: 5 warmup + 50 steps)",
        "",
        "| Measure | Bytes | MiB |",
        "|---|---|---|",
        f"| `torch.cuda.max_memory_reserved()` | {reserved} | {mib(reserved)} |",
        f"| `torch.cuda.max_memory_allocated()` | {res_a['max_memory_allocated']} | {mib(res_a['max_memory_allocated'])} |",
        f"| cuscope peak, all device-resident kinds | {peaks['device_total']} | {mib(peaks['device_total'])} |",
        f"| cuscope peak, `cuMemAlloc` (device) only | {peaks['device']} | {mib(peaks['device'])} |",
        f"| **gap** (cuscope total - reserved) | {peaks['device_total'] - reserved} | "
        f"{mib(peaks['device_total'] - reserved)} |",
        "",
        "## Where the device memory came from (allocations live at exit, by allocating library)",
        "",
        "PyTorch's caching allocator never returns its blocks before exit, so every allocation it made is",
        "still live at exit; cuscope attributes each one to the first library in its call stack outside",
        "the CUDA runtime.",
        "",
        "| Library | Allocations | Bytes | MiB |",
        "|---|---|---|---|",
        *[f"| `{who}` | {n} | {b} | {mib(b)} |" for who, (n, b) in owners_a.items()],
        "",
        f"Caching allocator (`libc10_cuda.so`): {mib(caching)} "
        f"({'matches' if caching == reserved else 'differs from'} max_memory_reserved {mib(reserved)}). "
        f"Everything else: {mib(outside)}.",
        "",
        "## Caching allocator disabled (run C: `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, 10 steps)",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| cuscope allocations / frees | {trace_c['alloc_count']} / {trace_c['free_count']} |",
        f"| cuscope peak device memory | {mib(trace_c['peak_bytes']['device_total'])} |",
        f"| `max_memory_allocated()` | {mib(res_c['max_memory_allocated'])} |",
        f"| allocations with caching on (run A, 55 steps) | {trace_a['alloc_count']} |",
        "",
        "## Kernels",
        "",
        "| Measure | cuscope | torch.profiler |",
        "|---|---|---|",
    ]
    prof = res_d.get("profiler")
    lines.append(f"| kernel launches per step | {per_step:.1f} | "
                 f"{prof['kernels_per_step']:.1f} |" if prof else
                 f"| kernel launches per step | {per_step:.1f} | unavailable: {res_d.get('profiler_error', '?')} |")
    if prof:
        lines.append(f"| GPU kernel time per step | {gpu_us_per_step / 1e3:.2f} ms | "
                     f"{prof['kernel_us_per_step'] / 1e3:.2f} ms |")
    lines += [
        f"| completeness (run A) | dropped {trace_a['dropped_records']}, unresolved {trace_a['unresolved_events']},"
        f" untimed {trace_a['gpu_untimed']} | |",
        "",
        "## Gap analysis",
        "",
        "_Written by the author after reading the tables above: what each difference is and why._",
        "",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(Path(args.out).read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
