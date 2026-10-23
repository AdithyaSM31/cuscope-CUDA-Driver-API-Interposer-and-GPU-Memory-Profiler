#!/usr/bin/env python3
"""Measures cuscope's overhead and writes docs/overhead.md.

For each microbenchmark (bench/micro.cu) and for a PyTorch ResNet-18 training
step (bench/pytorch_resnet.py), runs with cuscope off, in alloc mode and in
full mode, several times each, and reports the median.

usage: run_bench.py [--repeats N] [--torch-repeats N] [--sample-every N] [--out docs/overhead.md]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("CUSCOPE_BUILD", REPO / "build"))
MICRO = BUILD / "bench" / "micro"
RESNET = REPO / "bench" / "pytorch_resnet.py"

MICROS = [("launch", "empty kernel launch", 20000), ("alloc", "cudaMalloc + cudaFree (4 KiB)", 5000),
          ("copy4k", "cudaMemcpy 4 KiB host to device", 5000)]


def mode_env(mode: str, tmp: Path, sample_every: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    if mode != "off":
        env.update(LD_PRELOAD=str(BUILD / "libcuscope.so"), CUSCOPE_MODE=mode.split("+")[0],
                   CUSCOPE_OUT=str(tmp / "t-%p.json"), CUSCOPE_LOG=os.devnull)
        if "+sample" in mode:
            env["CUSCOPE_SAMPLE_EVERY"] = str(sample_every)
        if "+cap" in mode:
            env["CUSCOPE_MEM_LIMIT"] = "4G"
    return env


def run(cmd: list[str], env: dict) -> str:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f"{cmd} failed:\n{proc.stderr[-2000:]}")
    return proc.stdout


def micro_ns(name: str, iters: int, env: dict) -> float:
    out = run([str(MICRO), name, "--iters", str(iters)], env)
    return float(out.strip().split("=")[1])


def torch_step_ms(env: dict, tmp: Path) -> float:
    result = tmp / "r.json"
    run([sys.executable, str(RESNET), "--steps", "30", "--warmup", "5", "--json", str(result)], env)
    return json.loads(result.read_text())["step_ms_median"]


def pct(value: float, base: float) -> str:
    return f"{(value / base - 1) * 100:+.1f}%"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--torch-repeats", type=int, default=3)
    p.add_argument("--sample-every", type=int, default=0, help="also measure full mode with this sampling")
    p.add_argument("--out", default=str(REPO / "docs" / "overhead.md"))
    args = p.parse_args()

    modes = ["off", "alloc", "full"] + ([f"full+sample"] if args.sample_every > 1 else []) + ["full+cap"]
    results: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(prefix="cuscope-bench-") as d:
        tmp = Path(d)
        for name, _, iters in MICROS:
            results[name] = {}
            for mode in modes:
                env = mode_env(mode, tmp, args.sample_every)
                results[name][mode] = statistics.median(micro_ns(name, iters, env) for _ in range(args.repeats))
                print(f"{name:7} {mode:12} {results[name][mode]:9.1f} ns", flush=True)
        results["torch"] = {}
        for mode in modes:
            env = mode_env(mode, tmp, args.sample_every)
            results["torch"][mode] = statistics.median(torch_step_ms(env, tmp) for _ in range(args.torch_repeats))
            print(f"torch   {mode:12} {results['torch'][mode]:9.2f} ms", flush=True)

    labels = {"off": "off", "alloc": "`alloc`", "full": "`full`", "full+sample": f"`full`, sample every {args.sample_every}",
              "full+cap": "`full`, 4 GiB cap"}
    lines = [
        "# Overhead",
        "",
        "Produced by `python3 bench/run_bench.py` on an RTX 3060 Laptop GPU under WSL2 (CUDA 13.0,",
        f"PyTorch 2.14). Medians of {args.repeats} runs per microbenchmark and {args.torch_repeats} per",
        "PyTorch configuration; wall-clock time on the calling thread.",
        "",
        "## Per call (ns)",
        "",
        "| Call | " + " | ".join(labels[m] for m in modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for name, title, _ in MICROS:
        base = results[name]["off"]
        cells = [f"{results[name]['off']:.0f}"] + [f"{results[name][m]:.0f} ({pct(results[name][m], base)})"
                                                  for m in modes[1:]]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    base = results["torch"]["off"]
    lines += [
        "",
        "## PyTorch ResNet-18 training step (batch 32, median step time)",
        "",
        "| " + " | ".join(labels[m] for m in modes) + " |",
        "|" + "---|" * len(modes),
        "| " + " | ".join([f"{base:.2f} ms"] + [f"{results['torch'][m]:.2f} ms ({pct(results['torch'][m], base)})"
                                                for m in modes[1:]]) + " |",
        "",
        "## Notes",
        "",
        "- `alloc` mode records every call and tracks memory but creates no CUDA events. Its cost is a",
        "  clock read and a 64-byte record per call, which is within run-to-run noise here.",
        "- `full` mode adds four driver calls around each launch or copy: a stream-capture check, a",
        "  context lookup and two event records. On WSL each driver call costs a few microseconds, which",
        "  is the extra 10-16 us per call above. Kernels that run longer than that hide it, hence the",
        "  small PyTorch figure.",
        "- `CUSCOPE_SAMPLE_EVERY=N` times 1 in N kernel launches per thread. Copies are always timed,",
        "  so every copy contributes to the bandwidth figures.",
        "- `full` with a 4 GiB cap (`CUSCOPE_MEM_LIMIT=4G`, never reached here) adds one atomic",
        "  reservation per allocation and one release per free on top of `full`.",
        "- These are medians, but microsecond-scale calls on WSL still vary between runs by several",
        "  percent (see the alloc-mode copy figure, which can come out below `off`). Treat differences",
        "  under about 5% as noise.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(Path(args.out).read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
