#!/usr/bin/env python3
"""Day 2 symbol census: which driver functions do real programs look up, and how?

  census.py run      run hello_static, hello_shared and a PyTorch matmul with and
                     without cuscope, writing docs/census/<program>.tsv and runs.json
  census.py check    evaluate the Day 2 gate; exit 1 if any line fails
  census.py summary  print a Markdown summary of the census (for SUMMARY.md)

TSV rows come from src/census.cpp:
  dlsym  symbol  caller-module  action
  gpa    symbol  cudaVersion  flags  CUresult  symbolStatus  action
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PYTORCH_SNIPPET = ("import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; "
                   "torch.cuda.synchronize(); print(y.sum().item()!=0)")


def programs(build: Path) -> dict[str, list[str]]:
    return {
        "static": [str(build / "tests/cuda/hello_static")],
        "shared": [str(build / "tests/cuda/hello_shared")],
        "pytorch": [sys.executable, "-c", PYTORCH_SNIPPET],
    }


# Spec §5 Phase 1 hook list (base names).
HOOK_LIST = [
    "cuMemAlloc", "cuMemAllocPitch", "cuMemFree", "cuMemAllocAsync", "cuMemAllocFromPoolAsync",
    "cuMemFreeAsync", "cuMemAllocManaged", "cuMemAllocHost", "cuMemHostAlloc", "cuMemFreeHost",
    "cuMemcpy", "cuMemcpyAsync", "cuMemcpyHtoD", "cuMemcpyDtoH", "cuMemcpyDtoD",
    "cuMemcpyHtoDAsync", "cuMemcpyDtoHAsync", "cuMemcpyDtoDAsync",
    "cuLaunchKernel", "cuLaunchKernelEx", "cuGraphLaunch",
    "cuStreamSynchronize", "cuCtxSynchronize", "cuCtxDestroy", "cuDevicePrimaryCtxRelease",
    "cuDevicePrimaryCtxReset",
    "cuFuncGetName", "cuModuleGetFunction", "cuLibraryGetKernel",
]

# Families where a function missing from the hook list could hide allocations,
# copies or launches.
RELEVANT = re.compile(r"^cu(Mem(Alloc|Free|cpy|set|Host|Pool|Create|Map|Release|AddressReserve)"
                      r"|Launch|Graph(Exec)?Launch|Library(Load|GetKernel)|Module(Load|GetFunction)"
                      r"|KernelGetFunction|StreamSynchronize|CtxSynchronize|CtxDestroy|DevicePrimaryCtx)")

FLAG_NAMES = {0: "default", 1: "legacy", 2: "per-thread"}


def read_tsv(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines() if line]


def run_one(argv: list[str], census: Path | None, build: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    env.pop("CUSCOPE_CENSUS", None)
    if census is not None:
        env["LD_PRELOAD"] = str(build / "libcuscope.so")
        env["CUSCOPE_CENSUS"] = str(census)
        env["CUSCOPE_LOG"] = os.devnull
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout


def cmd_run(args) -> int:
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, argv in programs(Path(args.build)).items():
        tsv = out / f"{name}.tsv"
        tsv.unlink(missing_ok=True)
        base_rc, base_out = run_one(argv, None, Path(args.build))
        rc, stdout = run_one(argv, tsv, Path(args.build))
        runs[name] = {"argv": argv[-1] if name == "pytorch" else Path(argv[0]).name,
                      "rc_without": base_rc, "rc_with": rc, "stdout_identical": stdout == base_out}
        print(f"{name}: rc {base_rc}/{rc}, stdout {'identical' if stdout == base_out else 'DIFFERENT'}, "
              f"{len(read_tsv(tsv)) if tsv.exists() else 0} census rows")
    (out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    return 0


def cmd_check(args) -> int:
    d = Path(args.dir)
    runs = json.loads((d / "runs.json").read_text())
    tsv = {name: read_tsv(d / f"{name}.tsv") for name in runs}
    results = []

    for name, r in runs.items():
        ok = r["rc_without"] == 0 and r["rc_with"] == 0 and r["stdout_identical"]
        results.append((ok, f"{name}: exit 0 and stdout identical with and without cuscope"))

    results.append((any(row[0] == "gpa" and row[1] == "cuMemAlloc" for row in tsv["static"]),
                    "static: cuMemAlloc arrives through cuGetProcAddress"))

    for name in ("static", "shared"):
        results.append((any(row[:2] == ["dlsym", "cuGetProcAddress_v2"] and row[3] == "wrap"
                            for row in tsv[name]),
                        f"{name}: the runtime's dlsym(cuGetProcAddress_v2) gets cuscope's wrapper"))

    for name, rows in tsv.items():
        results.append((any(row[0] == "dlsym" and row[2] == "libcuda.so.1" and row[3] == "caller-libcuda"
                            for row in rows),
                        f"{name}: the WSL shim's own lookups pass through as caller-libcuda"))

    for ok, text in results:
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    return 0 if all(ok for ok, _ in results) else 1


def gpa_requests(rows: list[list[str]]) -> dict[str, set[tuple[int, str]]]:
    """symbol -> {(cudaVersion, flags name)} for successful lookups."""
    out = defaultdict(set)
    for row in rows:
        if row[0] == "gpa" and row[4] == "0":
            out[row[1]].add((int(row[2]), FLAG_NAMES.get(int(row[3]), row[3])))
    return out


def fmt_requests(reqs: set[tuple[int, str]]) -> str:
    return ", ".join(f"{v} {f}" for v, f in sorted(reqs)) if reqs else "—"


def cmd_summary(args) -> int:
    d = Path(args.dir)
    runs = json.loads((d / "runs.json").read_text())
    names = list(runs)
    req = {name: gpa_requests(read_tsv(d / f"{name}.tsv")) for name in names}
    dlsym_rows = {name: [r for r in read_tsv(d / f"{name}.tsv") if r[0] == "dlsym"] for name in names}

    print("## Lookups per program\n")
    print("| Program | Driver functions looked up | dlsym(cuGetProcAddress*) wrapped for | Shim/driver lookups passed through |")
    print("|---|---|---|---|")
    for name in names:
        wrapped = sorted({r[2] for r in dlsym_rows[name] if r[3] == "wrap"})
        passed = sum(1 for r in dlsym_rows[name] if r[3] == "caller-libcuda")
        print(f"| {name} | {len(req[name])} | {', '.join(wrapped) or '—'} | {passed} |")

    print("\n## Phase 1 hook list: requested versions and flags\n")
    print("| Function | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for fn in HOOK_LIST:
        print(f"| `{fn}` | " + " | ".join(fmt_requests(req[n].get(fn, set())) for n in names) + " |")

    seen = set().union(*(set(r) for r in req.values()))
    extra = sorted(s for s in seen if RELEVANT.match(s) and s not in HOOK_LIST)
    print("\n## Requested, relevant, and not on the hook list\n")
    print("| Function | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for fn in extra:
        print(f"| `{fn}` | " + " | ".join(fmt_requests(req[n].get(fn, set())) for n in names) + " |")

    per_thread = sorted({s for n in names for s, reqs in req[n].items() if any(f == "per-thread" for _, f in reqs)})
    print(f"\n## Requested with the per-thread-stream flag\n\n{len(per_thread)} functions: "
          + ", ".join(f"`{s}`" for s in per_thread) + "\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "check", "summary"])
    p.add_argument("--build", default=str(REPO / "build"))
    p.add_argument("--dir", default=str(REPO / "docs/census"))
    args = p.parse_args()
    return {"run": cmd_run, "check": cmd_check, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
