"""Runs the GPU test programs with and without cuscope and reads their reports.

Programs live in $CUSCOPE_BUILD/tests/cuda (default: <repo>/build).
"""

import json
import os
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("CUSCOPE_BUILD", REPO / "build"))
CUDA_DIR = BUILD / "tests" / "cuda"
SOURCE_DIR = REPO / "tests" / "cuda"
TIMEOUT_S = 120


@dataclass
class Run:
    stdout: str
    stderr: str
    code: int
    trace: dict | None  # the whole trace file; trace["cuscope"] is the report section


def run(prog: str, *args: str, cuscope: bool = True, mode: str = "full",
        out: Path | None = None, env: dict | None = None) -> Run:
    """Runs build/tests/cuda/<prog>. With cuscope, the trace goes to `out` (a temp file by default)."""
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    tmp = None
    if cuscope:
        if out is None:
            tmp = tempfile.TemporaryDirectory(prefix=f"cuscope-{prog}-")
            out = Path(tmp.name) / "trace.json"
        run_env.update(LD_PRELOAD=str(BUILD / "libcuscope.so"), CUSCOPE_MODE=mode, CUSCOPE_OUT=str(out))
    run_env.update(env or {})

    try:
        proc = subprocess.run([str(CUDA_DIR / prog), *args], env=run_env, capture_output=True,
                              text=True, timeout=TIMEOUT_S)
        trace = None
        if cuscope and out is not None and out.exists():
            trace = json.loads(out.read_text())
        return Run(stdout=proc.stdout, stderr=proc.stderr, code=proc.returncode, trace=trace)
    finally:
        if tmp is not None:
            tmp.cleanup()


def assert_passthrough(prog: str, *args: str) -> None:
    """The program's stdout and exit code are the same with and without cuscope."""
    plain = run(prog, *args, cuscope=False)
    assert plain.code == 0, f"{prog} fails even without cuscope:\n{plain.stderr}"
    under = run(prog, *args)
    assert under.code == plain.code, f"exit code {plain.code} -> {under.code}\n{under.stderr}"
    assert under.stdout == plain.stdout, f"stdout changed under cuscope:\n{plain.stdout!r}\n{under.stdout!r}"


def normalize(report: dict) -> dict:
    """The parts of a report that must match between two runs of the same program:
    no pointers, times or stack ids."""
    return {
        "leaks": sorted(Counter((l["bytes"], l["kind"]) for l in report["leaks"]).items()),
        "invalid_frees": len(report["invalid_frees"]),
        "peak_bytes": report["peak_bytes"],
        "alloc_count": report["alloc_count"],
        "free_count": report["free_count"],
        "kernels": sorted((k["name"], k["count"]) for k in report["kernels"]),
        "copies": sorted((c["dir"], c["pinned"], c["count"], c["bytes"]) for c in report["copies"]),
        "graph_launches": report["graph_launches"],
    }


def marker_line(src: str, marker: str) -> int:
    """1-based line number of the line in tests/cuda/<src> containing `// <marker>`."""
    for number, line in enumerate((SOURCE_DIR / src).read_text().splitlines(), start=1):
        if f"// {marker}" in line:
            return number
    raise ValueError(f"marker {marker!r} not found in {src}")
