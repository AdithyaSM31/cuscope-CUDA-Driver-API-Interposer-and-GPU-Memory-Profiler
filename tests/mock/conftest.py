"""Fixtures that run the mock-driver test programs with and without cuscope.

The mock libcuda lives in build/mock/<mode>/libcuda.so.1:
  direct - the mock driver itself
  shim   - a loader that dlopen()s the mock driver and dlsym()s every function
           in its constructor, the way the WSL libcuda.so.1 does
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("CUSCOPE_BUILD", REPO / "build"))
TIMEOUT_S = 10

SCENARIOS = ["dlsym_gpa", "dlsym_default", "ptsz", "old_version", "gpa_self",
             "capture", "launch", "copy", "teardown", "fork"]
MODES = ["direct", "shim"]


@dataclass
class Result:
    code: int
    stdout: str
    stderr: str
    stats: dict | None
    mock: dict | None
    census: list = field(default_factory=list)   # rows of the census TSV
    trace: dict | None = None                     # trace written by the main process
    traces: dict = field(default_factory=dict)    # pid -> trace, every process


def _tagged_json(stdout: str, tag: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    return None


def _run(argv: list[str], mode: str, preload: bool, env: dict | None, tmp: Path) -> Result:
    run_env = dict(os.environ)
    run_env.pop("LD_PRELOAD", None)
    run_env["LD_LIBRARY_PATH"] = str(BUILD / "mock" / mode)
    run_env["CUSCOPE_OUT"] = str(tmp / "t-%p.json")
    run_env["CUSCOPE_CENSUS"] = str(tmp / "census.tsv")
    if preload:
        run_env["LD_PRELOAD"] = str(BUILD / "libcuscope.so")
    run_env.update(env or {})

    try:
        proc = subprocess.Popen(argv, env=run_env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(f"{argv} timed out after {TIMEOUT_S}s (deadlock or recursion?)\n{err}")

    census_file = tmp / "census.tsv"
    census = [l.split("\t") for l in census_file.read_text().splitlines()] if census_file.exists() else []

    traces = {}
    for f in tmp.glob("t-*.json"):
        traces[int(f.stem[2:])] = json.loads(f.read_text())

    return Result(code=proc.returncode, stdout=out, stderr=err,
                  stats=_tagged_json(out, "STATS"), mock=_tagged_json(out, "MOCK"),
                  census=census, trace=traces.get(proc.pid), traces=traces)


@pytest.fixture
def mimic(tmp_path_factory):
    """mimic(scenario, mode="direct", preload=True, env=None) -> Result"""
    def run(scenario: str, mode: str = "direct", preload: bool = True, env: dict | None = None) -> Result:
        tmp = tmp_path_factory.mktemp(f"{scenario}-{mode}")
        return _run([str(BUILD / "mock" / "runtime_mimic"), scenario], mode, preload, env, tmp)
    return run


@pytest.fixture
def direct_mimic(tmp_path_factory):
    """direct_mimic(preload=True, env=None) -> Result; the program is linked with -lcuda."""
    def run(preload: bool = True, env: dict | None = None) -> Result:
        tmp = tmp_path_factory.mktemp("direct_mimic")
        return _run([str(BUILD / "mock" / "direct_mimic")], "direct", preload, env, tmp)
    return run
