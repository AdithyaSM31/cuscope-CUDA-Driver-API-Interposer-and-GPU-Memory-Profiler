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

SCENARIOS = ["dlsym_gpa", "dlsym_default", "via_cudart", "ptsz", "old_version", "gpa_self",
             "capture", "launch", "launch8", "launch200", "copy", "teardown", "release_destroys", "release_keeps", "release_keeps_launch", "fork", "fork_quiet",
             "cap_exact", "cap_threads", "cap_pinned", "cap_pitch", "cap_driver_fail", "cap_double_free", "cap_teardown", "cap_fork", "cap_meminfo", "cap_reuse", "share_loop", "launch_fail"]
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
        # Sanitizer builds: the sanitizer runtime must be loaded before any instrumented library.
        if os.environ.get("CUSCOPE_PRELOAD_FIRST"):
            run_env["LD_PRELOAD"] = os.environ["CUSCOPE_PRELOAD_FIRST"] + " " + run_env["LD_PRELOAD"]
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


# ---- Compute sharing: the scheduler daemon ----

import signal
import time
import uuid

SCHED = BUILD / "cuscope-sched"


def wait_until(predicate, timeout: float, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        time.sleep(step)


class SchedDaemon:
    """A cuscope-sched with a segment name of its own (CUSCOPE_SCHED_NAME)."""

    def __init__(self) -> None:
        self.name = f"/cuscope-test-{uuid.uuid4().hex[:12]}"
        self.env = {"CUSCOPE_SCHED_NAME": self.name}
        self.proc = None

    def start(self) -> None:
        env = {**os.environ, **self.env}
        self.proc = subprocess.Popen([str(SCHED)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     text=True)
        wait_until(lambda: self.proc.poll() is not None or Path("/dev/shm" + self.name).exists(), timeout=5)
        assert self.proc.poll() is None, self.proc.stderr.read()
        wait_until(lambda: self._status_run().returncode == 0, timeout=5)

    def _status_run(self) -> subprocess.CompletedProcess:
        return subprocess.run([str(SCHED), "--status"], env={**os.environ, **self.env}, capture_output=True,
                              text=True, timeout=5)

    def status(self) -> list[dict]:
        """One dict per used slot: slot, pid, weight, balance_ns, solo, waiting."""
        r = self._status_run()
        assert r.returncode == 0, r.stderr
        keys = ["slot", "pid", "weight", "balance_ns", "solo", "waiting"]
        return [dict(zip(keys, map(int, line.split()))) for line in r.stdout.splitlines() if line.strip()]

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=5)

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def cleanup(self) -> None:
        self.kill()
        Path("/dev/shm" + self.name).unlink(missing_ok=True)


@pytest.fixture
def sched_daemon():
    daemon = SchedDaemon()
    daemon.start()
    yield daemon
    daemon.cleanup()


@pytest.fixture
def tmp_name_env():
    """A segment name no daemon owns."""
    name = f"/cuscope-test-{uuid.uuid4().hex[:12]}"
    yield {"CUSCOPE_SCHED_NAME": name}
    Path("/dev/shm" + name).unlink(missing_ok=True)


@pytest.fixture
def mimic_popen(tmp_path_factory):
    """mimic_popen(scenario, mode="shim", env=None) -> Popen, preloaded like mimic()."""
    procs = []

    def start(scenario: str, mode: str = "shim", env: dict | None = None) -> subprocess.Popen:
        tmp = tmp_path_factory.mktemp(f"{scenario}-bg")
        run_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        run_env.update(LD_LIBRARY_PATH=str(BUILD / "mock" / mode), CUSCOPE_OUT=str(tmp / "t-%p.json"),
                       LD_PRELOAD=str(BUILD / "libcuscope.so"))
        if os.environ.get("CUSCOPE_PRELOAD_FIRST"):
            run_env["LD_PRELOAD"] = os.environ["CUSCOPE_PRELOAD_FIRST"] + " " + run_env["LD_PRELOAD"]
        run_env.update(env or {})
        proc = subprocess.Popen([str(BUILD / "mock" / "runtime_mimic"), scenario], env=run_env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(proc)
        return proc

    yield start
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.fixture
def mimic_pair(tmp_path_factory):
    """mimic_pair(scenario, weights=(a, b), mode="shim", env=None) -> (Result, Result), run at the same time."""
    from concurrent.futures import ThreadPoolExecutor

    def run(scenario: str, weights: tuple, mode: str = "shim", env: dict | None = None):
        def one(weight):
            tmp = tmp_path_factory.mktemp(f"{scenario}-w{weight}")
            return _run([str(BUILD / "mock" / "runtime_mimic"), scenario], mode, True,
                        {**(env or {}), "CUSCOPE_SHARE": str(weight)}, tmp)
        with ThreadPoolExecutor(max_workers=2) as pool:
            return tuple(pool.map(one, weights))
    return run
