"""The CLI on real programs: file:line for leaks, and a clean stdout."""

import os
import subprocess
import sys
from pathlib import Path

from harness import BUILD, CUDA_DIR, REPO, marker_line

CLI = REPO / "cli" / "cuscope.py"


def cuscope(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    env["CUSCOPE_LIB"] = str(BUILD / "libcuscope.so")
    return subprocess.run([sys.executable, str(CLI), *args], env=env, capture_output=True, text=True, timeout=120)


def test_leak_file_line(tmp_path):
    out = tmp_path / "leak.json"
    r = cuscope("run", "--out", str(out), "--", str(CUDA_DIR / "leak"))
    assert r.returncode == 0, r.stderr
    s = cuscope("summary", str(out))
    assert s.returncode == 0, s.stderr
    for marker in ("LEAK_A", "LEAK_C"):
        assert f"leak.cu:{marker_line('leak.cu', marker)}" in s.stdout, s.stdout


def test_summary_on_stderr_only(tmp_path):
    plain = subprocess.run([str(CUDA_DIR / "multistream")], capture_output=True, text=True, timeout=120)
    r = cuscope("run", "--out", str(tmp_path / "t.json"), "--", str(CUDA_DIR / "multistream"))
    assert r.returncode == plain.returncode == 0
    assert r.stdout == plain.stdout
    assert "== cuscope summary" in r.stderr and "spin_kernel(long long)" in r.stderr
    assert f"trace written to {tmp_path / 't.json'}" in r.stderr
