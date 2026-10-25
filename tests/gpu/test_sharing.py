"""Compute sharing on the real GPU: a short version of tools/bench_share.py."""

import json
import os
import subprocess
import uuid
from pathlib import Path

from harness import BUILD, CUDA_DIR

SCHED = BUILD / "cuscope-sched"


def test_two_jobs_share_equally(tmp_path):
    name = f"/cuscope-gputest-{uuid.uuid4().hex[:12]}"
    base = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    base["CUSCOPE_SCHED_NAME"] = name
    daemon = subprocess.Popen([str(SCHED)], env=base, stderr=subprocess.PIPE, text=True)
    try:
        jobs = []
        for job in ("a", "b"):
            env = {**base, "LD_PRELOAD": str(BUILD / "libcuscope.so"), "CUSCOPE_SHARE": "50",
                   "CUSCOPE_OUT": str(tmp_path / f"{job}.json"), "CUSCOPE_LOG": str(tmp_path / f"{job}.log")}
            jobs.append(subprocess.Popen([str(CUDA_DIR / "share_load"), "3", "2", "10"], env=env,
                                         stdout=subprocess.PIPE, text=True))
        for proc in jobs:
            out, _ = proc.communicate(timeout=60)
            assert proc.returncode == 0
            assert json.loads(out.splitlines()[-1])["kernels"] > 0
    finally:
        daemon.terminate()
        daemon.wait(timeout=5)
        Path("/dev/shm" + name).unlink(missing_ok=True)
    charged = [json.loads((tmp_path / f"{job}.json").read_text())["cuscope"]["sched"]["gpu_ns_charged"]
               for job in ("a", "b")]
    share_a = charged[0] / sum(charged)
    assert 0.40 <= share_a <= 0.60, charged
