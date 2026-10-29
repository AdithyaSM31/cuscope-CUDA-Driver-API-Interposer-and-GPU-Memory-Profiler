"""Compute sharing on the real GPU: a short version of tools/bench_share.py."""

import json
import os
import subprocess
import uuid
from pathlib import Path

from harness import BUILD, CUDA_DIR

SCHED = BUILD / "cuscope-sched"


def test_two_jobs_share_by_weight_3_to_1(tmp_path):
    name = f"/cuscope-gputest-{uuid.uuid4().hex[:12]}"
    base = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    base["CUSCOPE_SCHED_NAME"] = name
    daemon = subprocess.Popen([str(SCHED)], env=base, stderr=subprocess.PIPE, text=True)
    try:
        jobs = []
        for job, weight in (("a", "75"), ("b", "25")):
            env = {**base, "LD_PRELOAD": str(BUILD / "libcuscope.so"), "CUSCOPE_SHARE": weight,
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
    scheds = [json.loads((tmp_path / f"{job}.json").read_text())["cuscope"]["sched"]
              for job in ("a", "b")]
    assert scheds[0]["registered"] is True, scheds[0]
    assert scheds[1]["registered"] is True, scheds[1]
    charged = [s["gpu_ns_charged"] for s in scheds]
    share_a = charged[0] / sum(charged)
    assert 0.65 <= share_a <= 0.85, (share_a, charged)
