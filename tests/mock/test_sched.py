"""Compute sharing: the scheduler daemon (cuscope-sched) and the jobs that use it."""

import os
import subprocess
from pathlib import Path

from conftest import SCHED


def test_second_daemon_refuses(sched_daemon):
    r = subprocess.run([str(SCHED)], env={**os.environ, **sched_daemon.env}, capture_output=True, text=True,
                       timeout=5)
    assert r.returncode == 1 and "already running" in r.stderr


def test_status_without_daemon(tmp_name_env):
    r = subprocess.run([str(SCHED), "--status"], env={**os.environ, **tmp_name_env}, capture_output=True,
                       text=True, timeout=5)
    assert r.returncode == 1 and "no scheduler running" in r.stderr


def test_sigterm_unlinks_the_segment(sched_daemon):
    sched_daemon.stop()
    assert not Path("/dev/shm" + sched_daemon.name).exists()


def test_takes_over_after_a_crashed_daemon(sched_daemon):
    sched_daemon.kill()  # SIGKILL: the segment stays behind with a dead daemon pid
    assert Path("/dev/shm" + sched_daemon.name).exists()
    sched_daemon.start()  # a new daemon must start, not refuse
    assert sched_daemon.status() == []


def test_cli_sched_finds_the_daemon(tmp_name_env):
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, **tmp_name_env}
    env.pop("CUSCOPE_SCHED", None)
    r = subprocess.run(["python3", str(repo / "cli" / "cuscope.py"), "sched", "--status"], env=env,
                       capture_output=True, text=True, timeout=10)
    assert r.returncode == 1 and "no scheduler running" in r.stderr


# ---- Task 4: jobs register, release, and fail open ----

from conftest import wait_until  # noqa: E402


def test_registers_and_frees_its_slot(sched_daemon, mimic):
    r = mimic("share_loop", "shim", env={**sched_daemon.env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "0.3"})
    assert r.code == 0, r.stderr
    assert r.trace["cuscope"]["sched"]["registered"] is True
    assert r.trace["cuscope"]["sched"]["weight"] == 50
    assert sched_daemon.status() == []  # released at exit


def test_no_daemon_runs_unthrottled(mimic, tmp_name_env):
    r = mimic("share_loop", "shim", env={**tmp_name_env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "0.3"})
    assert r.code == 0, r.stderr
    sched = r.trace["cuscope"]["sched"]
    assert (sched["registered"], sched["reason"], sched["waits"]) == (False, "no scheduler running", 0)
    assert r.stderr.count("compute sharing off: no scheduler running") == 1


def test_killed_job_slot_is_freed(sched_daemon, mimic_popen):
    proc = mimic_popen("share_loop", env={**sched_daemon.env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "30"})
    wait_until(lambda: len(sched_daemon.status()) == 1, timeout=5)
    proc.kill()
    proc.wait()
    wait_until(lambda: sched_daemon.status() == [], timeout=1.0)


def test_no_share_no_sched(mimic):
    r = mimic("share_loop", "shim", env={"MIMIC_SECONDS": "0.2"})
    assert r.code == 0 and r.trace["cuscope"]["sched"] is None
