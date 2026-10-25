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
