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


# ---- Task 5: launches wait when a job is over its share ----

import threading  # noqa: E402

# Back-to-back launches of 1 ms (mock) kernels: far more GPU time than any
# share. (usleep(10) sleeps ~70 us on WSL, too slow for the 75% job.)
FAST = {"MOCK_KERNEL_NS": "1000000", "MIMIC_GAP_US": "0"}


def test_two_jobs_split_by_weight(sched_daemon, mimic_pair):
    a, b = mimic_pair("share_loop", weights=(3, 1), env={**sched_daemon.env, **FAST, "MIMIC_SECONDS": "2.0"})
    assert a.code == 0 and b.code == 0, a.stderr + b.stderr
    ga, gb = (x.trace["cuscope"]["sched"]["gpu_ns_charged"] for x in (a, b))
    assert 0.65 <= ga / (ga + gb) <= 0.85, (ga, gb)
    assert a.trace["cuscope"]["sched"]["waits"] > 0 and b.trace["cuscope"]["sched"]["waits"] > 0
    for r in (a, b):  # each wait is on the timeline
        throttled = [e for e in r.trace["traceEvents"] if e["name"] == "throttled"]
        assert len(throttled) == r.trace["cuscope"]["sched"]["waits"]
        assert "Sharing: weight" in r.stderr


def test_alone_is_never_held_back(sched_daemon, mimic):
    r = mimic("share_loop", "shim", env={**sched_daemon.env, **FAST, "MIMIC_GAP_US": "10", "CUSCOPE_SHARE": "10",
                                         "MIMIC_SECONDS": "0.5"})
    assert r.code == 0, r.stderr
    assert r.trace["cuscope"]["sched"]["waits"] == 0


def test_daemon_stopped_jobs_continue(sched_daemon, mimic_pair):  # Review Focus 2
    # Two jobs, so neither is solo and both wait; the daemon stops mid-run.
    threading.Timer(0.3, sched_daemon.stop).start()
    # A 10 us gap still asks for far more than any share (each mock kernel is
    # 1 ms), and keeps the unthrottled tail after the stop small.
    a, b = mimic_pair("share_loop", weights=(1, 1),
                      env={**sched_daemon.env, **FAST, "MIMIC_GAP_US": "10", "MIMIC_SECONDS": "0.5"})
    for r in (a, b):
        assert r.code == 0, r.stderr
        assert r.stderr.count("compute sharing off: scheduler stopped") == 1, r.stderr
        assert "scheduler stopped during the run" in r.stderr  # the summary says so too


def test_capture_launches_not_charged(sched_daemon, mimic):  # Review Focus 5
    r = mimic("capture", "shim", env={**sched_daemon.env, "CUSCOPE_SHARE": "50"})
    assert r.code == 0, r.stderr
    assert r.trace["cuscope"]["sched"]["registered"] is True
    assert r.trace["cuscope"]["sched"]["gpu_ns_charged"] == 0


# ---- Task 6: charges are corrected to the measured GPU time ----


def test_charged_time_matches_measured_kernels(sched_daemon, mimic):
    r = mimic("share_loop", "shim", env={**sched_daemon.env, "CUSCOPE_SHARE": "50", "MOCK_KERNEL_NS": "1000000",
                                         "MIMIC_SECONDS": "0.5"})
    assert r.code == 0, r.stderr
    launches = int(next(l for l in r.stdout.splitlines() if l.startswith("LAUNCHES ")).split()[1])
    charged = r.trace["cuscope"]["sched"]["gpu_ns_charged"]
    # Each mock kernel measures 1 ms; cuEventElapsedTime is float milliseconds,
    # so allow 1 us per launch of rounding.
    assert abs(charged - launches * 1_000_000) <= launches * 1_000, (charged, launches)


# ---- Final review C1: a daemon restarted after a crash ----


def test_jobs_survive_a_daemon_crash_and_restart(sched_daemon, mimic_pair):
    # SIGKILL leaves the segment behind; a new daemon takes over. Jobs that
    # were registered must not hang: they fail open (or are served again).
    import threading

    def crash_and_restart():
        sched_daemon.kill()
        sched_daemon.start()

    threading.Timer(0.3, crash_and_restart).start()
    a, b = mimic_pair("share_loop", weights=(1, 1),
                      env={**sched_daemon.env, **FAST, "MIMIC_GAP_US": "10", "MIMIC_SECONDS": "0.8"})
    for r in (a, b):
        assert r.code == 0, r.stderr


def test_daemon_stops_on_sighup(sched_daemon):
    # Closing the terminal the daemon runs in sends SIGHUP: it must close the
    # segment like SIGTERM does, not leave it behind.
    import signal
    sched_daemon.proc.send_signal(signal.SIGHUP)
    sched_daemon.proc.wait(timeout=5)
    assert not Path("/dev/shm" + sched_daemon.name).exists()


# ---- Final review I3: the segment must be private to its user ----


def _plant_open_segment(name: str) -> None:
    """What another local user could do: create the victim's segment, writable by all."""
    import os
    old = os.umask(0)
    try:
        fd = os.open("/dev/shm" + name, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, 1 << 20)
        os.close(fd)
    finally:
        os.umask(old)


def test_daemon_refuses_a_segment_open_to_others(tmp_name_env):
    _plant_open_segment(tmp_name_env["CUSCOPE_SCHED_NAME"])
    r = subprocess.run([str(SCHED)], env={**os.environ, **tmp_name_env}, capture_output=True, text=True, timeout=5)
    assert r.returncode == 1 and "not private" in r.stderr, r.stderr


def test_job_ignores_a_segment_open_to_others(mimic, tmp_name_env):
    _plant_open_segment(tmp_name_env["CUSCOPE_SCHED_NAME"])
    r = mimic("share_loop", "shim", env={**tmp_name_env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "0.2"})
    assert r.code == 0, r.stderr
    sched = r.trace["cuscope"]["sched"]
    assert (sched["registered"], sched["reason"]) == (False, "scheduler segment is not private to this user")


def test_failed_launch_is_refunded(sched_daemon, mimic):
    r = mimic("launch_fail", "shim", env={**sched_daemon.env, "CUSCOPE_SHARE": "50", "MOCK_LAUNCH_ERROR": "1"})
    assert r.code == 0, r.stderr
    assert r.trace is not None, (r.code, r.stdout, r.stderr)
    sched = r.trace["cuscope"]["sched"]
    assert sched["registered"] is True
    assert sched["gpu_ns_charged"] == 0, sched


def _plant_segment(name: str, version: int, size: int = 4096) -> None:
    import struct
    fd = os.open("/dev/shm" + name, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, size)
        if size >= 12:
            os.write(fd, struct.pack("<QI", 0x6375736368656421, version))
    finally:
        os.close(fd)


def test_version_mismatch_daemon(tmp_name_env):
    name = tmp_name_env["CUSCOPE_SCHED_NAME"]
    _plant_segment(name, version=2, size=4096)
    r = subprocess.run([str(SCHED)], env={**os.environ, **tmp_name_env}, capture_output=True, text=True, timeout=5)
    assert r.returncode == 1, r.stdout
    assert "version 2" in r.stderr or "version" in r.stderr, r.stderr


def test_version_mismatch_job(mimic, tmp_name_env):
    name = tmp_name_env["CUSCOPE_SCHED_NAME"]
    _plant_segment(name, version=2, size=4096)
    r = mimic("share_loop", "shim", env={**tmp_name_env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "0.2"})
    assert r.code == 0, r.stderr
    sched = r.trace["cuscope"]["sched"]
    assert sched["registered"] is False
    assert sched["reason"] == "scheduler is a different version (restart cuscope sched)"


def test_too_small_segment_daemon(tmp_name_env):
    name = tmp_name_env["CUSCOPE_SCHED_NAME"]
    _plant_segment(name, version=1, size=64)
    r = subprocess.run([str(SCHED)], env={**os.environ, **tmp_name_env}, capture_output=True, text=True, timeout=5)
    assert r.returncode == 1, r.stdout
    assert "File exists" not in r.stderr, r.stderr
    assert "too small" in r.stderr or "version" in r.stderr, r.stderr


def test_too_small_segment_job(mimic, tmp_name_env):
    name = tmp_name_env["CUSCOPE_SCHED_NAME"]
    _plant_segment(name, version=1, size=64)
    r = mimic("share_loop", "shim", env={**tmp_name_env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "0.2"})
    assert r.code == 0, r.stderr
    sched = r.trace["cuscope"]["sched"]
    assert sched["registered"] is False
    assert sched["reason"] == "scheduler is a different version (restart cuscope sched)"


