"""The cuscope command line, without a GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("CUSCOPE_BUILD", REPO / "build"))
CLI = REPO / "cli" / "cuscope.py"
FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.json"


def cuscope(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    run_env["CUSCOPE_LIB"] = str(BUILD / "libcuscope.so")
    run_env.update(env or {})
    return subprocess.run([sys.executable, str(CLI), *args], env=run_env, capture_output=True, text=True, timeout=60)


def child_env(*cuscope_args: str, env: dict | None = None) -> dict:
    code = "import json, os; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith(('CUSCOPE_', 'LD_PRELOAD'))}))"
    r = cuscope("run", *cuscope_args, "--", sys.executable, "-c", code, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_run_passes_exit_code():
    r = cuscope("run", "--", sys.executable, "-c", "import sys; sys.exit(3)")
    assert r.returncode == 3


def test_run_keeps_stdout_clean():
    r = cuscope("run", "--", sys.executable, "-c", "print('hello')")
    assert r.stdout == "hello\n"


def test_run_prepends_ld_preload():
    env = child_env(env={"LD_PRELOAD": "/x.so"})
    assert env["LD_PRELOAD"].split()[0].endswith("libcuscope.so")
    assert env["LD_PRELOAD"].split()[1] == "/x.so"


def test_run_sets_config():
    env = child_env("--mode", "alloc", "--out", "/tmp/t-%p.json", "--stack-depth", "8", "--sample-every", "4")
    assert env["CUSCOPE_MODE"] == "alloc"
    assert env["CUSCOPE_OUT"] == "/tmp/t-%p.json"
    assert env["CUSCOPE_STACK_DEPTH"] == "8"
    assert env["CUSCOPE_SAMPLE_EVERY"] == "4"


def test_no_report_without_cuda(tmp_path):
    # Child processes inherit LD_PRELOAD; those that never touch CUDA (shells,
    # compilers, workers) must not leave traces or summaries behind.
    out = tmp_path / "t-%p.json"
    r = cuscope("run", "--out", str(out), "--", sys.executable, "-c", "print(1)")
    assert r.returncode == 0 and r.stdout == "1\n"
    assert "cuscope summary" not in r.stderr
    assert list(tmp_path.iterdir()) == []


def test_ctrl_c_lets_the_program_finish_its_own_shutdown(tmp_path):
    # Review finding I3: on Ctrl-C the CLI killed the program 0.25 s later,
    # before it (and cuscope's exit handler) could finish. Ctrl-C reaches the
    # whole foreground process group; the program decides what to do with it.
    import signal
    import time
    marker = tmp_path / "shut-down"
    code = ("import signal, sys, time\n"
            "def stop(*_):\n"
            "    time.sleep(1)\n"
            f"    open({str(marker)!r}, 'w').close()\n"
            "    sys.exit(5)\n"
            "signal.signal(signal.SIGINT, stop)\n"
            "print('ready', flush=True)\n"
            "time.sleep(30)\n")
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    run_env["CUSCOPE_LIB"] = str(BUILD / "libcuscope.so")
    proc = subprocess.Popen([sys.executable, str(CLI), "run", "--", sys.executable, "-c", code], env=run_env,
                            stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert proc.stdout.readline() == "ready\n"
        os.killpg(proc.pid, signal.SIGINT)
        assert proc.wait(timeout=10) == 5
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(0.2)
    assert marker.exists()


def test_run_of_missing_command_exits_127():
    r = cuscope("run", "--", "/nonexistent/program")
    assert r.returncode == 127 and "command not found" in r.stderr


def test_run_without_command_is_an_error():
    r = cuscope("run")
    assert r.returncode == 2 and "command" in r.stderr


def test_missing_library_is_a_clear_error():
    r = cuscope("run", "--", "true", env={"CUSCOPE_LIB": "/nonexistent/libcuscope.so"})
    assert r.returncode == 2 and "libcuscope.so" in r.stderr


def test_summary_sections():
    r = cuscope("summary", str(FIXTURE))
    assert r.returncode == 0, r.stderr
    last = 0
    for header in ["== cuscope summary (pid 1234, mode full) ==", "Memory", "Leaks (1, 1.0 MiB)", "Invalid frees: none",
                   "Top kernels", "spin_kernel(long long)", "Copies", "HtoD pinned",
                   "Completeness: dropped=0 unresolved=0 untimed=0"]:
        at = r.stdout.find(header, last)
        assert at >= 0, (header, r.stdout)
        last = at


def test_summary_of_missing_file():
    r = cuscope("summary", "/nonexistent/trace.json")
    assert r.returncode == 2 and "trace.json" in r.stderr
