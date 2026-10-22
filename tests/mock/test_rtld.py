"""cuscope's dlsym override must not change what dlsym returns to anyone else.

glibc resolves RTLD_NEXT, and the local scope of RTLD_DEFAULT, from the
return address of dlsym's caller. If the override called the real dlsym, glibc
would see libcuscope as the caller. These programs need no CUDA at all: every
process that inherits LD_PRELOAD is affected.
"""

import os
import subprocess

from conftest import BUILD

APP = BUILD / "mock" / "rtld_app"
EXPECTED = "rtld_next: ok\nrtld_default from a local plugin: found\n"


def run_app(preload: bool) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    if preload:
        env["LD_PRELOAD"] = str(BUILD / "libcuscope.so")
        if os.environ.get("CUSCOPE_PRELOAD_FIRST"):
            env["LD_PRELOAD"] = os.environ["CUSCOPE_PRELOAD_FIRST"] + " " + env["LD_PRELOAD"]
    return subprocess.run([str(APP)], env=env, capture_output=True, text=True, timeout=10)


def test_without_cuscope():
    r = run_app(preload=False)
    assert (r.returncode, r.stdout) == (0, EXPECTED), r.stderr


def test_rtld_next_and_rtld_default_unchanged_under_cuscope():
    r = run_app(preload=True)
    assert r.returncode == 0, f"exit {r.returncode} (99 = RTLD_NEXT returned the wrapper itself)\n{r.stderr}"
    assert r.stdout == EXPECTED


def run_probe(mode: str, preload: bool) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    env["LD_LIBRARY_PATH"] = str(BUILD / "mock" / "direct")  # a libcuda exists, but nothing loads it
    if preload:
        env["LD_PRELOAD"] = str(BUILD / "libcuscope.so")
        if os.environ.get("CUSCOPE_PRELOAD_FIRST"):
            env["LD_PRELOAD"] = os.environ["CUSCOPE_PRELOAD_FIRST"] + " " + env["LD_PRELOAD"]
    return subprocess.run([str(BUILD / "mock" / "dlsym_probe"), mode], env=env, capture_output=True, text=True,
                          timeout=10)


def test_driver_name_from_another_library_keeps_its_own_function():
    # A library that is not libcuda but exports cuMemAlloc_v2 keeps its own.
    for preload in (False, True):
        r = run_probe("foreign", preload)
        assert (r.returncode, r.stdout) == (0, "foreign cuMemAlloc_v2: 999\n"), (preload, r.stdout, r.stderr)


def test_cugetprocaddress_probe_does_not_load_libcuda():
    # Without cuscope, RTLD_DEFAULT finds nothing when libcuda is not loaded.
    # cuscope must not load it to answer.
    for preload in (False, True):
        r = run_probe("probe", preload)
        assert (r.returncode, r.stdout) == (0, "probe: null null, libcuda loaded: no\n"), (preload, r.stdout)
