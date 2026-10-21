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
