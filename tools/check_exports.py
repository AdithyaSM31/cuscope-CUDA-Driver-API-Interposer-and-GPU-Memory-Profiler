#!/usr/bin/env python3
"""Checks src/hooks.def against the installed driver (needs the WSL NVIDIA driver).

1. Every export name in hooks.def exists in /usr/lib/wsl/lib/libcuda.so.1.
2. Signature safety. For every hook and stream variant, the real driver's
   cuGetProcAddress_v2(base, version, flags) returns the same function as
   dlsym(driver, export_name). Each wrapper is written against the cuda.h
   prototype of that export, so this proves the pointer cuscope matches is a
   function with the wrapper's signature.
3. Negative control: cuCtxSynchronize at 13000 is cuCtxSynchronize_v2(CUcontext),
   not the (void) cuCtxSynchronize, which is why hooks.def resolves it at 2000.

usage: check_exports.py [path/to/hooks.def]
"""

import ctypes
import glob
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHIM = "/usr/lib/wsl/lib/libcuda.so.1"
CUDA_VERSION = 13000
LEGACY, PER_THREAD = 1, 2

HOOK_LINE = re.compile(r'^\s*HOOK\(\s*(\w+)\s*,\s*"(\w+)"\s*,\s*(nullptr|"\w+")\s*,\s*(\w+)\s*\)')


def parse_hooks(path: Path) -> list[tuple[str, str, str | None, int]]:
    hooks = []
    for line in path.read_text().splitlines():
        m = HOOK_LINE.match(line)
        if m is None:
            continue
        base, legacy, per_thread, version = m.groups()
        per_thread = None if per_thread == "nullptr" else per_thread.strip('"')
        hooks.append((base, legacy, per_thread, CUDA_VERSION if version == "CUDA_VERSION" else int(version)))
    return hooks


class Driver:
    def __init__(self):
        paths = glob.glob("/usr/lib/wsl/drivers/*/libcuda.so.1.1")
        if not paths:
            raise SystemExit("real driver libcuda.so.1.1 not found under /usr/lib/wsl/drivers")
        self.lib = ctypes.CDLL(paths[0], mode=ctypes.RTLD_LOCAL)
        self.gpa = self.lib.cuGetProcAddress_v2
        self.gpa.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                             ctypes.c_uint64, ctypes.POINTER(ctypes.c_int)]
        self.gpa.restype = ctypes.c_int

    def lookup(self, base: str, version: int, flags: int) -> int | None:
        fn, status = ctypes.c_void_p(), ctypes.c_int(-1)
        if self.gpa(base.encode(), ctypes.byref(fn), version, flags, ctypes.byref(status)) != 0:
            return None
        return fn.value

    def export(self, name: str) -> int | None:
        try:
            return ctypes.cast(getattr(self.lib, name), ctypes.c_void_p).value
        except AttributeError:
            return None


def main() -> int:
    hooks_def = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "src" / "hooks.def"
    hooks = parse_hooks(hooks_def)
    if not hooks:
        print(f"FAIL  no HOOK(...) lines in {hooks_def}")
        return 1

    nm = subprocess.run(["nm", "-D", "--defined-only", SHIM], check=True, capture_output=True, text=True).stdout
    exported = {line.split()[-1] for line in nm.splitlines() if line.strip()}
    driver = Driver()
    failures = 0

    def report(ok: bool, text: str) -> None:
        nonlocal failures
        failures += not ok
        if not ok:
            print(f"FAIL  {text}")

    for base, legacy, per_thread, version in hooks:
        for variant, (export, flags) in enumerate([(legacy, LEGACY), (per_thread, PER_THREAD)]):
            if export is None:
                continue
            report(export in exported, f"{base}: export {export} missing from {SHIM}")
            expected, got = driver.export(export), driver.lookup(base, version, flags)
            report(expected is not None and got == expected,
                   f"{base}@{version} variant {variant}: cuGetProcAddress does not return {export}")

    v1, v2 = driver.export("cuCtxSynchronize"), driver.lookup("cuCtxSynchronize", CUDA_VERSION, LEGACY)
    report(v1 is not None and v2 is not None and v1 != v2,
           "negative control: cuCtxSynchronize@13000 should not be the (void) cuCtxSynchronize")

    checked = sum(1 for h in hooks for e in (h[1], h[2]) if e)
    print(f"{'PASS' if failures == 0 else 'FAIL'}  {len(hooks)} hooks, {checked} variants checked, {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
