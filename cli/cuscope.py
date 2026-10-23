#!/usr/bin/env python3
"""cuscope: GPU allocations, copies and kernel launches of an unmodified CUDA program.

  cuscope run [options] -- <command> [args...]
      Runs the command with libcuscope.so preloaded. At exit the program's
      trace is written (Chrome Trace JSON: open it in https://ui.perfetto.dev)
      and a summary is printed to stderr. The exit code is the command's.

  cuscope summary <trace.json> [--frames N]
      Prints the summary of a trace, with source file:line for each leak and
      invalid free (programs built with -g or -lineinfo).

The library is found through $CUSCOPE_LIB, then <cli>/../lib/libcuscope.so,
then <cli>/../build/libcuscope.so.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

TOP_KERNELS = 10
MAX_LISTED = 20


def find_library() -> tuple[Path | None, list[Path]]:
    here = Path(__file__).resolve().parent
    if os.environ.get("CUSCOPE_LIB"):  # named explicitly: no fallback
        candidates = [Path(os.environ["CUSCOPE_LIB"])]
    else:
        candidates = [here.parent / "lib" / "libcuscope.so", here.parent / "build" / "libcuscope.so"]
    for path in candidates:
        if path.is_file():
            return path.resolve(), candidates
    return None, candidates


def fail(message: str, code: int = 2) -> int:
    print(f"cuscope: {message}", file=sys.stderr)
    return code


# ---- run ----

def cmd_run(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        return fail("run needs a command, e.g. cuscope run -- ./app")
    lib, looked = find_library()
    if lib is None:
        return fail("cannot find libcuscope.so (looked in " + ", ".join(map(str, looked)) + "); set CUSCOPE_LIB")

    env = dict(os.environ)
    env["LD_PRELOAD"] = f"{lib} {env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else str(lib)
    for option, var in [("mode", "CUSCOPE_MODE"), ("out", "CUSCOPE_OUT"), ("stack_depth", "CUSCOPE_STACK_DEPTH"),
                        ("sample_every", "CUSCOPE_SAMPLE_EVERY"), ("log", "CUSCOPE_LOG"),
                        ("mem_limit", "CUSCOPE_MEM_LIMIT")]:
        value = getattr(args, option)
        if value is not None:
            env[var] = str(value)

    # Become the command, like `env` does: its exit code and signals are its
    # own, and Ctrl-C reaches it alone, so it can shut down (and cuscope can
    # write its report) at its own pace.
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execvpe(command[0], command, env)
    except FileNotFoundError:
        return fail(f"command not found: {command[0]}", 127)
    except PermissionError:
        return fail(f"command not executable: {command[0]}", 126)


# ---- summary ----

def bytes_text(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1 << 20:
        return f"{n / 1024:.1f} KiB"
    if n < 1 << 30:
        return f"{n / (1 << 20):.1f} MiB"
    return f"{n / (1 << 30):.2f} GiB"


def time_text(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.1f} us"
    return f"{ns} ns"


def is_cuda_frame(frame: dict) -> bool:
    # Same rule as src/report.cpp: the static runtime lives inside the program,
    # and its API functions are cudaX with X uppercase (cudaMalloc, not cuda_check).
    symbol, module = frame["symbol"], Path(frame["module"]).name
    runtime_api = symbol.startswith("cuda") and symbol[4:5].isupper()
    return (runtime_api or symbol.startswith(("libcudart_static", "__cuda", "cudart::"))
            or module.startswith(("libcudart", "libcuda.")))


def program_frames(frames: list[dict], count: int) -> list[dict]:
    """The first `count` frames from the first one that belongs to the program."""
    for i, frame in enumerate(frames):
        if not is_cuda_frame(frame):
            return frames[i:i + count]
    return frames[:count]


def resolve_lines(frames: list[dict]) -> dict[tuple[str, int], str]:
    """(module, offset) -> "file:line", via addr2line, one call per module."""
    if shutil.which("addr2line") is None:
        return {}
    by_module = defaultdict(set)
    for f in frames:
        if f["offset"] > 0 and Path(f["module"]).is_file():
            by_module[f["module"]].add(f["offset"])
    lines = {}
    for module, offsets in by_module.items():
        ordered = sorted(offsets)
        # A return address points just past its call: look up the call itself.
        out = subprocess.run(["addr2line", "-e", module, *(hex(o - 1) for o in ordered)],
                             capture_output=True, text=True).stdout.splitlines()
        for offset, line in zip(ordered, out):
            location = line.split(" (discriminator")[0].strip()
            if location and not location.startswith("??"):
                lines[(module, offset)] = f"{Path(location.rsplit(':', 1)[0]).name}:{location.rsplit(':', 1)[1]}"
    return lines


def describe(frame: dict, lines: dict) -> str:
    where = lines.get((frame["module"], frame["offset"]))
    where = where or f"{Path(frame['module']).name}+{frame['offset']:#x}"
    return f"{frame['symbol'] or '?'} ({where})"


def print_summary(report: dict, frames_per_site: int) -> None:
    stacks = {int(k): v for k, v in report["stacks"].items()}
    sites = {}
    denied = [report["mem_limit"]["largest_denied"]] if report.get("mem_limit") else []
    for item in report["leaks"][:MAX_LISTED] + report["invalid_frees"][:MAX_LISTED] + denied:
        sites[item["stack"]] = program_frames(stacks.get(item["stack"], []), frames_per_site)
    lines = resolve_lines([f for frames in sites.values() for f in frames])

    def site_text(stack_id: int) -> str:
        frames = sites.get(stack_id) or []
        if not frames:
            return "unknown site"
        return describe(frames[0], lines) + "".join(f"\n      called from {describe(f, lines)}" for f in frames[1:])

    peaks = report["peak_bytes"]
    sample_every = report.get("sample_every", 1)
    sampling = f", timing 1 in {sample_every} launches" if sample_every > 1 else ""
    print(f"== cuscope summary (pid {report['pid']}, mode {report['mode']}{sampling}) ==")
    print("Memory")
    print(f"  peak device memory  {bytes_text(peaks['device_total'])}  (device {bytes_text(peaks['device'])}, "
          f"managed {bytes_text(peaks['managed'])}, async {bytes_text(peaks['async'])}; "
          f"pinned host {bytes_text(peaks['host_pinned'])})")
    print(f"  allocations {report['alloc_count']}, frees {report['free_count']}")

    cap = report.get("mem_limit")
    if cap:
        pct = 100.0 * cap["peak_charged_bytes"] / cap["limit_bytes"]
        print(f"Memory cap: {bytes_text(cap['limit_bytes'])}, peak charged {bytes_text(cap['peak_charged_bytes'])} "
              f"({pct:.1f}%)")
        if cap["denied_count"]:
            n, largest = cap["denied_count"], cap["largest_denied"]
            print(f"  {n} allocation{'' if n == 1 else 's'} denied ({bytes_text(cap['denied_bytes'])}); "
                  f"largest {bytes_text(largest['bytes'])} {largest['kind']} at {site_text(largest['stack'])}")

    leaks = report["leaks"]
    if not leaks:
        print("Leaks: none")
    else:
        print(f"Leaks ({len(leaks)}, {bytes_text(sum(l['bytes'] for l in leaks))})")
        for leak in leaks[:MAX_LISTED]:
            teardown = ", freed at context teardown" if leak["at_ctx_teardown"] else ""
            print(f"  {bytes_text(leak['bytes'])} {leak['kind']} at {site_text(leak['stack'])}{teardown}")
        if len(leaks) > MAX_LISTED:
            print(f"  ... and {len(leaks) - MAX_LISTED} more")

    frees = report["invalid_frees"]
    if not frees:
        print("Invalid frees: none")
    else:
        print(f"Invalid frees ({len(frees)})")
        for free in frees[:MAX_LISTED]:
            print(f"  {free['ptr']} at {site_text(free['stack'])}")

    kernels = report["kernels"]
    if not kernels:
        print("Top kernels: none")
    else:
        # With sampling, GPU time covers only the timed launches.
        timed_header = f"  {'timed':>6}" if sample_every > 1 else ""
        print(f"Top kernels\n  {'GPU time':>12}  {'calls':>6}{timed_header}  name")
        for k in kernels[:TOP_KERNELS]:
            gpu = time_text(k["gpu_ns_total"]) if k["gpu_ns_total"] else "untimed"
            timed = f"  {k.get('timed', k['count']):>6}" if sample_every > 1 else ""
            print(f"  {gpu:>12}  {k['count']:>6}{timed}  {k['name']}")
    if report["graph_launches"]:
        print(f"  graph launches: {report['graph_launches']}")

    copies = report["copies"]
    if not copies:
        print("Copies: none")
    else:
        print("Copies")
        for c in copies:
            rate = f"{c['gbps']:.2f} GB/s" if c["gbps"] > 0 else "untimed"
            size = bytes_text(c["bytes"] // c["count"]) if c["count"] else "0 B"
            print(f"  {c['dir']:<4} {'pinned' if c['pinned'] else 'pageable':<8}  {c['count']:>6} x {size:<10} {rate}")

    print(f"Completeness: dropped={report['dropped_records']} unresolved={report['unresolved_events']} "
          f"untimed={report['gpu_untimed']}")


def cmd_summary(args: argparse.Namespace) -> int:
    try:
        report = json.loads(Path(args.trace).read_text())["cuscope"]
    except (OSError, ValueError, KeyError) as e:
        return fail(f"cannot read {args.trace}: {e}")
    print_summary(report, max(1, args.frames))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="cuscope", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a command under cuscope")
    run.add_argument("--mode", choices=["full", "alloc"], help="full (default) or alloc: no GPU timing")
    run.add_argument("--out", help="trace path; %%p becomes the pid (default cuscope-%%p.json)")
    run.add_argument("--stack-depth", type=int, help="frames per call-site stack (default 16)")
    run.add_argument("--sample-every", type=int, help="time every Nth launch per thread (default 1)")
    run.add_argument("--log", help="write cuscope's messages and summary here instead of stderr")
    run.add_argument("--mem-limit", help="cap GPU memory per process, e.g. 2G (default: none)")
    run.add_argument("command", nargs=argparse.REMAINDER, help="-- <command> [args...]")
    run.set_defaults(func=cmd_run)

    summary = sub.add_parser("summary", help="print the summary of a trace file")
    summary.add_argument("trace")
    summary.add_argument("--frames", type=int, default=1, help="program frames to show per call site")
    summary.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
