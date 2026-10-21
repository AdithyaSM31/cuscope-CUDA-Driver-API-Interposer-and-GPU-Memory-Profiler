"""Allocation tracking on real programs (Task 8)."""

from pathlib import Path

from harness import assert_passthrough, normalize, run


def report(prog: str) -> dict:
    r = run(prog)
    assert r.code == 0, r.stderr
    assert r.trace is not None, f"no trace written\n{r.stderr}"
    return r.trace["cuscope"]


def test_leak():
    r = report("leak")
    assert sorted(l["bytes"] for l in r["leaks"]) == [1 << 20, 3 << 20]
    assert all(l["kind"] == "device" for l in r["leaks"])
    for leak in r["leaks"]:
        frames = r["stacks"][str(leak["stack"])]
        assert any(f["module"].endswith("/leak") and "main" in f["symbol"] for f in frames), frames


def test_double_free():
    r = report("double_free")
    assert len(r["invalid_frees"]) == 1  # the second free; cudaFree(nullptr) is not one
    assert_passthrough("double_free")


def test_static_vs_shared():
    # The test that proves the driver-level approach: same program, same report.
    assert normalize(report("hello_static")) == normalize(report("hello_shared"))


def test_multithread_exact_counts():
    r = report("multithread")
    assert r["alloc_count"] == 8 * 1001
    assert r["free_count"] == 8000
    assert sorted(l["bytes"] for l in r["leaks"]) == [k * 1024 for k in range(1, 9)]
    assert r["invalid_frees"] == []


def test_peaks_by_kind():
    r = report("bandwidth")  # 64 MiB device + 64 MiB pinned host, both freed
    assert r["peak_bytes"]["device"] >= 64 << 20
    assert r["peak_bytes"]["host_pinned"] >= 64 << 20
    assert r["leaks"] == []


def test_unwritable_out_keeps_program_behaviour():
    # Review Focus 5: a bad CUSCOPE_OUT costs the trace, never the program.
    plain = run("leak", cuscope=False)
    under = run("leak", out=Path("/nonexistent/dir/t.json"))
    assert (under.stdout, under.code) == (plain.stdout, plain.code)
    assert "[cuscope] cannot write trace to /nonexistent/dir/t.json" in under.stderr
