"""Memory caps on the real driver, through the CUDA runtime (static and shared)."""

import pytest

from harness import run

MiB = 1 << 20


@pytest.mark.parametrize("prog", ["cap_exact_static", "cap_exact_shared"])
def test_cap_denies_at_the_limit(prog):
    r = run(prog, env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert r.code == 0, r.stderr
    lines = r.stdout.splitlines()
    oks = lines.index("cudaErrorMemoryAllocation")
    cap = r.trace["cuscope"]["mem_limit"]
    assert cap["denied_count"] == 1
    assert cap["peak_charged_bytes"] <= 256 * MiB < cap["peak_charged_bytes"] + 64 * MiB
    assert oks == cap["peak_charged_bytes"] // (64 * MiB)  # any runtime allocations are under 64 MiB
    assert lines[-1] == "after free: cudaSuccess"
    free_b, total = map(int, next(l for l in lines if l.startswith("MEMINFO ")).split()[1:])
    assert total == 256 * MiB and free_b < 64 * MiB


@pytest.mark.parametrize("prog", ["hello_static", "multistream", "async_pool"])
def test_generous_cap_changes_nothing(prog):
    plain, capped = run(prog), run(prog, env={"CUSCOPE_MEM_LIMIT": "4G"})
    assert (capped.code, capped.stdout) == (plain.code, plain.stdout)
    assert capped.trace["cuscope"]["mem_limit"]["denied_count"] == 0
