"""Memory caps (CUSCOPE_MEM_LIMIT) on the mock driver.

Scenarios print "RESULT <what> <CUresult>" for each call whose result is the
point of the test; the mock's live_hwm counter is the highest number of device
bytes it ever held at once.
"""

import pytest

from conftest import MODES

MiB = 1 << 20
OK, INVALID, OOM = 0, 1, 2  # CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY


def results(r) -> list[int]:
    return [int(line.split()[2]) for line in r.stdout.splitlines() if line.startswith("RESULT ")]


@pytest.mark.parametrize("mode", MODES)
def test_fifth_allocation_is_denied_without_reaching_the_driver(mimic, mode):
    r = mimic("cap_exact", mode, env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert r.code == 0, r.stderr
    assert results(r) == [OK, OK, OK, OK, OOM, OK]
    assert r.mock["alloc_v2"] == 5  # the denied call never reached the driver
    assert r.mock["live_hwm"] == 256 * MiB


@pytest.mark.parametrize("mode", MODES)
def test_threads_never_overshoot(mimic, mode):
    r = mimic("cap_threads", mode, env={"CUSCOPE_MEM_LIMIT": "4M"})
    assert r.code == 0, r.stderr
    assert r.mock["live_hwm"] <= 4 * MiB
    assert OOM in results(r)


def test_pinned_host_is_not_capped(mimic):
    assert results(mimic("cap_pinned", "shim", env={"CUSCOPE_MEM_LIMIT": "1M"})) == [OK]


def test_pitch_charges_the_padded_size(mimic):
    # 1000-byte rows pad to 1024: 1 MiB charged, so exactly 1 MiB is left of
    # 2 MiB and one more byte is over. Charging width x height (1,024,000)
    # would wrongly leave room for the last byte.
    r = mimic("cap_pitch", "shim", env={"CUSCOPE_MEM_LIMIT": "2M"})
    assert results(r) == [OK, OK, OOM]


def test_driver_failure_gives_the_reservation_back(mimic):  # Review Focus 2
    r = mimic("cap_driver_fail", "shim", env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert results(r) == [INVALID, OK, OK, OK, OK]


def test_double_free_releases_nothing_twice(mimic):  # Review Focus 3
    r = mimic("cap_double_free", "shim", env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert results(r) == [OK, OK, INVALID, OK, OK, OK, OK, OOM]


def test_no_cap_changes_nothing(mimic):
    assert results(mimic("cap_exact", "shim")) == [OK] * 6


def test_context_teardown_gives_bytes_back(mimic):
    assert results(mimic("cap_teardown", "shim", env={"CUSCOPE_MEM_LIMIT": "128M"})) == [OK, OK]


def test_forked_child_starts_with_an_empty_budget(mimic):
    assert results(mimic("cap_fork", "direct", env={"CUSCOPE_MEM_LIMIT": "128M"})) == [OK, OK]


def meminfo(r) -> tuple[int, int]:
    line = next(l for l in r.stdout.splitlines() if l.startswith("MEMINFO "))
    free_bytes, total_bytes = (int(x) for x in line.split()[1:])
    return free_bytes, total_bytes


@pytest.mark.parametrize("mode", MODES)
def test_meminfo_reports_the_cap(mimic, mode):
    r = mimic("cap_meminfo", mode, env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert r.code == 0, r.stderr
    assert meminfo(r) == (192 * MiB, 256 * MiB)


def test_meminfo_uncapped_is_untouched(mimic):  # Review Focus 5
    r = mimic("cap_meminfo", "shim")
    assert meminfo(r) == ((6 << 30) - 64 * MiB, 6 << 30)


def test_trace_reports_the_cap_end_to_end(mimic):
    r = mimic("cap_exact", "shim", env={"CUSCOPE_MEM_LIMIT": "256M"})
    cap = r.trace["cuscope"]["mem_limit"]
    assert (cap["limit_bytes"], cap["peak_charged_bytes"], cap["denied_count"]) == (256 * MiB, 256 * MiB, 1)
    denials = [e for e in r.trace["traceEvents"] if e["ph"] == "i"]
    assert [e["name"] for e in denials] == ["denied cuMemAlloc 64.0 MiB"]
    assert "Memory cap: 256.0 MiB" in r.stderr
    assert mimic("cap_exact", "shim").trace["cuscope"]["mem_limit"] is None


def test_reused_address_gives_the_stale_bytes_back(mimic):
    # Final review I1: 96 MiB freed behind cuscope's back (an unhooked context
    # release, say) stayed charged when the driver reused its address, so the
    # budget shrank for good: the 128 MiB allocation was denied.
    r = mimic("cap_reuse", "shim", env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert results(r) == [OK, OK, OK], r.stderr
