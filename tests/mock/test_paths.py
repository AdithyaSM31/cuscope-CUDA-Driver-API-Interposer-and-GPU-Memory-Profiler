"""Interception-path tests against the mock libcuda (no GPU needed)."""

import pytest

from conftest import MODES, SCENARIOS


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_baseline_without_cuscope(mimic, scenario, mode):
    r = mimic(scenario, mode, preload=False)
    assert r.code == 0, r.stderr
    assert r.stdout.startswith(f"OK {scenario}")
    assert r.mock is not None
    assert r.stats is None  # cuscope not loaded


def test_direct_baseline_without_cuscope(direct_mimic):
    r = direct_mimic(preload=False)
    assert r.code == 0, r.stderr
    assert r.stdout.startswith("OK direct")
    assert r.mock["alloc_v2"] == 1 and r.mock["free_"] == 1


# ---- Task 3: dlsym override and census ----

@pytest.mark.parametrize("mode", MODES)
def test_dlsym_returns_gpa_wrapper(mimic, mode):
    r = mimic("dlsym_gpa", mode)
    assert r.code == 0, r.stderr
    assert r.stats["dlsym_wrapped"] >= 1


def test_shim_lookups_pass_through(mimic):
    # The WSL-style shim resolves cuGetProcAddress_v2 from its constructor.
    # Handing it a wrapper would make the wrapper call itself forever.
    r = mimic("dlsym_gpa", "shim")
    assert r.code == 0, r.stderr
    assert ["dlsym", "cuGetProcAddress_v2", "libcuda.so.1", "caller-libcuda"] in r.census


@pytest.mark.parametrize("mode", MODES)
def test_dlsym_default_during_init(mimic, mode):
    # Review Focus 1: dlsym(RTLD_DEFAULT, "cuGetProcAddress_v2") before libcuda
    # is loaded. cuscope must load libcuda itself (the shim's constructor then
    # calls dlsym re-entrantly) and must not return its own export as "real".
    r = mimic("dlsym_default", mode)
    assert r.code == 0, r.stderr
    assert r.stdout.startswith("OK dlsym_default")
    assert r.stats["dlsym_wrapped"] >= 1


@pytest.mark.parametrize("mode", MODES)
def test_libcudart_caller_is_not_libcuda(mimic, mode):
    # Found by the real-GPU census: "libcudart.so.13" starts with "libcuda", so a
    # prefix match passed the shared runtime's lookups through unrecorded.
    r = mimic("via_cudart", mode)
    assert r.code == 0, r.stderr
    assert r.stats["dlsym_wrapped"] >= 1
    assert ["dlsym", "cuGetProcAddress_v2", "libcudart.so.13", "wrap"] in r.census


def test_census_records_gpa(mimic):
    r = mimic("dlsym_gpa", "direct")
    assert r.code == 0, r.stderr
    assert any(row[:3] == ["gpa", "cuMemAlloc", "13000"] for row in r.census), r.census


# ---- Task 6: loader, hook table, variant-matched wrappers ----

@pytest.mark.parametrize("mode", MODES)
def test_gpa_wraps_matching_variant(mimic, mode):
    r = mimic("dlsym_gpa", mode)
    assert r.code == 0, r.stderr
    assert r.stats["gpa_wrapped"] >= 2 and r.stats["hooked_calls"] >= 2  # cuMemAlloc, cuMemFree
    assert r.mock["alloc_v2"] == 1 and r.mock["free_"] == 1


def test_ptsz_forwards_to_ptsz(mimic):
    # Each wrapper forwards to the variant the program asked for.
    r = mimic("ptsz", "shim")
    assert r.code == 0, r.stderr
    assert r.mock["memcpy_async_ptsz"] == 1 and r.mock["launch_ptsz"] == 1
    assert r.mock["memcpy_async"] == 0 and r.mock["launch"] == 0


def test_old_version_passthrough(mimic):
    # cuMemAlloc at version 3000 is the 32-bit v1 function: wrapping it with the
    # v2 signature would corrupt arguments, so it must come back untouched.
    r = mimic("old_version", "shim")
    assert r.code == 0, r.stderr
    assert r.stats["gpa_passthrough"] >= 1 and r.mock["alloc_v1"] == 1
    assert r.stats["hooked_calls"] == 0


def test_gpa_self_lookup(mimic):
    r = mimic("gpa_self", "shim")
    assert r.code == 0, r.stderr
    assert r.stats["hooked_calls"] >= 1


def test_direct_exports(direct_mimic):
    r = direct_mimic()
    assert r.code == 0, r.stderr
    assert r.stats["direct_calls"] >= 2  # cuMemAlloc_v2, cuMemFree_v2 by symbol name
    assert r.mock["alloc_v2"] == 1 and r.mock["free_"] == 1


def test_real_gpa_is_never_own_wrapper(mimic):
    # Review Focus 1: RTLD_DEFAULT finds cuscope's own export first.
    r = mimic("dlsym_default", "direct")
    assert r.code == 0, r.stderr
    assert r.stats["hooked_calls"] >= 2


# ---- Task 9: recorder, collector, fork safety ----

@pytest.mark.parametrize("mode", MODES)
def test_fork_child_has_no_inherited_leaks(mimic, mode):
    # Review Focus 4: a forked child (a dataloader worker, a subprocess that
    # inherits LD_PRELOAD) writes its own trace and does not report the
    # parent's allocations as its leaks.
    r = mimic("fork", mode)
    assert r.code == 0, r.stderr
    assert len(r.traces) == 2, sorted(r.traces)
    parent = r.trace["cuscope"]
    child = next(t["cuscope"] for pid, t in r.traces.items() if t is not r.trace)
    assert [l["bytes"] for l in parent["leaks"]] == [1 << 20]
    assert child["leaks"] == []
    assert child["alloc_count"] == 0


def test_api_calls_recorded_in_mock(mimic):
    r = mimic("dlsym_gpa", "shim")
    names = [e["name"] for e in r.trace["traceEvents"] if e["ph"] == "X"]
    assert names.count("cuMemAlloc") == 1 and names.count("cuMemFree") == 1


# ---- Task 10: GPU timing ----

def gpu_spans(trace: dict) -> list[dict]:
    return [e for e in trace["traceEvents"] if e["ph"] == "X" and e.get("cat") == "gpu"]


def test_launch_timed_full_mode(mimic):
    r = mimic("launch", "shim")
    assert r.code == 0, r.stderr
    spans = gpu_spans(r.trace)
    assert len(spans) == 1
    launch = next(e for e in r.trace["traceEvents"] if e["ph"] == "X" and e["name"] == "cuLaunchKernel")
    assert spans[0]["args"]["corr"] == launch["args"]["corr"] >= 1
    assert r.mock["event_record"] >= 2  # start and end of the launch, plus the clock reference


def test_alloc_mode_has_no_events(mimic):
    r = mimic("launch", "shim", env={"CUSCOPE_MODE": "alloc"})
    assert r.code == 0, r.stderr
    assert r.mock["event_record"] == 0
    assert gpu_spans(r.trace) == []
    assert r.trace["cuscope"]["kernels"][0]["count"] == 1  # still counted, just not timed


def test_capturing_stream_not_timed(mimic):
    # Review Focus 2: an event recorded on a capturing stream would become a
    # node in the program's graph.
    r = mimic("capture", "shim")
    assert r.code == 0, r.stderr
    assert r.mock["event_record_on_capturing"] == 0
    assert gpu_spans(r.trace) == []
    assert r.trace["cuscope"]["gpu_untimed"] == 1


def test_preloaded_runs_keep_program_behaviour(mimic):
    # Every call the program made reaches the driver unchanged. Event calls are
    # left out: from Task 10 on, cuscope records its own timing events.
    def program_calls(mock: dict) -> dict:
        return {k: v for k, v in mock.items() if not k.startswith("event_record")}

    for scenario in ["dlsym_gpa", "ptsz", "old_version", "gpa_self", "fork"]:
        for mode in ["direct", "shim"]:
            base, under = mimic(scenario, mode, preload=False), mimic(scenario, mode)
            assert under.code == base.code, (scenario, mode)
            assert program_calls(under.mock) == program_calls(base.mock), (scenario, mode)
