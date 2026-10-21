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


def test_preloaded_runs_keep_program_behaviour(mimic):
    for scenario in ["dlsym_gpa", "ptsz", "old_version", "gpa_self", "fork"]:
        for mode in ["direct", "shim"]:
            base, under = mimic(scenario, mode, preload=False), mimic(scenario, mode)
            assert (under.code, under.mock) == (base.code, base.mock), (scenario, mode)
