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
