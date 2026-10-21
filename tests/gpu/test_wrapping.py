"""Every hooked function the CUDA runtime looks up must reach cuscope's wrapper.

A "mismatch" in the census means cuGetProcAddress returned a variant the
wrapper was not written for, so cuscope passed it through and saw nothing.
"""

import pytest

from harness import run

# (symbol, cudaVersion) lookups that must not be wrapped: a different function
# with a different signature under the same base name.
EXPECTED_MISMATCHES = {("cuCtxSynchronize", "13000")}  # cuCtxSynchronize_v2(CUcontext)


@pytest.mark.parametrize("prog", ["hello_static", "hello_shared"])
def test_runtime_lookups_of_hooked_functions_are_wrapped(prog, tmp_path):
    census = tmp_path / "census.tsv"
    r = run(prog, env={"CUSCOPE_CENSUS": str(census)})
    assert r.code == 0, r.stderr
    rows = [line.split("\t") for line in census.read_text().splitlines()]
    gpa = [row for row in rows if row[0] == "gpa"]
    assert any(row[6] == "wrap" for row in gpa)
    mismatches = {(row[1], row[2]) for row in gpa if row[6] == "mismatch"}
    assert mismatches == EXPECTED_MISMATCHES
