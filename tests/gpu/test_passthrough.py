"""Rule one: cuscope never changes what the program does."""

import pytest

from harness import assert_passthrough

PROGRAMS = ["hello_static", "hello_shared", "leak", "double_free", "async_pool",
            "multistream", "multithread", "bandwidth", "graph_capture", "device_reset", "capture_race"]


@pytest.mark.parametrize("prog", PROGRAMS)
def test_passthrough(prog):
    assert_passthrough(prog)
