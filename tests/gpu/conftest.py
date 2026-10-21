"""GPU tests run on the real driver; skip them where there is none (CI)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for `import harness`

HAS_GPU = Path("/usr/lib/wsl/lib/libcuda.so.1").exists() or Path("/dev/nvidiactl").exists()


def pytest_collection_modifyitems(config, items):
    if HAS_GPU:
        return
    skip = pytest.mark.skip(reason="no NVIDIA driver on this machine")
    for item in items:
        item.add_marker(skip)
