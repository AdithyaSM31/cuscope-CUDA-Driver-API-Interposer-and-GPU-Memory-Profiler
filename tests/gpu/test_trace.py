"""The timeline: API spans, the device-memory counter, and (later) GPU spans."""

import pytest

from harness import assert_passthrough, run

MiB = 1 << 20


def trace_of(prog: str, mode: str = "full") -> dict:
    r = run(prog, mode=mode)
    assert r.code == 0, r.stderr
    assert r.trace is not None, r.stderr
    return r.trace


def counter_values(trace: dict, name: str = "device memory") -> list[int]:
    return [e["args"]["bytes"] for e in trace["traceEvents"] if e["ph"] == "C" and e["name"] == name]


def test_api_spans_and_memory_counter():
    trace = trace_of("hello_static")
    api = [e for e in trace["traceEvents"] if e["ph"] == "X" and e.get("cat") == "api"]
    names = [e["name"] for e in api]
    assert names.count("cuMemAlloc") == 3 and names.count("cuMemFree") == 3
    assert "cuLaunchKernel" in names and "cuMemcpyHtoD" in names
    assert all(e["dur"] >= 0 for e in api)
    values = counter_values(trace)
    assert max(values) == 3 * 4 * MiB and values[-1] == 0  # three 4 MiB buffers, all freed


def test_async_pool_live_bytes():
    trace = trace_of("async_pool")
    assert [v // MiB for v in counter_values(trace)] == [8, 12, 4, 20, 16, 0]
    assert trace["cuscope"]["peak_bytes"]["async"] == 20 * MiB


def test_no_records_dropped():
    assert trace_of("multithread")["cuscope"]["dropped_records"] == 0


# ---- Task 10: GPU timing ----

def gpu_spans(trace: dict) -> list[dict]:
    return [e for e in trace["traceEvents"] if e["ph"] == "X" and e.get("cat") == "gpu"]


def test_multistream_tracks():
    trace = trace_of("multistream")
    spans = gpu_spans(trace)
    assert len(spans) == 4
    assert len({s["tid"] for s in spans}) == 4  # one track per stream
    assert all(s["dur"] >= 1000 for s in spans), [s["dur"] for s in spans]  # ~5 ms kernels, in microseconds

    # Streams run concurrently: at least two kernels overlap in time.
    windows = sorted((s["ts"], s["ts"] + s["dur"]) for s in spans)
    assert any(b_start < a_end for (_, a_end), (b_start, _) in zip(windows, windows[1:]))

    # Clock alignment: a kernel cannot start before its launch call did.
    launch_ts = {e["args"]["corr"]: e["ts"] for e in trace["traceEvents"]
                 if e["ph"] == "X" and e["name"] == "cuLaunchKernel"}
    for s in spans:
        assert s["ts"] >= launch_ts[s["args"]["corr"]] - 100, (s["ts"], launch_ts[s["args"]["corr"]])


def test_alloc_mode_skips_gpu_timing():
    trace = trace_of("multistream", mode="alloc")
    assert gpu_spans(trace) == []
    assert sum(k["count"] for k in trace["cuscope"]["kernels"]) == 4


PROGRAMS = ["hello_static", "hello_shared", "leak", "double_free", "async_pool",
            "multistream", "multithread", "bandwidth", "graph_capture", "device_reset"]


@pytest.mark.parametrize("prog", PROGRAMS)
def test_every_event_resolved(prog):
    report = trace_of(prog)["cuscope"]
    assert report["unresolved_events"] == 0
    assert report["dropped_records"] == 0


def test_context_reset_marks_leaks_at_teardown():
    # cudaDeviceReset destroys the primary context while the driver is alive:
    # the 1 MiB still held then was freed by the driver with the context.
    leaks = trace_of("device_reset")["cuscope"]["leaks"]
    assert sorted((l["bytes"], l["at_ctx_teardown"]) for l in leaks) == [(1 << 20, True), (2 << 20, False)]


def test_leaks_at_exit_on_wsl_are_exit_leaks():
    # On WSL the driver shuts itself down at exit before the CUDA runtime
    # releases its context (that release returns CUDA_ERROR_DEINITIALIZED),
    # so no teardown is observed: unfreed memory is reported as leaked at exit.
    leaks = trace_of("leak")["cuscope"]["leaks"]
    assert sorted(l["bytes"] for l in leaks) == [1 << 20, 3 << 20]
    assert not any(l["at_ctx_teardown"] for l in leaks), leaks


def test_graph_capture():
    # Review Focus 2 on real hardware: if cuscope recorded events inside the
    # capture, the graph would have more than 3 nodes and stdout would change.
    assert_passthrough("graph_capture")
    trace = trace_of("graph_capture")
    assert trace["cuscope"]["graph_launches"] == 2
    names = [s["name"] for s in gpu_spans(trace)]
    assert len(names) == 2 and all(n.startswith("graph ") for n in names), names
