"""The timeline: API spans, the device-memory counter, and (later) GPU spans."""

from harness import run

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
