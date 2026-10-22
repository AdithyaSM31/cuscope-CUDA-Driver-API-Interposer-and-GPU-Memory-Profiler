"""Interception-path tests against the mock libcuda (no GPU needed)."""

import json

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
    assert (child["alloc_count"], child["free_count"]) == (1, 1)  # its own calls only
    # Nor does it inherit the parent's GPU timing state (final review, I2).
    assert len(gpu_spans(r.trace)) == 1
    child_trace = next(t for t in r.traces.values() if t is not r.trace)
    assert gpu_spans(child_trace) == []
    assert (child["unresolved_events"], child["gpu_untimed"]) == (0, 0)


@pytest.mark.parametrize("mode", MODES)
def test_fork_child_without_cuda_work_writes_nothing(mimic, mode):
    # Final review, I2: a child that only runs inherited cleanup (the CUDA
    # runtime releasing its context at exit) did no CUDA work of its own, so
    # it writes no trace, let alone one holding the parent's data.
    r = mimic("fork_quiet", mode)
    assert r.code == 0, r.stderr
    assert r.trace is not None
    assert len(r.traces) == 1, sorted(r.traces)


def test_fixed_out_path_is_not_overwritten_by_a_child(mimic, tmp_path):
    # --out without %p: the process that started the run writes that file; a
    # child that also reports writes trace.<pid>.json instead of overwriting it.
    r = mimic("fork", "direct", env={"CUSCOPE_OUT": str(tmp_path / "trace.json")})
    assert r.code == 0, r.stderr
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 2 and names[-1] == "trace.json", names
    assert names[0].startswith("trace.") and names[0].endswith(".json"), names
    parent = json.loads((tmp_path / "trace.json").read_text())["cuscope"]
    assert [l["bytes"] for l in parent["leaks"]] == [1 << 20]  # the parent's report, not the child's


def test_fixed_out_path_through_a_shell_wrapper(tmp_path):
    # cuscope run -- ./train.sh: the shell loads cuscope first but does no CUDA
    # work; the program it runs is the one whose trace gets the exact path.
    import os
    import subprocess
    from conftest import BUILD
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUSCOPE_") and k != "LD_PRELOAD"}
    env.update(LD_LIBRARY_PATH=str(BUILD / "mock" / "direct"), LD_PRELOAD=str(BUILD / "libcuscope.so"),
               CUSCOPE_OUT=str(tmp_path / "trace.json"))
    if os.environ.get("CUSCOPE_PRELOAD_FIRST"):
        env["LD_PRELOAD"] = os.environ["CUSCOPE_PRELOAD_FIRST"] + " " + env["LD_PRELOAD"]
    r = subprocess.run(["sh", "-c", f"{BUILD / 'mock' / 'runtime_mimic'} launch"], env=env, capture_output=True,
                       text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trace.json"], r.stderr


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


# ---- Task 11: context teardown ----

def test_ctx_destroy_flushes_pending(mimic):
    # Events complete only after 200 queries here, far more than the collector
    # makes before the program destroys its context, so only the flush in the
    # cuCtxDestroy hook (polling them before the real destroy) resolves them.
    r = mimic("teardown", "shim", env={"MOCK_EVENT_DELAY_QUERIES": "200"})
    assert r.code == 0, r.stderr
    report = r.trace["cuscope"]
    assert len(gpu_spans(r.trace)) == 1
    assert report["unresolved_events"] == 0
    assert [(l["bytes"], l["at_ctx_teardown"]) for l in report["leaks"]] == [(1 << 20, True)]


@pytest.mark.parametrize("mode", MODES)
def test_primary_release_that_destroys_marks_leaks(mimic, mode):
    r = mimic("release_destroys", mode)
    assert r.code == 0, r.stderr
    assert [(l["bytes"], l["at_ctx_teardown"]) for l in r.trace["cuscope"]["leaks"]] == [(1 << 20, True)]


def test_primary_release_that_keeps_context_does_not(mimic):
    # Release is reference counted: with another reference held, the context
    # and its memory live on, so the allocation is a plain leak at exit.
    r = mimic("release_keeps", "shim")
    assert r.code == 0, r.stderr
    assert [(l["bytes"], l["at_ctx_teardown"]) for l in r.trace["cuscope"]["leaks"]] == [(1 << 20, False)]


# ---- Task 12: kernel names and flow arrows ----

def test_kernel_name_from_driver(mimic):
    # The mock's cuFuncGetName returns the mangled "_Z11mock_kernelv".
    r = mimic("launch", "shim")
    assert r.code == 0, r.stderr
    assert [s["name"] for s in gpu_spans(r.trace)] == ["mock_kernel()"]
    assert r.trace["cuscope"]["kernels"][0]["name"] == "mock_kernel()"


def test_flow_arrow_links_launch_to_span(mimic):
    r = mimic("launch", "shim")
    flows = [e for e in r.trace["traceEvents"] if e["ph"] in ("s", "f")]
    assert sorted(e["ph"] for e in flows) == ["f", "s"]
    assert flows[0]["id"] == flows[1]["id"] == gpu_spans(r.trace)[0]["args"]["corr"]


# ---- Task 13: copy tracking ----

def copies(trace: dict) -> list[tuple]:
    return [(c["dir"], c["pinned"], c["count"], c["bytes"]) for c in trace["cuscope"]["copies"]]


def test_copy_direction(mimic):
    # A synchronous host-to-device copy from a std::vector (pageable memory).
    r = mimic("copy", "shim")
    assert r.code == 0, r.stderr
    assert copies(r.trace) == [("HtoD", False, 1, 4096)]
    assert [s["name"] for s in gpu_spans(r.trace)] == ["HtoD 4096 B"]


def test_generic_copy_direction_from_pointers(mimic):
    # cuMemcpyAsync says nothing about direction; cuscope asks the driver what
    # each pointer is. Both are device allocations here.
    r = mimic("ptsz", "shim")
    assert r.code == 0, r.stderr
    assert copies(r.trace) == [("DtoD", False, 1, 16)]


# ---- Task 16: sampling ----

def test_sample_every(mimic):
    r = mimic("launch8", "shim", env={"CUSCOPE_SAMPLE_EVERY": "4"})
    assert r.code == 0, r.stderr
    report = r.trace["cuscope"]
    assert len(gpu_spans(r.trace)) == 2
    assert report["gpu_untimed"] == 6
    assert (report["kernels"][0]["count"], report["kernels"][0]["timed"]) == (8, 2)
    assert report["sample_every"] == 4


def test_copies_are_not_sampled(mimic):
    # Bandwidth divides all bytes by GPU time, so every copy must be timed.
    r = mimic("copy", "shim", env={"CUSCOPE_SAMPLE_EVERY": "4"})
    assert len(gpu_spans(r.trace)) == 1 and r.trace["cuscope"]["gpu_untimed"] == 0


# ---- Final review: polling cost and bounded exit ----

def test_polling_stops_at_first_unfinished_event_per_stream(mimic):
    # Events on one stream complete in order, so one unfinished event means
    # every later one on that stream is unfinished too: querying them all on
    # every 2 ms poll (with the lock held) stalls the program's launches.
    r = mimic("launch200", "shim", env={"MOCK_EVENT_DELAY_QUERIES": "1000000", "CUSCOPE_FLUSH_TIMEOUT_MS": "50"})
    assert r.code == 0, r.stderr
    assert r.mock["event_query"] < 1000, r.mock["event_query"]  # 200 per poll before the fix


def test_exit_flush_is_bounded(mimic):
    # Work that never completes must not hold the process at exit: after the
    # flush timeout the remaining events are counted as unresolved.
    r = mimic("launch200", "shim", env={"MOCK_EVENT_DELAY_QUERIES": "1000000", "CUSCOPE_FLUSH_TIMEOUT_MS": "50"})
    assert r.code == 0, r.stderr
    assert r.trace["cuscope"]["unresolved_events"] == 200
    assert gpu_spans(r.trace) == []


def test_banner_once_per_process_that_does_cuda_work(mimic):
    # The banner says cuscope is watching this process: printed at its first
    # allocation, free, copy or launch, and only then.
    r = mimic("launch", "shim")
    assert r.code == 0, r.stderr
    assert r.stderr.count(" loaded (pid ") == 1, r.stderr


def test_preloaded_runs_keep_program_behaviour(mimic):
    # Every call the program made reaches the driver unchanged. Event calls are
    # left out: from Task 10 on, cuscope records its own timing events.
    def program_calls(mock: dict) -> dict:
        return {k: v for k, v in mock.items() if not k.startswith(("event_record", "event_query"))}

    for scenario in ["dlsym_gpa", "ptsz", "old_version", "gpa_self", "fork"]:
        for mode in ["direct", "shim"]:
            base, under = mimic(scenario, mode, preload=False), mimic(scenario, mode)
            assert under.code == base.code, (scenario, mode)
            assert program_calls(under.mock) == program_calls(base.mock), (scenario, mode)
