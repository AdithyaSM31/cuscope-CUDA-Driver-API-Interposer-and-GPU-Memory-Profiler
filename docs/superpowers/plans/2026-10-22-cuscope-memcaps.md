# cuscope Phase 2a: Memory Caps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each process a GPU memory budget (`CUSCOPE_MEM_LIMIT` / `--mem-limit`): allocations past it fail with `CUDA_ERROR_OUT_OF_MEMORY`, `cuMemGetInfo` reports the capped GPU, and denials show up in the trace and summary. Release as `v0.2.0`.

**Architecture:** A new CUDA-free `MemLimit` class holds the cap and an atomic `charged` byte count. Allocation wrappers reserve bytes before the real call (deny if over), give them back on failure, free, teardown and fork. A new `cuMemGetInfo` hook rewrites the driver's answer. Denials are `AllocDenied` records, so they flow through the existing ring, trace writer and report.

**Tech Stack:** C++17, CMake, GoogleTest, pytest, the mock libcuda (direct and shim modes), CUDA 13.0 on the RTX 3060 (WSL2), PyTorch for the demo.

**Spec:** `docs/superpowers/specs/2026-10-22-cuscope-memcaps-design.md` (and the Phase 1 spec it builds on)

## Global Constraints

- Counted kinds: `Device`, `Async`, `Managed` (`is_device_memory(kind)`); `HostPinned` never counts.
- A denial returns `CUDA_ERROR_OUT_OF_MEMORY`, does not call the driver, and does not write the output pointer.
- Exactly at the cap is allowed; one byte over is denied.
- `cuMemAllocPitch` charges `width × height` before the call and trues up to `pitch × height` after; the true-up never denies.
- `cuMemGetInfo` with a cap: `total = min(cap, real_total)`, `free = min(real_free, total − charged)` floored at 0; with no cap it is untouched. It records no API event.
- `CUSCOPE_MEM_LIMIT`: positive number, optional suffix `K`/`M`/`G`/`T` (binary, case-insensitive, optional trailing `B` or `iB`), decimals allowed with a suffix; empty, zero, negative or junk → one warning, uncapped.
- No cap set: Phase 1 behaviour is identical and the trace's `mem_limit` is `null`.
- Never deny a free, a copy or a launch.
- Same rules as Phase 1: TDD (watch every test fail first), mock suite in CI, GPU tests on the 3060, commit at least once per task, every deviation ledgered as `Ruling:`. Steps marked **[You]** are the user's; the user may hand them to Claude.

## Review Focus

1. A request so large that adding it to `charged` would overflow `uint64_t` (`SIZE_MAX`) must be denied, not wrap to a small total. → Task 2 test `MemLimit.HugeRequestDenied`.
2. A driver failure under the cap (bad argument, real OOM) must give the reservation back, or the budget leaks a little on every failure. → Task 3 scenario `cap_driver_fail`.
3. A double or stray free must not release bytes it never charged, or a buggy program gains budget. → Task 3 scenario `cap_double_free`.
4. When pitch padding has pushed `charged` past the cap, `cuMemGetInfo` must report `free = 0`, not a wrapped huge number. → Task 2 test `MemLimit.CapInfoFloorsFree`.
5. With no cap, `cuMemGetInfo` must return exactly the driver's numbers. → Task 5 test `test_meminfo_uncapped_is_untouched`.

---

### Task 1: `CUSCOPE_MEM_LIMIT` and `--mem-limit`

**Files:**
- Modify: `src/config.hpp`, `src/config.cpp`
- Modify: `cli/cuscope.py` (the `run` option list)
- Test: `tests/unit/test_config.cpp`, `tests/cli/test_cli.py`

**Interfaces:**
- Produces: `uint64_t Config::mem_limit_bytes = 0;` (0 = no cap); `bool parse_size(const char* s, uint64_t* out);` declared in `config.hpp` (true on a valid positive size).

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(Config, MemLimitSizes) {
  auto limit = [](const char* v) { return parse_config(env({{"CUSCOPE_MEM_LIMIT", v}})).mem_limit_bytes; };
  EXPECT_EQ(parse_config(env({})).mem_limit_bytes, 0u);
  EXPECT_EQ(limit("268435456"), 268435456u);
  EXPECT_EQ(limit("512M"), 512ull << 20);
  EXPECT_EQ(limit("2G"), 2ull << 30);
  EXPECT_EQ(limit("2g"), 2ull << 30);
  EXPECT_EQ(limit("1.5GiB"), 3ull << 29);
  EXPECT_EQ(limit("64KB"), 64ull << 10);
  EXPECT_EQ(limit("1T"), 1ull << 40);
}

TEST(Config, MemLimitRejectsJunk) {
  for (const char* v : {"", "0", "-1G", "G", "2X", "1.5", "two"}) {
    Config c = parse_config(env({{"CUSCOPE_MEM_LIMIT", v}}));
    EXPECT_EQ(c.mem_limit_bytes, 0u) << v;
    EXPECT_EQ(c.warnings.size(), 1u) << v;
  }
}
```

```python
def test_run_sets_mem_limit():
    assert child_env("--mem-limit", "2G")["CUSCOPE_MEM_LIMIT"] == "2G"
```

- [ ] **Step 2: Run to verify they fail**

Run: `cmake --build build && ./build/tests/unit/cuscope_unit_tests --gtest_filter='Config.MemLimit*'` and `python3 -m pytest -q tests/cli -k mem_limit`
Expected: compile error (`mem_limit_bytes` missing); CLI test fails (unknown option).

- [ ] **Step 3: Implement `parse_size` and the config field; add `("mem_limit", "CUSCOPE_MEM_LIMIT")` and `run.add_argument("--mem-limit", help="cap GPU memory per process, e.g. 2G (default: none)")` to the CLI**

`1.5` with no suffix is junk (decimals need a suffix). The warning text follows the existing `warn()` helper: `ignoring CUSCOPE_MEM_LIMIT="…", using no cap`.

- [ ] **Step 4: Run to verify they pass**

Same commands. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.hpp src/config.cpp cli/cuscope.py tests/unit/test_config.cpp tests/cli/test_cli.py
git commit -m "feat: CUSCOPE_MEM_LIMIT and --mem-limit"
```

---

### Task 2: The `MemLimit` class

**Files:**
- Create: `src/limits.hpp`, `src/limits.cpp` (add to `cuscope_core` in `CMakeLists.txt`)
- Modify: `src/model.hpp` (`MemLimitStats`, `Snapshot::mem_limit`)
- Test: `tests/unit/test_limits.cpp` (add to `tests/unit/CMakeLists.txt`)

**Interfaces:**
- Consumes: `Config::mem_limit_bytes` (Task 1).
- Produces:

```cpp
// model.hpp
struct MemLimitStats {
  bool enabled = false;
  uint64_t limit_bytes = 0, peak_charged_bytes = 0, denied_count = 0, denied_bytes = 0;
  uint64_t largest_denied_bytes = 0;
  AllocKind largest_denied_kind = AllocKind::Device;
  uint32_t largest_denied_stack = 0;
};
// Snapshot gains: MemLimitStats mem_limit;

// limits.hpp
class MemLimit {
 public:
  explicit MemLimit(uint64_t limit_bytes);           // 0 = no cap
  bool enabled() const;
  uint64_t limit() const;
  uint64_t charged() const;
  bool try_charge(AllocKind kind, uint64_t bytes);  // false = over the cap; uncounted kinds always true
  void charge(AllocKind kind, uint64_t bytes);      // never denies (restore after a rejected free, pitch true-up)
  void release(AllocKind kind, uint64_t bytes);
  void on_denied(AllocKind kind, uint64_t bytes, uint32_t stack_id);
  void cap_info(size_t* free_bytes, size_t* total_bytes) const;  // rewrites the driver's figures in place
  void fill(Snapshot& s) const;
};
MemLimit& mem_limit();                    // process-wide, from config(); never destroyed
void mem_limit_after_fork_child();        // fresh MemLimit with the same cap
```

- [ ] **Step 1: Write the failing tests** (`tests/unit/test_limits.cpp`)

```cpp
constexpr uint64_t MiB = 1ull << 20;

TEST(MemLimit, ChargesUpToTheCapExactly) {
  MemLimit m(256 * MiB);
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(m.try_charge(AllocKind::Device, 64 * MiB));
  EXPECT_FALSE(m.try_charge(AllocKind::Device, 1));
  EXPECT_EQ(m.charged(), 256 * MiB);
  m.release(AllocKind::Device, 64 * MiB);
  EXPECT_TRUE(m.try_charge(AllocKind::Async, 64 * MiB));
}

TEST(MemLimit, PinnedHostNeverCounts) {
  MemLimit m(1 * MiB);
  EXPECT_TRUE(m.try_charge(AllocKind::HostPinned, 64 * MiB));
  EXPECT_EQ(m.charged(), 0u);
  m.release(AllocKind::HostPinned, 64 * MiB);
  EXPECT_EQ(m.charged(), 0u);
}

TEST(MemLimit, HugeRequestDenied) {  // Review Focus 1
  MemLimit m(256 * MiB);
  EXPECT_TRUE(m.try_charge(AllocKind::Device, 64 * MiB));
  EXPECT_FALSE(m.try_charge(AllocKind::Device, UINT64_MAX));
  EXPECT_FALSE(m.try_charge(AllocKind::Managed, UINT64_MAX - 32 * MiB));
  EXPECT_EQ(m.charged(), 64 * MiB);
}

TEST(MemLimit, NoCapNeverDenies) {
  MemLimit m(0);
  EXPECT_FALSE(m.enabled());
  EXPECT_TRUE(m.try_charge(AllocKind::Device, UINT64_MAX));
}

TEST(MemLimit, CapInfo) {
  MemLimit m(2048 * MiB);
  m.charge(AllocKind::Device, 512 * MiB);
  size_t free_b = 5000 * MiB, total = 6144 * MiB;
  m.cap_info(&free_b, &total);
  EXPECT_EQ(total, 2048 * MiB);
  EXPECT_EQ(free_b, 1536 * MiB);
  free_b = 100 * MiB, total = 6144 * MiB;  // the GPU itself is nearly full
  m.cap_info(&free_b, &total);
  EXPECT_EQ(free_b, 100 * MiB);
}

TEST(MemLimit, CapInfoFloorsFree) {  // Review Focus 4
  MemLimit m(256 * MiB);
  m.charge(AllocKind::Device, 300 * MiB);  // pitch padding past the cap
  size_t free_b = 5000 * MiB, total = 6144 * MiB;
  m.cap_info(&free_b, &total);
  EXPECT_EQ(free_b, 0u);
}

TEST(MemLimit, StatsAndPeak) {
  MemLimit m(256 * MiB);
  m.try_charge(AllocKind::Device, 200 * MiB);
  m.release(AllocKind::Device, 200 * MiB);
  m.on_denied(AllocKind::Device, 512 * MiB, 7);
  m.on_denied(AllocKind::Managed, 300 * MiB, 9);
  Snapshot s;
  m.fill(s);
  EXPECT_TRUE(s.mem_limit.enabled);
  EXPECT_EQ(s.mem_limit.limit_bytes, 256 * MiB);
  EXPECT_EQ(s.mem_limit.peak_charged_bytes, 200 * MiB);
  EXPECT_EQ(s.mem_limit.denied_count, 2u);
  EXPECT_EQ(s.mem_limit.denied_bytes, 812 * MiB);
  EXPECT_EQ(s.mem_limit.largest_denied_bytes, 512 * MiB);
  EXPECT_EQ(s.mem_limit.largest_denied_stack, 7u);
}

TEST(MemLimit, ThreadsNeverOvershoot) {
  MemLimit m(4 * MiB);
  std::atomic<uint64_t> held{0}, worst{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 20000; ++i) {
        if (!m.try_charge(AllocKind::Device, 1 * MiB)) continue;
        uint64_t now = held.fetch_add(1 * MiB) + 1 * MiB;
        uint64_t w = worst.load();
        while (now > w && !worst.compare_exchange_weak(w, now)) {}
        held.fetch_sub(1 * MiB);
        m.release(AllocKind::Device, 1 * MiB);
      }
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_LE(worst.load(), 4 * MiB);
  EXPECT_EQ(m.charged(), 0u);
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cmake -S . -B build && cmake --build build`
Expected: compile error (`limits.hpp` not found).

- [ ] **Step 3: Implement `MemLimit` in `src/limits.cpp`**

`try_charge` is a compare-exchange loop on `charged_` that refuses when `bytes > limit_ - current` (no addition, so no overflow), then updates the peak with a CAS maximum. `on_denied` takes a small mutex only for the "largest" fields; counts are atomics. `mem_limit()` follows `tracker()`: heap-allocated on first use from `config().mem_limit_bytes`, never destroyed; `mem_limit_after_fork_child()` replaces it.

- [ ] **Step 4: Run to verify they pass**

Run: `./build/tests/unit/cuscope_unit_tests --gtest_filter='MemLimit.*'`
Expected: 8 tests PASS. Also run the TSan unit build: `cmake --build build-thread && setarch -R ctest --test-dir build-thread -L unit` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/limits.hpp src/limits.cpp src/model.hpp CMakeLists.txt tests/unit/test_limits.cpp tests/unit/CMakeLists.txt
git commit -m "feat: MemLimit, the per-process GPU memory budget"
```

---

### Task 3: Allocation hooks charge, deny and release

**Files:**
- Modify: `src/model.hpp` (`RecType::AllocDenied`, appended last), `src/wrappers/alloc.cpp`
- Modify (mock): `tests/mock/mock_driver.cpp` (size map, `cuMemAllocHost`/`cuMemFreeHost`/`cuMemAllocPitch` with pitch rounded up to 512, `live_hwm` counter), `tests/mock/mock_counters.h`, `tests/mock/mimic_common.hpp` (print `live_hwm`), `tests/mock/runtime_mimic.cpp`, `tests/mock/conftest.py` (scenario list)
- Test: `tests/mock/test_memcaps.py` (new)

**Interfaces:**
- Consumes: `mem_limit()`, `MemLimit::try_charge/charge/release/on_denied` (Task 2).
- Produces: `RecType::AllocDenied` records (`aux` = kind, `bytes`, `stack_id`, `api`); mock counter `live_hwm` (highest live device bytes); mimic helper `CUresult try_alloc(size_t bytes, CUdeviceptr* out)`.

Mock scenarios (all print `RESULT <name> <CUresult>` lines for each allocation the test checks, plus the usual `MOCK` line). Each test runs with `env={"CUSCOPE_MEM_LIMIT": ...}`, in both `direct` and `shim` modes:

| Scenario | Program | Cap |
|---|---|---|
| `cap_exact` | five `try_alloc(64 MiB)`; free the first; one more `try_alloc(64 MiB)` | `256M` |
| `cap_threads` | eight threads × 500 iterations: `try_alloc(1 MiB)`, free on success | `4M` |
| `cap_pinned` | `cuMemAllocHost(64 MiB)` then free it | `1M` |
| `cap_pitch` | `cuMemAllocPitch(width=1000, height=1024)`, then `try_alloc(1 MiB)`, then `try_alloc(1)` | `2M` |
| `cap_driver_fail` | `cuMemAlloc(nullptr, 64 MiB)` (mock: INVALID_VALUE), then four `try_alloc(64 MiB)` | `256M` |
| `cap_double_free` | `try_alloc(64 MiB)`, free it twice, then four `try_alloc(64 MiB)` and a fifth | `256M` |

- [ ] **Step 1: Write the failing tests** (`tests/mock/test_memcaps.py`)

```python
MiB = 1 << 20
OOM, OK, INVALID = 2, 0, 1  # CUDA_ERROR_OUT_OF_MEMORY, CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

def results(r):
    return [int(l.split()[2]) for l in r.stdout.splitlines() if l.startswith("RESULT ")]

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
    assert results(mimic("cap_driver_fail", "shim", env={"CUSCOPE_MEM_LIMIT": "256M"})) == [INVALID, OK, OK, OK, OK]

def test_double_free_releases_nothing_twice(mimic):  # Review Focus 3
    r = mimic("cap_double_free", "shim", env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert results(r) == [OK, OK, INVALID, OK, OK, OK, OK, OOM]

def test_no_cap_changes_nothing(mimic):
    assert results(mimic("cap_exact", "shim")) == [OK] * 6
```

(`cap_double_free` prints a result for the allocation, both frees and the five allocations.)

- [ ] **Step 2: Run to verify they fail**

Run: `cmake --build build && CUSCOPE_BUILD=build python3 -m pytest -q tests/mock/test_memcaps.py`
Expected: FAIL: every allocation returns OK under the cap (`[0, 0, 0, 0, 0, 0]`).

- [ ] **Step 3: [You] Charge, deny and release in `src/wrappers/alloc.cpp`**

Add one helper the allocation wrappers call before forwarding:

```cpp
// True if the allocation may go ahead. Reserves `bytes` against the cap; on
// denial records an AllocDenied record (with the caller's stack) and the
// caller returns CUDA_ERROR_OUT_OF_MEMORY without calling the driver.
bool reserve(ApiRecord& rec, AllocKind kind, uint64_t bytes);
```

- `reserve` is a no-op returning true when `!mem_limit().enabled()` or `in_internal()`.
- `after_alloc` releases the reservation when the real call failed.
- `cuMemAllocPitch` reserves `WidthInBytes * Height`, then after success `charge`s the difference to `pitch * Height` (or `release`s it if smaller).
- `tracked_free` releases `freed->bytes` after the driver accepts the free; if it rejects it, `restore` the tracker entry and leave the charge as it was (it was never released). A free of an untracked pointer releases nothing.

- [ ] **Step 4: Run to verify they pass**

Run: `CUSCOPE_BUILD=build python3 -m pytest -q tests/mock/test_memcaps.py` then `ctest --test-dir build`
Expected: new tests PASS; full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model.hpp src/wrappers/alloc.cpp tests/mock
git commit -m "feat: allocation hooks deny past the memory cap"
```

---

### Task 4: Teardown and fork give bytes back

**Files:**
- Modify: `src/tracker.hpp`, `src/tracker.cpp` (`on_ctx_teardown` returns released bytes per kind), `src/wrappers/teardown.cpp` (both call sites release them), `src/runtime.cpp` (`after_fork_child` calls `mem_limit_after_fork_child()`)
- Modify (mock): `tests/mock/runtime_mimic.cpp`, `tests/mock/conftest.py`
- Test: `tests/mock/test_memcaps.py`, `tests/unit/test_tracker.cpp`

**Interfaces:**
- Consumes: `mem_limit()`, `mem_limit_after_fork_child()` (Task 2).
- Produces: `std::array<uint64_t, kAllocKinds> Tracker::on_ctx_teardown(uint64_t ctx);` (bytes per kind that died with the context).

| Scenario | Program | Cap |
|---|---|---|
| `cap_teardown` | `try_alloc(96 MiB)`; `cuDevicePrimaryCtxRelease(0)` (destroys it); retain and set current again; `try_alloc(96 MiB)` | `128M` |
| `cap_fork` | `try_alloc(96 MiB)`; fork; the child prints its own `try_alloc(96 MiB)` result and exits 0 | `128M` |

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(Tracker, TeardownReportsReleasedBytesPerKind) {
  Tracker t;
  t.on_alloc({0x1000, 100, AllocKind::Device, 0, 7});
  t.on_alloc({0x2000, 50, AllocKind::Managed, 0, 7});
  t.on_alloc({0x3000, 30, AllocKind::Device, 0, 8});
  auto gone = t.on_ctx_teardown(7);
  EXPECT_EQ(gone[static_cast<int>(AllocKind::Device)], 100u);
  EXPECT_EQ(gone[static_cast<int>(AllocKind::Managed)], 50u);
}
```

```python
def test_context_teardown_gives_bytes_back(mimic):
    assert results(mimic("cap_teardown", "shim", env={"CUSCOPE_MEM_LIMIT": "128M"})) == [OK, OK]

def test_forked_child_starts_with_an_empty_budget(mimic):
    assert results(mimic("cap_fork", "direct", env={"CUSCOPE_MEM_LIMIT": "128M"})) == [OK, OK]
```

- [ ] **Step 2: Run to verify they fail**

Expected: tracker test fails to compile (`on_ctx_teardown` returns void); mock tests give `[OK, OOM]`.

- [ ] **Step 3: Implement** the return value, `mem_limit().release(kind, bytes)` for each kind at both `on_ctx_teardown` call sites, and the fork reset.

- [ ] **Step 4: Run to verify they pass**

Run: `ctest --test-dir build`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tracker.hpp src/tracker.cpp src/wrappers/teardown.cpp src/runtime.cpp tests
git commit -m "feat: context teardown and fork reset the memory budget"
```

---

### Task 5: The `cuMemGetInfo` hook

**Files:**
- Create: `src/wrappers/meminfo.cpp` (add to `CUSCOPE_SOURCES`)
- Modify: `src/hooks.def`, `src/wrappers/wrappers.hpp` and `src/hook_table.cpp` (`register_meminfo_hooks()`)
- Modify (mock): `tests/mock/mock_driver.cpp` (`cuMemGetInfo`: total 6 GiB, free = total − live bytes; `ENTRY(cuMemGetInfo)`), `runtime_mimic.cpp`, `conftest.py`
- Test: `tests/mock/test_memcaps.py`

**Interfaces:**
- Consumes: `MemLimit::cap_info` (Task 2).
- Produces: hook entry `HOOK(cuMemGetInfo, "cuMemGetInfo_v2", nullptr, 3020)`. The census (`docs/census/*.tsv`) shows the static runtime, shared runtime and PyTorch all request `cuMemGetInfo` at version 3020 with flags 0; resolving the hook there makes the pointers match. Export: `CS_EXPORT(cuMemGetInfo_v2, cuMemGetInfo, 0, (size_t* free, size_t* total), (free, total))`.

Scenario `cap_meminfo`: `try_alloc(64 MiB)`, then `cuMemGetInfo`, printing `MEMINFO <free> <total>`.

- [ ] **Step 1: Write the failing tests**

```python
def meminfo(r):
    line = next(l for l in r.stdout.splitlines() if l.startswith("MEMINFO "))
    return tuple(int(x) for x in line.split()[1:])

@pytest.mark.parametrize("mode", MODES)
def test_meminfo_reports_the_cap(mimic, mode):
    r = mimic("cap_meminfo", mode, env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert meminfo(r) == (192 * MiB, 256 * MiB)

def test_meminfo_uncapped_is_untouched(mimic):  # Review Focus 5
    r = mimic("cap_meminfo", "shim")
    assert meminfo(r) == ((6 << 30) - 64 * MiB, 6 << 30)
```

- [ ] **Step 2: Run to verify they fail**

Expected: `test_meminfo_reports_the_cap` gets the mock's 6 GiB figures.

- [ ] **Step 3: [You] Implement `cs_cuMemGetInfo` in `src/wrappers/meminfo.cpp`**

Forward; if the real call succeeded and `mem_limit().enabled()`, call `cap_info` on the results. `note_call()` only: no `ApiRecord`.

- [ ] **Step 4: Run to verify they pass**

Run: `ctest --test-dir build` and `python3 tools/check_exports.py src/hooks.def` (on the 3060)
Expected: all PASS; `check_exports` reports `cuMemGetInfo` OK at 3020.

- [ ] **Step 5: Commit**

```bash
git add src/wrappers/meminfo.cpp src/hooks.def src/wrappers/wrappers.hpp src/hook_table.cpp CMakeLists.txt tests/mock
git commit -m "feat: cuMemGetInfo reports the capped GPU"
```

---

### Task 6: Denials in the trace, summary and CLI

**Files:**
- Modify: `src/runtime.cpp` (`mem_limit().fill(s)` in `write_report_at_exit`), `src/trace_writer.cpp`, `src/report.cpp`, `cli/cuscope.py`
- Modify: `tests/cli/fixtures/sample_trace.json` (add a `mem_limit` object and one denial)
- Test: `tests/unit/test_trace_writer.cpp`, `tests/unit/test_report.cpp`, `tests/cli/test_cli.py`, `tests/mock/test_memcaps.py`

**Interfaces:**
- Consumes: `Snapshot::mem_limit` (Task 2), `RecType::AllocDenied` (Task 3).
- Produces: trace `"mem_limit"` (`null` or the object in spec §3.6); instant events `{"ph":"i","s":"t","name":"denied cuMemAlloc 512.0 MiB","args":{"bytes":…,"kind":"device","stack":…}}`; summary section `Memory cap`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(TraceWriter, MemLimitSectionAndDenials) {
  Snapshot s;
  s.mem_limit = {true, 2ull << 30, 2000000000, 1, 512ull << 20, 512ull << 20, AllocKind::Device, 3};
  Record d{};
  d.type = RecType::AllocDenied;
  d.api = H_cuMemAlloc;
  d.aux = static_cast<uint8_t>(AllocKind::Device);
  d.bytes = 512ull << 20;
  d.stack_id = 3;
  s.records = {d};
  json j = json::parse(trace_text(s));
  EXPECT_EQ(j["cuscope"]["mem_limit"]["limit_bytes"], 2ull << 30);
  EXPECT_EQ(j["cuscope"]["mem_limit"]["largest_denied"]["bytes"], 512ull << 20);
  auto denied = std::count_if(j["traceEvents"].begin(), j["traceEvents"].end(),
                              [](const json& e) { return e["ph"] == "i" && e["name"] == "denied cuMemAlloc 512.0 MiB"; });
  EXPECT_EQ(denied, 1);
}

TEST(TraceWriter, NoCapIsNull) {
  EXPECT_TRUE(json::parse(trace_text(Snapshot{}))["cuscope"]["mem_limit"].is_null());
}

TEST(Report, MemoryCapSection) {
  Snapshot s = sample();
  s.mem_limit = {true, 2ull << 30, 2140000000, 3, 1610612736, 512ull << 20, AllocKind::Device, 7};
  const std::string text = render(s);
  EXPECT_TRUE(contains(text, "Memory cap: 2.0 GiB, peak charged")) << text;
  EXPECT_TRUE(contains(text, "3 allocations denied (1.5 GiB); largest 512.0 MiB device at main")) << text;
  EXPECT_LT(text.find("Memory cap"), text.find("Leaks"));
}

TEST(Report, NoCapNoSection) { EXPECT_FALSE(contains(render(sample()), "Memory cap")); }
```

```python
def test_trace_reports_the_cap_end_to_end(mimic):
    r = mimic("cap_exact", "shim", env={"CUSCOPE_MEM_LIMIT": "256M"})
    cap = r.trace["cuscope"]["mem_limit"]
    assert (cap["limit_bytes"], cap["peak_charged_bytes"], cap["denied_count"]) == (256 * MiB, 256 * MiB, 1)
    assert mimic("cap_exact", "shim").trace["cuscope"]["mem_limit"] is None
```

(`trace_text` and `render` are the existing test helpers.) CLI: extend `test_summary_sections` with `"Memory cap: 2.0 GiB"` after `"Memory"`, using the fixture's new `mem_limit`.

- [ ] **Step 2: Run to verify they fail**

Expected: compile or assertion failures (`mem_limit` absent).

- [ ] **Step 3: Implement**

The summary line format is exactly spec §3.6: `Memory cap: <cap>, peak charged <peak> (<pct>%)`, then, if any, `  <n> allocations denied (<bytes>); largest <bytes> <kind> at <call site>`, using the existing `bytes_text` and `call_site` helpers (and their Python twins).

- [ ] **Step 4: Run to verify they pass**

Run: `ctest --test-dir build`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src tests cli
git commit -m "feat: memory cap section and denials in the trace and summary"
```

---

### Task 7: GPU test on the 3060

**Files:**
- Create: `tests/cuda/cap_exact.cu` (add static and shared builds to `tests/cuda/CMakeLists.txt` as `cap_exact_static`, `cap_exact_shared`)
- Test: `tests/gpu/test_memcaps.py` (new)

`cap_exact.cu`: `cudaFree(0)` to start the runtime; `cudaMalloc` 64 MiB in a loop (at most 16) until it fails, printing `ok` per success and `cudaGetErrorName` of the first failure on its own line; `cudaMemGetInfo` printing `MEMINFO <free> <total>`; free the first buffer; one more `cudaMalloc(64 MiB)` printing `after free: <error name>` as the last line; free everything; exit 0.

- [ ] **Step 1: Write the failing tests**

```python
MiB = 1 << 20

@pytest.mark.parametrize("prog", ["cap_exact_static", "cap_exact_shared"])
def test_cap_denies_at_the_limit(prog):
    r = run(prog, env={"CUSCOPE_MEM_LIMIT": "256M"})
    assert r.code == 0, r.stderr
    lines = r.stdout.splitlines()
    oks = lines.index("cudaErrorMemoryAllocation")
    cap = r.trace["cuscope"]["mem_limit"]
    assert cap["denied_count"] == 1
    assert cap["peak_charged_bytes"] <= 256 * MiB < cap["peak_charged_bytes"] + 64 * MiB
    assert oks == cap["peak_charged_bytes"] // (64 * MiB)  # any runtime allocations are under 64 MiB
    assert lines[-1] == "after free: cudaSuccess"
    free_b, total = map(int, next(l for l in lines if l.startswith("MEMINFO ")).split()[1:])
    assert total == 256 * MiB and free_b < 64 * MiB

@pytest.mark.parametrize("prog", ["hello_static", "multistream", "async_pool"])
def test_generous_cap_changes_nothing(prog):
    plain, capped = run(prog), run(prog, env={"CUSCOPE_MEM_LIMIT": "4G"})
    assert (capped.code, capped.stdout) == (plain.code, plain.stdout)
    assert capped.trace["cuscope"]["mem_limit"]["denied_count"] == 0
```

- [ ] **Step 2: Run to verify it fails before the build exists, then passes**

Run: `cmake -S . -B build && cmake --build build && CUSCOPE_BUILD=build python3 -m pytest -q tests/gpu/test_memcaps.py`
Expected: PASS (Tasks 3–6 already make it pass; this task's RED is the missing program: run the test before adding `cap_exact` to CMake and see `FileNotFoundError`). Ledger any runtime allocation the trace shows.

- [ ] **Step 3: Commit**

```bash
git add tests/cuda tests/gpu/test_memcaps.py
git commit -m "test: memory cap on the GPU through the CUDA runtime"
```

---

### Task 8: The two-job PyTorch demo

**Files:**
- Create: `tools/memcaps_job.py`, `tools/demo_memcaps.py`, `docs/memcaps.md`

**Interfaces:**
- `memcaps_job.py --steps N --batch B --min-res R1 --max-res R2 --seed S --json OUT`: ResNet-18 training on random images whose resolution is drawn per step from `[R1, R2]` (varying shapes fragment PyTorch's cache, so reserved memory grows well past the working set, as with variable-length inputs). Writes `{"exit": "ok"|"oom", "steps_done", "max_allocated", "max_reserved", "step_ms_median"}`; catches `torch.OutOfMemoryError` and exits 3.
- `demo_memcaps.py --cap SIZE [--job-args ...]`: runs the pair twice, uncapped then each job capped at `SIZE`, starting job B 10 s after job A, both with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`; prints a table and writes `docs/memcaps.json`.

- [ ] **Step 1: Calibrate one job alone** (uncapped, under cuscope): choose `--batch`, `--min-res`, `--max-res` so `max_reserved` is above half the free VRAM (`nvidia-smi`) while `max_allocated` is below `SIZE`. Record the numbers in the ledger.

- [ ] **Step 2: Run the demo**

Run: `python3 tools/demo_memcaps.py --cap 2.5G <calibrated job args>`
Expected: uncapped, at least one job reports `oom`; capped, both report `ok`. If not, adjust per Step 1 (spec §8) and ledger the change.

- [ ] **Step 3: Write `docs/memcaps.md`**: setup (GPU, driver, model, sizes), the two-run table (exit, steps, max allocated, max reserved, median step time per job), what the cap does and does not cover (spec §2 non-goals, §8), and the measured context overhead (`nvidia-smi` per-process usage minus cuscope's `peak_charged_bytes`).

- [ ] **Step 4: Commit**

```bash
git add tools/memcaps_job.py tools/demo_memcaps.py docs/memcaps.md docs/memcaps.json
git commit -m "docs: two-job memory cap demo on the RTX 3060"
```

---

### Task 9: Overhead, docs and release

**Files:**
- Modify: `bench/run_bench.py` (mode `full+cap`, sets `CUSCOPE_MEM_LIMIT=4G`), `docs/overhead.md`, `README.md`, `docs/DESIGN.md`, `CMakeLists.txt` (`VERSION 0.2.0`)

- [ ] **Step 1: Rerun the microbenchmarks** with `python3 bench/run_bench.py` including `full+cap`; add the alloc/free rows to `docs/overhead.md`. Expected: `full+cap` alloc/free within noise of `full`.
- [ ] **Step 2: Docs**: README quickstart line for `--mem-limit`, a "Memory caps" section linking `docs/memcaps.md`, limitations (context/driver memory outside the cap, pool reservation, VMM APIs, per-process only, pitch overshoot), roadmap now lists only Phase 2b; DESIGN.md gets a "Memory caps" section (charging, release points, `cuMemGetInfo` at 3020, the one relaxed safety rule).
- [ ] **Step 3: Full verification**: `ctest --test-dir build` (with GPU), ASan and TSan unit builds, `tools/census.py run && tools/census.py check`.
- [ ] **Step 4: Commit, push, check CI**

```bash
git add -A
git commit -m "docs: memory caps; version 0.2.0"
git push origin main
```

- [ ] **Step 5: Final whole-branch review** (executing-plans), fix pass, then the user tags `v0.2.0`.
