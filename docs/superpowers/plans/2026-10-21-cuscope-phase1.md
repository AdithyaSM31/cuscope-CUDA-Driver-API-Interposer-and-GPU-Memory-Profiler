# cuscope Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Pairing model (overrides the sub-skill where they conflict):** every step is tagged **[You]** (Adithya writes it) or **[Claude]**. Claude never writes a **[You]** step's code. On those steps Claude writes the failing tests first, then waits, then reviews the code (explaining problems and reasons, not rewriting), then runs verification. In every task, test steps come before code steps.

**Goal:** Ship cuscope v0.1.0, an `LD_PRELOAD` library and CLI that records GPU allocations, copies and kernel launches from unmodified CUDA programs on WSL2, and validate it against PyTorch.

**Architecture:** `libcuscope.so` catches driver calls on three paths: exported symbols, a `dlsym` override (with a libcuda caller check), and a `cuGetProcAddress` hook that wraps a pointer only when it matches the variant resolved at init. Wrappers forward the call unchanged, then feed an allocation tracker and per-thread ring buffers. A collector thread resolves GPU event timings. At exit cuscope writes a Chrome Trace JSON (with a `cuscope` report section) and prints a summary to stderr.

**Tech Stack:** C++17, CMake ≥ 3.28 + Ninja, gcc 11.4, CUDA 13.0 headers (`cuda.h`) and nvcc for test programs, GoogleTest 1.15.2 and nlohmann/json 3.11.3 (tests only, via FetchContent), Python 3.10 + pytest for the CLI and harness, PyTorch 2.14+cu130.

**Spec:** `docs/superpowers/specs/2026-10-21-cuscope-phase1-design.md` (read it with this plan).

## Global Constraints

- All commands run inside WSL Ubuntu-22.04 from `~/cuscope`. Build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build`.
- CUDA toolkit pinned to **13.0** (`cuda-toolkit-13-0`). Requested driver version everywhere is `CUDA_VERSION` from `cuda.h` (13000). Never install a Linux NVIDIA driver in WSL.
- `libcuscope.so` **never links `-lcuda`**. It compiles against `cuda.h` only and links `-ldl -pthread`.
- `-fvisibility=hidden`. Only these are exported: `dlsym`, `cuGetProcAddress`, `cuGetProcAddress_v2`, the export names listed in `hooks.def`, and `cuscope_get_stats`.
- Never define `CUDA_API_PER_THREAD_DEFAULT_STREAM` when compiling `libcuscope`, so `cuda.h` macros map to the legacy names.
- Code reachable from `dlsym`/`cuGetProcAddress` must work **before cuscope's own constructors run** (the WSL shim can call `dlsym` from its constructor first). That means no namespace-scope objects with dynamic initialization: use POD, zero-initialized atomics, function-local statics, `pthread_once`.
- Safety rules from spec §7: pass through exactly, fail open, never write to stdout. Logs are prefixed `[cuscope] ` and go to `CUSCOPE_LOG` or stderr.
- Env vars and defaults: `CUSCOPE_OUT=cuscope-%p.json` (`%p` → pid), `CUSCOPE_MODE=full|alloc` (default `full`), `CUSCOPE_STACK_DEPTH=16` (clamped to 1..64), `CUSCOPE_SAMPLE_EVERY=1` (≥ 1), `CUSCOPE_LOG` (unset = stderr), `CUSCOPE_CENSUS` (unset = off).
- `sizeof(Record) == 64`.
- CUDA test programs: `CMAKE_CUDA_ARCHITECTURES=86`, `-g -lineinfo`, static runtime unless stated.
- Commit at the end of every task. Never push unless asked.

## Review Focus

1. **Re-entrant `dlsym` during cuscope's own init.** The shim's constructor calls `dlsym` while cuscope is `dlopen`ing libcuda, and `dlsym(RTLD_DEFAULT, "cuGetProcAddress_v2")` would find cuscope's own export. Expected: no deadlock, no self-recursion. Pinned by `test_dlsym_default_during_init` (Task 3) and `test_real_gpa_is_never_own_wrapper` (Task 6).
2. **Launch or copy on a stream under graph capture.** Expected: cuscope records no events into the capture, and the graph behaves identically. Pinned by `test_capturing_stream_not_timed` (Task 10) and `graph_capture` GPU assertions (Task 10).
3. **A free racing with re-allocation of the same address on another thread.** Expected: exact counts, no false leak or invalid free. Pinned by `Tracker.RestoreAfterFailedFree`, `Tracker.AddressReuseAfterFree` and `multithread` (Task 8).
4. **Forked children inheriting `LD_PRELOAD`** (dataloader workers, subprocesses). Expected: each process writes its own trace, and the child reports none of the parent's allocations. Pinned by `test_fork_child_has_no_inherited_leaks` (Task 9).
5. **Unwritable `CUSCOPE_OUT` or bad env values.** Expected: the program's stdout and exit code are unchanged, plus one stderr warning. Pinned by `Config.InvalidValuesFallBack` (Task 1) and `test_unwritable_out_keeps_program_behaviour` (Task 8).

## File Map

```
CMakeLists.txt                     root build; options CUSCOPE_GPU_TESTS, CUSCOPE_SANITIZE
include/cuscope/cuscope.h          C API: cuscope_stats, cuscope_get_stats (tests use it)
src/log.{hpp,cpp}                  log_msg, CS_LOG_ONCE
src/config.{hpp,cpp}               Config, parse_config, config(), expand_out_path
src/stats.{hpp,cpp}                atomic counters behind cuscope_get_stats
src/census.{hpp,cpp}               CUSCOPE_CENSUS TSV writer
src/loader.{hpp,cpp}               real_dlsym, libcuda_handle, ensure_loader, real gpa
src/intercept.cpp                  exported dlsym, cuGetProcAddress(_v2), caller check
src/hooks.def                      X-macro HOOK(base, legacy_export, ptds_export)
src/hook_table.{hpp,cpp}           HookId, HookEntry, find_hook_*, real_fn, hook_name
src/guard.hpp                      InternalScope, in_internal()
src/wrappers/wrappers.hpp          declarations of every cs_<base><V> template
src/wrappers/{alloc,memcpy,launch,teardown,names}.cpp
src/model.hpp                      AllocKind, CopyDir, RecType, Record, GpuSpan, Snapshot
src/tracker.{hpp,cpp}              allocation tracker
src/stacks.{hpp,cpp}               capture_stack, symbolize_all, demangle
src/ring.hpp                       SpscRing<T,N>
src/recorder.{hpp,cpp}             per-thread rings, collector thread, atfork
src/gputimer.{hpp,cpp}             event pool, clock alignment, pending resolution
src/summary.{hpp,cpp}              pure aggregation: kernels, copies, memory
src/trace_writer.{hpp,cpp}         Chrome Trace JSON + "cuscope" section
src/report.{hpp,cpp}               text summary
src/runtime.cpp                    constructor banner, exit handler, snapshot assembly
cli/cuscope.py                     run / summary, addr2line symbolization
tests/unit/*.cpp                   GoogleTest
tests/mock/                        mock driver, mock shim, runtime_mimic, direct_mimic, pytest
tests/cuda/*.cu, multithread.cpp   GPU known-answer programs
tests/harness.py, tests/gpu/       GPU pytest suite
tests/cli/                         CLI pytest
tools/census.py, tools/check_exports.py, tools/validate_pytorch.py
bench/micro.cu, bench/run_bench.py, bench/pytorch_resnet.py
docs/SETUP.md, docs/census/, docs/validation.md, docs/overhead.md, docs/DESIGN.md
.github/workflows/ci.yml, README.md
```

## Shared Interfaces (defined once, used everywhere)

```cpp
// src/model.hpp  (Task 7, but the enums are needed from Task 8; Task 7 lands first)
enum class AllocKind : uint8_t { Device, Managed, HostPinned, Async };
enum class CopyDir : uint8_t { HtoD, DtoH, DtoD, HtoH, Unknown };   // aux = dir | (pinned << 4)
enum class RecType : uint8_t { ApiCall, Alloc, AllocFailed, Free, InvalidFree, Launch, Copy, GraphLaunch };
struct alignas(64) Record {
  RecType type; uint8_t aux; uint16_t api;   // api = HookId
  uint32_t tid; uint64_t cpu_ns; uint32_t dur_ns; uint32_t stack_id;
  uint64_t ptr; uint64_t bytes; uint64_t stream; uint64_t func; uint64_t corr;
};  // static_assert(sizeof(Record) == 64)
struct GpuSpan { uint64_t corr, stream, start_ns, end_ns, func, bytes; uint16_t api; uint8_t aux; };
struct StackFrame { std::string module; uint64_t offset; std::string symbol; };
struct Leak { uint64_t ptr, bytes; AllocKind kind; uint32_t stack_id; bool at_ctx_teardown; };
struct InvalidFree { uint64_t ptr; uint32_t stack_id; };
struct Snapshot {
  int pid = 0; std::string mode;
  std::vector<Record> records; std::vector<GpuSpan> spans;
  std::map<uint64_t, std::string> kernel_names;            // func handle -> demangled name
  std::array<uint64_t, 4> peak_bytes{}; uint64_t peak_device_total = 0;  // Device+Managed+Async
  uint64_t alloc_count = 0, free_count = 0;
  std::vector<Leak> leaks; std::vector<InvalidFree> invalid_frees;
  std::map<uint32_t, std::vector<StackFrame>> stacks;
  uint64_t dropped_records = 0, unresolved_events = 0, gpu_untimed = 0;
};
```

```c
/* include/cuscope/cuscope.h  (Task 2) */
typedef struct cuscope_stats {
  uint64_t dlsym_wrapped;    /* wrappers handed out by the dlsym override */
  uint64_t gpa_wrapped;      /* wrappers handed out by the cuGetProcAddress hook */
  uint64_t gpa_passthrough;  /* hooked base names returned unwrapped (variant mismatch) */
  uint64_t direct_calls;     /* calls entering through exported driver symbols (path 1) */
  uint64_t hooked_calls;     /* wrapper invocations not marked internal */
  uint64_t live_bytes, live_allocs, invalid_frees;
} cuscope_stats;
int cuscope_get_stats(cuscope_stats* out);   /* returns 0 */
```

**`cuscope` JSON section** (Task 7 writes it; the harness and CLI read it):
```json
"cuscope": {"version": "0.1.0", "pid": 0, "mode": "full",
  "peak_bytes": {"device": 0, "managed": 0, "host_pinned": 0, "async": 0, "device_total": 0},
  "alloc_count": 0, "free_count": 0,
  "leaks": [{"ptr": "0x0", "bytes": 0, "kind": "device", "stack": 0, "at_ctx_teardown": false}],
  "invalid_frees": [{"ptr": "0x0", "stack": 0}],
  "stacks": {"1": [{"module": "/abs/path", "offset": 0, "symbol": "main"}]},
  "kernels": [{"name": "k", "count": 0, "gpu_ns_total": 0, "gpu_ns_max": 0}],
  "copies": [{"dir": "HtoD", "pinned": true, "count": 0, "bytes": 0, "gpu_ns_total": 0, "gbps": 0.0}],
  "graph_launches": 0, "dropped_records": 0, "unresolved_events": 0, "gpu_untimed": 0}
```
`kernels` is sorted by `gpu_ns_total` descending. `gbps = bytes / gpu_ns_total` (bytes per ns equals GB/s).

---

### Task 1: Toolchain, repo skeleton, banner (Day 1)

**Files:** Create `docs/SETUP.md`, `CMakeLists.txt`, `.gitignore`, `src/log.{hpp,cpp}`, `src/config.{hpp,cpp}`, `src/runtime.cpp`, `tests/unit/CMakeLists.txt`, `tests/unit/test_config.cpp`, `tests/cuda/CMakeLists.txt`, `tests/cuda/hello.cu`.

**Interfaces:**
- Produces: `struct Config { std::string out, log_path, census_path; enum class Mode { Alloc, Full } mode; int stack_depth; uint32_t sample_every; std::vector<std::string> warnings; };`
  `Config parse_config(const std::function<const char*(const char*)>& getenv_fn);`
  `const Config& config();` (function-local static, parsed once)
  `std::string expand_out_path(const std::string& pattern, int pid);`
  `void log_msg(const char* fmt, ...)` (printf-style, prefix `[cuscope] `), and `CS_LOG_ONCE(fmt, ...)` (a per-call-site `static std::atomic<bool>`).
  CMake target `cuscope` → `build/libcuscope.so`. `CUSCOPE_VERSION` comes from `project(cuscope VERSION 0.1.0)`.

- [ ] **Step 1 [Claude]: Write `docs/SETUP.md`** with these exact commands:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0          # never `cuda` or `cuda-drivers`
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
python3 -m pip install --user "cmake>=3.28,<4" ninja pytest
```
- [ ] **Step 2 [You]: Run the setup.** In a new shell, expected output: `nvcc --version` shows `release 13.0`; `cmake --version` ≥ 3.28; `ninja --version` prints a version; `python3 -m pytest --version` prints a version; `ls /usr/local/cuda-13.0/include/cuda.h` exists.
- [ ] **Step 3 [Claude]: Write failing `tests/unit/test_config.cpp`**
```cpp
TEST(Config, Defaults) { auto c = parse_config(env({}));
  EXPECT_EQ(c.out, "cuscope-%p.json"); EXPECT_EQ(c.mode, Config::Mode::Full);
  EXPECT_EQ(c.stack_depth, 16); EXPECT_EQ(c.sample_every, 1u); EXPECT_TRUE(c.warnings.empty()); }
TEST(Config, ReadsAll) { auto c = parse_config(env({{"CUSCOPE_MODE","alloc"},{"CUSCOPE_STACK_DEPTH","8"},
  {"CUSCOPE_SAMPLE_EVERY","10"},{"CUSCOPE_OUT","/tmp/t.json"},{"CUSCOPE_CENSUS","/tmp/c.tsv"}}));
  EXPECT_EQ(c.mode, Config::Mode::Alloc); EXPECT_EQ(c.stack_depth, 8); EXPECT_EQ(c.sample_every, 10u);
  EXPECT_EQ(c.out, "/tmp/t.json"); EXPECT_EQ(c.census_path, "/tmp/c.tsv"); }
TEST(Config, InvalidValuesFallBack) {   // Review Focus 5
  auto c = parse_config(env({{"CUSCOPE_MODE","bogus"},{"CUSCOPE_STACK_DEPTH","abc"},{"CUSCOPE_SAMPLE_EVERY","0"}}));
  EXPECT_EQ(c.mode, Config::Mode::Full); EXPECT_EQ(c.stack_depth, 16); EXPECT_EQ(c.sample_every, 1u);
  EXPECT_EQ(c.warnings.size(), 3u);
  EXPECT_EQ(parse_config(env({{"CUSCOPE_STACK_DEPTH","1000"}})).stack_depth, 64); }
TEST(Config, ExpandsPid) { EXPECT_EQ(expand_out_path("a-%p.json", 42), "a-42.json");
  EXPECT_EQ(expand_out_path("plain.json", 42), "plain.json"); }
```
(`env(...)` is a test helper that returns a lambda over a `std::map`.)
- [ ] **Step 4 [Claude]: Run.** `cmake --build build && ctest --test-dir build -L unit --output-on-failure`. Expected: build fails, because `config.hpp` doesn't exist yet.
- [ ] **Step 5 [Claude]: Write the root `CMakeLists.txt`, `log.*`, `config.*` and `tests/cuda/hello.cu`.** The root CMake does `find_path(CUDA_DRIVER_INCLUDE cuda.h HINTS /usr/local/cuda-13.0/include $ENV{CUDA_HOME}/include REQUIRED)`; builds the `cuscope` SHARED target with `-fvisibility=hidden`, linked to `dl` and `Threads::Threads`; adds GoogleTest via FetchContent; turns `CUSCOPE_GPU_TESTS` ON when `CMAKE_CUDA_COMPILER` is found; and adds ctest labels `unit`, `mock`, `gpu`. `hello.cu` (vector add of 1M floats: cudaMalloc, H→D, kernel, D→H, free) prints `hello: sum=<n>` and is built twice, as `hello_static` (default) and `hello_shared` (`CUDA_RUNTIME_LIBRARY Shared`).
- [ ] **Step 6 [Claude]: Run the unit tests.** Expected: 4 tests PASS.
- [ ] **Step 7 [You]: Write `src/runtime.cpp`.** Add `__attribute__((constructor)) static void cuscope_ctor()`, which logs `cuscope 0.1.0 loaded (pid <pid>, mode <full|alloc>)` once through `log_msg` and logs each `config().warnings` entry.
- [ ] **Step 8 [You]: Verify.** `LD_PRELOAD=build/libcuscope.so build/tests/cuda/hello_static` prints the banner on stderr and `hello: sum=...` on stdout, and `diff <(build/tests/cuda/hello_static) <(LD_PRELOAD=build/libcuscope.so build/tests/cuda/hello_static 2>/dev/null)` prints nothing.
- [ ] **Step 9: Commit.** `git add -A && git commit -m "build: skeleton, config, logging, preload banner"`

### Task 2: Mock driver, mock shim, runtime mimic (Day 2) [Claude]

**Files:** Create `include/cuscope/cuscope.h`, `src/stats.{hpp,cpp}`, `tests/mock/{CMakeLists.txt,mock_driver.cpp,mock_shim.cpp,mock_counters.h,runtime_mimic.cpp,direct_mimic.cpp,conftest.py,test_paths.py}`.

**Interfaces:**
- Produces: `cuscope_get_stats` (exported). In `stats.hpp`, `namespace cuscope::stats { inline std::atomic<uint64_t> dlsym_wrapped, gpa_wrapped, ...; }` (zero-initialized atomics).
- Mock build layout: `build/mock/driver/libcuda.so.1.1` (the mock driver), `build/mock/direct/libcuda.so.1` (the same mock with SONAME `libcuda.so.1`), `build/mock/shim/libcuda.so.1` (the shim). The shim `dlopen`s the absolute driver path baked in with `-DMOCK_DRIVER_PATH`, `dlsym`s every function **in its constructor** (as WSL does), and forwards each export, including `cuGetProcAddress(_v2)` and `mock_get_counters`.
- Mock driver exports (signatures from `cuda.h`): `cuInit, cuDriverGetVersion, cuGetProcAddress, cuGetProcAddress_v2, cuCtxGetCurrent, cuCtxSetCurrent, cuCtxGetDevice, cuCtxDestroy, cuDevicePrimaryCtxRetain, cuDevicePrimaryCtxRelease, cuDevicePrimaryCtxGetState, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, cuMemcpyAsync(+_ptsz), cuLaunchKernel(+_ptsz), cuStreamIsCapturing(+_ptsz), cuEventCreate, cuEventRecord(+_ptsz), cuEventQuery, cuEventSynchronize, cuEventElapsedTime, cuFuncGetName, cuPointerGetAttribute`, plus `extern "C" void mock_get_counters(mock_counters*)`.
- Mock semantics: memory is `malloc`-backed. `cuGetProcAddress_v2` maps (base, version, flags) to an internal function: flag `PER_THREAD_DEFAULT_STREAM` → the `_ptsz` function where one exists; `cuMemAlloc` with version < 3020 → a separate v1 function (`unsigned int` size). Stream `(CUstream)0xCA97` reports `CU_STREAM_CAPTURE_STATUS_ACTIVE`. `cuFuncGetName` returns `"_Z11mock_kernelv"`. Event queries return `CUDA_ERROR_NOT_READY` for the first `MOCK_EVENT_DELAY_QUERIES` (env, default 0) queries per event, and `cuEventSynchronize` makes an event ready. After `cuCtxDestroy` every event of that context returns `CUDA_ERROR_INVALID_CONTEXT`.
- `mock_counters`: `alloc_v1, alloc_v2, free_, memcpy_async, memcpy_async_ptsz, launch, launch_ptsz, event_record, event_record_on_capturing, ctx_destroy` (all `uint64_t`).
- `runtime_mimic <scenario>` (not linked to libcuda) behaves like cudart: `dlopen("libcuda.so.1")`, `dlsym(h, "cuGetProcAddress_v2")`, then lookups. Scenarios: `dlsym_gpa` (alloc 4096 + free via gpa at 13000/LEGACY), `dlsym_default` (`dlsym(RTLD_DEFAULT, "cuGetProcAddress_v2")` **before** any `dlopen`, then the same as `dlsym_gpa`), `ptsz` (gpa with PER_THREAD flags for `cuMemcpyAsync` and `cuLaunchKernel`, then call them), `old_version` (gpa `cuMemAlloc` at version 3000, call it), `gpa_self` (gpa `"cuGetProcAddress"` at 13000, then use the result to fetch and call `cuMemAlloc`), `capture` (launch on stream `0xCA97`), `launch` (one launch on stream 0, then sync), `copy` (4096-byte HtoD), `teardown` (alloc 1 MiB without freeing, launch, `cuCtxDestroy`), `fork` (alloc 1 MiB without freeing, `fork()`; the child makes no CUDA calls and returns `exit(0)` normally so its exit handler runs; the parent `waitpid`s and exits normally). On success it prints `OK <scenario>`, then `STATS <json>` if `dlsym(RTLD_DEFAULT,"cuscope_get_stats")` is non-null, then `MOCK <json>`. Exit code 0 means every CUDA call returned `CUDA_SUCCESS`.
- `direct_mimic` links `-lcuda` against `build/mock/direct`, calls `cuInit`, `cuMemAlloc` and `cuMemFree` by name, then prints the same lines.
- `conftest.py` fixture `mimic(scenario, mode="direct"|"shim", preload=True, env=None) -> Result(code, stdout, stats: dict|None, mock: dict, census: list[list[str]], trace: dict|None)`. It sets `LD_LIBRARY_PATH=build/mock/<mode>`, `CUSCOPE_OUT=<tmp>/t-%p.json`, `CUSCOPE_CENSUS=<tmp>/census.tsv`, and a 10 s timeout (a timeout fails the test). A second fixture, `direct_mimic(preload=True, env=None) -> Result`, runs `direct_mimic` with `LD_LIBRARY_PATH=build/mock/direct`. Both parse `trace` from the newest `t-<child pid>.json`, and `traces` lists every trace file written in the run.

- [ ] **Step 1: Write `tests/mock/test_paths.py::test_baseline_without_cuscope`.** It is parametrized over every scenario × {direct, shim} with `preload=False` and asserts `code == 0` and `stdout.startswith("OK ")`.
- [ ] **Step 2: Run** `python3 -m pytest tests/mock -v`. Expected: FAIL (mock not built).
- [ ] **Step 3: Write the mock driver, mock shim, mimics, `stats.*` and `cuscope.h` as specified above; register `ctest` label `mock` running `python3 -m pytest tests/mock`.**
- [ ] **Step 4: Run** `cmake --build build && python3 -m pytest tests/mock -v`. Expected: all baseline cases PASS.
- [ ] **Step 5: Commit.** `git commit -am "test: mock libcuda (direct and WSL-shim modes) and runtime mimic"`

### Task 3: Census interception: `dlsym` override and logging `cuGetProcAddress` (Day 2) [You]

**Files:** Create `src/loader.{hpp,cpp}` (stage 1 only), `src/intercept.cpp`, `src/census.{hpp,cpp}`. Modify `tests/mock/test_paths.py`.

**Interfaces:**
- Consumes: `config().census_path`, `stats::*`, `log_msg`.
- Produces (loader.hpp):
  `using DlsymFn = void* (*)(void*, const char*);`
  `DlsymFn real_dlsym();` uses `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34")`, then `"GLIBC_2.2.5"`, behind `pthread_once`. It never calls `dlsym`.
  `void* libcuda_handle();` runs `dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD)`, then `RTLD_LAZY`, behind its own `pthread_once`; null on failure.
  `enum class GpaVer { V1, V2 }; void set_real_gpa(GpaVer, void*); void* real_gpa(GpaVer);` (atomics).
- Produces (census.hpp, written by **[Claude]** in Step 1): `void census_dlsym(const char* sym, const char* caller_module, const char* action);` and `void census_gpa(const char* sym, int version, uint64_t flags, int result, int status, const char* action);`. These are TSV lines `dlsym\t<sym>\t<caller basename>\t<action>` and `gpa\t<sym>\t<version>\t<flags>\t<result>\t<status>\t<action>`. They are no-ops when `census_path` is empty, and the file is opened once with line buffering.
- Exported behaviour in this task:
  `dlsym(h, sym)`: if the caller (from `__builtin_return_address(0)` + `dladdr`) has a basename starting with `libcuda`, return `real_dlsym()(h, sym)` and log `caller-libcuda`. If `sym` is `cuGetProcAddress` or `cuGetProcAddress_v2`: for `RTLD_DEFAULT`/`RTLD_NEXT`, resolve the real one from `libcuda_handle()`; otherwise use `h`. If the result is null or **equals cuscope's own wrapper address**, return the real lookup unchanged. Otherwise `set_real_gpa`, `stats::dlsym_wrapped++`, log `wrap`, and return the matching wrapper. All other names go to `real_dlsym()(h, sym)`.
  `cuGetProcAddress_v2` / `cuGetProcAddress` wrappers: call the real one, `census_gpa(..., "pass")`, and return the result unchanged, except that a successful lookup of `"cuGetProcAddress"` returns cuscope's v2 wrapper if `version >= 12000` and the v1 wrapper otherwise.

- [ ] **Step 1 [Claude]: Write `census.*` and the failing tests in `tests/mock/test_paths.py`.**
```python
@pytest.mark.parametrize("mode", ["direct", "shim"])
def test_dlsym_returns_gpa_wrapper(mimic, mode):
    r = mimic("dlsym_gpa", mode); assert r.code == 0 and r.stats["dlsym_wrapped"] >= 1
def test_shim_lookups_pass_through(mimic):
    r = mimic("dlsym_gpa", "shim"); assert r.code == 0
    assert ["dlsym", "cuGetProcAddress_v2", "libcuda.so.1", "caller-libcuda"] in r.census
@pytest.mark.parametrize("mode", ["direct", "shim"])
def test_dlsym_default_during_init(mimic, mode):          # Review Focus 1
    r = mimic("dlsym_default", mode); assert r.code == 0 and r.stdout.startswith("OK dlsym_default")
def test_census_records_gpa(mimic):
    r = mimic("dlsym_gpa", "direct")
    assert any(l[:2] == ["gpa", "cuMemAlloc"] and l[2] == "13000" for l in r.census)
```
- [ ] **Step 2 [Claude]: Run** `python3 -m pytest tests/mock -v -k "dlsym or census or shim"`. Expected: the new tests FAIL (`stats` is None or census lines are missing).
- [ ] **Step 3 [You]: Implement `loader.cpp` stage 1 and `intercept.cpp`** as specified. The caller check is the key line: decide from the `dladdr` basename, never from the handle.
- [ ] **Step 4 [Claude]: Review, then run** `python3 -m pytest tests/mock -v`. Expected: all PASS, with no timeouts.
- [ ] **Step 5: Commit.** `git commit -am "feat: dlsym override with libcuda caller check, census logging"`

### Task 4: Census on real programs, Day 2 gate (Day 2) [Claude runs, You decide]

**Files:** Create `tools/census.py`, `docs/census/{static,shared,pytorch}.tsv`, `docs/census/SUMMARY.md`.

**Interfaces:**
- Produces: `python3 tools/census.py run` runs `hello_static`, `hello_shared` and `python3 -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); print(y.sum().item()>0)"` under `LD_PRELOAD=build/libcuscope.so` with `CUSCOPE_CENSUS` set, and writes the three TSVs. `python3 tools/census.py check` prints `PASS`/`FAIL` per gate line and exits non-zero on any FAIL.
- Gate lines: (1) all three programs exit 0 with stdout identical to a run without cuscope; (2) `static.tsv` has a `gpa` line for `cuMemAlloc`; (3) `static.tsv` and `shared.tsv` each have a `dlsym cuGetProcAddress_v2 ... wrap` line; (4) the WSL shim's own lookups appear as `caller-libcuda`.

- [ ] **Step 1 [Claude]: Write `tools/census.py`, then run** `python3 tools/census.py run && python3 tools/census.py check`. Expected: 4 PASS lines.
- [ ] **Step 2 [Claude]: Write `docs/census/SUMMARY.md`.** It covers every distinct gpa symbol per program with version and flags, which Phase 1 hook-list names were seen or not seen, any requested names that aren't in the list but matter (allocation, copy or launch families), and whether `_ptsz`/`_ptds` variants were requested.
- [ ] **Step 3 [You]: Gate decision.** If the check passes, update the hook list in spec §5 with the SUMMARY delta and continue. If it fails, **stop**: Claude writes a fallback-B (proxy library) addendum to this plan before Task 6.
- [ ] **Step 4: Commit.** `git add tools docs/census && git commit -m "docs: Day 2 symbol census (static, shared, PyTorch)"`

### Task 5: GPU test programs and passthrough harness (Day 3) [Claude]

**Files:** Create `tests/cuda/{leak.cu,double_free.cu,async_pool.cu,multistream.cu,multithread.cpp,bandwidth.cu,graph_capture.cu}`, `tests/harness.py`, `tests/gpu/{conftest.py,test_passthrough.py}`.

**Interfaces:**
- Produces (`tests/harness.py`): `BUILD = Path(os.environ.get("CUSCOPE_BUILD", "build"))`; `run(prog: str, *args, cuscope=True, mode="full", out: Path|None=None, env: dict|None=None) -> Run` with `Run(stdout, stderr, code, trace: dict|None)`, where `trace["cuscope"]` is the report section; `assert_passthrough(prog, *args)`; `normalize(report) -> dict` (keeps leak `(bytes, kind)` multiset, invalid-free count, peaks, alloc/free counts, kernel `(name, count)`, copy `(dir, pinned, count, bytes)`, graph_launches; drops pointers, times and stack ids); `marker_line(src: str, marker: str) -> int`.
- Programs. Each prints one deterministic line and exits 0:
  - `leak.cu`: `cudaMalloc` 1 MiB `// LEAK_A`, 2 MiB `// FREED`, 3 MiB `// LEAK_C`; frees the 2 MiB one.
  - `double_free.cu`: 4096 bytes freed twice, plus `cudaFree(nullptr)`. Prints the second free's error name.
  - `async_pool.cu`: two streams; on call order, A=8 MiB on s1, B=4 MiB on s2, free A on s1, C=16 MiB on s2, free B, free C.
  - `multistream.cu`: `__global__ void spin_kernel(long long cycles)` busy-waits about 5 ms, launched once on each of 4 streams.
  - `multithread.cpp`: 8 threads; thread t does 1000 × (`cudaMalloc((t+1)*4096)`, `cudaFree`), then leaks one `(t+1)*1024`-byte allocation.
  - `bandwidth.cu`: 64 MiB HtoD ×5 and DtoH ×5 from pageable memory, then the same from `cudaMallocHost` memory.
  - `graph_capture.cu`: captures 3 `spin_kernel` launches (1000 cycles) on a stream into a graph, instantiates it, launches it twice, syncs.
- GPU tests are skipped unless `/usr/lib/wsl/lib/libcuda.so.1` exists. ctest label `gpu` runs `python3 -m pytest tests/gpu`.

- [ ] **Step 1: Write `tests/gpu/test_passthrough.py`.** `test_passthrough[prog]` is parametrized over `hello_static, hello_shared, leak, double_free, async_pool, multistream, multithread, bandwidth, graph_capture` and calls `assert_passthrough(prog)`.
- [ ] **Step 2: Run** `python3 -m pytest tests/gpu -v`. Expected: FAIL (programs not built).
- [ ] **Step 3: Write the programs, CMake targets and `harness.py`.**
- [ ] **Step 4: Run** `cmake --build build && python3 -m pytest tests/gpu -v`. Expected: 9 PASS, with the Task 3 interception active.
- [ ] **Step 5: Commit.** `git commit -am "test: GPU known-answer programs and passthrough harness"`

### Task 6: Loader, hook table, pass-through wrappers (Day 3) [You]

**Files:** Modify `src/loader.{hpp,cpp}` and `src/intercept.cpp`. Create `src/hooks.def`, `src/hook_table.{hpp,cpp}`, `src/guard.hpp`, `src/wrappers/{wrappers.hpp,alloc.cpp,memcpy.cpp,launch.cpp,teardown.cpp}`, `tools/check_exports.py`.

**Interfaces:**
- Consumes: Task 3 loader stage 1, `real_gpa`, `stats`.
- Produces:
  - `src/hooks.def`, one line per hook: `HOOK(base, "legacy_export", "ptds_export_or_nullptr", version)`, for example `HOOK(cuMemAlloc, "cuMemAlloc_v2", nullptr, CUDA_VERSION)`, `HOOK(cuMemcpyAsync, "cuMemcpyAsync", "cuMemcpyAsync_ptsz", CUDA_VERSION)` and `HOOK(cuCtxSynchronize, "cuCtxSynchronize", nullptr, 2000)`. `version` is the cudaVersion whose returned function has the signature `cuda.h` declares for `base` (amended after the Day 2 census; see `docs/census/SUMMARY.md` finding 5). The list is spec §5. Export names come from `nm -D --defined-only /usr/lib/wsl/lib/libcuda.so.1` and the `#define` in `cuda.h`.
  - `hook_table.hpp`: `enum HookId : uint16_t { H_<base>..., H_COUNT };` and `struct HookEntry { const char* base; const char* export_name[2]; void* wrapper[2]; std::atomic<void*> real[2]; };`. Functions: `HookEntry& hook(HookId)`, `HookEntry* find_hook_by_base(const char*)`, `HookEntry* find_hook_by_export(const char* sym, int* variant)`, `const char* hook_name(uint16_t)`, `void hook_table_init()` (`pthread_once`; calls `register_alloc_hooks()`, `register_memcpy_hooks()`, `register_launch_hooks()`, `register_teardown_hooks()`, `register_name_hooks()`; hooks left without wrappers are logged once and stay pass-through), and `template <class Fn> Fn real_fn(HookId id, int v)` (calls `ensure_loader()`, then loads `real[v]` with acquire).
  - In each wrapper .cpp, registration uses:
```cpp
#define CS_REGISTER(b) do { \
  static_assert(std::is_same_v<decltype(&cs_##b<0>), decltype(&::b)>, #b ": wrapper signature != cuda.h"); \
  set_wrappers(H_##b, reinterpret_cast<void*>(&cs_##b<0>), reinterpret_cast<void*>(&cs_##b<1>)); } while (0)
```
  - Wrapper shape: `template <int V> CUresult cs_<base>(<cuda.h params>)` with `V` = 0 (legacy) or 1 (per-thread). It forwards to `real_fn<decltype(&::<base>)>(H_<base>, V)` and does `stats::hooked_calls++` unless `in_internal()`. Every exported name in `hooks.def` gets an `extern "C"` definition that does `stats::direct_calls++` and calls `cs_<base><0>` (or `<1>` for the ptds export).
  - `loader.hpp` additions: `void ensure_loader();` (`pthread_once`, inside an `InternalScope`: `hook_table_init()`; for every hook, `real[0] = gpa(base, version, CU_GET_PROC_ADDRESS_LEGACY_STREAM)` and `real[1] = gpa(base, version, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)`, with `version` taken from `hooks.def`; a failed lookup falls back to `real_dlsym()(libcuda_handle(), export_name[v])`) and `bool loader_ready();`.
  - `guard.hpp`: `struct InternalScope { InternalScope(); ~InternalScope(); };` and `bool in_internal();` over a `thread_local int` depth.
  - The gpa wrapper becomes: real call → census → if it succeeded and the hook exists and `loader_ready()`: when `*pfn == real[v]`, set `*pfn = wrapper[v]` and `gpa_wrapped++`; otherwise `gpa_passthrough++` (the pointer is left untouched). The `dlsym` override also returns `wrapper[v]` for any `find_hook_by_export` match.

- [ ] **Step 1 [Claude]: Write the failing tests**, in `tests/mock/test_paths.py` plus `tools/check_exports.py` (registered as ctest label `gpu`):
```python
@pytest.mark.parametrize("mode", ["direct", "shim"])
def test_gpa_wraps_matching_variant(mimic, mode):
    r = mimic("dlsym_gpa", mode); assert r.stats["gpa_wrapped"] >= 2 and r.stats["hooked_calls"] >= 2
    assert r.mock["alloc_v2"] == 1
def test_ptsz_forwards_to_ptsz(mimic):
    r = mimic("ptsz", "shim"); assert r.mock["memcpy_async_ptsz"] == 1 and r.mock["launch_ptsz"] == 1
    assert r.mock["memcpy_async"] == 0 and r.mock["launch"] == 0
def test_old_version_passthrough(mimic):
    r = mimic("old_version", "shim"); assert r.stats["gpa_passthrough"] >= 1 and r.mock["alloc_v1"] == 1
    assert r.stats["hooked_calls"] == 0
def test_gpa_self_lookup(mimic):
    r = mimic("gpa_self", "shim"); assert r.code == 0 and r.stats["hooked_calls"] >= 1
def test_direct_exports(direct_mimic):
    r = direct_mimic(); assert r.code == 0 and r.stats["direct_calls"] >= 2
def test_real_gpa_is_never_own_wrapper(mimic):        # Review Focus 1
    r = mimic("dlsym_default", "direct"); assert r.code == 0 and r.stats["hooked_calls"] >= 2
```
`tools/check_exports.py` checks two things: (a) every export name in `hooks.def` appears in `nm -D --defined-only /usr/lib/wsl/lib/libcuda.so.1`; (b) **signature safety**. It loads the real driver (`/usr/lib/wsl/drivers/*/libcuda.so.1.1`) directly with ctypes, and for every hook and both variants asserts that `cuGetProcAddress_v2(base, version, flags)` returns the same address as `dlsym(driver, export_name[v])`. That proves the pointer each wrapper matches is the function whose prototype the wrapper was written against. A negative control asserts that `cuCtxSynchronize` at 13000 does **not** equal `dlsym(driver, "cuCtxSynchronize")`.
- [ ] **Step 2 [Claude]: Run** `python3 -m pytest tests/mock -v`. Expected: the new tests FAIL.
- [ ] **Step 3 [You]: Implement** guard, hooks.def, hook table, loader stage 2, pass-through wrappers for every hook, the gpa variant matching, and the direct exports.
- [ ] **Step 4 [Claude]: Review, then run** `cmake --build build && ctest --test-dir build --output-on-failure`. Expected: unit, mock and gpu all PASS. The gpu label covers the 9 passthrough tests and `check_exports`.
- [ ] **Step 5: Commit.** `git commit -am "feat: loader, hook table, variant-matched wrappers (pass-through)"`

### Task 7: Data model, summary, trace writer (Day 3–4) [Claude]

**Files:** Create `src/model.hpp`, `src/summary.{hpp,cpp}`, `src/trace_writer.{hpp,cpp}`, `src/hook_names.cpp` (names from `hooks.def` with no `cuda.h` dependency), `tests/unit/{test_summary.cpp,test_trace_writer.cpp}`.

**Interfaces:**
- Produces: the model types in "Shared Interfaces".
  `struct KernelStat { std::string name; uint64_t count, gpu_ns_total, gpu_ns_max; };`
  `struct CopyStat { CopyDir dir; bool pinned; uint64_t count, bytes, gpu_ns_total; double gbps; };`
  `struct Summary { std::vector<KernelStat> kernels; std::vector<CopyStat> copies; uint64_t graph_launches; };`
  `Summary summarize(const Snapshot&);`
  `void write_trace(const Snapshot&, std::ostream&);`
  `bool write_trace_file(const Snapshot&, const std::string& path);` (false and one log line on failure).
- Trace mapping: every record → `ph:"X"`, `name = hook_name(api)`, `ts = cpu_ns/1000.0`, `dur = dur_ns/1000.0`, `pid`, `tid`. Records are sorted by `cpu_ns`, and each Alloc/Free emits `ph:"C"`, `name:"device memory"`, `args:{"bytes": running Device+Managed+Async total}`. Each GPU span → `ph:"X"` with `tid = 1000000 + stream index` and one `ph:"M"` `thread_name` event, `"GPU stream 0x<hex>"`. The top-level object is `{"traceEvents": [...], "displayTimeUnit": "ns", "cuscope": {...}}`. Kernel name lookup is `kernel_names[func]`, falling back to `"0x<func hex>"`.

- [ ] **Step 1: Write the failing tests.** They use nlohmann/json to parse writer output:
  - `TraceWriter.ApiCallBecomesCompleteEvent`: one ApiCall (cpu 1000 ns, dur 500, `api=H_cuMemAlloc`, tid 7) → event `{ph:X, name:"cuMemAlloc", ts:1.0, dur:0.5, tid:7}`.
  - `TraceWriter.MemoryCounterRisesAndFalls`: Alloc 1 MiB Device, then Free → counter values `[1048576, 0]`.
  - `TraceWriter.GpuSpanOnStreamTrack`: span on stream `0xabc` → the event's `tid ≥ 1000000`, plus a metadata event named `"GPU stream 0xabc"`.
  - `TraceWriter.EscapesNames`: a kernel name containing `"`, `\` and `\n` still parses.
  - `TraceWriter.ReportSectionMatchesSchema`: every key of the `cuscope` section in Shared Interfaces is present.
  - `TraceWriter.UnwritablePathReturnsFalse`: `/nonexistent/x.json` → `false`.
  - `Summary.KernelsSortedByTotal`.
  - `Summary.CopyBandwidth`: 1e9 bytes over 1e8 ns → `gbps == 10.0`.
- [ ] **Step 2: Run** `ctest --test-dir build -L unit`. Expected: FAIL.
- [ ] **Step 3: Implement.** JSON is hand-written with one `json_escape(std::string_view)` helper and no dependency in `libcuscope`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat: data model, summary aggregation, Chrome Trace writer"`

### Task 8: Allocation tracker, stack capture, allocation wrappers, exit report (Day 4) [You]

**Files:** Create `src/tracker.{hpp,cpp}`, `src/stacks.{hpp,cpp}`, `tests/unit/{test_tracker.cpp,test_stacks.cpp}`, `tests/gpu/test_memory.py`. Modify `src/wrappers/alloc.cpp`, `src/wrappers/teardown.cpp` (exit path only) and `src/runtime.cpp`.

**Interfaces:**
- Consumes: model types (Task 7), `write_trace_file`, `config()`.
- Produces:
```cpp
struct AllocInfo { uint64_t ptr, bytes; AllocKind kind; uint32_t stack_id; uint64_t ctx; };
class Tracker {  // 16 shards, std::mutex each, shard = hash(ptr) & 15
 public:
  void on_alloc(const AllocInfo&);
  std::optional<AllocInfo> on_free(uint64_t ptr);   // removes; nullopt if unknown
  void restore(const AllocInfo&);                    // re-insert after a failed real free
  void on_invalid_free(uint64_t ptr, uint32_t stack_id);
  void on_ctx_teardown(uint64_t ctx);                // live allocs of ctx -> leaks(at_ctx_teardown=true)
  void finalize();                                   // remaining live -> leaks
  void fill(Snapshot&) const;                        // peaks, counts, leaks, invalid frees
  uint64_t live_bytes() const; uint64_t live_allocs() const;
};
Tracker& tracker();
uint32_t capture_stack(int skip);   // backtrace() to config().stack_depth, FNV-1a over frames, dedup; 0 = none
std::map<uint32_t, std::vector<StackFrame>> symbolize_all();  // dladdr: module, offset from dli_fbase, demangled symbol
std::string demangle(const char* mangled);   // __cxa_demangle, returns input on failure
void warm_up_backtrace();
```
- Wrapper rules: `on_alloc` only on `CUDA_SUCCESS` (failure → nothing tracked). Free = `on_free` **before** the real call; if the real call fails and an entry existed, `restore`. When `ptr != 0` and there's no entry, `on_invalid_free(ptr, capture_stack(1))`. `ptr == 0` is ignored. ctx comes from `cuCtxGetCurrent` inside an `InternalScope`. Kinds: `cuMemAlloc`/`cuMemAllocPitch` (bytes = pitch × height) → Device; `cuMemAllocManaged` → Managed; `cuMemAllocHost`/`cuMemHostAlloc` → HostPinned; `cuMemAllocAsync`/`cuMemAllocFromPoolAsync` → Async. `stats::live_*` and `invalid_frees` mirror the tracker.
- `runtime.cpp`: `atexit` handler registered in the constructor builds the `Snapshot` (`tracker().finalize(); tracker().fill(s); s.stacks = symbolize_all();`), then calls `write_trace_file(s, expand_out_path(config().out, getpid()))`.

- [ ] **Step 1 [Claude]: Write the failing tests.**
  - `Tracker.AllocFreeUpdatesLiveAndPeak`: 100 + 50 Device, free the first → live 50, peak device 150.
  - `Tracker.PeakPerKind`: 10/20/30/40 across Device/Managed/HostPinned/Async → each peak equals its value; `peak_device_total == 70`.
  - `Tracker.FreeUnknownReturnsNullopt`.
  - `Tracker.RestoreAfterFailedFree` (Review Focus 3): on_free, then restore → live 1, the entry keeps its stack_id.
  - `Tracker.AddressReuseAfterFree` (Review Focus 3).
  - `Tracker.CtxTeardownMovesToLeaks`: ctx A and B; teardown A → only A's leaks, flagged; finalize adds B's.
  - `Tracker.ConcurrentAllocFree`: 8 threads × 10000 unique pointers → live 0, `alloc_count == free_count == 80000`.
  - `Stacks.SameSiteSameId`, `Stacks.DifferentSitesDifferentIds`, `Stacks.DepthRespected` (depth 4 → ≤ 4 frames), `Stacks.SymbolizeHasModuleAndSymbol` (test built with `-rdynamic`).
  - `tests/gpu/test_memory.py`:
```python
def test_leak():
    r = run("leak").trace["cuscope"]; assert sorted(l["bytes"] for l in r["leaks"]) == [1 << 20, 3 << 20]
    assert all(l["kind"] == "device" for l in r["leaks"])
    assert all(any(f["module"].endswith("/leak") and "main" in f["symbol"] for f in r["stacks"][str(l["stack"])]) for l in r["leaks"])
def test_double_free():
    r = run("double_free"); assert len(r.trace["cuscope"]["invalid_frees"]) == 1; assert_passthrough("double_free")
def test_static_vs_shared():
    assert normalize(run("hello_static").trace["cuscope"]) == normalize(run("hello_shared").trace["cuscope"])
def test_multithread_exact_counts():
    r = run("multithread").trace["cuscope"]; assert r["alloc_count"] == 8 * 1001 and r["free_count"] == 8000
    assert sorted(l["bytes"] for l in r["leaks"]) == [k * 1024 for k in range(1, 9)] and r["invalid_frees"] == []
def test_unwritable_out_keeps_program_behaviour():      # Review Focus 5
    a, b = run("leak", cuscope=False), run("leak", out=Path("/nonexistent/dir/t.json"))
    assert (a.stdout, a.code) == (b.stdout, b.code) and "[cuscope]" in b.stderr
```
- [ ] **Step 2 [Claude]: Run** `ctest --test-dir build -L unit && python3 -m pytest tests/gpu/test_memory.py -v`. Expected: FAIL.
- [ ] **Step 3 [You]: Implement** the tracker, stacks, allocation and free wrappers, and the exit handler. Call `warm_up_backtrace()` from the constructor.
- [ ] **Step 4 [Claude]: Review, then run** the full `ctest --test-dir build --output-on-failure`. Expected: all PASS. If `static_vs_shared` differs, stop and investigate: that difference is a finding, not noise.
- [ ] **Step 5: Commit.** `git commit -am "feat: allocation tracker, stack capture, leak and invalid-free report"`

### Task 9: Ring buffers, collector thread, API spans, fork safety (Day 5) [You]

**Files:** Create `src/ring.hpp`, `src/recorder.{hpp,cpp}`, `tests/unit/{test_ring.cpp,test_recorder.cpp}`, `tests/gpu/test_trace.py`. Modify every wrapper (push a Record) and `src/runtime.cpp`.

**Interfaces:**
- Produces:
```cpp
template <class T, size_t N> class SpscRing {   // N power of two; one writer, one reader
 public: bool push(const T&); bool pop(T&); uint64_t drops() const; };   // push false => drop counted
void record(Record r);           // fills tid and pushes to this thread's ring (16384 slots); starts collector lazily
uint64_t next_corr();            // atomic, starts at 1
uint64_t now_ns();               // CLOCK_MONOTONIC
void collector_stop_and_drain(Snapshot&);   // joins the thread, drains rings into s.records, sets dropped_records
```
- The collector thread runs inside `InternalScope` and wakes every 2 ms to drain rings into an internal vector (and, from Task 10, to call `poll_pending()`). The thread registry is a mutex-protected vector of ring pointers. Rings are never freed before exit.
- Every wrapper pushes a Record with `api`, `cpu_ns`, `dur_ns` (real call duration, saturating at `UINT32_MAX`) and the relevant fields: Alloc (ptr, bytes, aux=kind, stack), Free (bytes = freed size), InvalidFree, AllocFailed, Launch (stream, func, corr), GraphLaunch, ApiCall for sync and teardown calls.
- `pthread_atfork` child handler: abandons parent state without freeing (other threads' locks may be held), resets the tracker, registry and collector-running flag, and leaves the collector to restart on next `record()`.

- [ ] **Step 1 [Claude]: Write the failing tests.**
  - `Ring.PushPopOrder`.
  - `Ring.FullDropsAndCounts`: N pushes succeed, push N+1 → false, `drops()==1`.
  - `Ring.SpscStress`: 1e6 items, the consumer's sum equals the producer's.
  - `Recorder.MultiThreadAllDrained`: 8 threads × 1000 → 8000 records, unique corr ids.
  - `tests/mock/test_paths.py::test_fork_child_has_no_inherited_leaks` (Review Focus 4): scenario `fork` writes two `t-<pid>.json` files; the parent's has `len(leaks)==1`, the child's has `leaks == []`.
  - `tests/gpu/test_trace.py::test_api_spans_and_memory_counter`: `hello_static` trace has an `X` event named `cuMemAlloc` and `C` "device memory" events whose last value is 0.
  - `test_async_pool_live_bytes`: the Async counter sequence in MiB is `[8, 12, 4, 20, 16, 0]` and `peak_bytes.async == 20 << 20`.
- [ ] **Step 2 [Claude]: Run** the unit, mock and gpu suites. Expected: the new tests FAIL.
- [ ] **Step 3 [You]: Implement** the ring, recorder, collector, record pushes in all wrappers, atfork, and the exit-handler drain.
- [ ] **Step 4 [Claude]: Review, then run** `ctest --test-dir build --output-on-failure`. Expected: PASS.
- [ ] **Step 5 [You]: Manual check.** Open the `hello_static` trace in https://ui.perfetto.dev and confirm the API spans on the main thread track and the "device memory" counter graph.
- [ ] **Step 6: Commit.** `git commit -am "feat: per-thread ring buffers, collector thread, API spans, memory counter"`

### Task 10: GPU timer and launch timing (Day 6) [You]

**Files:** Create `src/gputimer.{hpp,cpp}`. Modify `src/wrappers/launch.cpp`, `src/recorder.cpp` (collector calls poll) and `tests/gpu/test_trace.py`.

**Interfaces:**
- Produces:
```cpp
struct Timing { CUevent start = nullptr, end = nullptr; uint64_t ctx = 0; };
bool gpu_begin(int variant, CUstream s, Timing* t);  // false if mode==alloc, stream capturing, pool empty, or sampled out
void gpu_end(int variant, CUstream s, Timing& t, const GpuSpan& meta);  // records end, queues pending
void poll_pending();                 // collector: cuEventQuery with ctx current; resolved -> span; events back to pool
void flush_ctx(uint64_t ctx);        // Task 11
void gputimer_fill(Snapshot&);       // spans, unresolved_events, gpu_untimed
```
- Per-context state is created on first use inside an `InternalScope`: a pool of **4096** events created with `CU_EVENT_DEFAULT`, the device from `cuCtxGetDevice`, and a reference event recorded on stream 0, synchronized, with `cpu_ref_ns = now_ns()`. GPU time = `cpu_ref_ns + elapsed_ms(ref, ev) * 1e6`.
- Events on variant 1 use `cuEventRecord` resolved with the PER_THREAD flag (`real_fn(..., 1)`), and capture checks use `cuStreamIsCapturing` of the same variant. Pending entries go in a per-context `std::vector` behind a mutex.
- Launch wrappers (`cuLaunchKernel`, `cuLaunchKernelEx`): `corr = next_corr()`, then `gpu_begin`, the real call, `gpu_end` (only if the call succeeded) and the Launch record. `cuGraphLaunch`: GraphLaunch record plus one timed span.

- [ ] **Step 1 [Claude]: Write the failing tests.**
  - Mock `test_launch_timed_full_mode`: scenario `launch` → the trace has 1 span whose `corr` equals the launch record's.
  - Mock `test_alloc_mode_has_no_events`: `CUSCOPE_MODE=alloc` → `mock.event_record == 0` and no spans.
  - Mock `test_capturing_stream_not_timed` (Review Focus 2): scenario `capture` → `mock.event_record_on_capturing == 0`.
  - GPU `test_multistream_tracks`: 4 distinct GPU stream tracks each with 1 span ≥ 1 ms, at least one pair overlapping in time, and each span starting at or after its launch's CPU `ts - 100 µs`.
  - GPU `test_graph_capture` (Review Focus 2): passthrough holds, `graph_launches == 2`, and no GPU spans for the 3 captured launches.
- [ ] **Step 2 [Claude]: Run.** Expected: FAIL.
- [ ] **Step 3 [You]: Implement** the GPU timer and launch timing.
- [ ] **Step 4 [Claude]: Review, then run the full ctest.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat: GPU event timing with clock alignment and capture safety"`

### Task 11: Teardown flush (Day 7) [You]

**Files:** Modify `src/wrappers/teardown.cpp` and `src/gputimer.cpp`.

**Interfaces:**
- Consumes: `flush_ctx`, `tracker().on_ctx_teardown`.
- Behaviour: `cuCtxDestroy(ctx)` → `flush_ctx(ctx)` (synchronize and resolve all of that context's pending events, then drop its state without destroying events) **before** the real call; on success, `on_ctx_teardown(ctx)`. `cuDevicePrimaryCtxRelease`/`Reset(dev)` → flush the context whose state has `device == dev`, make the real call, then call `cuDevicePrimaryCtxGetState(dev, &flags, &active)` and treat `active == 0` as torn down. At exit, flush every remaining context, and count events that can't be resolved as unresolved.

- [ ] **Step 1 [Claude]: Write the failing tests.**
  - Mock `test_ctx_destroy_flushes_pending`: scenario `teardown` with `MOCK_EVENT_DELAY_QUERIES=1000000` → 1 span, `unresolved_events == 0`, and one leak with `at_ctx_teardown == true`.
  - GPU `test_multistream_no_unresolved`: `unresolved_events == 0` for every program in the passthrough list.
- [ ] **Step 2 [Claude]: Run.** Expected: FAIL.
- [ ] **Step 3 [You]: Implement** the teardown hooks.
- [ ] **Step 4 [Claude]: Review, then run the full ctest.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat: resolve pending GPU events in context teardown hooks"`

### Task 12: Kernel names and flow arrows (Day 7) [Claude]

**Files:** Create `src/wrappers/names.cpp`. Modify `src/hooks.def` (adds `cuModuleGetFunction`, `cuLibraryGetKernel`, `cuKernelGetFunction`), `src/trace_writer.cpp`, `src/runtime.cpp` and the tests.

**Interfaces:**
- Produces: `void resolve_kernel_names(Snapshot&);`, called in the exit handler for every distinct `func` in Launch records. It tries `cuFuncGetName` inside an `InternalScope`, then the fallback map (handle → mangled name recorded by the three hooks), then `"0x<hex>"`, and demangles each name.
- Flow arrows in the writer: for each Launch/Copy record with `corr != 0` whose span exists, emit `ph:"s"` (id = corr) on the CPU event and `ph:"f"` with `"bp":"e"` on the GPU span.

- [ ] **Step 1: Write the failing tests.** `Stacks.Demangle`: `"_Z11spin_kernelx"` → `"spin_kernel(long long)"`. `TraceWriter.FlowPairs`: one launch with a span → one `s` and one `f` event with equal ids. Mock `test_kernel_name_from_driver`: the span name is `"mock_kernel()"`. GPU `test_multistream_named`: 4 spans named `"spin_kernel(long long)"`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full ctest.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat: kernel names with fallback map, launch-to-GPU flow arrows"`

### Task 13: Copy tracking and bandwidth (Day 8) [You: wrappers, Claude: tests]

**Files:** Modify `src/wrappers/memcpy.cpp` and `tests/gpu/test_trace.py`.

**Interfaces:**
- Behaviour: every hooked copy gets `corr`, GPU timing through `gpu_begin`/`gpu_end` (sync copies use variant-matched stream 0), and a Copy record with `bytes` and `aux = dir | (pinned << 4)`. The direction comes from the API name. For the generic `cuMemcpy`/`cuMemcpyAsync`, call `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_MEMORY_TYPE)` on src and dst inside an `InternalScope`. On the host side, `pinned` = the attribute call succeeded with `CU_MEMORYTYPE_HOST`; an error means pageable.

- [ ] **Step 1 [Claude]: Write the failing tests.** Mock `test_copy_direction`: scenario `copy` → `copies` has `{"dir":"HtoD","bytes":4096,"count":1}`. GPU `test_bandwidth_pinned_faster`: `HtoD` pinned and pageable both have `count == 5`, and pinned `gbps` > pageable `gbps`; the same for `DtoH`.
- [ ] **Step 2 [Claude]: Run.** Expected: FAIL.
- [ ] **Step 3 [You]: Implement** the copy wrappers.
- [ ] **Step 4 [Claude]: Review, then run the full ctest.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat: copy tracking with direction, pinned detection and bandwidth"`

### Task 14: CLI and text summary (Day 9) [Claude]

**Files:** Create `src/report.{hpp,cpp}`, `cli/cuscope.py`, `tests/cli/{test_cli.py,fixtures/sample_trace.json}`. Modify `src/runtime.cpp` and `tests/gpu/test_memory.py`.

**Interfaces:**
- Produces: `void print_summary(const Snapshot&, const Summary&, FILE* out);`, called at exit with stderr (or the `CUSCOPE_LOG` file). Sections, in order: `== cuscope summary (pid N, mode M) ==`, `Memory`, `Leaks`, `Invalid frees`, `Top kernels` (10), `Copies`, `Completeness: dropped=X unresolved=Y untimed=Z`.
- CLI: `cuscope run [--mode alloc|full] [--out PATH] [--stack-depth N] [--sample-every N] -- CMD...` sets the env vars, **prepends** the library to `LD_PRELOAD`, and exits with the child's exit code. `cuscope summary TRACE.json` prints the same sections from JSON, with leak and invalid-free frames resolved to `file:line` by `addr2line -f -C -e <module> 0x<offset-1>`. Library lookup order: `$CUSCOPE_LIB`, then `<cli dir>/../lib/libcuscope.so`, then `<cli dir>/../build/libcuscope.so`.

- [ ] **Step 1: Write the failing tests.**
  - `test_run_passes_exit_code`: `cuscope run -- python3 -c "import sys; sys.exit(3)"` → 3.
  - `test_run_prepends_ld_preload`: with `LD_PRELOAD=/x.so`, the child sees `libcuscope.so` first and `/x.so` second.
  - `test_summary_sections`: the fixture prints all 7 section headers.
  - GPU `test_leak_file_line`: `cuscope summary` on the `leak` trace shows `leak.cu:<marker_line LEAK_A>` and `leak.cu:<marker_line LEAK_C>`.
  - GPU `test_summary_on_stderr_only`: stdout is unchanged under `cuscope run`.
- [ ] **Step 2: Run** `python3 -m pytest tests/cli tests/gpu -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full ctest and `python3 -m pytest tests/cli`.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat: cuscope CLI and text summary"`

### Task 15: PyTorch cross-check (Day 10) [Claude: scripts, You: write-up]

**Files:** Create `bench/pytorch_resnet.py`, `tools/validate_pytorch.py`, `docs/validation.md`.

**Interfaces:**
- `bench/pytorch_resnet.py --steps 50 --batch 32 [--json OUT]` trains torchvision ResNet-18 on random 3×224×224 data with SGD. It prints the median step ms and writes `{"max_memory_reserved", "max_memory_allocated", "step_ms_median"}`. It always sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, because expandable segments use VMM, which is a non-goal.
- `tools/validate_pytorch.py` runs it under `cuscope run --mode full`, compares `peak_bytes.device_total` (and the Device kind alone) with `max_memory_reserved`, and repeats with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`. If `torch.profiler` works, it also compares kernel counts and total kernel time. It writes the numbers table into `docs/validation.md`.

- [ ] **Step 1 [Claude]: Check dependencies.** `python3 -c "import torchvision"`. If that fails, run `python3 -m pip install --user torchvision --index-url https://download.pytorch.org/whl/cu130`. If batch 32 doesn't fit in 6 GB, halve it and record the batch size used.
- [ ] **Step 2 [Claude]: Write both scripts and run** `python3 tools/validate_pytorch.py`. Expected: exit 0 and `docs/validation.md` containing measured values for both runs.
- [ ] **Step 3 [You]: Explain each gap in writing in `docs/validation.md`** (for example memory allocated outside the caching allocator, or a `device_total` vs Device-only difference). Fix any gap that's a cuscope bug (Claude reviews the fixes).
- [ ] **Step 4: Commit.** `git commit -am "docs: PyTorch peak-memory cross-check"`

### Task 16: Overhead benchmarks and sampling (Day 11) [Claude: bench, You: sampling if needed]

**Files:** Create `bench/micro.cu`, `bench/run_bench.py`, `docs/overhead.md`. Modify `src/gputimer.cpp` (sampling).

**Interfaces:**
- `micro <launch|alloc|copy4k> --iters N` prints `ns_per_call=<float>`. `run_bench.py` runs each case 7 times in modes off/`alloc`/`full`, takes the median, adds the PyTorch step median from Task 15's script per mode, and writes the `docs/overhead.md` table.
- Sampling: when `config().sample_every > 1`, `gpu_begin` returns false except on every Nth call per thread (a `thread_local` counter), and skipped calls increment `gpu_untimed`.

- [ ] **Step 1 [Claude]: Write and run the benchmarks.** Expected: `docs/overhead.md` with real numbers.
- [ ] **Step 2 [You]: Decide on sampling.** If `full` adds more than 2× launch cost or more than 10% PyTorch step time, implement sampling. Test first ([Claude]): mock `test_sample_every`, where `CUSCOPE_SAMPLE_EVERY=4` with 8 launches → 2 spans and `gpu_untimed == 6`. Then re-run the benchmarks with `CUSCOPE_SAMPLE_EVERY=10`.
- [ ] **Step 3: Commit.** `git commit -am "bench: overhead microbenchmarks and PyTorch step timing"`

### Task 17: CI and sanitizers (Day 12) [Claude]

**Files:** Create `.github/workflows/ci.yml`. Modify `CMakeLists.txt` (`CUSCOPE_SANITIZE=address|thread`).

**Interfaces:**
- CI runs on `ubuntu-22.04` with a `sanitize: [none, address, thread]` matrix. It installs `cuda.h` only, from NVIDIA's ubuntu2204 repo package that ships `/usr/local/cuda-13.0/include/cuda.h` (identify it on the dev box with `dpkg -S /usr/local/cuda-13.0/include/cuda.h` and pin it). It configures with `-DCUSCOPE_GPU_TESTS=OFF` and runs `ctest -L "unit|mock"`.
- Sanitizer mock runs set `LD_PRELOAD="$(gcc -print-file-name=libasan.so) build/libcuscope.so"` (or `libtsan.so`). If ASan's own early `dlsym` calls can't run cleanly through the override, the sanitizer job runs the `unit` label only, and `docs/DESIGN.md` says so.

- [ ] **Step 1: Write the workflow and the sanitizer option. Run the three configurations locally** (`-DCUSCOPE_SANITIZE=address`, `thread`). Expected: `ctest -L "unit|mock"` PASS.
- [ ] **Step 2: Commit.** `git commit -am "ci: GPU-free build, unit and mock tests, ASan/TSan"`
- [ ] **Step 3 [You]: Create the GitHub repo, push, and confirm CI is green.**

### Task 18: README and design docs (Day 13) [You: How it works, Claude: the rest]

**Files:** Create `README.md`, `docs/DESIGN.md` (the spec trimmed), `docs/img/architecture.svg`, `docs/img/perfetto.gif`.

- [ ] **Step 1 [You]: Write the README "How it works" section in your own words.** Cover the three paths, the WSL shim and caller check, and pointer-matched variants.
- [ ] **Step 2 [Claude]: Write the remaining README sections, in this order:** one sentence plus the GIF, a 3-command quickstart, How it works, validation (from `docs/validation.md`), overhead (from `docs/overhead.md`), limitations (non-goals, `_exit`/signals write no trace, approximate async-free timing, VMM not tracked, CUDA 13.0 only), and roadmap (Phase 2). Also write `DESIGN.md` and the SVG diagram.
- [ ] **Step 3 [You]: Record the Perfetto GIF** of the `multistream` trace.
- [ ] **Step 4: Verify.** In a fresh clone (`git clone ~/cuscope /tmp/cs && cd /tmp/cs`), follow the README quickstart literally. Expected: a trace and a summary.
- [ ] **Step 5: Commit.** `git commit -am "docs: README, DESIGN, diagram, Perfetto GIF"`

### Task 19: Release v0.1.0 (Day 14) [You]

- [ ] **Step 1:** Fix anything still broken. Run the full ctest, `pytest tests/cli`, and `tools/census.py check`. Expected: all PASS.
- [ ] **Step 2:** `git tag -a v0.1.0 -m "cuscope v0.1.0"`, then push the tag and pin the repo.
- [ ] **Step 3:** Fill the X% in the resume line from `docs/overhead.md`.
