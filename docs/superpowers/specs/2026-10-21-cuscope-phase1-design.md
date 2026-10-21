# cuscope Phase 1 — Design Spec

Date: 2026-10-21 · Author: Adithya Sankar Menon · Status: draft for review

Source: *cuscope: CUDA Driver-API Interposer and GPU Memory Profiler* (the original 14-page design doc, Oct 21 2026). This spec keeps that design and records every decision made since, including what the environment check on the target laptop found. Where the two disagree, this spec wins.

## 1. What cuscope is

cuscope is a drop-in Linux library, loaded with `LD_PRELOAD`, that records every GPU allocation, copy and kernel launch an unmodified CUDA program makes. `cuscope run -- ./app` produces:

1. a leak and memory report (live allocations, peak per kind, leaks at exit with call sites, invalid frees);
2. per-kernel and per-copy GPU timings per stream, with copy bandwidth;
3. a Chrome Trace Event JSON timeline that opens in Perfetto.

It sits at the driver layer (`libcuda.so.1`), so it works whether the CUDA runtime is linked statically (nvcc default), shared (`-cudart shared`), or bundled by PyTorch.

**Purpose of the project:** a portfolio project for a systems/GPU job application. The README and the "How it works" section are aimed at an NVIDIA reviewer. Every core component must be explainable by the author line by line.

## 2. Scope

### Goals (Phase 1, ends at tag `v0.1.0`)
- Intercept driver calls from: nvcc default static runtime, `-cudart shared`, and PyTorch.
- Track device, managed, pinned-host and stream-ordered (async pool) allocations with call-site stacks.
- Time every kernel launch and copy on the GPU, per stream.
- Export a Perfetto-compatible timeline and print a text summary at exit.
- Publish measured overhead and validation results in the README.

### Non-goals
- Per-tensor tracking inside PyTorch (cuscope sees caching-allocator segments).
- Per-node timing inside CUDA graphs (one event per `cuGraphLaunch`).
- VMM APIs (`cuMemCreate`, `cuMemMap`).
- Multi-GPU, native Windows, replacing Nsight.
- **Phase 2 (memory caps, compute sharing) is out of scope for this spec.** It gets its own spec after `v0.1.0`.

### Success criteria
| Criterion | Target | Checked by |
|---|---|---|
| Static vs shared runtime | Identical reports for the same program | `static_vs_shared` test |
| Program behaviour | stdout and exit code unchanged | every GPU test run with and without cuscope, diffed |
| Leak detection | Exact count, sizes and file:line call site | `leak.cu` |
| PyTorch peak memory | Matches `torch.cuda.max_memory_reserved()`, every gap explained in writing | validation script |
| Overhead, `alloc` mode | < 5% on a PyTorch training step (target, measured) | benchmark suite |
| Overhead, `full` mode | Measured and published per call type | microbenchmarks |
| GPU-free tests | Mock suite green, including ASan and TSan builds | GitHub Actions |

## 3. Environment (verified 2026-10-21)

| Item | Value | Notes |
|---|---|---|
| GPU | RTX 3060 Laptop, 6 GB, Ampere sm_86 | sizes PyTorch workloads |
| Windows driver | 617.14 (CUDA UMD 13.4) | never install a Linux NVIDIA driver in WSL |
| WSL distro | Ubuntu 22.04.5, kernel 6.6 WSL2 | |
| glibc | 2.35 | `dlsym@GLIBC_2.34` exists; fallback `GLIBC_2.2.5` |
| Compiler | gcc/g++ 11.4 | C++17 |
| CMake | 3.22 installed → **upgrade to ≥ 3.28** via `pip install --user cmake` | older CMake has weak CUDA 13 support |
| CUDA toolkit | **not installed → install CUDA 13.0** (`cuda-toolkit-13-0` from NVIDIA's WSL-Ubuntu repo; never the `cuda` or `cuda-drivers` metapackages) | pinned; recorded in README |
| PyTorch | 2.14.0+cu130, `nvidia-cuda-runtime` 13.0.96 | toolkit pin matches its runtime |
| Repo | `~/cuscope` on the WSL ext4 filesystem | open from Windows via `\\wsl.localhost\Ubuntu-22.04\home\adith\cuscope` |

**Pin rationale:** CUDA 13.0 matches PyTorch's bundled runtime, so the static build, shared build and PyTorch all request the same `cuGetProcAddress` versions. One version set to support and document.

### WSL finding: libcuda is a loader shim
`/usr/lib/wsl/lib/libcuda.so.1` is a 188 KB `libcuda_loader.so`, not the driver. It exports all 728 driver symbols as trampolines, `dlopen`s the real 28 MB `/usr/lib/wsl/drivers/nvlti.inf_amd64_*/libcuda.so.1.1`, and resolves each function with `dlsym@GLIBC_2.2.5`.

Consequence: cuscope's exported `dlsym` override **also receives the shim's own lookups**. If it returns a cuscope wrapper for `cuGetProcAddress` to the shim, the wrapper calls the shim, the shim calls the wrapper, and the program recurses until it crashes on its first CUDA call. This is not covered by the original doc and is handled in §4.2.

## 4. Architecture

### 4.1 Overview

```
 program / CUDA runtime (static, shared, PyTorch)
        │ direct symbol      │ dlsym(handle, ...)     │ cuGetProcAddress(name, ver, flags)
        ▼                    ▼                         ▼
 ┌─────────────────────── libcuscope.so (LD_PRELOAD) ───────────────────────┐
 │ 1 exported symbols   2 dlsym override        3 cuGetProcAddress hook      │
 │                 └──────────────┬───────────────────┘                      │
 │                        hook wrappers (forward, then record)               │
 │        tracker · stacks · recorder(ring buffers) · gputimer               │
 └──────────────┬──────────────────────────────────┬─────────────────────────┘
                ▼ real call                         ▼ records
   /usr/lib/wsl/lib/libcuda.so.1 (shim)      collector thread
                ▼                                   ▼
   real libcuda.so.1.1 → RTX 3060            trace.json + summary
```

**Chosen approach (A):** `LD_PRELOAD` with three entry paths, as in the original design, plus a caller check in the `dlsym` override.
**Fallback (B):** a proxy `libcuda.so.1` placed first on `LD_LIBRARY_PATH`, with forwarding stubs generated from `nm -D` of the real library and the same wrappers. Used only if the Day 2 gate fails.
**Rejected (C):** CUPTI callbacks. That removes the core problem the project exists to demonstrate, and support on WSL is unreliable.

### 4.2 The three entry paths

1. **Direct symbol binding.** A program linked with `-lcuda` calls e.g. `cuMemAlloc_v2` by name. cuscope exports the same names, so the dynamic linker binds them to cuscope first.
2. **`dlsym` override.** The runtime does `dlopen("libcuda.so.1")` and `dlsym(handle, "cuGetProcAddress_v2")`. cuscope exports `dlsym`:
   - **Caller check (new):** take `__builtin_return_address(0)` and `dladdr` it. If the caller lives in a libcuda object (the WSL shim or the real driver), forward to the real `dlsym` untouched.
   - Otherwise, if the name is in the hook table, return cuscope's wrapper. Everything else goes to the real `dlsym`.
3. **`cuGetProcAddress` hook** (both `cuGetProcAddress` and `cuGetProcAddress_v2` signatures). Call the real function first. If the base name is hooked **and** the returned pointer is the variant cuscope's wrapper was written for (see 4.3), return the wrapper, which forwards to that real pointer. Requests for `cuGetProcAddress` itself return the wrapper. Anything else passes through untouched.

### 4.3 Matching the right variant (version and flags)

The original doc keys by name, version and flags. Under WSL the shim returns real-driver pointers from `cuGetProcAddress`, not the shim's own export addresses, so variants can't be identified by comparing against `dlsym` results. The rule:

- At init, for each hook and each flags value (`CU_GET_PROC_ADDRESS_DEFAULT`, `..._LEGACY_STREAM`, `..._PER_THREAD_DEFAULT_STREAM`), the loader calls the real `cuGetProcAddress(base, &p, 13000, flags)` and stores `p` as the expected real pointer for `(base, flags)`.
- At lookup time, if the real call returns exactly that pointer, the request is for the variant the wrapper's signature matches. Return the wrapper, which forwards to `p`.
- If it returns a different pointer (for example an older version was requested), return the real pointer untouched. A wrapper with the wrong signature would corrupt arguments.
- The Day 2 census confirms this rule against real traffic before Day 3 builds on it.

### 4.4 Getting the real functions
On first use (guarded by `pthread_once`): find the real `dlsym` via `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34")`, falling back to `"GLIBC_2.2.5"`. Then `dlopen("libcuda.so.1")`, get the real `cuGetProcAddress_v2`, and resolve every hooked function through it. All three paths share this one resolution route.

## 5. Components

| Component | File | Job | Interface |
|---|---|---|---|
| Loader | `src/loader.cpp` | real `dlsym`, open libcuda, resolve real functions once | `real<T>(name, flags)` → `T*` or null; glibc only |
| Hook table | `src/hooks.def` | single list of hooked functions | X-macro `HOOK(base, ret, params, args)`; adding a hook = one line + a wrapper |
| Interception | `src/intercept.cpp` | exported `dlsym` and `cuGetProcAddress(_v2)` | caller check, variant matching (§4.2–4.3) |
| Wrappers | `src/wrappers/{alloc,memcpy,launch,teardown}.cpp` | time, forward, report | call real fn → tracker + recorder; return the real result unchanged |
| Allocation tracker | `src/tracker.cpp` | live allocations, peak per kind, leaks, invalid frees | `on_alloc(ptr, bytes, kind, stack_id)`, `on_free(ptr)`; 16 mutex-guarded shards by pointer hash; no CUDA dependency |
| Stack capture | `src/stacks.cpp` | deduplicated call-site backtraces | `backtrace()` to `CUSCOPE_STACK_DEPTH`, stored by hash; at exit each frame is written as module path + offset with a `dladdr` + `__cxa_demangle` function name; the CLI resolves file:line with `addr2line` (programs built with `-g`) |
| Event recorder | `src/recorder.cpp` | move events off the hot path | per-thread SPSC ring buffers of 64-byte records; drops counted when full |
| GPU timer | `src/gputimer.cpp` | GPU start/end for kernels and copies | preallocated CUDA event pool; collector resolves with `cuEventQuery`; per-context reference event for clock alignment |
| Trace writer | `src/trace_writer.cpp` | Perfetto timeline | Chrome Trace Event JSON; pure C++, no CUDA |
| Reporter | `src/report.cpp` | text summary at exit | top kernels, peak memory, leaks with stacks, copy bandwidth, drop and unresolved counts |
| CLI | `cli/cuscope.py` | `cuscope run -- <cmd>`, `cuscope summary trace.json` | sets `LD_PRELOAD` and config variables |

**Configuration (env vars):** `CUSCOPE_OUT` (trace path), `CUSCOPE_MODE` (`alloc` | `full`, default `full`), `CUSCOPE_STACK_DEPTH` (default 16), `CUSCOPE_SAMPLE_EVERY` (time every Nth launch, default 1), `CUSCOPE_LOG` (log path, default stderr).

**Build rules:** C++17, CMake ≥ 3.28, `-fvisibility=hidden` with explicit exports for hooked symbols and `dlsym`. Compile against `cuda.h` only, link `-ldl -pthread`, **never `-lcuda`**, so CI builds without a GPU. CUDA test programs use `CMAKE_CUDA_ARCHITECTURES=86`.

### Phase 1 hook list (starting point; the Day 2 census finalises it)
- **Allocation:** `cuMemAlloc`, `cuMemAllocPitch`, `cuMemFree`, `cuMemAllocAsync`, `cuMemAllocFromPoolAsync`, `cuMemFreeAsync`, `cuMemAllocManaged`, `cuMemAllocHost`, `cuMemHostAlloc`, `cuMemFreeHost`
- **Copies:** `cuMemcpy`, `cuMemcpyAsync`, `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD` and their `Async` variants
- **Launches:** `cuLaunchKernel`, `cuLaunchKernelEx`, `cuGraphLaunch`
- **Sync and teardown:** `cuStreamSynchronize`, `cuCtxSynchronize`, `cuCtxDestroy`, `cuDevicePrimaryCtxRelease`, `cuDevicePrimaryCtxReset`
- **Kernel names:** `cuFuncGetName`, with a fallback map from hooking `cuModuleGetFunction` and `cuLibraryGetKernel`

## 6. Data flow

**Program thread, per hooked call:** CPU timestamp (`CLOCK_MONOTONIC`) → in `full` mode for launches and copies, record a start event on the call's stream → call the real function → record the end event, update the tracker on success → push one 64-byte record to this thread's ring → return the real result unchanged.

**Collector thread (every few ms):** drain all rings → for pending GPU events, make their context current and `cuEventQuery`; resolved events become GPU spans → return events to the pool.

**Teardown (inside `cuCtxDestroy` / `cuDevicePrimaryCtxRelease` / `Reset` hooks, before calling the real function) and at exit:** resolve or count every pending event for that context, then write the trace and print the summary at exit.

**Clock alignment:** on first use of each context, record a reference event, synchronise it, and note the CPU time. GPU time = that CPU time + `cuEventElapsedTime(reference, event)`. Resolution is about 0.5 µs.

**Record (64 bytes):** type, thread id, CPU time, pointer, bytes, stream, function handle, correlation id, stack id.

**Perfetto mapping:** API call → `ph:X` on the thread track · kernel/copy → `ph:X` on a per-stream track · launch→execution → flow `ph:s`/`ph:f` · device memory in use → counter `ph:C`.

## 7. Safety rules (override everything else)
1. **Pass through exactly.** Same arguments in, real result out, errors included. Failed allocations are recorded, never retried or masked.
2. **Fail open.** If init or a real lookup fails, that hook becomes a plain pass-through and the problem is logged once.
3. **No recursion.** A thread-local depth flag marks cuscope's own calls (event create/record, stack capture) so they skip instrumentation. The `dlsym` caller check (§4.2) prevents shim recursion.
4. **Nothing slow on the hot path.** Rings and event pool are preallocated. `backtrace()` is warmed up once at init.
5. **Full buffers drop, never block.** Drop counts appear in the summary.
6. **Teardown before exit.** Flush pending events inside the context-teardown hooks.
7. **Fork safety.** A `pthread_atfork` child handler resets state; the collector restarts on next use.
8. **Thread safety.** One mutex per tracker shard; each ring has one writer and one reader.
9. **Output hygiene.** Logs go to `CUSCOPE_LOG` or stderr, never stdout.

## 8. Testing and validation

**Layer 0: mock suite (no GPU, every change, `ctest`).** Built from Day 2, not Day 12.
- `tests/mock/libcuda_real_mock.so`: about 15 driver functions on host memory (`cuInit`, `cuDriverGetVersion`, `cuGetProcAddress(_v2)`, `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, `cuLaunchKernel`, `cuEventCreate/Record/Query/ElapsedTime`, `cuStreamSynchronize`, `cuCtxGetCurrent/SetCurrent`, `cuFuncGetName`).
- `tests/mock/libcuda.so.1` in two modes: **direct** (the mock itself) and **shim** (a loader that `dlopen`s the real mock and `dlsym`s every function, copying the WSL loader).
- `tests/mock/runtime_mimic`: `dlopen` → `dlsym("cuGetProcAddress_v2")` → lookups by base name, version and flags, plus direct `-lcuda` calls. This covers all three paths, both flags, the older-version pass-through, and the shim-recursion case.
- Unit tests for tracker, stack dedup, ring buffer and trace writer. ASan and TSan builds of the whole suite.

**Layer 1: known-answer GPU programs (`tests/cuda/`, run by `tests/harness.py` on the 3060).**
| Test | Assertion |
|---|---|
| `static_vs_shared` | static and `-cudart shared` builds give identical reports |
| `leak.cu` | 3 allocs, 1 freed → 2 leaks, exact sizes, file:line call site |
| `double_free.cu` | 1 invalid free reported; output unchanged |
| `async_pool.cu` | correct live bytes over time across 2 streams |
| `multistream.cu` | 4 named GPU tracks with visible overlap; 0 unresolved events |
| `multithread.cpp` | 8 threads alloc/free concurrently → exact counts, no crash |
| `bandwidth.cu` | pinned and pageable bandwidth both reported; pinned faster |
| passthrough | every test with and without cuscope → identical stdout and exit code |

**Layer 2: PyTorch cross-check.** ResNet-18 on random data, 50 steps, sized for 6 GB. Peak device memory vs `torch.cuda.max_memory_reserved()`, every gap explained. Repeat with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` and document the difference. Compare kernel counts with `torch.profiler` if it works on WSL.

**Layer 3: overhead.** Microbenchmarks (empty launch, alloc/free pair, 4 KB copy) in modes off / `alloc` / `full`, ns per call, median of repeated runs. PyTorch step time in each mode.

**Definition of done for each day:** that day's "done when" holds, the mock suite is green, and the passthrough diff is clean.

## 9. Plan and ownership

Working model: **the author writes the interview-critical core; Claude writes support code and reviews.** Claude's reviews explain problems and reasons and do not rewrite core code unless asked.

| Day | Author writes | Claude writes | Done when |
|---|---|---|---|
| 1 | Run installs (`sudo`); preload constructor banner | Install steps, repo skeleton, CMake, `hello.cu` | `LD_PRELOAD=./libcuscope.so ./hello` prints banner to stderr |
| 2 | `dlsym` override with caller check; logging `cuGetProcAddress` wrapper | Mock v0 (direct and shim), `runtime_mimic`, census runner script | **Gate:** census logs committed for static, shared and PyTorch, showing `cuMemAlloc` arriving via `cuGetProcAddress` in the static build, with no shim recursion. Otherwise switch to fallback B |
| 3 | Loader, `hooks.def`, wrappers keyed by name and flags, recursion guard | Mock tests for all paths; passthrough-diff harness | All programs run with identical output under cuscope |
| 4 | Allocation tracker; stack capture | Tracker tests, `leak.cu`, `double_free.cu`, `static_vs_shared` | those three pass |
| 5 | Ring buffers; collector thread | Trace writer and tests | Trace opens in Perfetto with API spans and memory graph |
| 6 | Event pool; clock alignment | `multistream.cu`, `async_pool.cu`, `multithread.cpp` | Kernel spans on per-stream tracks |
| 7 | Teardown flush | Kernel-name map, demangling, flow arrows | `multistream.cu`: named kernels, 4 tracks, 0 unresolved |
| 8 | Copy wrappers | `bandwidth.cu`, bandwidth reporting | pinned > pageable reported |
| 9 | Review | `cli/cuscope.py`, `report.cpp` | one command → trace + summary |
| 10 | Investigate and write up gaps | ResNet-18 validation script | peak matches or each gap explained |
| 11 | Sampling mode if needed | Microbenchmarks, PyTorch step benchmark | overhead table with real numbers |
| 12 | Review | GitHub Actions, ASan/TSan builds | CI green on push |
| 13 | README "How it works" | Rest of README, `DESIGN.md`, diagram, Perfetto GIF | a stranger can run it from the README |
| 14 | Buffer; tag `v0.1.0` | Buffer | release tagged |

**Cut order if behind:** sampling mode → CI wiring (the mock suite itself stays) → flow arrows. Never cut the PyTorch cross-check or the README.

**Git:** commit at least daily on `main`. The author creates the GitHub remote; Claude never pushes without being asked.

## 10. Risks and fallbacks
| Risk | Fallback |
|---|---|
| `dlsym` override misses runtime lookups under WSL | Proxy `libcuda.so.1` on `LD_LIBRARY_PATH` (approach B) |
| `dlsym` override recurses through the WSL shim | Caller check (§4.2); covered by the shim-mode mock test |
| Pointer-identity variant matching (§4.3) doesn't hold for some function | Fall back to version-number comparison for that hook, informed by census logs |
| Runtime exit cleanup destroys the context before events resolve | Flush in teardown hooks; count leftovers as unresolved |
| Function versions differ from what the doc assumed | Census shows the real requests; support CUDA 13.0 only, stated in README |
| Event per launch is too expensive | `alloc` mode; `CUSCOPE_SAMPLE_EVERY` |
| Kernel names unavailable | name map → function address |
| 6 GB VRAM too tight for the PyTorch workload | smaller batch size; record it in the README |

## 11. Repo layout
```
cuscope/
  src/ loader.cpp intercept.cpp hooks.def wrappers/ tracker.cpp stacks.cpp
       recorder.cpp gputimer.cpp trace_writer.cpp report.cpp
  cli/cuscope.py
  tests/ cuda/ mock/ unit/ harness.py
  bench/
  docs/ DESIGN.md census/ img/ superpowers/specs/
  .github/workflows/ci.yml
  CMakeLists.txt README.md
```
README order: one sentence + Perfetto GIF → 3-command quickstart → how it works (3 paths + WSL shim) → validation → overhead → limitations → roadmap (Phase 2).
