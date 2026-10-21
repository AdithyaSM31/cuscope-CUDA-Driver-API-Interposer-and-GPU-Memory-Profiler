# cuscope design

This is the design as built for v0.1.0. It started from the Phase 1 spec
([superpowers/specs/2026-10-21-cuscope-phase1-design.md](superpowers/specs/2026-10-21-cuscope-phase1-design.md))
and records the decisions and findings made while building it on an RTX 3060 Laptop GPU
under WSL2 (Windows driver 617.14, CUDA 13.0).

![architecture](img/architecture.svg)

## 1. Goal and the hard part

cuscope is an `LD_PRELOAD` library that sees every driver call an unmodified CUDA program
makes, whether the CUDA runtime is linked statically (nvcc's default), shared, or bundled by
PyTorch. All of them reach the GPU through `libcuda.so.1`, loaded dynamically, so the driver
is the one layer common to every program.

The hard part is that the runtime does not call driver functions by name. It `dlopen`s
libcuda, fetches `cuGetProcAddress` with `dlsym` on that handle, and then asks for each
function by base name, CUDA version and stream flags. A handle lookup never consults
`LD_PRELOAD`, so preloading alone sees none of it.

## 2. Three paths into the driver

| Path | Who uses it | How cuscope gets in |
|---|---|---|
| 1. Exported symbols | programs linked with `-lcuda` | cuscope exports the same names; the preloaded library binds first |
| 2. `dlsym(handle, ...)` | the CUDA runtime fetching `cuGetProcAddress` | cuscope exports `dlsym` and returns its own `cuGetProcAddress` |
| 3. `cuGetProcAddress(name, version, flags)` | the runtime, for every driver function | cuscope's `cuGetProcAddress` calls the real one, then swaps in a wrapper when the pointer is the expected variant |

### The WSL loader shim

On WSL, `/usr/lib/wsl/lib/libcuda.so.1` is a 188 KB loader. It `dlopen`s the real driver
(`libcuda.so.1.1`) and fills a table of 728 functions with `dlsym`, through cuscope's override.
Handing it a wrapper would make the wrapper call the shim and the shim call the wrapper,
forever. So the override first asks `dladdr` which module called it, and passes through
everything libcuda asks for itself. The census counted 739 such lookups per process.

That check must match `libcuda.` or `libcuda_` exactly. The first version matched the prefix
`libcuda`, which also matches `libcudart.so.13`: every lookup from the shared runtime, and
PyTorch's main runtime, went through unrecorded. The real-GPU census caught it
(`hello_shared` showed 0 lookups); a fake `libcudart.so.13` in the mock suite now covers it.

### Wrapping the right variant

The runtime asks for each function at its own minimum version (`cuMemAlloc` at 3020,
`cuLaunchKernel` at 4000, or at 7000 with the per-thread-stream flag), never at 13000. A
wrapper has one signature, so cuscope must never hand it out for a different variant. At
start-up the loader resolves each hook's expected pointer through the real
`cuGetProcAddress`, for both stream variants, at the version in `hooks.def`. At lookup time
it compares pointers, not version numbers: an exact match gets the wrapper, anything else
passes through untouched and is counted as a `mismatch` in the census.

Two hooks need a version other than `CUDA_VERSION`:

- `cuCtxSynchronize` at 2000. At 13000 the driver returns `cuCtxSynchronize_v2(CUcontext)`,
  while `cuda.h` declares the base name `(void)`. Resolving at 13000 would hand a `(void)`
  wrapper the `_v2` pointer.
- `cuDevicePrimaryCtxRelease` and `cuDevicePrimaryCtxReset` at 7000, with v1 export names.
  The runtime releases its context through v1; hooks resolved at 13000 never saw it.

Each wrapper's signature is checked against `cuda.h` at compile time (`CS_REGISTER`'s
`static_assert`). `tools/check_exports.py` checks the other half against the installed driver:
for every hook and both variants, `cuGetProcAddress(base, version, flags)` on the real
`libcuda.so.1.1` must return the same function as `dlsym` of the export the wrapper was
written for. A negative control asserts that `cuCtxSynchronize` at 13000 is *not* the `(void)`
function.

cuscope's own driver calls (events, context queries) use `CS_DRIVER_FN(fn)`, which builds both
the C++ type and the exported symbol name from the same token, so they cannot disagree. The
first version resolved them through `cuGetProcAddress` at 13000 and crashed inside the driver
on every run: `cuCtxGetDevice` at 13000 is `cuCtxGetDevice_v2(CUdevice*, CUcontext)`.

### Initialisation order

`dlsym` can be called before cuscope's constructors run: the WSL shim calls it from its own
constructor. Everything reachable from the hooks is therefore constant-initialised (POD,
zero-initialised atomics, `constexpr` tables) or set up on first use (`pthread_once`,
function-local statics). Start-up has two stages that never wait on each other: finding the
real `dlsym` (with `dlvsym`, so it never re-enters the override) and loading libcuda. A
`dlsym(RTLD_DEFAULT, "cuGetProcAddress_v2")` would find cuscope's own export, so pseudo-handle
lookups resolve through libcuda, and no pointer inside `libcuscope.so` is ever accepted as
"real".

## 3. Recording

```
wrapper (program thread)                     collector thread (every 2 ms)
  capture check, start event ── real call ──   drain every ring into one list
  end event, 64-byte Record ─► per-thread  ──► query pending events, convert to spans
                               SPSC ring        return events to the pool
```

- **Records.** One 64-byte `Record` per call: type, hook id, thread, CPU time and duration,
  pointer, bytes, stream, function, correlation id, stack id. Each thread pushes into its own
  single-producer, single-consumer ring of 16,384 records; a full ring drops the record and
  counts it, never blocking the program.
- **Allocations.** A tracker with 16 mutex-guarded shards keyed by pointer hash keeps live
  allocations and per-kind peaks. A free removes the allocation *before* the real call, so the
  driver can hand the address to another thread at once, and restores it if the driver rejects
  the free. An invalid free is a pointer never seen allocated that the driver also rejects.
- **Stacks.** `backtrace()` at each allocation site, deduplicated by hash. Capture anchors on
  the caller's return address (sanitizers add a frame above it) and drops cuscope's own
  frames. At exit, frames are named from the module's ELF symbol table, which knows functions
  like `main` that `dladdr` cannot; the CLI adds file:line with `addr2line`.
- **GPU time.** In `full` mode each launch and copy gets a CUDA event before and after it on
  its own stream, from a per-context pool. GPU time is
  `reference_cpu_ns + cuEventElapsedTime(reference, event)`, where the reference event was
  recorded and waited for when the context was first seen. Calls on a stream being captured
  into a graph are never timed: an event there would become a node in the program's graph.
  `CUSCOPE_SAMPLE_EVERY=N` times 1 in N launches; copies are always timed.
- **Kernel names** are looked up at first launch (`cuFuncGetName`, then `cuKernelGetName`),
  with names from `cuModuleGetFunction` and `cuLibraryGetKernel` as a fallback.

## 4. Exit, teardown and fork

- **WSL exit order.** The driver registers its own exit cleanup when CUDA starts; on WSL it
  runs before the CUDA runtime's cleanup and before cuscope's handler, and every call after it
  returns `CUDA_ERROR_DEINITIALIZED`. cuscope therefore registers a second exit handler when
  it first sees a context (after the driver's), which runs first and resolves every pending
  event while the driver still answers. Without it, `multistream` lost all four spans.
- **Context teardown.** `cuCtxDestroy`, `cuDevicePrimaryCtxRelease` and `Reset` flush the
  context's pending events before the real call and forget its timing state. When the context
  is really gone (checked with `cuDevicePrimaryCtxGetState` after a release), the memory still
  allocated there becomes leaks marked `at_ctx_teardown`. The primary context's handle is read
  with a Retain and Release only when it is active: Retain on an inactive device would create
  a context.
- **Fork.** Every cuscope lock is taken before `fork()` and released on both sides; the child
  starts with empty rings, records, tracker and counters. The collector is a raw `pthread`
  with signals blocked, so the child never inherits a `std::thread` it cannot join.
- **Quiet processes.** A process that never made a CUDA call writes no trace and no summary.

## 5. Components

| File | Job |
|---|---|
| `src/loader.cpp` | real `dlsym`, libcuda handle, real `cuGetProcAddress`, stage-2 resolution, `driver_proc` |
| `src/intercept.cpp` | exported `dlsym` and `cuGetProcAddress(_v2)`: caller check, variant matching |
| `src/hooks.def`, `src/hook_table.cpp` | the hook list (base, exports, version); wrapper and real pointers |
| `src/wrappers/*.cpp` | alloc, memcpy, launch, teardown and name hooks; path-1 exports |
| `src/tracker.cpp`, `src/stacks.cpp` | allocations, peaks, leaks, invalid frees; call-site stacks |
| `src/ring.hpp`, `src/recorder.cpp` | SPSC rings, collector thread, fork handling |
| `src/gputimer.cpp`, `src/names.hpp` | event pool, clock alignment, exit flush; kernel names |
| `src/summary.cpp`, `src/trace_writer.cpp`, `src/report.cpp` | aggregation, Chrome Trace JSON, text summary |
| `cli/cuscope.py` | `run` and `summary` |

## 6. Testing

- **Mock driver** (`tests/mock`): about 30 driver functions on host memory, in two modes: the
  driver itself, and a WSL-style loader shim generated from its exports. A runtime-like program
  and a `-lcuda` program exercise all three paths, both stream variants, version mismatch, the
  shim's re-entrant `dlsym`, `RTLD_DEFAULT`, fork, capture, teardown and sampling.
- **GPU programs** (`tests/cuda`): each runs with and without cuscope, and stdout and exit code
  must match. Known answers: `leak` (two leaks, exact sizes and lines), `double_free`,
  `async_pool` (live bytes 8, 12, 4, 20, 16, 0 MiB), `multistream`, `multithread`, `bandwidth`
  (pinned faster), `graph_capture` (still 3 nodes), `device_reset` (a teardown leak), and
  `hello` built static and shared, which must give identical reports.
- **Driver checks**: `tools/check_exports.py`, and `tools/census.py` for the Day 2 gate.
- **PyTorch** ([validation.md](validation.md)) and **overhead** ([overhead.md](overhead.md)).
- **CI**: the GPU-free suites on every push, plain and under ASan and TSan. Under a sanitizer
  only the unit tests run, because the sanitizer runtime's own early `dlsym` calls reach the
  instrumented override before the sanitizer is initialised.

## 7. Limitations

See the [README](../README.md#limitations).
