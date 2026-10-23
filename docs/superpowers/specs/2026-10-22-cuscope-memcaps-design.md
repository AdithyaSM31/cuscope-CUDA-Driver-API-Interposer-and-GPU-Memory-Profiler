# cuscope Phase 2a: Per-Process Memory Caps (Design Spec)

Date: 2026-10-22 · Author: Adithya Sankar Menon · Status: draft for review

Source: the Phase 2 section of *cuscope: CUDA Driver-API Interposer and GPU Memory Profiler*
(week 3, "per-process memory caps") and the README roadmap. Builds on the Phase 1 spec
(`2026-10-21-cuscope-phase1-design.md`) and on v0.1.1 as shipped. Where this spec and the
source disagree, this spec wins.

## 1. What this adds

Phase 1 watches driver calls. Phase 2 makes cuscope decide whether some of them go through.
This first part gives each process a GPU memory budget:

```
cuscope run --mem-limit 2G -- python3 train.py
CUSCOPE_MEM_LIMIT=2G LD_PRELOAD=libcuscope.so ./app
```

Allocations that would take the process past its cap fail with `CUDA_ERROR_OUT_OF_MEMORY`,
exactly as if the GPU were that size, and `cuMemGetInfo` reports a GPU that size. Two jobs
that would fight over one 6 GB laptop GPU can each be given a share and both finish.

Compute sharing (the week-4 token-bucket scheduler) is Phase 2b. It gets its own spec after
this ships as `v0.2.0`.

## 2. Scope

### Goals (ends at tag `v0.2.0`)

- A per-process cap on GPU-resident memory, set with `CUSCOPE_MEM_LIMIT` or `--mem-limit`.
- Denied allocations return `CUDA_ERROR_OUT_OF_MEMORY` without reaching the driver.
- `cuMemGetInfo` reports the capped total and free memory.
- Denials appear in the trace and the summary, with call sites.
- A reproducible two-job PyTorch demo on the RTX 3060, written up in `docs/memcaps.md`.

### Non-goals

- Compute sharing or launch throttling (Phase 2b).
- A cap shared across processes (a job-wide budget). Each process has its own cap; a forked
  child starts at zero against the same cap.
- Capping memory cuscope cannot see: CUDA context overhead, driver-internal allocations, and
  the virtual memory management APIs (`cuMemCreate`, `cuMemMap`), which stay unhooked as in
  Phase 1.
- Changing PyTorch. Its caching allocator reacts to the out-of-memory error on its own.

### Success criteria

| Criterion | Target | How it's checked |
|---|---|---|
| No cap, no change | With `CUSCOPE_MEM_LIMIT` unset, every Phase 1 test passes unchanged and the trace's `mem_limit` is `null` | Full Phase 1 suite |
| Under the cap, no change | A capped run that stays under its cap has the same stdout, exit code and report (apart from the cap section) as an uncapped run | Passthrough tests rerun with a generous cap |
| Exact denial | 64 MiB allocations under a 256 MiB cap succeed until the next one would pass the cap, and that one fails with `cudaErrorMemoryAllocation`; after freeing one, the next succeeds. On the mock that is exactly the fifth; on the GPU, any device memory the CUDA runtime allocates itself counts too, so the test checks against the charged bytes in the trace | `cap_exact.cu` on the GPU, and the mock equivalent |
| No overshoot under concurrency | Eight threads allocating against one cap never hold more than the cap at once | Mock test with a live-bytes high-water mark |
| Capped `cuMemGetInfo` | `cudaMemGetInfo` reports total = cap and free = cap − charged | `cap_exact.cu` and mock |
| Demo | Two PyTorch jobs sharing the GPU: uncapped, together they reserve more than the GPU has (on native Linux one dies with `torch.OutOfMemoryError`; WSL2 pages the excess to system RAM instead); capped, both exit 0 and stay within their caps | `tools/demo_memcaps.py`, results in `docs/memcaps.md` |
| Overhead | Uncapped: unchanged. Capped: one atomic add per allocation and per free, measured on the alloc/free microbenchmark | `bench/` rerun, numbers in `docs/overhead.md` |

## 3. Behaviour

### 3.1 What counts

The cap counts GPU-resident allocations, the same set Phase 1 already counts as "device
memory":

| Kind (`AllocKind`) | Hooks | Counted |
|---|---|---|
| Device | `cuMemAlloc`, `cuMemAllocPitch` | yes |
| Async (stream-ordered) | `cuMemAllocAsync`, `cuMemAllocFromPoolAsync` | yes |
| Managed | `cuMemAllocManaged` | yes |
| HostPinned | `cuMemAllocHost`, `cuMemHostAlloc` | no: it lives in system RAM |

### 3.2 Charging (reserve, then call)

A process-wide atomic `charged_bytes` holds what the process has allocated of the counted
kinds. A counted allocation of `n` bytes:

1. Adds `n` to `charged_bytes` (one `fetch_add`).
2. If the new total exceeds the cap: subtracts `n` again, records the denial, and returns
   `CUDA_ERROR_OUT_OF_MEMORY` **without calling the driver**. The output pointer is not
   written.
3. Otherwise calls the real function. If it fails, subtracts `n` and returns its result.

Because the reservation happens before the call, concurrent allocations can never all slip
under the cap together.

`cuMemAllocPitch` is charged `width × height` before the call (the pitch is not known yet)
and trued up to `pitch × height` after it succeeds. The true-up never denies; it can leave a
process over its cap by at most the row padding of that one allocation.

### 3.3 Giving bytes back

- **Free** (`cuMemFree`, `cuMemFreeAsync`): once the tracker removes the allocation, its
  bytes are subtracted if its kind counts. A free the driver rejects restores the tracker
  entry and charges the bytes again. Stream-ordered frees release at call time, matching how
  Phase 1 counts them.
- **Context teardown**: allocations that die with a context (`on_ctx_teardown`) release
  their bytes.
- **Fork**: the child's `charged_bytes` starts at 0, as its tracker does.

The tracker remains the single record of what is allocated; `charged_bytes` is always equal
to the tracker's live bytes of the counted kinds, except during the instant between a
reservation and its call.

### 3.4 `cuMemGetInfo`

A new hook, `cuMemGetInfo` (export `cuMemGetInfo_v2`, at the toolkit's version). With no cap
it forwards and returns the driver's answer untouched. With a cap, after a successful real
call:

- `total = min(cap, real_total)`
- `free = min(real_free, total − charged_bytes)`, floored at 0

It records no API event: it is a query, and PyTorch calls it on every out-of-memory retry.

### 3.5 Configuration

- `CUSCOPE_MEM_LIMIT`: a positive number with an optional binary suffix `K`, `M`, `G` or `T`
  (case-insensitive, optional trailing `B` or `iB`), or plain bytes. Decimals are allowed with
  a suffix (`1.5G`). Examples: `2G`, `512M`, `1.5GiB`, `268435456`.
- An empty, zero, negative or unparseable value is a configuration warning (printed at load,
  like Phase 1's) and leaves the process uncapped. A cap of zero would deny every allocation,
  which is never what someone meant.
- `cuscope run --mem-limit SIZE` sets the variable.
- The cap is per process. Children that inherit the environment get the same cap of their
  own.

### 3.6 Reporting

- **Trace**: each denial is an instant event (`ph: "i"`) on the calling thread's track, named
  `denied cuMemAlloc 512.0 MiB`, with `bytes`, `kind` and `stack` args. The `cuscope` section
  gains `mem_limit`:

  ```json
  "mem_limit": {"limit_bytes": 2147483648, "peak_charged_bytes": 2145000000,
                "denied_count": 3, "denied_bytes": 1610612736,
                "largest_denied": {"bytes": 536870912, "kind": "device", "stack": 12}}
  ```

  `mem_limit` is `null` when no cap is set.
- **Summary**: a "Memory cap" section after "Memory", only when a cap is set:

  ```
  Memory cap: 2.0 GiB, peak charged 1.99 GiB (99.6%)
    3 allocations denied (1.5 GiB); largest 512.0 MiB device at train_step (train.py:41)
  ```

- **Records**: a new `RecType::AllocDenied` carries the kind, bytes and stack id, so denials
  flow through the existing ring, trace writer and report code. A denial counts as CUDA work
  for the "quiet process" rule.

## 4. Architecture

```
alloc wrapper ──► limits::try_charge(kind, n) ──denied──► record AllocDenied, return OOM
      │                   │ granted
      │                   ▼
      └──────────► real driver call ──failed──► limits::release(kind, n)
                          │ ok
                          ▼
                tracker.on_alloc(...)          (unchanged from Phase 1)

free wrapper ───► tracker.on_free(ptr) ─► limits::release(kind, bytes) ─► real free
                                              (re-charged if the driver rejects it)
cuMemGetInfo ───► real call ─► limits::cap_info(&free, &total)
```

### New and changed components

| File | Job |
|---|---|
| `src/limits.hpp`, `src/limits.cpp` (new) | cap, `charged_bytes`, peak, denial stats; `try_charge`, `release`, `cap_info`, `after_fork_child`; no CUDA, unit-testable |
| `src/config.*` | parse `CUSCOPE_MEM_LIMIT` into `Config::mem_limit_bytes` (0 = none) |
| `src/wrappers/alloc.cpp` | charge before, release on failure and on free, record denials |
| `src/wrappers/meminfo.cpp` (new) | the `cuMemGetInfo` hook |
| `src/hooks.def` | one line for `cuMemGetInfo` |
| `src/tracker.*` | report the bytes and kinds released at context teardown |
| `src/model.hpp`, `src/trace_writer.cpp`, `src/report.cpp`, `src/summary.*` | `AllocDenied` records, `mem_limit` section, "Memory cap" summary |
| `cli/cuscope.py` | `--mem-limit`; the `summary` command prints the cap section |

`limits` is lock-free: one atomic for charged bytes, one for the peak (compare-exchange
loop), atomic counters for denials, and a small mutex only around "largest denied" (taken on
denial, never on the success path).

## 5. Safety rules

Phase 1's rules all still hold; this part relaxes exactly one of them on purpose.

- **"Pass through exactly" now has one exception**: an allocation past the cap is denied.
  Nothing else changes: under the cap every call forwards its exact arguments and returns
  the driver's result, and with no cap the cap code is never entered.
- **Deny only as the driver would.** The denial is `CUDA_ERROR_OUT_OF_MEMORY`, the same error
  a full GPU returns, so programs take their existing out-of-memory path.
- **Never deny a free, a copy or a launch.** Only allocations of counted kinds can be denied.
- **Fail open.** If cap setup fails, the process runs uncapped and cuscope says so once.

## 6. Testing

- **Unit** (`tests/unit/test_limits.cpp`, `test_config.cpp`): size parsing (each suffix,
  decimals, junk, zero); charge and deny at the boundary (exactly at the cap is allowed, one
  byte over is denied); release; peak; uncounted kinds never charge; eight threads hammering
  `try_charge` never exceed the cap.
- **Mock** (`tests/mock`): new scenarios, each in direct and shim modes:
  - `cap_exact`: 64 MiB allocations under a 256 MiB cap; the fifth is denied and the mock's
    alloc counter shows it never reached the driver; free one, allocate again.
  - `cap_meminfo`: `cuMemGetInfo` reports capped total and free (the mock gains
    `cuMemGetInfo_v2` with a fixed 6 GiB total).
  - `cap_threads`: eight threads allocate and free against one cap; the mock's live-bytes
    high-water mark never exceeds the cap.
  - `cap_teardown`: allocations released by `cuCtxDestroy` give their bytes back.
  - Plus: a pinned-host allocation over the cap is not denied; an invalid `CUSCOPE_MEM_LIMIT`
    warns and runs uncapped; no cap set means the Phase 1 suite is unchanged.
- **GPU** (`tests/cuda/cap_exact.cu`): the exact-denial and `cudaMemGetInfo` assertions on the
  3060, through the CUDA runtime (static and shared builds).
- **Demo** (`tools/demo_memcaps.py`): starts two PyTorch training jobs together on the 3060,
  first uncapped, then each capped, and records exit codes, peaks and step times. The model
  and batch sizes are chosen in the plan so the uncapped pair runs out of memory and each job
  fits its cap.

## 7. Plan and ownership

Same workflow as Phase 1: an implementation plan with TDD steps, mock CI on every push,
GPU tests on the 3060, a final whole-branch review, and the user tags the release. Same
pairing model: steps marked `[You]` are the user's to write, and the user can hand any of
them to Claude.

Expected task outline (the plan finalises it):

1. Census: confirm which `cuMemGetInfo` variants the static runtime, shared runtime and
   PyTorch request.
2. Config: `CUSCOPE_MEM_LIMIT` parsing and `--mem-limit`.
3. `limits` module with unit tests.
4. Allocation hooks charge, release and deny; mock tests.
5. Free, teardown and fork release; mock tests.
6. `cuMemGetInfo` hook; mock and GPU tests.
7. Reporting: `AllocDenied` records, trace section, summary section, CLI summary.
8. GPU test `cap_exact.cu`.
9. PyTorch demo and `docs/memcaps.md`.
10. Overhead rerun, README, DESIGN.md, final review, tag `v0.2.0`.

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| PyTorch fetches `cuMemGetInfo` through a variant or version cuscope does not hook | Task 1's census decides the hook entries before any code; if PyTorch never calls it, the demo still works, because PyTorch reacts to the out-of-memory error, not to `cuMemGetInfo` |
| PyTorch uses the VMM APIs (`expandable_segments:True`) and bypasses the cap | Detect the setting in the demo and run with `expandable_segments:False`, as Phase 1 validation already does; document it as a limitation |
| The demo pair does not run out of memory uncapped, or a job does not fit its cap | Found during Task 8: WSL2 oversubscribes device memory (one process reached 12.5 GiB on the 6 GiB card), so an uncapped pair survives on paged memory. The demo shows oversubscription uncapped versus staying in budget capped, and documents the native-Linux outcome (amended 2026-10-23 with the user) |
| Stream-ordered pools keep freed memory reserved, so real usage exceeds charged bytes | Documented: the cap counts live allocations; pool reservation is not visible without hooking pool management |
| Context and driver memory make the process use more VRAM than its cap | Documented: the cap bounds what the program allocates; `docs/memcaps.md` reports the measured per-process overhead on the 3060 |
