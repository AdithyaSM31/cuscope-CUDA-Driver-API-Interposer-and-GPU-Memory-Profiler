# cuscope Phase 2b: Compute Sharing (Design Spec)

Date: 2026-10-25 · Author: Adithya Sankar Menon · Status: draft for review

Source: the Phase 2 section of *cuscope: CUDA Driver-API Interposer and GPU Memory Profiler*
(week 4, "compute sharing") and the README roadmap. Builds on v0.2.0 (memory caps) and the
Phase 1 spec. Where this spec and the source disagree, this spec wins.

## 1. What this adds

Two jobs sharing one GPU each get a configured share of its time:

```
cuscope sched &                                   # the scheduler daemon
cuscope run --share 80 -- python3 train.py        # gets 80% when both are busy
cuscope run --share 20 -- python3 eval.py         # gets 20%, or everything when alone
```

A share is a **guaranteed minimum**, not a ceiling: when both jobs keep the GPU busy, GPU time
splits by weight; when one is idle, the other gets the whole GPU. cuscope enforces it at the
only place it can, the kernel launch: a job that has used more than its share waits before
its next launch until the scheduler has given it more time.

## 2. Scope

### Goals (ends at tag `v0.3.0`)

- A scheduler daemon, `cuscope sched`, that hands out GPU time to registered jobs by weight.
- `cuscope run --share N` / `CUSCOPE_SHARE=N`: a job registers with weight N and its kernel
  and graph launches wait when it is over its share.
- Work conservation: a job alone, or the only busy job, is never held back.
- Fail open: no daemon, a dead daemon or a full client table means the job runs unthrottled,
  and cuscope says so once.
- Waits in the trace and summary.
- A benchmark on the RTX 3060 at no scheduler, 50/50 and 80/20, written up in
  `docs/sharing.md`.

### Non-goals

- Pre-empting a running kernel. cuscope controls when kernels are *submitted*; a kernel
  already queued or running finishes regardless.
- Charging copies. Copies use the copy engines, which run alongside kernels.
- A daemon config file, job names, priorities or ceilings. Weights come from the jobs.
- More than one GPU, and sharing across users (the shared-memory segment is private to the
  user who started the daemon).
- Memory: that is the memory cap's job. The two compose (`--share 20 --mem-limit 2G`).

### Success criteria

| Criterion | Target | How it's checked |
|---|---|---|
| No share, no change | Without `--share`, every v0.2.0 test passes unchanged and the trace's `sched` is `null` | Full suite |
| Fail open | `--share` with no daemon running: same throughput as without cuscope's scheduler (within 5%), one log line | GPU benchmark run, mock test |
| Shares hold | Two saturating jobs at 80/20 get 80% ± 5 and 20% ± 5 of the GPU time the pair used; at 50/50, 50% ± 5 each | `tools/bench_share.py`, cuscope's measured GPU time per job |
| Work conservation | A job alone under the scheduler reaches at least 95% of its throughput without it | `tools/bench_share.py` |
| Latency reported | p50 and p99 time per batch for each job in each configuration | `docs/sharing.md` |
| Dead clients | A job killed while registered frees its slot within one second | Mock test |
| Overhead | Launch cost with `--share` measured against `full` mode | `bench/run_bench.py` |

## 3. Behaviour

### 3.1 The token model

Each registered job has a **balance** of GPU time, in nanoseconds, in shared memory.

- **Refill (daemon, every 1 ms).** The elapsed wall time Δ since the last tick is GPU time to
  hand out. It goes to the **active** jobs only (those waiting, or that launched within the
  last 50 ms), split by weight: job *i* gets `Δ × wᵢ / Σw_active`. A balance is capped at
  20 ms of credit, so an idle job cannot bank a burst. Inactive jobs get nothing.
- **Alone means unthrottled.** When at most one job is active, its balance is set to the cap
  every tick, forgiving any debt. A job alone is never held back, whatever it launches.
- **Charge at launch (client).** Before forwarding a kernel or graph launch, the job
  subtracts an **estimate** of that launch's GPU time: a running average (α = 1/8) of the
  measured GPU time of the same function or graph, or 50 µs for one never measured.
- **Wait (client).** If the balance is then negative, the launching thread marks the job
  waiting and sleeps in 200 µs steps until the balance is back at zero or above. The launch
  then goes ahead, unchanged.
- **Correct on completion (client).** When the collector resolves the launch's GPU span, it
  adds back `estimate − actual` and updates the running average. Over time a job pays
  exactly the GPU time it used, whatever the estimates were.

Charges and corrections are atomic adds on the balance; the daemon's refill is a
compare-and-swap that applies the cap. Multi-stream jobs whose kernels overlap are charged
the sum of their kernel times, which can exceed wall time: that is GPU use, and sharing
counts it.

### 3.2 The daemon

`cuscope sched [--verbose]` runs in the foreground.

- It creates the POSIX shared-memory segment `/cuscope-sched-<uid>` (mode 0600) with a header
  (magic, layout version, daemon pid, heartbeat time) and 32 client slots (pid, weight,
  balance, last-launch time, waiting flag, GPU time charged, number and total time of waits).
- It refuses to start if a live daemon already owns the segment, and takes over a segment
  whose daemon pid is dead.
- Every tick it updates the heartbeat, refills balances and, every 100 ms, frees slots whose
  pid no longer exists.
- `--verbose` prints a table once a second: pid, weight, share of GPU time over the last
  second, time spent waiting.
- On SIGINT or SIGTERM it marks the segment closed and unlinks it, so clients fail open.

### 3.3 The client

- **Registration.** With `CUSCOPE_SHARE` set, the process opens the segment at its first
  kernel or graph launch and claims a free slot (compare-and-swap of its pid into an empty
  slot). It releases the slot in its exit handler. A forked child registers on its own at
  its first launch.
- **Fail open.**
  - No segment, a wrong magic or version, or no free slot: run unthrottled, log once.
  - A heartbeat older than 100 ms while waiting: stop throttling for the rest of the
    process, log once.
- **What is never throttled.** cuscope's own calls; launches on a stream being captured
  (they do not run on the GPU then); copies and every other call.
- **Mode.** Charging needs measured GPU time, so `--share` needs `full` mode. The CLI rejects
  `--share` with `--mode alloc`; with `CUSCOPE_MODE=alloc` set directly, the library logs
  once and does not register. With sampling on, unsampled launches pay their estimate only.
- **No locks held while waiting.** The wait happens before the launch takes any cuscope lock.

### 3.4 Configuration

- `CUSCOPE_SHARE`: an integer weight from 1 to 1000. Empty, zero, out of range or junk is a
  warning and no registration. `cuscope run --share N` sets it, and rejects a bad value with
  exit code 2 (like `--mem-limit`).
- Weights are relative: 80 and 20 is the same split as 4 and 1.

### 3.5 Reporting

- **Trace.** Each wait is a complete event (`ph: "X"`, `cat: "sched"`, name `throttled`) on the
  launching thread. The `cuscope` section gains:

  ```json
  "sched": {"weight": 80, "registered": true, "gpu_ns_charged": 4210000000,
            "waits": 1534, "wait_ns": 812000000}
  ```

  `sched` is `null` without `CUSCOPE_SHARE`; `registered` is `false` when it failed open.
- **Summary.** A "Sharing" section after "Memory cap", when `CUSCOPE_SHARE` is set:

  ```
  Sharing: weight 80, 4.21 s GPU time; 1534 launches waited, 0.81 s in total
  ```

  or `Sharing: weight 80, not registered (no scheduler running)`.

## 4. Architecture

```
            cuscope sched (daemon)                       job (libcuscope.so)
  ┌─────────────────────────────────┐        ┌───────────────────────────────────┐
  │ every 1 ms: heartbeat; refill   │        │ launch hook:                      │
  │ active slots by weight (cap     │        │   est = estimate(func)            │
  │ 20 ms); alone → cap             │◄──────►│   balance -= est; wait while < 0  │
  │ every 100 ms: free dead pids    │  shm   │   forward the launch              │
  └─────────────────────────────────┘ slots  │ collector, span resolved:         │
                                             │   balance += est - actual         │
                                             └───────────────────────────────────┘
```

### Components

| File | Job |
|---|---|
| `src/sched_shm.hpp` (new) | the shared layout: header and slots as `std::atomic` fields; magic and version |
| `src/sched_core.hpp`, `src/sched_core.cpp` (new) | pure token math: `refill`, `charge`, `correct`, `Estimator`; no CUDA, no shm; unit-tested |
| `src/sched_client.hpp`, `src/sched_client.cpp` (new) | registration, `before_launch` (charge and wait), `on_span` (correct), fail-open, release at exit |
| `tools/cuscope_sched.cpp` (new binary `cuscope-sched`) | the daemon loop |
| `src/wrappers/launch.cpp` | call `before_launch` ahead of each kernel and graph launch |
| `src/gputimer.cpp` | pass each resolved launch span (corr, func, duration) to `on_span` |
| `src/config.*`, `cli/cuscope.py` | `CUSCOPE_SHARE`, `--share`, `cuscope sched` (runs the daemon binary) |
| `src/trace_writer.cpp`, `src/report.cpp`, `src/model.hpp` | `sched` section, throttled spans, "Sharing" summary |

The daemon is a separate small C++ binary, so it does not load CUDA or cuscope's hooks;
`cuscope sched` finds it next to `libcuscope.so`, as it finds the library.

## 5. Safety rules

The Phase 1 and memory-cap rules still hold. Sharing adds one more deliberate change: a
launch may be **delayed**. It is never failed, reordered, changed or dropped.

- Without `CUSCOPE_SHARE`, the launch hooks do not touch the scheduler code.
- Every failure of the scheduler (no daemon, dead daemon, full table, bad segment) means
  unthrottled, never stuck: the only unbounded wait is on a daemon whose heartbeat is
  fresh.
- A wait never holds a cuscope lock or happens inside cuscope's own calls.
- The segment is private to the user (mode 0600, name includes the uid).

## 6. Testing

- **Unit** (`tests/unit/test_sched_core.cpp`): refill splits Δ by weight among active jobs
  only; the cap; a single active job is set to the cap; correction; the estimator's running
  average and default; weights that do not divide evenly lose no time beyond rounding.
- **Mock** (`tests/mock`, with the daemon binary):
  - registration claims a slot and the exit handler frees it;
  - with the daemon paused (`SIGSTOP`), a launch whose balance is negative waits; after
    `SIGCONT` it proceeds;
  - after the daemon is killed, a waiting launch proceeds within ~100 ms and logs once;
  - a client killed with `SIGKILL` has its slot freed within one second;
  - no daemon: `--share` runs unthrottled and logs once;
  - no `CUSCOPE_SHARE`: `sched` is `null` and the suite is unchanged;
  - corrections: the mock's measured span times are added back (mock spans have a known
    duration when `MOCK_KERNEL_NS` is set).
- **GPU** (`tests/cuda/share_load.cu` and `tools/bench_share.py`): two processes, each
  launching 2 ms spin kernels in batches of 10 for 20 s, under no scheduler, scheduler with
  one job alone, 50/50 and 80/20. Reported per job: kernels per second, share of the pair's
  GPU time (from cuscope's traces), p50 and p99 time per batch. The success criteria in §2
  are checked by the script.

## 7. Plan and ownership

Same workflow as Phases 1 and 2a: TDD plan, mock CI, GPU tests on the 3060, final
whole-branch review, the user tags. Steps marked `[You]` are the user's; they may hand them
to Claude.

Expected task outline (the plan finalises it):

1. `CUSCOPE_SHARE` and `--share` (with the alloc-mode check).
2. Shared layout and `sched_core` token math, unit-tested.
3. The daemon binary and `cuscope sched`.
4. Client registration, fail-open and release.
5. Charging and waiting in the launch hooks.
6. Correction from resolved spans; the estimator.
7. Reporting: throttled spans, `sched` section, "Sharing" summary.
8. GPU benchmark and `docs/sharing.md`.
9. Overhead, README, DESIGN.md, final review, tag `v0.3.0`.

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| WSL timer and sleep jitter makes 1 ms ticks and 200 µs waits coarse | The model is in GPU time, not ticks: a late tick hands out the whole elapsed Δ. Measure the achieved shares; if they miss ±5, raise the cap or the tick and record the numbers |
| The Windows driver's own time-slicing of the two processes interferes | Measure the no-scheduler split first and report it; the shares criterion is about the scheduler's effect, judged against that baseline |
| Deep launch queues: a job submits many kernels before its debt shows | The estimate is charged at launch, so debt shows at once; kernels already queued run regardless (documented limit) |
| Estimates far off for kernels whose time varies by input | Correction makes the long-run charge exact; only the size of a burst depends on the estimate |
| A crashed daemon leaves a stale segment | The next daemon takes over a segment whose pid is dead; clients fail open on a stale heartbeat |
| PyTorch workloads are too irregular for a clean benchmark | The benchmark uses fixed spin kernels; a PyTorch pair is an optional extra in `docs/sharing.md` |
