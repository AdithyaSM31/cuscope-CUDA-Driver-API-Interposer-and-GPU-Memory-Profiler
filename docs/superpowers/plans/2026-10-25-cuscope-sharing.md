# cuscope Phase 2b: Compute Sharing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Jobs launched with `cuscope run --share N` get a weighted, work-conserving share of GPU time, enforced by delaying kernel and graph launches against token balances that a small daemon (`cuscope sched`) refills. Release as `v0.3.0`.

**Architecture:**
- **Shared layout.** A per-user POSIX shared-memory segment holds a header and 32 client slots of atomics.
- **Pure core.** The token math (refill planning, the estimator) and a `SchedClient` class that charges, waits and corrects against a `SchedShm*`. It has no CUDA and no shm calls, so it is unit-tested on an in-memory segment.
- **Glue in `libcuscope.so`.** Opens the segment, checks for stream capture, records the waits, and connects the launch hooks and the collector's resolved spans.
- **Daemon.** A separate binary, `cuscope-sched`, owns the segment and ticks every millisecond.

**Tech Stack:** C++17, POSIX shm (`shm_open`/`mmap`), CMake, GoogleTest, pytest, the mock libcuda (direct and shim), CUDA 13.0 on the RTX 3060 (WSL2).

**Spec:** `docs/superpowers/specs/2026-10-25-cuscope-sharing-design.md` (and the Phase 1 and memory-cap specs it builds on)

## Global Constraints

- Constants (spec §3.1–§3.3):
  - tick `kTickNs = 1'000'000`;
  - active window `kActiveWindowNs = 50'000'000`;
  - credit cap `kCapNs = 20'000'000`;
  - default estimate `kDefaultEstimateNs = 50'000`;
  - estimator α = 1/8;
  - wait step `kWaitStepNs = 200'000`;
  - stale heartbeat `kStaleNs = 100'000'000`;
  - reaping every `kReapEveryNs = 100'000'000`;
  - 32 slots.
- Segment name: `/cuscope-sched-<uid>`, mode 0600, magic `kSchedMagic = 0x6375736368656421` ("cusched!"), layout `kSchedVersion = 1`.
- A share is a guaranteed minimum. Refill goes only to active jobs, split by weight.
- **A job alone is never held back.** The daemon sets the only active job's balance to the cap and raises its `solo` flag every tick, and a client does not wait while `solo` is set. *(Plan decision: the spec's "set to the cap every tick" alone would still let a fast job exhaust 20 ms within one tick.)*
- `CUSCOPE_SHARE`: integer 1..1000. Anything else warns and does not register. `--share` rejects a bad value with exit 2, and rejects `--mode alloc`.
- Without `CUSCOPE_SHARE` the launch hooks never enter scheduler code, and the trace's `sched` is `null`.
- Fail open: no segment, bad magic or version, no free slot, a stale heartbeat while waiting, or a closed segment all mean unthrottled. Each is logged once.
- Never throttle cuscope's own calls, launches on a capturing stream, copies or other calls. Never hold a cuscope lock while waiting. A launch is only ever delayed.
- Same rules as before: TDD (watch every test fail), mock suite in CI, GPU work on the 3060, commit per task, every deviation ledgered as `Ruling:`. **[You]** steps are the user's and may be handed to Claude.

## Review Focus

1. **Extreme weights (1 vs 1000).** Integer division must not starve the weight-1 job of refill. → Task 2 `SchedCore.ExtremeWeightsStillRefill`.
2. **Daemon stopped (SIGTERM) or restarted while jobs run.** Clients holding the old, now closed segment must fail open, not hang. → Task 5 `test_daemon_stopped_jobs_continue`.
3. **More than 32 jobs.** The 33rd must run unthrottled with one log line, not crash or overwrite a slot. → Task 4 `SchedClient.FullTableFailsOpen`.
4. **A second job arriving while the first is solo.** The `solo` flag must clear and both must be throttled from the next tick. → Task 3 `SchedCore.SoloClearsWhenSecondJobActive`.
5. **Launches during stream capture under `--share`.** They must not be charged or delayed, or graph capture slows and charges twice. → Task 5 `test_capture_launches_not_charged`.

---

### Task 1: `CUSCOPE_SHARE` and `--share`

**Files:**
- Modify: `src/config.hpp`, `src/config.cpp`, `cli/cuscope.py`
- Test: `tests/unit/test_config.cpp`, `tests/cli/test_cli.py`

**Interfaces:**
- Produces: `uint32_t Config::share_weight = 0;` (0 = not sharing).

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(Config, ShareWeight) {
  EXPECT_EQ(parse_config(env({})).share_weight, 0u);
  EXPECT_EQ(parse_config(env({{"CUSCOPE_SHARE", "80"}})).share_weight, 80u);
  EXPECT_EQ(parse_config(env({{"CUSCOPE_SHARE", "1"}})).share_weight, 1u);
  EXPECT_EQ(parse_config(env({{"CUSCOPE_SHARE", "1000"}})).share_weight, 1000u);
  for (const char* bad : {"", "0", "1001", "-5", "4.5", "lots"}) {
    Config c = parse_config(env({{"CUSCOPE_SHARE", bad}}));
    EXPECT_EQ(c.share_weight, 0u) << bad;
    EXPECT_EQ(c.warnings.size(), 1u) << bad;
  }
}
```

```python
def test_run_sets_share():
    assert child_env("--share", "80")["CUSCOPE_SHARE"] == "80"

def test_bad_share_is_an_error():
    for bad in ["0", "1001", "4.5", "lots"]:
        r = cuscope("run", "--share", bad, "--", sys.executable, "-c", "print(1)")
        assert r.returncode == 2 and "--share" in r.stderr and r.stdout == "", bad

def test_share_needs_full_mode():
    r = cuscope("run", "--share", "50", "--mode", "alloc", "--", sys.executable, "-c", "print(1)")
    assert r.returncode == 2 and "full" in r.stderr and r.stdout == ""
```

- [ ] **Step 2: Run to verify they fail**

Run: `cmake --build build && ./build/tests/unit/cuscope_unit_tests --gtest_filter='Config.Share*'` and `python3 -m pytest -q tests/cli -k share`
Expected: compile error (`share_weight`); CLI tests fail (unknown option).

- [ ] **Step 3: Implement** the field and parsing (reuse `parse_int`; warning `ignoring CUSCOPE_SHARE="…", not sharing`). In the CLI, add `--share` to the option mapping (`("share", "CUSCOPE_SHARE")`, help `share of GPU time when jobs compete, a weight 1-1000 (needs cuscope sched)`), validate it before `find_library`, and reject it with `--mode alloc`.

- [ ] **Step 4: Run to verify they pass.** Expected: PASS; `ctest --test-dir build` all pass.

- [ ] **Step 5: Commit** `feat: CUSCOPE_SHARE and --share`

---

### Task 2: Shared layout and the token math

**Files:**
- Create: `src/sched_shm.hpp`, `src/sched_core.hpp`, `src/sched_core.cpp` (add to `cuscope_core`)
- Test: `tests/unit/test_sched_core.cpp` (add to `tests/unit/CMakeLists.txt`)

**Interfaces:**
- Produces:

```cpp
// sched_shm.hpp: the segment, mapped by the daemon and every client.
constexpr uint64_t kSchedMagic = 0x6375736368656421ULL;
constexpr uint32_t kSchedVersion = 1;
constexpr int kSchedSlots = 32;
struct SchedSlot {
  std::atomic<int32_t> pid;             // 0 = free
  std::atomic<uint32_t> weight;
  std::atomic<int64_t> balance_ns;
  std::atomic<uint64_t> last_launch_ns; // CLOCK_MONOTONIC
  std::atomic<uint32_t> waiting, solo;
  std::atomic<uint64_t> charged_ns, waits, wait_ns;
};
struct SchedShm {
  uint64_t magic;
  uint32_t version;
  std::atomic<int32_t> daemon_pid;
  std::atomic<uint64_t> heartbeat_ns;
  std::atomic<uint32_t> closed;
  SchedSlot slots[kSchedSlots];
};
std::string sched_shm_name();  // "/cuscope-sched-<uid>"
void sched_shm_init(SchedShm* shm, int32_t daemon_pid, uint64_t now_ns);  // magic, version, zeroed slots

// sched_core.hpp: pure math.
struct ClientView { bool used; bool active; uint32_t weight; };
struct RefillStep { int64_t add_ns; bool solo; };
std::vector<RefillStep> plan_refill(const std::vector<ClientView>& clients, uint64_t delta_ns);
void apply_refill(SchedShm* shm, uint64_t now_ns, uint64_t delta_ns);  // reads slots, plans, applies with the cap

class Estimator {   // per launch key (CUfunction or CUgraphExec)
 public:
  uint64_t estimate(uint64_t key) const;          // kDefaultEstimateNs if never measured
  void observe(uint64_t key, uint64_t actual_ns); // first sets it; then avg += (actual - avg) / 8
};
```

`apply_refill`:
- A slot is active if `waiting` is set, or if `now − last_launch_ns ≤ kActiveWindowNs`.
- When exactly one slot is active: `balance_ns = kCapNs` and `solo = 1`. Every other slot gets `solo = 0`.
- Otherwise, each active slot gets `fetch_add(add_ns)` and is then clamped to `kCapNs` with a CAS loop.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(SchedCore, SplitsByWeightAmongActive) {
  auto steps = plan_refill({{true, true, 80}, {true, true, 20}, {true, false, 50}, {false, false, 0}}, 1'000'000);
  EXPECT_EQ(steps[0].add_ns, 800'000);
  EXPECT_EQ(steps[1].add_ns, 200'000);
  EXPECT_EQ(steps[2].add_ns, 0);  // idle: nothing
  EXPECT_EQ(steps[3].add_ns, 0);
  EXPECT_FALSE(steps[0].solo);
}

TEST(SchedCore, AloneIsSolo) {
  auto steps = plan_refill({{true, true, 5}, {true, false, 95}}, 1'000'000);
  EXPECT_TRUE(steps[0].solo);
  EXPECT_FALSE(steps[1].solo);
}

TEST(SchedCore, ExtremeWeightsStillRefill) {  // Review Focus 1
  auto steps = plan_refill({{true, true, 1}, {true, true, 1000}}, 1'000'000);
  EXPECT_GT(steps[0].add_ns, 0);
  EXPECT_GE(steps[0].add_ns + steps[1].add_ns, 1'000'000 - 2);  // only rounding is lost
}

TEST(SchedCore, ApplyRefillCapsAndForgivesSolo) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10; shm.slots[0].weight = 1; shm.slots[0].balance_ns = 19'900'000; shm.slots[0].last_launch_ns = 1;
  shm.slots[1].pid = 11; shm.slots[1].weight = 1; shm.slots[1].balance_ns = -5'000'000; shm.slots[1].last_launch_ns = 1;
  apply_refill(&shm, 2, 1'000'000);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs);               // capped
  EXPECT_EQ(shm.slots[1].balance_ns.load(), -5'000'000 + 500'000); // debt pays down
  shm.slots[1].pid = 0;                                            // the other job leaves
  apply_refill(&shm, 3, 1'000'000);
  EXPECT_EQ(shm.slots[0].solo.load(), 1u);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs);
}

TEST(SchedCore, SoloClearsWhenSecondJobActive) {  // Review Focus 4
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10; shm.slots[0].weight = 1; shm.slots[0].last_launch_ns = 1;
  apply_refill(&shm, 2, 1'000'000);
  ASSERT_EQ(shm.slots[0].solo.load(), 1u);
  shm.slots[1].pid = 11; shm.slots[1].weight = 1; shm.slots[1].waiting = 1;
  apply_refill(&shm, 3, 1'000'000);
  EXPECT_EQ(shm.slots[0].solo.load(), 0u);
  EXPECT_EQ(shm.slots[1].solo.load(), 0u);
}

TEST(SchedCore, IdleSlotsAreNotActive) {
  SchedShm shm{};
  sched_shm_init(&shm, 1, 0);
  shm.slots[0].pid = 10; shm.slots[0].weight = 1; shm.slots[0].last_launch_ns = 0;
  apply_refill(&shm, kActiveWindowNs + 1, 1'000'000);
  EXPECT_EQ(shm.slots[0].solo.load(), 0u);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), 0);
}

TEST(Estimator, DefaultThenRunningAverage) {
  Estimator e;
  EXPECT_EQ(e.estimate(7), kDefaultEstimateNs);
  e.observe(7, 800'000);
  EXPECT_EQ(e.estimate(7), 800'000u);
  e.observe(7, 1'600'000);
  EXPECT_EQ(e.estimate(7), 900'000u);  // 800 + (1600 - 800) / 8
}
```

- [ ] **Step 2: Run to verify they fail.** Expected: compile error (headers missing).
- [ ] **Step 3: Implement** the layout, `sched_shm_name` (`getuid()`), `sched_shm_init`, `plan_refill`, `apply_refill` and `Estimator` (an `unordered_map` behind a mutex). `plan_refill` computes `delta × w / Σw` in 128-bit (`unsigned __int128`) to avoid overflow.
- [ ] **Step 4: Run to verify they pass.** Run the unit tests and the TSan unit build. Expected: PASS.
- [ ] **Step 5: Commit** `feat: the scheduler's shared layout and token math`

---

### Task 3: The daemon, `cuscope sched`

**Files:**
- Create: `tools/cuscope_sched.cpp` (CMake target `cuscope-sched`, output next to `libcuscope.so`, linked with the `cuscope_core` objects and `rt`)
- Modify: `cli/cuscope.py` (`sched` subcommand: find `cuscope-sched` like the library, through `$CUSCOPE_SCHED`, `<cli>/../lib`, `<cli>/../build`, then `os.execv` it with the remaining arguments)
- Test: `tests/mock/test_sched.py` (new; fixture `sched_daemon` starts and stops the binary with a private uid-free name via `CUSCOPE_SCHED_NAME`)

**Interfaces:**
- Consumes: `SchedShm`, `sched_shm_init`, `apply_refill` (Task 2).
- Produces:
  - `cuscope-sched [--verbose] [--status]`. `--status` prints one line per used slot, `slot pid weight balance_ns solo waiting`, and exits 0, or exits 1 with `no scheduler running`.
  - `CUSCOPE_SCHED_NAME` overrides `sched_shm_name()` for the daemon and clients alike, so tests run in parallel without clashing. `sched_shm_name()` reads it.

Daemon behaviour (spec §3.2):
- Refuse with exit 1 if the segment exists and its `daemon_pid` is alive (`kill(pid, 0) == 0`).
- Otherwise create it (`shm_open(O_CREAT|O_RDWR, 0600)`, `ftruncate`, `mmap`, `sched_shm_init`).
- Tick every `kTickNs` with `clock_nanosleep` to an absolute deadline: heartbeat, then `apply_refill(now, now − last)`.
- Every `kReapEveryNs`, clear slots whose pid is dead.
- On SIGINT/SIGTERM, set `closed = 1`, `shm_unlink`, and exit 0.
- `--verbose` prints, once a second, per slot: pid, weight, share of `charged_ns` growth over the last second, and `wait_ns` growth.

- [ ] **Step 1: Write the failing tests** (`tests/mock/test_sched.py`)

```python
def test_second_daemon_refuses(sched_daemon):
    r = subprocess.run([str(SCHED)], env=sched_daemon.env, capture_output=True, text=True, timeout=5)
    assert r.returncode == 1 and "already running" in r.stderr

def test_status_without_daemon(tmp_name_env):
    r = subprocess.run([str(SCHED), "--status"], env=tmp_name_env, capture_output=True, text=True, timeout=5)
    assert r.returncode == 1 and "no scheduler running" in r.stderr

def test_sigterm_unlinks_the_segment(sched_daemon):
    sched_daemon.stop()  # SIGTERM, wait
    assert not Path("/dev/shm" + sched_daemon.name).exists()

def test_takes_over_after_a_crashed_daemon(sched_daemon):
    sched_daemon.kill()  # SIGKILL: the segment stays behind with a dead daemon pid
    assert Path("/dev/shm" + sched_daemon.name).exists()
    sched_daemon.start()  # a new daemon must start, not refuse
    assert sched_daemon.status() == []
```

- [ ] **Step 2: Run to verify they fail.** Expected: `cuscope-sched` not found.
- [ ] **Step 3: [You] Implement the daemon loop in `tools/cuscope_sched.cpp`** as described above. It must not load CUDA or link `libcuscope.so`.
- [ ] **Step 4: Run to verify they pass.** Also `python3 cli/cuscope.py sched --status` prints `no scheduler running`.
- [ ] **Step 5: Commit** `feat: cuscope sched, the compute-sharing daemon`

---

### Task 4: Client registration, release and fail-open

**Files:**
- Modify: `src/sched_core.hpp/.cpp` (`SchedClient`)
- Create: `src/sched_client.hpp`, `src/sched_client.cpp` (process glue; in `CUSCOPE_SOURCES`)
- Modify: `src/runtime.cpp` (release at exit; reset in the fork child)
- Test: `tests/unit/test_sched_core.cpp`, `tests/mock/test_sched.py`, `tests/mock/runtime_mimic.cpp` (scenario `share_loop`: launch kernels for `MIMIC_SECONDS` (default 1.0) seconds, print `LAUNCHES <n>`)

**Interfaces:**
- Consumes: `SchedShm` (Task 2); the daemon and `CUSCOPE_SCHED_NAME` (Task 3); `Config::share_weight` (Task 1).
- Produces:

```cpp
// sched_core.hpp
enum class SchedState { Off, Registered, NoScheduler, NoFreeSlot, AllocMode, Stopped };
class SchedClient {
 public:
  using Clock = uint64_t (*)();
  using Sleep = void (*)(uint64_t ns);
  SchedClient(SchedShm* shm, int32_t pid, uint32_t weight, Clock now, Sleep sleep);  // claims a slot
  SchedState state() const;
  uint64_t before_launch(uint64_t key);  // Task 5
  void on_span(uint64_t key, uint64_t charged_ns, uint64_t actual_ns);  // Task 6
  void release();
  SchedStats stats() const;
};
// model.hpp
struct SchedStats {
  bool enabled = false, registered = false;
  uint32_t weight = 0;
  uint64_t charged_ns = 0, waits = 0, wait_ns = 0;
  std::string reason;  // why not registered: "no scheduler running", "no free slot", "alloc mode", "scheduler stopped"
};
// Snapshot gains: SchedStats sched;

// sched_client.hpp (glue, libcuscope.so only)
uint64_t sched_before_launch(int variant, CUstream stream, uint64_t key);  // 0 when not sharing
void sched_on_span(const GpuSpan& span);
void sched_fill(Snapshot& s);
void sched_release_at_exit();
void sched_after_fork_child();
```

Glue behaviour:
- Lazily create a `SchedClient` on the first `sched_before_launch` when `config().share_weight > 0`.
- Map the segment read-write (`shm_open(O_RDWR)`) and validate the magic, version and `closed`.
- In alloc mode, do not register: state `AllocMode`, logged once.
- Log once for each fail-open reason, for example `compute sharing off: no scheduler running (start cuscope sched)`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(SchedClient, ClaimsAndReleasesASlot) {
  SchedShm shm{}; sched_shm_init(&shm, 1, 0);
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);
  EXPECT_EQ(c.state(), SchedState::Registered);
  EXPECT_EQ(shm.slots[0].pid.load(), 4242);
  EXPECT_EQ(shm.slots[0].weight.load(), 80u);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs);  // starts with full credit
  c.release();
  EXPECT_EQ(shm.slots[0].pid.load(), 0);
}

TEST(SchedClient, FullTableFailsOpen) {  // Review Focus 3
  SchedShm shm{}; sched_shm_init(&shm, 1, 0);
  for (auto& s : shm.slots) s.pid = 99;
  SchedClient c(&shm, 4242, 80, fake_now, fake_sleep);
  EXPECT_EQ(c.state(), SchedState::NoFreeSlot);
  EXPECT_EQ(c.before_launch(7), 0u);  // never charges or waits
}

TEST(SchedClient, ClosedSegmentFailsOpen) {
  SchedShm shm{}; sched_shm_init(&shm, 1, 0);
  shm.closed = 1;
  EXPECT_EQ(SchedClient(&shm, 4242, 80, fake_now, fake_sleep).state(), SchedState::NoScheduler);
}
```

(`fake_now`/`fake_sleep` are test helpers: a static clock that `fake_sleep` advances.)

```python
def test_registers_and_frees_its_slot(sched_daemon, mimic):
    r = mimic("share_loop", "shim", env={**sched_daemon.env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "0.3"})
    assert r.code == 0 and r.trace["cuscope"]["sched"]["registered"] is True
    assert sched_daemon.status() == []  # released at exit

def test_no_daemon_runs_unthrottled(mimic, tmp_name_env):
    r = mimic("share_loop", "shim", env={**tmp_name_env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "0.3"})
    sched = r.trace["cuscope"]["sched"]
    assert (sched["registered"], sched["reason"], sched["waits"]) == (False, "no scheduler running", 0)
    assert r.stderr.count("no scheduler running") == 1

def test_killed_job_slot_is_freed(sched_daemon, mimic_popen):
    proc = mimic_popen("share_loop", env={**sched_daemon.env, "CUSCOPE_SHARE": "50", "MIMIC_SECONDS": "30"})
    wait_until(lambda: len(sched_daemon.status()) == 1, timeout=5)
    proc.kill()
    wait_until(lambda: sched_daemon.status() == [], timeout=1.0)

def test_no_share_no_sched(mimic):
    assert mimic("share_loop", "shim", env={"MIMIC_SECONDS": "0.2"}).trace["cuscope"]["sched"] is None
```

The fixtures (`sched_daemon` with `.env`, `.name`, `.status()`, `.stop()`; `tmp_name_env`; `mimic_popen`; `wait_until`) go in `tests/mock/conftest.py`. `trace["cuscope"]["sched"]` needs Task 7's trace section: put the `sched` JSON section in this task (the writer part only), and leave its summary line and throttled spans to Task 7.

- [ ] **Step 2: Run to verify they fail.** Expected: compile errors, then registration assertions failing.
- [ ] **Step 3: Implement** `SchedClient` registration and release, the glue, the trace's `sched` object (`null` without `CUSCOPE_SHARE`; otherwise `weight`, `registered`, `gpu_ns_charged`, `waits`, `wait_ns`, `reason`), the exit release (registered in the constructor *before* `write_report_at_exit`, so it runs after it: the report reads the slot's counters first), and the fork reset.
- [ ] **Step 4: Run to verify they pass.** Run `ctest --test-dir build`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: jobs register with the scheduler and fail open without it`

---

### Task 5: Charging and waiting at launch

**Files:**
- Modify: `src/sched_core.cpp` (`SchedClient::before_launch`), `src/sched_client.cpp`, `src/wrappers/launch.cpp`, `src/model.hpp` (`RecType::Throttled`, `GpuSpan::charged_ns`)
- Modify (mock): `tests/mock/mock_driver.cpp`. With `MOCK_KERNEL_NS` set, each launch on a stream advances that stream's virtual clock by that many ns, and `cuEventRecord` stamps events with the stream's clock. Measured spans then last `MOCK_KERNEL_NS`, and nothing else changes.
- Test: `tests/unit/test_sched_core.cpp`, `tests/mock/test_sched.py`

**Interfaces:**
- Consumes: `SchedClient`, `Estimator`, glue (Task 4).
- Produces: `uint64_t SchedClient::before_launch(uint64_t key)` returns the estimate charged (0 when not registered). It records waits into the slot (`waits`, `wait_ns`) and calls an optional `on_wait(start_ns, dur_ns)` callback, which the glue uses to `record()` a `RecType::Throttled` record. `GpuSpan` gains `uint64_t charged_ns`.

`before_launch` behaviour:
1. If not registered, return 0.
2. Otherwise:
   1. `est = estimator.estimate(key)`.
   2. `balance −= est`; `charged_ns += est`; `last_launch_ns = now`.
   3. While `balance < 0`, `solo == 0` and the heartbeat is fresh: set `waiting = 1`, `sleep(kWaitStepNs)`.
   4. Clear `waiting`.
   5. If the heartbeat went stale or the segment closed, switch to `Stopped` (logged once) and return.
   6. Return `est`.

In `launch.cpp`:
- Call `sched_before_launch(V, stream, key)` before `gpu_begin`. The glue returns 0, with no charge, when the stream is capturing (via `cuStreamIsCapturing`) or the call is internal.
- Store the result in `meta.charged_ns`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(SchedClient, ChargesTheEstimate) {
  SchedShm shm{}; sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  EXPECT_EQ(c.before_launch(7), kDefaultEstimateNs);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs - int64_t(kDefaultEstimateNs));
}

TEST(SchedClient, WaitsUntilRefilled) {
  SchedShm shm{}; sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, refill_on_sleep);  // each sleep: heartbeat + 100 us credit
  shm.slots[0].balance_ns = -300'000;
  c.before_launch(7);
  EXPECT_GE(shm.slots[0].waits.load(), 1u);
  EXPECT_GE(shm.slots[0].balance_ns.load(), -int64_t(kDefaultEstimateNs));
}

TEST(SchedClient, SoloNeverWaits) {
  SchedShm shm{}; sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  shm.slots[0].balance_ns = -50'000'000;
  shm.slots[0].solo = 1;
  c.before_launch(7);
  EXPECT_EQ(shm.slots[0].waits.load(), 0u);
}

TEST(SchedClient, StaleHeartbeatStopsThrottling) {
  SchedShm shm{}; sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);  // fake_sleep advances time, nobody beats
  shm.slots[0].balance_ns = -50'000'000;
  c.before_launch(7);
  EXPECT_EQ(c.state(), SchedState::Stopped);
  EXPECT_EQ(c.before_launch(7), 0u);
}
```

```python
def test_two_jobs_split_by_weight(sched_daemon, mimic_pair):
    # Mock kernels "take" 1 ms of GPU time each and the mock GPU never saturates,
    # so only the scheduler limits the jobs: their GPU time splits by weight.
    a, b = mimic_pair("share_loop", weights=(3, 1),
                      env={**sched_daemon.env, "MOCK_KERNEL_NS": "1000000", "MIMIC_SECONDS": "2.0"})
    ga, gb = (x.trace["cuscope"]["sched"]["gpu_ns_charged"] for x in (a, b))
    assert 0.65 <= ga / (ga + gb) <= 0.85
    assert a.trace["cuscope"]["sched"]["waits"] > 0 and b.trace["cuscope"]["sched"]["waits"] > 0

def test_alone_is_never_held_back(sched_daemon, mimic):
    r = mimic("share_loop", "shim",
              env={**sched_daemon.env, "CUSCOPE_SHARE": "10", "MOCK_KERNEL_NS": "1000000", "MIMIC_SECONDS": "0.5"})
    assert r.trace["cuscope"]["sched"]["waits"] == 0

def test_daemon_stopped_jobs_continue(sched_daemon, mimic_pair):  # Review Focus 2
    # Two jobs so both are throttled (neither is solo); the daemon stops mid-run.
    threading.Timer(0.5, sched_daemon.stop).start()
    a, b = mimic_pair("share_loop", weights=(1, 1),
                      env={**sched_daemon.env, "MOCK_KERNEL_NS": "1000000", "MIMIC_SECONDS": "1.5"})
    for r in (a, b):
        assert r.code == 0 and r.stderr.count("scheduler stopped") == 1

def test_capture_launches_not_charged(sched_daemon, mimic):  # Review Focus 5
    r = mimic("capture", "shim", env={**sched_daemon.env, "CUSCOPE_SHARE": "50"})
    assert r.code == 0 and r.trace["cuscope"]["sched"]["gpu_ns_charged"] == 0
```

(`mimic_pair` starts two mimics together with `CUSCOPE_SHARE` set to each weight and returns both results.)

- [ ] **Step 2: Run to verify they fail.** Expected: compile errors, then the split and wait assertions failing (no charging).
- [ ] **Step 3: [You] Implement `SchedClient::before_launch`** as specified, and wire it into the three launch wrappers.
- [ ] **Step 4: Run to verify they pass.** Run the full suite and the TSan unit build. Expected: PASS.
- [ ] **Step 5: Commit** `feat: kernel and graph launches wait when a job is over its share`

---

### Task 6: Correction from measured GPU time

**Files:**
- Modify: `src/sched_core.cpp` (`SchedClient::on_span`), `src/sched_client.cpp`, `src/gputimer.cpp` (call `sched_on_span` for each resolved launch or graph span, outside its locks)
- Test: `tests/unit/test_sched_core.cpp`, `tests/mock/test_sched.py`

**Interfaces:**
- Consumes: `GpuSpan::charged_ns` (Task 5).
- Produces: `on_span(key, charged_ns, actual_ns)`:
  - adds `charged − actual` to the balance;
  - adds `actual − charged` to `charged_ns`, so the slot's `charged_ns` ends up as measured GPU time;
  - calls `estimator.observe(key, actual)`.
  - A span with `charged_ns == 0` (not shared, or captured) is ignored.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(SchedClient, CorrectionMakesTheChargeExact) {
  SchedShm shm{}; sched_shm_init(&shm, 1, fake_now());
  SchedClient c(&shm, 4242, 50, fake_now, fake_sleep);
  uint64_t est = c.before_launch(7);
  c.on_span(7, est, 1'000'000);
  EXPECT_EQ(shm.slots[0].balance_ns.load(), kCapNs - 1'000'000);
  EXPECT_EQ(shm.slots[0].charged_ns.load(), 1'000'000u);
  EXPECT_EQ(c.before_launch(7), 1'000'000u);  // the estimate learned
}
```

```python
def test_charged_time_matches_measured_kernels(sched_daemon, mimic):
    r = mimic("share_loop", "shim",
              env={**sched_daemon.env, "CUSCOPE_SHARE": "50", "MOCK_KERNEL_NS": "1000000", "MIMIC_SECONDS": "0.5"})
    launches = int(next(l for l in r.stdout.splitlines() if l.startswith("LAUNCHES ")).split()[1])
    assert r.trace["cuscope"]["sched"]["gpu_ns_charged"] == launches * 1_000_000
```

- [ ] **Step 2: Run to verify they fail.** Expected: the charge stays at 50 µs per launch.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run to verify they pass.** Expected: PASS; full suite green.
- [ ] **Step 5: Commit** `feat: correct each launch's charge with its measured GPU time`

---

### Task 7: Throttled spans and the "Sharing" summary

**Files:**
- Modify: `src/trace_writer.cpp`, `src/report.cpp`, `cli/cuscope.py`, `tests/cli/fixtures/sample_trace.json` (one line: a `sched` object)
- Test: `tests/unit/test_trace_writer.cpp`, `tests/unit/test_report.cpp`, `tests/cli/test_cli.py`, `tests/mock/test_sched.py`

**Interfaces:**
- Consumes: `RecType::Throttled` (Task 5), `Snapshot::sched` (Task 4).
- Produces:
  - `Throttled` records become `{"name":"throttled","cat":"sched","ph":"X",…}` on the thread's track.
  - The summary section, after "Memory cap", using the existing `time_text`:
    - `Sharing: weight 80, 4.210 s GPU time; 1534 launches waited, 812.000 ms in total`
    - or `Sharing: weight 80, not registered (no scheduler running)`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(TraceWriter, ThrottledSpans) {
  Snapshot s;
  s.sched = {true, true, 80, 4'210'000'000, 1, 812'000};
  Record t = make_record(RecType::Throttled, H_cuLaunchKernel, 5000, 812'000);
  s.records = {t};
  json j = write(s);
  EXPECT_EQ(j["cuscope"]["sched"]["weight"], 80);
  int throttled = 0;
  for (const json& e : j["traceEvents"]) throttled += e["name"] == "throttled" && e["cat"] == "sched";
  EXPECT_EQ(throttled, 1);
}

TEST(Report, SharingSection) {
  Snapshot s = sample();
  s.sched = {true, true, 80, 4'210'000'000, 1534, 812'000'000};
  EXPECT_TRUE(contains(render(s), "Sharing: weight 80, 4.210 s GPU time; 1534 launches waited, 812.000 ms in total"));
  s.sched = {true, false, 80, 0, 0, 0, "no scheduler running"};
  EXPECT_TRUE(contains(render(s), "Sharing: weight 80, not registered (no scheduler running)"));
  EXPECT_FALSE(contains(render(sample()), "Sharing"));
}
```

CLI: `test_summary_sections` expects the fixture's `Sharing:` line after `Memory cap:`. Mock: `test_two_jobs_split_by_weight` also asserts each trace has at least one `throttled` event.

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** `feat: throttled spans and the Sharing summary`

---

### Task 8: GPU benchmark on the 3060

**Files:**
- Create: `tests/cuda/share_load.cu` (CMake `share_load`), `tools/bench_share.py`, `docs/sharing.md`, `docs/sharing.json`
- Test: `tests/gpu/test_sharing.py` (a short smoke version: 3 s per configuration, 50/50 within ±10)

**Interfaces:**
- `share_load <seconds> <kernel_ms> <batch>`: launches `batch` spin kernels of `kernel_ms`, synchronises, records the batch latency, and repeats until `seconds` pass. It prints one JSON line `{"kernels": N, "p50_ms": …, "p99_ms": …}`.
- `bench_share.py [--seconds 20] [--kernel-ms 2] [--batch 10]` starts the daemon where needed and runs four configurations, each with jobs A and B started together:
  1. no scheduler;
  2. scheduler with A alone;
  3. 50/50;
  4. 80/20.

  For each job it reports kernels/s, share of the pair's GPU time (from each trace's `sched.gpu_ns_charged`, or the kernel GPU totals when not sharing), and p50/p99 batch latency. It checks the spec §2 criteria (80±5 / 20±5, 50±5, solo ≥ 95% of unscheduled, fail-open within 5%), prints a table, writes `docs/sharing.json`, and exits 1 if a criterion fails.

- [ ] **Step 1: Write the smoke test** (`tests/gpu/test_sharing.py`: 50/50 for 3 s, both shares in 40–60%), and watch it fail (no `share_load`).
- [ ] **Step 2: Implement** `share_load.cu` and `bench_share.py`; make the smoke test pass.
- [ ] **Step 3: Run the full benchmark** (`python3 tools/bench_share.py`). Expected: all criteria met. If not, follow spec §8 (record the measured split; tune the tick or cap only with a ledgered ruling).
- [ ] **Step 4: Write `docs/sharing.md`** covering:
  - setup;
  - the four-configuration table;
  - what the no-scheduler split was (the Windows driver's own time-slicing);
  - latency;
  - what sharing cannot do (running kernels, queued work, copies).
- [ ] **Step 5: Commit** `docs: compute sharing benchmark on the RTX 3060`

---

### Task 9: Overhead, docs and release

**Files:**
- Modify: `bench/run_bench.py` (mode `full+share`: starts `cuscope-sched`, sets `CUSCOPE_SHARE=100`, a job alone), `docs/overhead.md`, `README.md` (quickstart lines for `cuscope sched` and `--share`; a "Compute sharing" section linking `docs/sharing.md`; limitations; roadmap), `docs/DESIGN.md` (§ compute sharing), `CMakeLists.txt` (`VERSION 0.3.0`), `.github/workflows/ci.yml` only if the new binary needs a dependency (`rt` is in glibc)

- [ ] **Step 1: Rerun the microbenchmarks** with `full+share` and update the tables.
- [ ] **Step 2: Docs** as listed.
- [ ] **Step 3: Full verification:** `ctest --test-dir build --output-on-failure` (with GPU), the ASan and TSan unit builds, `tools/census.py check`.
- [ ] **Step 4: Commit, push, check CI.**
- [ ] **Step 5: Final whole-branch review**, fix pass, then the user tags `v0.3.0`.
