# Compute sharing: two jobs on one GPU

`cuscope sched` runs a small daemon; jobs started with `cuscope run --share N` register with
it and get GPU time by weight when they compete. A share is a guaranteed minimum: a job
alone gets the whole GPU. cuscope enforces it at the kernel launch: a job that has used more
than its share waits before its next launch until the daemon has given it more time. This
page measures what that does on the RTX 3060 Laptop GPU.

## Setup

- RTX 3060 Laptop GPU (6 GiB), Windows driver 617.14, WSL2 Ubuntu 22.04, CUDA 13.0.
- The load is `tests/cuda/share_load`: batches of 10 kernels, each batch waited for. Each
  kernel fills the GPU (6 blocks of 256 threads per SM) with a multiply-add loop that takes
  2.0 ms alone, calibrated once (244,140 iterations) and used by every job, so both jobs
  run the same kernel.
- Each configuration runs 20 s after a 15 s warm-up. The laptop GPU's clock drops from its
  boost under sustained load; the warm-up makes every configuration run at the clock it
  settles at. That clock differs between sessions (an earlier session settled about 30%
  lower), so compare rows within one run, not across runs.
- Both jobs run under cuscope in `full` mode.

```
python3 tools/bench_share.py          # writes docs/sharing.json
```

Raw results: [sharing.json](sharing.json).

## Results

**Work** is each job's part of the kernels the pair completed: since both run the same
kernel, it is their real split of the GPU. **Measured** is each job's part of the GPU time
cuscope measured for its kernels, which is what the scheduler divides. **Busy** is a job's
measured GPU time over wall time.

| Configuration | Job | Weight | Kernels/s | Busy | Measured | Work | Batch p50 | Batch p99 |
|---|---|---|---|---|---|---|---|---|
| alone, no sharing | A | – | 497.0 | 98.9% | – | – | 19.9 ms | 29.2 ms |
| alone, `--share`, no daemon | A | 50 | 496.5 | 99.0% | – | – | 20.1 ms | 21.3 ms |
| alone, `--share`, daemon | A | 50 | 497.0 | 99.0% | – | – | 20.1 ms | 20.5 ms |
| pair, no scheduler | A | – | 185.5 | 99.1% | 50.0% | 50.0% | 53.9 ms | 54.7 ms |
| | B | – | 185.5 | 99.1% | 50.0% | 50.0% | 53.9 ms | 54.7 ms |
| pair, 50/50 | A | 50 | 184.5 | 99.0% | 50.0% | 50.0% | 54.0 ms | 54.8 ms |
| | B | 50 | 184.5 | 99.0% | 50.0% | 50.0% | 54.4 ms | 54.9 ms |
| pair, 80/20 | A | 80 | 408.5 | 99.4% | 79.4% | **89.2%** | 23.9 ms | 27.7 ms |
| | B | 20 | 49.5 | 25.8% | 20.6% | **10.8%** | 211.0 ms | 217.3 ms |

What the table shows:

- **A job alone is never held back.** With the daemon, without it (`--share` falls back to
  running unthrottled), and without sharing at all, a job alone runs at the same speed and
  keeps the GPU 99% busy.
- **Sharing does not cost throughput.** At 50/50 the pair completes 369 kernels/s, as many as
  with no scheduler (371). At 80/20 it completes 458.
- **The 80% job gets most of the GPU.** At 80/20 job A runs 408.5 kernels/s, 2.2 times what
  it gets with no scheduler, and its batches take 23.9 ms instead of 53.9 ms.
- **Measured GPU time splits by weight** (79.4/20.6 and 50.0/50.0).
- **Real work at 80/20 is 89.2/10.8, not 80/20.** See below.

## Why 80/20 becomes 90/10 in work

The driver time-slices the GPU between processes. (This was measured on WSL only; native
Linux without MPS also time-slices processes, but it was not measured here.) A kernel's
measured time runs from
when it starts to when it finishes, and that includes any time slices the other job ran in
between. With no scheduler both jobs measure 99% busy, 198% between them: each job's
measured time includes the other's.

The scheduler charges each job its measured time. At 80/20 the light job's kernels mostly run
while the heavy job also has work on the GPU, so they measure about 5.2 ms each against
2.0 ms alone: the light job is charged about 2.6 times its real GPU use and gets about half the
work its weight asks for. At 50/50 the two jobs are inflated alike and the work split is
exact.

Two other ways of charging were tried and rejected:
- **Charging each kernel its uncontended cost** (its shortest measured time) gave 50/50 work
  at both 80/20 and 50/50. Time-slicing itself costs about a quarter of the GPU here (the
  pair completes 371 kernels/s against 497 for one job alone), so jobs charged only their
  uncontended cost never run short of credit, and the driver's own even split decides.
- **Exclusive scheduling**, where jobs take turns so only one has kernels on the GPU at a
  time, would make measured time equal real work. It is on the roadmap.

## Keeping the GPU busy while jobs wait

Because both jobs' measured times are inflated, both can run out of credit at once. Then
nobody may launch and the GPU idles. The daemon prevents that: when every active job is
waiting, it grants credit by weight until the least indebted job can launch. The split is
unchanged and the scheduler never holds the GPU idle. Without this, an early 50/50 run
completed about a quarter fewer kernels than with no scheduler.

## What sharing cannot do

- **It delays launches; it cannot pre-empt.** A kernel already running, or already queued,
  finishes regardless. One long kernel holds the GPU until it is done.
- **Copies are not charged.** They use the copy engines, which run alongside kernels.
- **Time-slicing inflates measured time** (above): unequal weights come out more unequal in
  real work.
- **The scheduler needs GPU timing** (`full` mode) and one daemon per user; a job that cannot
  reach the daemon runs unthrottled and says so once.
