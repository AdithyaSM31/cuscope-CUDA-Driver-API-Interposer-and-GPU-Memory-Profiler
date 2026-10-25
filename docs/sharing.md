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
  2 ms alone, calibrated once (247,279 iterations) and used by every job, so both jobs run
  the same kernel.
- Each configuration runs 20 s after a 15 s warm-up. The laptop GPU's clock drops from its
  boost under sustained load; the warm-up makes every configuration run at the clock it
  settles at.
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
| alone, no sharing | A | – | 358.5 | 99.0% | – | – | 27.9 ms | 28.9 ms |
| alone, `--share`, no daemon | A | 50 | 357.5 | 99.1% | – | – | 28.0 ms | 29.4 ms |
| alone, `--share`, daemon | A | 50 | 357.5 | 99.1% | – | – | 28.0 ms | 28.8 ms |
| pair, no scheduler | A | – | 132.0 | 99.2% | 50.0% | 50.1% | 75.7 ms | 78.4 ms |
| | B | – | 131.5 | 99.2% | 50.0% | 49.9% | 75.9 ms | 78.3 ms |
| pair, 50/50 | A | 50 | 132.5 | 99.2% | 50.0% | 50.0% | 75.7 ms | 78.7 ms |
| | B | 50 | 132.5 | 99.2% | 50.0% | 50.0% | 75.7 ms | 79.1 ms |
| pair, 80/20 | A | 80 | 297.5 | 99.0% | 79.0% | **89.6%** | 32.9 ms | 38.6 ms |
| | B | 20 | 34.5 | 26.3% | 21.0% | **10.4%** | 300.9 ms | 303.7 ms |

What the table shows:

- **A job alone is never held back.** With the daemon, without it (`--share` falls back to
  running unthrottled), and without sharing at all, a job alone runs at the same speed and
  keeps the GPU 99% busy.
- **Sharing does not cost throughput.** At 50/50 the pair completes 265 kernels/s, as many as
  with no scheduler (263.5). At 80/20 it completes 332.
- **The 80% job gets most of the GPU.** At 80/20 job A runs 297.5 kernels/s, 2.3 times what
  it gets with no scheduler, and its batches take 32.9 ms instead of 75.7 ms.
- **Measured GPU time splits exactly by weight** (79.0/21.0 and 50.0/50.0).
- **Real work at 80/20 is 89.6/10.4, not 80/20.** See below.

## Why 80/20 becomes 90/10 in work

The Windows driver time-slices the GPU between processes. A kernel's measured time runs from
when it starts to when it finishes, and that includes any time slices the other job ran in
between. With no scheduler both jobs measure 99% busy, 198% between them: each job's
measured time includes the other's.

The scheduler charges each job its measured time. At 80/20 the light job's kernels mostly run
while the heavy job also has work on the GPU, so they measure about 7.6 ms each against
2.8 ms alone: the light job is charged about 2.5 times its real GPU use and gets about half the
work its weight asks for. At 50/50 the two jobs are inflated alike and the work split is
exact.

Two other ways of charging were tried and rejected:
- **Charging each kernel its uncontended cost** (its shortest measured time) gave 50/50 work
  at both 80/20 and 50/50. Time-slicing itself costs about a quarter of the GPU here (the
  pair completes 263.5 kernels/s against 358.5 for one job alone), so jobs charged only their
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
