# Overhead

Produced by `python3 bench/run_bench.py` on an RTX 3060 Laptop GPU under WSL2 (CUDA 13.0,
PyTorch 2.14). Medians of 7 runs per microbenchmark and 3 per
PyTorch configuration; wall-clock time on the calling thread.

## Per call (ns)

| Call | off | `alloc` | `full` | `full`, sample every 10 | `full`, 4 GiB cap | `full`, `--share` (alone) |
|---|---|---|---|---|---|---|
| empty kernel launch | 8898 | 9393 (+5.6%) | 24201 (+172.0%) | 9914 (+11.4%) | 26460 (+197.4%) | 26728 (+200.4%) |
| cudaMalloc + cudaFree (4 KiB) | 443998 | 425768 (-4.1%) | 393481 (-11.4%) | 396699 (-10.7%) | 410491 (-7.5%) | 410289 (-7.6%) |
| cudaMemcpy 4 KiB host to device | 22909 | 22141 (-3.4%) | 32530 (+42.0%) | 31143 (+35.9%) | 34264 (+49.6%) | 32576 (+42.2%) |

## PyTorch ResNet-18 training step (batch 32, median step time)

| off | `alloc` | `full` | `full`, sample every 10 | `full`, 4 GiB cap | `full`, `--share` (alone) |
|---|---|---|---|---|---|
| 84.53 ms | 84.77 ms (+0.3%) | 85.86 ms (+1.6%) | 85.02 ms (+0.6%) | 86.02 ms (+1.8%) | 86.35 ms (+2.2%) |

## Notes

- `alloc` mode records every call and tracks memory but creates no CUDA events. Its cost is a
  clock read and a 64-byte record per call, which is within run-to-run noise here.
- `full` mode adds four driver calls around each launch or copy: a stream-capture check, a
  context lookup and two event records. On WSL each driver call costs a few microseconds, which
  is the extra 10-16 us per call above. Kernels that run longer than that hide it, hence the
  small PyTorch figure.
- `CUSCOPE_SAMPLE_EVERY=N` times 1 in N kernel launches per thread. Copies are always timed,
  so every copy contributes to the bandwidth figures.
- `full` with a 4 GiB cap (`CUSCOPE_MEM_LIMIT=4G`, never reached here) adds one atomic
  reservation per allocation and one release per free on top of `full`.
- `full` with `--share` runs alone with the scheduler daemon: each kernel launch charges an
  estimate to the job's balance in shared memory (a capture check, a clock read, a lookup of
  the kernel's running average under a lock, and a few atomics); a job alone never waits.
- These are medians, but microsecond-scale calls on WSL still vary between runs by several
  percent (see the alloc-mode copy figure, which can come out below `off`). Treat differences
  under about 5% as noise.

