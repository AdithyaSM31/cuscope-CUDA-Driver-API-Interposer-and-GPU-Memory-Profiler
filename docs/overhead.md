# Overhead

Produced by `python3 bench/run_bench.py` on an RTX 3060 Laptop GPU under WSL2 (CUDA 13.0,
PyTorch 2.14). Medians of 7 runs per microbenchmark and 3 per
PyTorch configuration; wall-clock time on the calling thread.

## Per call (ns)

| Call | off | `alloc` | `full` | `full`, sample every 10 |
|---|---|---|---|---|
| empty kernel launch | 8022 | 7820 (-2.5%) | 20428 (+154.6%) | 10107 (+26.0%) |
| cudaMalloc + cudaFree (4 KiB) | 367889 | 363931 (-1.1%) | 371574 (+1.0%) | 366411 (-0.4%) |
| cudaMemcpy 4 KiB host to device | 20164 | 21299 (+5.6%) | 28818 (+42.9%) | 29914 (+48.4%) |

## PyTorch ResNet-18 training step (batch 32, median step time)

| off | `alloc` | `full` | `full`, sample every 10 |
|---|---|---|---|
| 60.65 ms | 60.58 ms (-0.1%) | 62.47 ms (+3.0%) | 60.81 ms (+0.3%) |

## Notes

- `alloc` mode records every call and tracks memory but creates no CUDA events. Its cost is a
  clock read and a 64-byte record per call, which is within run-to-run noise here.
- `full` mode adds four driver calls around each launch or copy: a stream-capture check, a
  context lookup and two event records. On WSL each driver call costs a few microseconds, which
  is the extra 10-16 us per call above. Kernels that run longer than that hide it, hence the
  small PyTorch figure.
- `CUSCOPE_SAMPLE_EVERY=N` times 1 in N kernel launches per thread. Copies are always timed:
  bandwidth divides all copied bytes by GPU time, so a sampled copy would overstate it.
- These are medians, but microsecond-scale calls on WSL still vary between runs by several
  percent (see the alloc-mode copy figure, which can come out below `off`). Treat differences
  under about 5% as noise.

