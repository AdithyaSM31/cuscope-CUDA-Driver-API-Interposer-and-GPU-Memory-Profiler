# Overhead

Produced by `python3 bench/run_bench.py` on an RTX 3060 Laptop GPU under WSL2 (CUDA 13.0,
PyTorch 2.14). Medians of 7 runs per microbenchmark and 3 per
PyTorch configuration; wall-clock time on the calling thread.

## Per call (ns)

| Call | off | `alloc` | `full` | `full`, sample every 10 | `full`, 4 GiB cap |
|---|---|---|---|---|---|
| empty kernel launch | 8144 | 6723 (-17.4%) | 19085 (+134.3%) | 8597 (+5.6%) | 19189 (+135.6%) |
| cudaMalloc + cudaFree (4 KiB) | 375846 | 358248 (-4.7%) | 354078 (-5.8%) | 336852 (-10.4%) | 335530 (-10.7%) |
| cudaMemcpy 4 KiB host to device | 34991 | 35374 (+1.1%) | 41709 (+19.2%) | 40755 (+16.5%) | 39075 (+11.7%) |

## PyTorch ResNet-18 training step (batch 32, median step time)

| off | `alloc` | `full` | `full`, sample every 10 | `full`, 4 GiB cap |
|---|---|---|---|---|
| 60.61 ms | 60.72 ms (+0.2%) | 62.15 ms (+2.6%) | 60.97 ms (+0.6%) | 62.15 ms (+2.5%) |

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
- These are medians, but microsecond-scale calls on WSL still vary between runs by several
  percent (see the alloc-mode copy figure, which can come out below `off`). Treat differences
  under about 5% as noise.

