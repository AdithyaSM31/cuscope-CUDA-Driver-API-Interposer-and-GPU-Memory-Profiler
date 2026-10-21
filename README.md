# cuscope

[![ci](https://github.com/AdithyaSM31/cuscope-CUDA-Driver-API-Interposer-and-GPU-Memory-Profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/AdithyaSM31/cuscope-CUDA-Driver-API-Interposer-and-GPU-Memory-Profiler/actions/workflows/ci.yml)

cuscope records every GPU allocation, copy and kernel launch of an unmodified CUDA program
(a static or shared CUDA runtime build, or PyTorch) by sitting between the program and the
CUDA driver, and gives you a leak report, memory and timing summaries, and a timeline for
[Perfetto](https://ui.perfetto.dev).

<!-- Perfetto GIF: docs/img/perfetto.gif (multistream: four GPU tracks, flow arrows, memory counter) -->

## Quickstart

On Linux or WSL2 with an NVIDIA driver and the CUDA 13.0 toolkit (see [docs/SETUP.md](docs/SETUP.md)):

```bash
git clone https://github.com/AdithyaSM31/cuscope-CUDA-Driver-API-Interposer-and-GPU-Memory-Profiler.git cuscope && cd cuscope
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build
python3 cli/cuscope.py run --out trace.json -- ./build/tests/cuda/multistream
```

The summary goes to stderr when the program exits; open `trace.json` in
[ui.perfetto.dev](https://ui.perfetto.dev). For any other program, put it after `--`:

```bash
python3 cli/cuscope.py run --out train.json -- python3 train.py
python3 cli/cuscope.py summary train.json        # leaks and invalid frees with file:line
```

Options: `--mode alloc` (memory only, no GPU timing), `--sample-every N` (time 1 in N kernel
launches), `--stack-depth N`, `--log FILE`. The program's stdout and exit code are unchanged.

What `cuscope summary` prints for `tests/cuda/leak.cu` (the summary at exit is the same,
with `main (leak+0x8a7f)` in place of the source line):

```text
== cuscope summary (pid 31714, mode full) ==
Memory
  peak device memory  6.0 MiB  (device 6.0 MiB, managed 0 B, async 0 B; pinned host 0 B)
  allocations 3, frees 1
Leaks (2, 4.0 MiB)
  1.0 MiB device at main (leak.cu:9)
  3.0 MiB device at main (leak.cu:11)
Invalid frees: none
Top kernels: none
Copies: none
Completeness: dropped=0 unresolved=0 untimed=0
```

## How it works

<!--
Written by the author. Points to cover (details in docs/DESIGN.md):
- the three ways a program reaches libcuda: exported symbols, dlsym on the libcuda handle,
  and cuGetProcAddress by base name, version and flags
- the WSL libcuda.so.1 is a loader shim that itself calls dlsym: the libcuda caller check, and
  the libcudart.so.13 prefix bug the census found
- wrapping a pointer only when it is exactly the variant resolved at start-up (per-hook
  version in hooks.def), and how check_exports.py proves every hook's signature
- recording: per-thread ring buffers, a collector thread, CUDA events with clock alignment,
  and why an extra exit handler is needed on WSL
-->

## Validation

Checked against PyTorch's own numbers on ResNet-18 training (batch 32, 50 steps, RTX 3060
Laptop GPU, WSL2). Full tables and the gap analysis: [docs/validation.md](docs/validation.md).

| Measure | cuscope | PyTorch | |
|---|---|---|---|
| Bytes from PyTorch's caching allocator | 1,256,194,048 | `max_memory_reserved()` 1,256,194,048 | exact |
| Kernel launches per training step | 335 | `torch.profiler` 335 | exact |
| GPU kernel time per step | 60.72 ms | `torch.profiler` 59.16 ms | +2.6% |
| Total device peak | 1,273,246,984 | | +17,052,936 |

The 17,052,936-byte gap is memory PyTorch does not manage: two cuBLAS handles allocate
8,520,704 bytes each inside `cublasCreate_v2`, and cuDNN handles add 11,528 bytes. cuscope
attributes every allocation to the library that made it, from its call stack.

## Overhead

Medians on the same machine; details and notes in [docs/overhead.md](docs/overhead.md).

| | off | `alloc` | `full` | `full`, 1 in 10 launches timed |
|---|---|---|---|---|
| Empty kernel launch | 8.0 µs | −2.5% | +155% | +26% |
| `cudaMalloc` + `cudaFree` (4 KiB) | 368 µs | −1.1% | +1.0% | −0.4% |
| `cudaMemcpy` 4 KiB host to device | 20.2 µs | +5.6% | +43% | +48% |
| **PyTorch ResNet-18 training step** | 60.65 ms | **−0.1%** | **+3.0%** | **+0.3%** |

`alloc` mode costs nothing measurable on a training step. `full` mode adds four driver calls
around each launch and copy (a capture check, a context lookup and two event records), a few
microseconds each on WSL; real kernels hide most of it.

## Limitations

- **Platform.** Linux and WSL2, one GPU, CUDA 13.0 (tested with Windows driver 617.14). The
  hook table records the driver function versions of that release.
- **Not hooked** (listed by the symbol census, [docs/census/SUMMARY.md](docs/census/SUMMARY.md)):
  memsets; 2D, 3D, peer and batched copies; cooperative and host-function launches;
  `cuMemHostRegister`; memory-pool management; the virtual memory APIs (`cuMemCreate`,
  `cuMemMap`). PyTorch must run with `expandable_segments:False`, which uses those VMM APIs.
- **PyTorch's caching allocator** is seen as the segments it requests, not as tensors.
- **CUDA graphs** are one span per graph launch. Launches made while a stream is being captured
  are recorded but not timed, so no event ever enters the program's graph.
- **Primary-context release and reset** are hooked in the v1 form the CUDA runtime uses. A
  driver-API program that calls `cuDevicePrimaryCtxRelease_v2` by name bypasses those two hooks.
- **Timing.** Event spans include any time the GPU waited for the kernel to be submitted, so
  on a starved GPU they read slightly long (+2.6% against `torch.profiler` above). Times are
  `float` milliseconds from a per-context reference event: resolution is about 0.5 µs, falling
  to a few microseconds after a minute of running. Bandwidth for pageable copies is GPU time
  only and excludes the driver's CPU staging copy, so it overstates the wall-clock rate.
- **Stream-ordered frees** (`cudaFreeAsync`) count as free when called, not when the stream
  reaches them.
- **Exit.** The trace is written by an exit handler: `_exit`, `abort` and fatal signals leave
  no trace. Records stay in memory until exit (64 bytes per call). On WSL the driver shuts
  itself down before the CUDA runtime releases its context, so memory the program never freed
  is reported as leaked at exit; leaks at an explicit `cudaDeviceReset` are reported as freed
  at context teardown.
- **Caller check.** cuscope tells libcuda's own `dlsym` calls apart by the caller's return
  address; a library that tail-calls `dlsym` would be attributed to its caller.
- **Processes that never call CUDA** (shells, compilers, worker processes that inherit
  `LD_PRELOAD`) write no trace and no summary.
- **Sanitizers.** The ASan and TSan CI jobs run the unit tests only: a sanitizer runtime calls
  `dlsym` while initialising, which reaches cuscope's instrumented override too early.

## Testing

- `ctest --test-dir build`: unit tests, the mock-driver suite (no GPU: a mock libcuda in direct
  and WSL-shim modes, plus a fake `libcudart.so.13`), CLI tests, and GPU tests on the real
  driver when one is present.
- `python3 tools/check_exports.py`: proves, against the installed driver, that every hook
  matches the function its wrapper was written for.
- `python3 tools/census.py run && python3 tools/census.py check`: the Day 2 symbol census.
- `python3 tools/validate_pytorch.py` and `python3 bench/run_bench.py`: the tables above.
- CI runs the GPU-free suites on every push, plain and under ASan and TSan.

Design, decisions and findings: [docs/DESIGN.md](docs/DESIGN.md).

## Roadmap (Phase 2)

- **Per-process memory caps.** `CUSCOPE_MEM_LIMIT=2G`: allocation hooks return
  `CUDA_ERROR_OUT_OF_MEMORY` past the cap, and `cuMemGetInfo` reports the capped figures so
  PyTorch sizes its cache to them.
- **Compute sharing.** A small daemon hands out launch tokens per process by configured share,
  measured against kernel time cuscope already records.
- More of the census list: memsets, 2D/3D copies and the VMM APIs.
