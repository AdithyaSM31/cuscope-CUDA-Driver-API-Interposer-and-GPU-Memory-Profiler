# Validation: cuscope vs PyTorch

Produced by `python3 tools/validate_pytorch.py` (ResNet-18, random 3x224x224 data, SGD with momentum,
batch 32, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`). RTX 3060 Laptop GPU, WSL2,
CUDA 13.0, PyTorch 2.14.0+cu130.

## Peak device memory (run A: 5 warmup + 50 steps)

| Measure | Bytes | MiB |
|---|---|---|
| `torch.cuda.max_memory_reserved()` | 1256194048 | 1198.0 MiB |
| `torch.cuda.max_memory_allocated()` | 861603840 | 821.7 MiB |
| cuscope peak, all device-resident kinds | 1273246984 | 1214.3 MiB |
| cuscope peak, `cuMemAlloc` (device) only | 1273246984 | 1214.3 MiB |
| **gap** (cuscope total - reserved) | 17052936 | 16.3 MiB |

## Where the device memory came from (allocations live at exit, by allocating library)

PyTorch's caching allocator never returns its blocks before exit, so every allocation it made is
still live at exit; cuscope attributes each one to the first library in its call stack outside
the CUDA runtime.

| Library | Allocations | Bytes | MiB |
|---|---|---|---|
| `libc10_cuda.so` | 46 | 1256194048 | 1198.0 MiB |
| `libcublas.so` | 6 | 17041408 | 16.3 MiB |
| `libcudnn_graph.so` | 4 | 11528 | 0.0 MiB |

Caching allocator (`libc10_cuda.so`): 1198.0 MiB (matches max_memory_reserved 1198.0 MiB). Everything else: 16.3 MiB.

## Caching allocator disabled (run C: `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, 10 steps)

| Measure | Value |
|---|---|
| cuscope allocations / frees | 4175 / 4161 |
| cuscope peak device memory | 834.0 MiB |
| `max_memory_allocated()` | 0.0 MiB |
| allocations with caching on (run A, 55 steps) | 58 |

## Kernels

| Measure | cuscope | torch.profiler |
|---|---|---|
| kernel launches per step | 335.0 | 335.0 |
| GPU kernel time per step | 60.72 ms | 59.16 ms |
| completeness (run A) | dropped 0, unresolved 0, untimed 0 | |

## Gap analysis

**Peak memory: the 16.3 MiB gap is cuBLAS and cuDNN, accounted for to the byte.** cuscope's
device peak is 1,273,246,984 bytes. PyTorch's `max_memory_reserved()` is 1,256,194,048. The
allocations cuscope attributes to PyTorch's caching allocator (`libc10_cuda.so`) add up to
exactly that second number, so the two agree on everything PyTorch manages. The difference,
17,052,936 bytes, is memory that libraries allocate for themselves, outside PyTorch's
allocator, so PyTorch's counters cannot see it:

- **cuBLAS, 17,041,408 bytes.** Each `cublasCreate_v2` allocates 1 KiB, 128 KiB and 8 MiB
  internally (8,520,704 bytes per handle). The training step creates two handles. cuscope's
  stacks and thread ids show one created on the main thread by the forward pass
  (`at::cuda::blas::gemm_and_bias`) and one on a second thread by the backward pass
  (`gemm_internal_cublas`), consistent with PyTorch keeping one cuBLAS handle per thread.
  2 × 8,520,704 = 17,041,408.
- **cuDNN, 11,528 bytes.** Small allocations made inside `cudnnCreate` when PyTorch creates
  its cuDNN handles (`at::native::getCudnnHandle`).

Both are fixed costs per handle: they do not grow with batch size or step count. In practice,
`max_memory_reserved()` understates the GPU memory the process holds by about 8.1 MiB for each
thread that runs cuBLAS work.

**Caching allocator disabled: cuscope sees every tensor, PyTorch reports nothing.** With
`PYTORCH_NO_CUDA_MEMORY_CACHING=1`, PyTorch sends each tensor straight to
`cudaMalloc`/`cudaFree`. cuscope records 4,175 allocations and 4,161 frees in 10 steps, against
58 allocations in 55 steps with caching on. The peak, 834 MiB, is close to the cached run's
`max_memory_allocated()` of 822 MiB, as per-tensor allocation should be.

PyTorch's own counters read 0 in this mode: `memory_allocated()`, `max_memory_allocated()`
and `memory_reserved()` all stay at 0 even with a 64 MiB tensor allocated (checked directly).
They belong to the caching allocator, which is bypassed. So this run has no PyTorch figure to
compare against, and that is not a disagreement.

The 14 allocations still live at exit are not model tensors, which Python frees during
interpreter shutdown, before cuscope's exit handler runs. They are the libraries' handle
memory (6 allocations inside cuBLAS, 6 inside cuDNN) plus 2 blocks made through PyTorch
totalling 17,039,360 bytes. That is 2 × (8 MiB + 128 KiB), consistent with PyTorch's default
cuBLAS workspace for each of the two handles, which PyTorch keeps cached past shutdown.

**Kernel launches agree exactly.** cuscope counts 335 launches per training step, and
`torch.profiler` counts 335 kernels per step. cuscope's figure is the difference between a
50-step and a 10-step run, which removes model setup and warmup.

**GPU kernel time is 2.6% higher than the profiler's, because the two measure different
things.** `torch.profiler` (CUPTI) reports how long each kernel executes. cuscope records a
CUDA event before and after each launch; a span runs from the start event completing to the end
event completing. On a busy GPU the start event completes when the previous work ends, so the
span is exactly the kernel. On an idle GPU the start event completes at once, and the span also
covers the wait until the CPU submits the kernel.

This run often left the GPU idle. Everything ran on one stream, and only 1,405 of 7,553 spans
began exactly where the previous one ended; the rest followed a gap in which the GPU had no
work. The 1.56 ms difference per step, spread over its 335 kernels, is about 4.7 µs per kernel,
the order of one kernel submission on WSL. Event timing is the spec's method because it needs
no profiling interface. The cost is that, when the GPU is starved, it measures slightly more
than kernel execution.

