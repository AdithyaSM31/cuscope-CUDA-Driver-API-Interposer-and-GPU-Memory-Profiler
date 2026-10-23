# Memory caps: two PyTorch jobs on one GPU

`CUSCOPE_MEM_LIMIT` (or `cuscope run --mem-limit`) gives a process a GPU memory budget.
Allocations that would pass it fail with `CUDA_ERROR_OUT_OF_MEMORY` without reaching the
driver, and `cuMemGetInfo` reports a GPU the size of the cap. This page shows what that does
for two training jobs sharing the RTX 3060 Laptop GPU (6 GiB).

## Setup

- RTX 3060 Laptop GPU, Windows driver 617.14, WSL2 Ubuntu 22.04, CUDA 13.0, PyTorch
  2.14.0+cu130.
- Each job is `tools/memcaps_job.py`: ResNet-18 trained on random images whose resolution is
  drawn per step, 60 steps. Varying shapes fragment PyTorch's caching allocator, as
  variable-length inputs do in real training, so the memory a job *reserves* grows well past
  what it *uses* at any moment. PyTorch gives cached memory back only when one of its own
  allocations fails.
  - job A: batch 32, resolution 128 to 320
  - job B: batch 48, resolution 128 to 352, started 8 s after A
- Both runs go through `cuscope run --mode alloc`, with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (expandable segments use the virtual
  memory APIs, which cuscope does not hook).

```
python3 tools/demo_memcaps.py --cap-a 2.25G --cap-b 3G --offset 8 --steps 60 \
    --a-args="--batch 32 --min-res 128 --max-res 320" \
    --b-args="--batch 48 --min-res 128 --max-res 352"
```

Raw results: [memcaps.json](memcaps.json). The seeds are fixed; two runs gave the same numbers.

## Results

| Run | Job | Exit | Steps | Max allocated | Max reserved | Median step | Denials |
|---|---|---|---|---|---|---|---|
| uncapped | A | ok | 60 | 1.49 GiB | 3.91 GiB | 60.1 ms | – |
| uncapped | B | ok | 60 | 2.60 GiB | 4.78 GiB | 102.9 ms | – |
| capped 2.25G | A | ok | 60 | 1.49 GiB | 2.21 GiB | 60.2 ms | 2 |
| capped 3G | B | ok | 60 | 2.61 GiB | 2.97 GiB | 103.3 ms | 4 |

- **Uncapped, the pair reserves 8.70 GiB on a 6 GiB GPU.** Neither job needs that much: they
  use 1.49 and 2.60 GiB at most. The rest is PyTorch's cache, which neither job will return
  while its own allocations succeed.
- **Capped, the pair stays at 5.18 GiB.** When a job reaches its cap, the denied allocation
  makes PyTorch free its cached blocks and retry, which succeeds: each job keeps the memory it
  actually uses and drops the rest. The capped jobs ran as fast as the uncapped ones.
- **PyTorch sees the cap.** Its own out-of-memory warnings during the capped run report
  `total: 2415919104` bytes (2.25 GiB) for job A, read through cuscope's `cuMemGetInfo`.

## Why the uncapped pair did not crash here

On native Linux, the GPU's memory is all there is: once job A's cache holds most of it, job
B's allocations fail, and B, which cannot make A release anything, dies with
`torch.OutOfMemoryError`. Under WSL2 the Windows driver (WDDM) pages GPU memory out to
shared system RAM instead. On this machine one process allocated **12.5 GiB** on the 6 GiB
card before getting `CUDA_ERROR_OUT_OF_MEMORY`. So the uncapped pair survived on
oversubscribed memory: idle cache blocks were paged out, which cost nothing visible here
because nothing touched them.

The cap is what keeps each job inside a budget you chose, on either platform. On native
Linux that is the difference between finishing and crashing; on WSL2 it keeps the jobs inside
physical GPU memory instead of relying on the driver to page.

## What the cap does not cover

- **Context and driver memory.** A fresh PyTorch process on this machine sees 5.0 of 6.0 GiB
  free: about 1 GiB is held by the Windows desktop and the CUDA context, none of it visible to
  cuscope. Under WSL2, `cuMemGetInfo` in one process does not reflect another process's
  allocations and `nvidia-smi` shows no per-process usage, so the per-process context cost
  could not be measured separately. The cap bounds what the program allocates, not the
  process's total VRAM.
- **Stream-ordered pools** can keep freed memory reserved; the cap counts live allocations.
- **The virtual memory APIs** (`cuMemCreate`, `cuMemMap`) are not hooked, so
  `expandable_segments:True` bypasses the cap.
- **Pitched allocations** are charged their padded size after the call, so the cap can be
  passed by one allocation's row padding.
- **One cap per process.** Each process (including each forked worker) has its own budget;
  there is no shared budget for a group of processes.
