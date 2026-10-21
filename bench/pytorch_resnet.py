#!/usr/bin/env python3
"""ResNet-18 training on random data: the PyTorch workload cuscope is checked against.

usage: pytorch_resnet.py [--steps N] [--warmup N] [--batch N] [--json OUT] [--profile N]

Prints the median step time and, with --json, PyTorch's own memory figures.
With --profile N, also runs N extra steps under torch.profiler and reports
kernel launches and kernel time per step.
"""

import os

# Expandable segments allocate through the VMM APIs (cuMemCreate/cuMemMap),
# which cuscope does not track (a spec non-goal). Must be set before CUDA starts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torchvision  # noqa: E402


def train_step(model, opt, loss_fn, x, y):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()


def profile_steps(model, opt, loss_fn, x, y, steps: int) -> dict:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            train_step(model, opt, loss_fn, x, y)
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
               and "memcpy" not in e.name.lower() and "memset" not in e.name.lower()]
    return {"kernels_per_step": len(kernels) / steps,
            "kernel_us_per_step": sum(e.device_time for e in kernels) / steps}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--json")
    p.add_argument("--profile", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(0)
    model = torchvision.models.resnet18(num_classes=1000).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(args.batch, 3, 224, 224, device="cuda")
    y = torch.randint(0, 1000, (args.batch,), device="cuda")

    for _ in range(args.warmup):
        train_step(model, opt, loss_fn, x, y)
    torch.cuda.synchronize()

    times = []
    for _ in range(args.steps):
        start = time.perf_counter()
        train_step(model, opt, loss_fn, x, y)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    result = {
        "steps": args.steps, "warmup": args.warmup, "batch": args.batch,
        "step_ms_median": statistics.median(times),
        "max_memory_reserved": torch.cuda.max_memory_reserved(),
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
        "memory_reserved_at_end": torch.cuda.memory_reserved(),
        "caching_disabled": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1",
    }
    if args.profile:
        try:
            result["profiler"] = profile_steps(model, opt, loss_fn, x, y, args.profile)
        except Exception as e:  # torch.profiler needs CUPTI, which WSL may not provide
            result["profiler_error"] = f"{type(e).__name__}: {e}"

    print(f"resnet18 batch {args.batch}: median step {result['step_ms_median']:.1f} ms, "
          f"max reserved {result['max_memory_reserved'] / 2**20:.1f} MiB")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
