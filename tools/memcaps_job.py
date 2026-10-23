#!/usr/bin/env python3
"""One training job for the memory-cap demo (tools/demo_memcaps.py).

ResNet-18 trained on random images whose resolution is drawn per step from
[--min-res, --max-res]. Varying shapes fragment PyTorch's caching allocator,
as variable-length inputs do in real training: the memory it reserves grows
well past what it actually uses at any moment, and it never gives it back
unless an allocation fails. Two such jobs on one GPU each hold far more than
they need.

Writes a JSON result:
  {"exit": "ok" | "oom", "steps_done", "max_allocated", "max_reserved", "step_ms_median"}
and exits 3 on torch.OutOfMemoryError.
"""

import argparse
import json
import random
import statistics
import sys
import time

import torch
import torchvision


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--min-res", type=int, default=128)
    p.add_argument("--max-res", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", required=True)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = torchvision.models.resnet18(num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    result = {"exit": "ok", "steps_done": 0}
    times = []
    try:
        for _ in range(args.steps):
            res = random.randrange(args.min_res, args.max_res + 1, 16)
            x = torch.randn(args.batch, 3, res, res, device=device)
            y = torch.randint(0, 10, (args.batch,), device=device)
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
            result["steps_done"] += 1
    except torch.OutOfMemoryError as e:
        result["exit"] = "oom"
        result["error"] = str(e).splitlines()[0]

    result["max_allocated"] = torch.cuda.max_memory_allocated()
    result["max_reserved"] = torch.cuda.max_memory_reserved()
    result["step_ms_median"] = statistics.median(times) if times else None
    with open(args.json, "w") as f:
        json.dump(result, f)
    return 3 if result["exit"] == "oom" else 0


if __name__ == "__main__":
    sys.exit(main())
