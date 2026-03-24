#!/usr/bin/env python
import argparse
import time
from typing import List

import torch
import torch.distributed as dist
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("all_to_all_single benchmark")
    parser.add_argument(
        "--backend", type=str, default="nccl", choices=["nccl", "gloo"],
        help="Process group backend (default: nccl)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Tensor dtype (default: float16)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Timed iterations (default: 50)",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    args = parse_args()

    dist.init_process_group(backend=args.backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())

    dtype = get_dtype(args.dtype)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    # Tensor sizes per rank (bytes). Adjust as desired.
    tensor_shapes = [
        (2, 4, 16, 2048),
        (2, 4, 16, 4096),
        (2, 4, 64, 14336),
        (4, 8, 64, 2048),
        (4, 8, 64, 4096),
        (4, 8, 64, 14336),
        (8, 8, 64, 2048),
        (8, 8, 64, 4096),
        (8, 8, 64, 14336),
    ]

    if rank == 0:
        print(
            f"Running all_to_all_single benchmark with world_size={world_size}, "
            f"dtype={dtype}, warmup={args.warmup}, iters={args.iters}"
        )

    for shape in tensor_shapes:
        # Number of elements per rank
        numel = np.prod(shape)

        # all_to_all_single with equal splits requires numel divisible by world_size
        if numel % world_size != 0:
            if rank == 0:
                print(
                    f"Skipping shape {shape}: "
                    f"numel={numel} not divisible by world_size={world_size}"
                )
            continue

        in_tensor = torch.empty(shape, device=device, dtype=dtype)
        out_tensor = torch.empty_like(in_tensor)

        # Warmup
        dist.barrier()
        for _ in range(args.warmup):
            dist.all_to_all_single(out_tensor, in_tensor)
        dist.barrier()

        # Timed iterations
        dist.barrier()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(args.iters):
            dist.all_to_all_single(out_tensor, in_tensor)
        end_event.record()
        torch.cuda.synchronize(device)

        total_ms = start_event.elapsed_time(end_event)  # milliseconds
        avg_ms = total_ms / args.iters
        avg_sec = avg_ms / 1e3
        gb = np.prod(shape) * bytes_per_elem / (1 << 30)
        gbps = gb / avg_sec

        if rank == 0:
            print(
                f"shape={shape}, "
                f"size={gb:.3f} GB per rank, "
                f"time={avg_ms:.3f} ms, "
                f"eff_bw={gbps:.2f} GB/s (per rank, one direction)"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()