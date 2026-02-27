"""
CUDA-event timing for pipeline parallel Schedule1F1B.
Measures per-rank forward and backward timings (backward not split) without using torch.profiler.
"""

import argparse
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda import Event
from torch.distributed.pipelining import (
    SplitPoint,
    Schedule1F1B,
    pipeline,
)

from model import Transformer, LLAMA_3B


def op_key(op) -> str:
    """Normalize schedule op to key 'F' | 'B'."""
    try:
        s = str(op)
    except Exception:
        s = f"{op}"
    s = s.upper()
    if "FORWARD" in s or s == "F":
        return "F"
    if "BACKWARD" in s or s == "B":
        return "B"
    return s  # fallback


class TimingTracker:
    """Collect CUDA event pairs for forward/backward timings per microbatch."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reset_iteration()

    def reset_iteration(self):
        self.fwd_pairs = []  # list of (mb, start, end)
        self.bwd_pairs = []
        self._fwd_ctr = 0
        self._bwd_ctr = 0

    def next_fwd_mb(self):
        mb = self._fwd_ctr
        self._fwd_ctr += 1
        return mb

    def next_bwd_input_mb(self):
        mb = self._bwd_ctr
        self._bwd_ctr += 1
        return mb

    # alias for clarity
    next_bwd_weight_mb = next_bwd_input_mb

    def add_fwd(self, mb: int):
        torch.cuda.set_device(self.device)
        start, end = Event(enable_timing=True), Event(enable_timing=True)
        self.fwd_pairs.append((mb, start, end))
        return start, end

    def add_bwd(self, mb: int):
        torch.cuda.set_device(self.device)
        start, end = Event(enable_timing=True), Event(enable_timing=True)
        self.bwd_pairs.append((mb, start, end))
        return start, end

    def summarize_iteration(self):
        """Return per-op ordered lists of seconds: {'F': [...], 'B': [...]}."""
        if not torch.cuda.is_available():
            return {"F": [], "B": []}
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)

        def to_list(pairs):
            return [s.elapsed_time(e) / 1000.0 for _, s, e in pairs]

        return {
            "F": to_list(self.fwd_pairs),
            "B": to_list(self.bwd_pairs),
        }


def wrap_stage_forward(stage, tracker: TimingTracker):
    """Wrap forward to record CUDA event timing."""
    orig_forward = stage.submod.forward

    def wrapped_forward(self, *args, **kwargs):
        mb = int(tracker.next_fwd_mb())
        f_start, f_end = tracker.add_fwd(mb)
        f_start.record()
        out = orig_forward(self, *args, **kwargs)
        f_end.record()
        return out

    stage.submod.forward = wrapped_forward


def wrap_stage_backward(stage, tracker: TimingTracker):
    """Wrap backward (activation/weight together in 1f1b) to time gradients."""
    orig_bwd_one = stage.__class__.backward_one_chunk

    def _normalize_mb(mb_val):
        try:
            return int(mb_val)
        except Exception:
            try:
                return int(mb_val.item())
            except Exception:
                return int(tracker.next_bwd_input_mb())

    def wrapped_bwd_one(self, *args, **kwargs):
        raw_mb = kwargs.get("bwd_chunk_id", args[0] if args else tracker.next_bwd_input_mb())
        mb = _normalize_mb(raw_mb)
        b_start, b_end = tracker.add_bwd(mb)
        b_start.record()
        result = orig_bwd_one(self, *args, **kwargs)
        b_end.record()
        return result

    stage.backward_one_chunk = wrapped_bwd_one.__get__(stage, stage.__class__)


def setup(rank, world_size):
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_pipeline(rank, world_size, mb_size, num_mbs, seq_len, model_cfg):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    setup(rank, world_size)
    dist.barrier()
    print(f"[Rank {rank}] using torch device {device}")

    model = Transformer(model_cfg).to(device)

    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_cfg.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)

    layers_per_rank = model_cfg.n_layers // world_size
    split_spec = {
        f"layers.{i * layers_per_rank}": SplitPoint.BEGINNING for i in range(1, world_size)
    }

    example_input = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    pipe = pipeline(module=model, mb_args=(example_input,), split_spec=split_spec)
    stage = pipe.build_stage(rank, device=device)

    tracker = TimingTracker(device)
    wrap_stage_forward(stage, tracker)
    wrap_stage_backward(stage, tracker)

    optimizer = torch.optim.Adam(stage.submod.parameters(), lr=1e-3)
    pipe_schedule = Schedule1F1B(stage, n_microbatches=num_mbs, loss_fn=tokenwise_loss_fn)

    x = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size * num_mbs, seq_len),
        dtype=torch.long,
        device=device,
    )
    y = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size * num_mbs, seq_len),
        dtype=torch.long,
        device=device,
    )

    warmup_steps = 2
    measure_steps = 1
    total_tokens = mb_size * num_mbs * seq_len

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        if rank != world_size - 1:
            pipe_schedule.step(x)
        else:
            _ = pipe_schedule.step(target=y, losses=[])
        optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()

    iter_times = []
    avg_accum = {"F": defaultdict(list), "B": defaultdict(list)}

    for _ in range(measure_steps):
        dist.barrier()
        torch.cuda.synchronize()

        tracker.reset_iteration()

        iter_start = Event(enable_timing=True)
        iter_end = Event(enable_timing=True)
        iter_start.record()

        optimizer.zero_grad(set_to_none=True)

        if rank != world_size - 1:
            pipe_schedule.step(x)
        else:
            losses = []
            _ = pipe_schedule.step(target=y, losses=losses)

        optimizer.step()
        torch.cuda.synchronize()
        iter_end.record()
        torch.cuda.synchronize()
        dist.barrier()

        per_op_lists = tracker.summarize_iteration()
        for key_in, times in per_op_lists.items():
            norm_key = op_key(key_in)
            for idx, t in enumerate(times):
                avg_accum[norm_key][idx].append(t)

        if rank == world_size - 1:
            iter_times.append(iter_start.elapsed_time(iter_end) / 1000.0)

    # Compute per-op per-occurrence averages across measured steps
    op_idx_avg = {
        op: {idx: sum(times) / len(times) for idx, times in idx_dict.items()}
        for op, idx_dict in avg_accum.items()
    }

    if rank == world_size - 1 and iter_times:
        avg_iter = sum(iter_times) / len(iter_times)
        tput = total_tokens / avg_iter

        print(f"[Rank {rank}] Total iter time: {sum(iter_times):.6f}s")
        print(f"[Rank {rank}] Avg iter time:   {avg_iter:.6f}s")
        print(f"[Rank {rank}] Throughput:      {tput:.2f} tokens/s")
        if 'losses' in locals():
            print(f"[Rank {rank}] final losses: {losses}")

    # Print pipeline order with average op timings for this rank
    print(f"\n=== Pipeline order for rank {rank} (avg over {measure_steps} steps) ===")
    op_counters = defaultdict(int)
    for step, action in enumerate(pipe_schedule.pipeline_order[rank]):
        if action is None:
            continue
        stage_idx, op, mb = action
        key = op_key(op)
        idx = op_counters[key]
        op_counters[key] += 1
        t_val = op_idx_avg.get(key, {}).get(idx)
        t_str = f"{t_val:.6f}s" if t_val is not None else "n/a"
        print(f"t={step:03d} | stage={stage_idx} | op={op} | microbatch={mb} | time={t_str}")

    dist.barrier()
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch_size", type=int, default=16)
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()

    world_size = 2
    mp.set_start_method("spawn", force=True)

    mp.spawn(
        run_pipeline,
        args=(world_size, args.microbatch_size, args.num_microbatches, args.seq_len, LLAMA_3B),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
