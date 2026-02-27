import os
import torch
import torch.multiprocessing as mp
import argparse
import torch.distributed as dist
import torch.nn as nn
import time
import subprocess
from torch.cuda import Event
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleInterleavedZeroBubble, Schedule1F1B, ScheduleInterleaved1F1B
from torch.profiler import profile, ProfilerActivity, record_function, schedule as profiler_schedule
from model import Transformer, LLAMA_DEBUG, LLAMA_3B, LLAMA_1B


def wrap_stage_forward(stage, name):
    """Tag each microbatch forward pass for profiling."""
    orig_forward = stage.submod.forward
    mb_counter = {"i": 0}

    def wrapped_forward(self, *args, **kwargs):
        i = mb_counter["i"]
        mb_counter["i"] += 1

        with record_function(f"{name}_F_mb{i}"):
            return orig_forward(self, *args, **kwargs)

    stage.submod.forward = wrapped_forward


def wrap_stage_backward(stage, name):
    """Tag activation and weight backward passes for profiling."""
    orig_bwd_one = stage.__class__.backward_one_chunk

    def wrapped_bwd_one(self, *args, **kwargs):
        mb = kwargs.get("bwd_chunk_id", args[0] if args else "unk")
        with record_function(f"{name}_B_mb{mb}"):
            return orig_bwd_one(self, *args, **kwargs)

    stage.backward_one_chunk = wrapped_bwd_one.__get__(stage, stage.__class__)

    orig_bwd_w = stage.__class__.backward_weight_one_chunk

    def wrapped_bwd_w(self, *args, **kwargs):
        mb = kwargs.get("bwd_chunk_id", args[0] if args else "unk")
        with record_function(f"{name}_W_mb{mb}"):
            return orig_bwd_w(self, *args, **kwargs)

    stage.backward_weight_one_chunk = wrapped_bwd_w.__get__(stage, stage.__class__)

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    backend = "nccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_pipeline(rank, world_size, mb_size, num_mbs, seq_len, model_cfg):
    setup(rank, world_size)
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"[Rank {rank}] using torch device {device}")

    model = Transformer(model_cfg)
    model.to(device)

    # Helper loss function that automatically reshapes output and targets tensors
    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_cfg.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)

    # Partition the model into world_size stages and create the stage for this rank
    layers_per_rank = model_cfg.n_layers // world_size

    split_spec = {
        f"layers.{i * layers_per_rank}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }

    example_input = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size, seq_len),
        dtype=torch.long,
        device=device
    )
    
    pipe = pipeline(
        module=model,
        mb_args=(example_input,),
        split_spec=split_spec,
    )

    stage = pipe.build_stage(rank, device=device)  # Currently assumes one stage per rank

    # Add profiling markers to forward / backward passes (names align with zerobubble.py)
    wrap_stage_forward(stage, f"stage{rank}")
    wrap_stage_backward(stage, f"stage{rank}")

    # Optimizer for this stage's parameters; profiled per-iteration
    optimizer = torch.optim.Adam(stage.submod.parameters(), lr=1e-3)

    # pipe_schedule = ScheduleInterleavedZeroBubble([stage], n_microbatches=num_mbs, loss_fn=tokenwise_loss_fn)
    pipe_schedule = Schedule1F1B(stage, n_microbatches=num_mbs, loss_fn=tokenwise_loss_fn)

    # Print the pipeline schedule
    print(f"\n=== Pipeline order for rank {rank} ===")
    for step, action in enumerate(pipe_schedule.pipeline_order[rank]):
        if action is None:
            continue
        stage_idx, op, mb = action
        print(f"t={step:03d} | stage={stage_idx} | op={op} | microbatch={mb}")

    # Generate random training tensors
    x = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size * num_mbs, seq_len),
        dtype=torch.long,
        device=device                 
    )

    y = torch.randint(
        low=0, 
        high=model_cfg.vocab_size, 
        size=(mb_size * num_mbs, seq_len), 
        dtype=torch.long,
        device=device
    )

    total_tokens = mb_size * num_mbs * seq_len

    warmup_steps = 2
    measure_steps = 4

    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        if rank != world_size - 1:
            pipe_schedule.step(x)
        else:
            _losses = []
            _ = pipe_schedule.step(target=y, losses=_losses)
        optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()

    elapsed_times = []

    # ---------------------------
    # Profiling + timing section
    # ---------------------------

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Profile the full measure_steps window (after warmup) and export chrome trace.
    prof_schedule = profiler_schedule(wait=0, warmup=0, active=measure_steps, repeat=1)

    trace_dir = os.path.join(os.path.dirname(__file__), "traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"trace_{rank}.json")

    def _remove_if_exists(path: str):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    with profile(
        activities=activities,
        schedule=prof_schedule,
        with_stack=False,
        record_shapes=False,
    ) as prof:
        for _ in range(measure_steps):
            dist.barrier()
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            optimizer.zero_grad(set_to_none=True)

            if rank != world_size - 1:
                pipe_schedule.step(x)
            else:
                losses = []
                _ = pipe_schedule.step(target=y, losses=losses)

            with record_function(f"optimizer_step_rank{rank}"):
                optimizer.step()
            
            torch.cuda.synchronize()
            end.record()
            torch.cuda.synchronize()
            dist.barrier()

            if rank == world_size - 1:
                elapsed_ms = start.elapsed_time(end)
                elapsed_times.append(elapsed_ms / 1000.0)

            prof.step()

    _remove_if_exists(trace_path)
    prof.export_chrome_trace(trace_path)
    dist.barrier()  # ensure both traces are written before merging

    if rank == world_size - 1:
        print(f"[Rank {rank}] chrome trace saved to {trace_path}")

        # Create user-only traces and merge (mirrors zerobubble.py workflow)
        trace0 = os.path.join(trace_dir, "trace_0.json")
        trace1 = os.path.join(trace_dir, "trace_1.json")
        user0 = os.path.join(trace_dir, "user_only_trace_0.json")
        user1 = os.path.join(trace_dir, "user_only_trace_1.json")
        merged = os.path.join(trace_dir, "merged_trace.json")
        merged_raw = os.path.join(trace_dir, "merged_trace_raw.json")

        for p in (user0, user1, merged, merged_raw):
            _remove_if_exists(p)

        subprocess.run(["python", "filter_trace.py", trace0, user0], check=True)
        subprocess.run(["python", "filter_trace.py", trace1, user1], check=True)
        subprocess.run(["python", "merge_traces.py", user0, user1, merged], check=True)
        subprocess.run(["python", "merge_traces.py", trace0, trace1, merged_raw], check=True)
        print(f"[Rank {rank}] merged trace available at {merged}")
        print(f"[Rank {rank}] raw merged trace available at {merged_raw}")

    if rank == world_size - 1:
        avg_elapsed = sum(elapsed_times) / len(elapsed_times)
        tokens_per_sec = total_tokens / avg_elapsed
        print(f"[Rank {rank}] Total elapsed: {sum(elapsed_times):.6f}s (CUDA events)")
        print(f"[Rank {rank}] Avg elapsed: {avg_elapsed:.6f}s (CUDA events)")
        print(f"[Rank {rank}] Throughput: {tokens_per_sec:.2f} tokens/s")
        print(f"[Rank {rank}] final losses: {losses}")

    dist.barrier()
    cleanup()

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch_size', type=int, default=16)
    parser.add_argument('--num_microbatches', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=256)
    # parser.add_argument("--schedule", type=str, default="1f1b", choices=["1f1b", "zb"])
    args = parser.parse_args()
    
    world_size = 2

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    mp.spawn(
        run_pipeline,
        args=(world_size, args.microbatch_size, args.num_microbatches, args.seq_len, LLAMA_3B),
        nprocs=world_size,
        join=True,
    )
