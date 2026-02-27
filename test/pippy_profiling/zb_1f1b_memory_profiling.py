import os
import torch
import torch.multiprocessing as mp
import argparse
import torch.distributed as dist
import torch.nn as nn
import subprocess
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleInterleavedZeroBubble, Schedule1F1B
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from model import Transformer, LLAMA_DEBUG, LLAMA_3B, LLAMA_1B

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def wrap_stage_forward(stage, name):
    orig_forward = stage.submod.forward
    mb_counter = {"i": 0}

    def wrapped_forward(self, *args, **kwargs):
        i = mb_counter["i"]
        mb_counter["i"] += 1

        with record_function(f"{name}_F_mb{i}"):
            return orig_forward(self, *args, **kwargs)

    stage.submod.forward = wrapped_forward

def wrap_stage_backward_ZB(stage, name):
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

def wrap_stage_backward_1f1b(stage, name):
    orig_bwd_one = stage.__class__.backward_one_chunk
    def wrapped_bwd_one(self, *args, **kwargs):
        mb = kwargs.get("bwd_chunk_id", args[0] if args else "unk")
        with record_function(f"{name}_B_mb{mb}"):
            return orig_bwd_one(self, *args, **kwargs)
    stage.backward_one_chunk = wrapped_bwd_one.__get__(stage, stage.__class__)


def run_pipeline(rank, world_size, schedule_name, mb_size, num_mbs, result_queue, model_cfg=LLAMA_DEBUG):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

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

    if rank == 0:
        print(model_cfg.n_layers)
        print(world_size)
        print(layers_per_rank)
        print(split_spec)

    example_input = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size, model_cfg.max_seq_len),
        dtype=torch.long,
        device=device
    )
    
    pipe = pipeline(
        module=model,
        mb_args=(example_input,),
        split_spec=split_spec,
    )
    
    stage = pipe.build_stage(rank, device=device) # Currently assumes one stage per rank  
    schedule = None      
    wrap_stage_forward(stage, f"stage{rank}")
    if schedule_name == "1f1b":
        wrap_stage_backward_1f1b(stage, f"stage{rank}")
        schedule = Schedule1F1B(stage, n_microbatches=num_mbs, loss_fn=tokenwise_loss_fn)
    else:
        wrap_stage_backward_ZB(stage, f"stage{rank}")
        schedule = ScheduleInterleavedZeroBubble([stage], n_microbatches=num_mbs, loss_fn=tokenwise_loss_fn)

    # Print the pipeline schedule
    print(f"\n=== Pipeline order for rank {rank} ===")
    for step, action in enumerate(schedule.pipeline_order[rank]):
        if action is None:
            continue
        stage_idx, op, mb = action
        print(f"t={step:03d} | stage={stage_idx} | op={op} | microbatch={mb}")

    prof_schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)

    # Generate random training tensors
    x = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size * num_mbs, model_cfg.max_seq_len),
        dtype=torch.long,
        device=device                 
    )

    y = torch.randint(
        low=0, 
        high=model_cfg.vocab_size, 
        size=(mb_size * num_mbs, model_cfg.max_seq_len), 
        dtype=torch.long,
        device=device
    )

    # Initialize profiler activity based on device availability
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities += [ProfilerActivity.CUDA]

    with profile(
        activities=activities, 
        schedule=prof_schedule,
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=True,
            profile_all_threads=True,
            capture_overload_names=True,
        )) as prof:
        if rank == 0:
            schedule.step(x)
        elif rank == 1:
            losses = []
            output = schedule.step(target=y, losses=losses)
            print(f"final losses: {losses}")
        prof.step()

    trace_dir = os.path.join(os.path.dirname(__file__), "traces")
    out_path = os.path.join(trace_dir, f"trace_{rank}.json")
    prof.export_chrome_trace(out_path)
    
    dist.barrier()
    if rank == 1:
        # Create traces folder if it doesn't exist
        os.makedirs(trace_dir, exist_ok=True)

        merged_out = os.path.join(trace_dir, "merged_trace.json")
        cmd = ["python", "filter_trace.py", os.path.join(trace_dir,"trace_0.json"), os.path.join(trace_dir,"user_only_trace_0.json")]
        subprocess.run(cmd, check=True)

        cmd = ["python", "filter_trace.py", os.path.join(trace_dir, "trace_1.json"),  os.path.join(trace_dir,"user_only_trace_1.json")]
        subprocess.run(cmd, check=True)

        cmd = ["python", "merge_traces.py", os.path.join(trace_dir,"user_only_trace_0.json"), os.path.join(trace_dir,"user_only_trace_1.json"), merged_out]
        subprocess.run(cmd, check=True)
        print(f"Merged trace available at {merged_out}")
    
    cleanup()
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch_size', type=int, default=64)
    parser.add_argument('--num_microbatches', type=int, default=256)
    parser.add_argument("--schedule", type=str, default="1f1b", choices=["1f1b", "zb"])
    args = parser.parse_args()
    
    world_size = 4

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_pipeline,
        args=(world_size, args.schedule, args.microbatch_size, args.num_microbatches, result_queue),
        nprocs=world_size,
        join=True,
    )
