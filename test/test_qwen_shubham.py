import ray
import torch
import torch.nn as nn
import argparse
import time
import os
import numpy as np
import json

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup, piper_shutdown
from src.piper_exec import piper_exec
from src.piper_utils import piper_metadata
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)

from .models.qwen3 import PiperQwen3Model, create_qwen3_config
from torchtitan.models.qwen3 import Qwen3Model, Qwen3TransformerBlock
from torchtitan.models.common import GQAttention, RoPE, FeedForward
from torchtitan.models.common.moe import MoE

from .schedule_helpers import (
    build_1f1b_schedule, 
    build_gpipe_schedule, 
    print_schedule,
    INTERLEAVED_1F1B_PP2_MB4_SCHEDULE,
    INTERLEAVED_1F1B_PP2_MB6_SCHEDULE,
    INTERLEAVED_1F1B_PP4_MB8_SCHEDULE,
    INTERLEAVED_GPIPE_PP2_MB4_SCHEDULE,
    NO_PP_SCHEDULE,
    DUALPIPEV_MB6_SCHEDULE,
    DUALPIPEV_NOZB_MB6_SCHEDULE,
    ZEROBUBBLE_MB4_SCHEDULE,
)


def main(args, pg):
    world_size = args.dp * args.pp
    mbs = args.mbs
    batch_size = args.batch_size
    seq_len = args.seq_len

    # Create model config
    config = create_qwen3_config(args.model)

    # Select schedule
    schedule = None
    match args.schedule:
        case "no-pp":
            schedule = NO_PP_SCHEDULE
        case "interleaved-1f1b":
            schedule = INTERLEAVED_1F1B_PP2_MB4_SCHEDULE if args.pp == 2 else INTERLEAVED_1F1B_PP4_MB8_SCHEDULE
        case "1f1b":
            schedule = build_1f1b_schedule(args.mbs, args.pp)
        case "gpipe":
            schedule = build_gpipe_schedule(args.mbs, args.pp)
    
    print("Schedule:")
    print_schedule(schedule)

    # Create model with pipeline stages
    model = PiperQwen3Model(config, num_stages=args.pp)
    model.to('cuda')
    

    # # Initialize weights using TorchTitan's method
    # with torch.no_grad():
    #     model.init_weights(buffer_device=torch.device("cuda:0"))
    
    # Create input tensors
    # Qwen3 forward signature: forward(tokens, attention_masks=None, positions=None)
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to('cuda')
    # For Qwen3, we don't need input_pos like Mixtral - positions can be None
    # attention_masks can also be None for causal attention
    
    # Create labels for loss (vocab_size)
    y = torch.randn(batch_size, seq_len, config.vocab_size).to('cuda')

    # Setup Piper
    piper_setup(
        PiperQwen3Model,
        model_args=(config, args.pp),
        optim_fn=torch.optim.Adam,
        example_inputs=[x],
        example_outputs=y,
        schedule=schedule,
        naive_gradient_sync=args.naive_gradient_sync,
        activation_checkpointing=args.activation_checkpointing,
        pg=pg,
    )

    # Send data to actors ahead of time
    actors = piper_metadata.actors
    ray.get(actors[0].load_input.remote([x]))
    ray.get(actors[len(actors)-1].load_labels.remote(y))

    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Running {args.warmup} warmup iterations...")
    for _ in range(args.warmup):
        piper_exec(schedule, loss_fn, args.dp, args.naive_gradient_sync)
        # piper_exec(schedule, [x], y, loss_fn, args.dp, args.naive_gradient_sync)
    
    print(f"Running {args.iters} timed iterations...")
    iter_times = []
    profile_iter = args.profile_iter
    

    # if profile_iter is not None:
    #     os.makedirs("/m-coriander/coriander/shubham/moe-scheduling/piper_profiling/memory_snapshots", exist_ok=True)
    #     os.makedirs("/m-coriander/coriander/shubham/moe-scheduling/piper_profiling/overlapped", exist_ok=True)
    #     os.makedirs("/m-coriander/coriander/shubham/moe-scheduling/piper_profiling/tensorboard", exist_ok=True)
    #     os.makedirs("/m-coriander/coriander/shubham/moe-scheduling/piper_profiling/chrome_traces", exist_ok=True)
    
    for iter_idx in range(args.iters):
        # Enable profiling for the specified iteration
        if profile_iter is not None and iter_idx == profile_iter:
            print(f"\n=== Enabling profiling for iteration {iter_idx} ===")
            # ray.get([actor.enable_profiling.remote(True) for actor in actors.values()])
            # ray.get([actor.set_tracing.remote(True) for actor in actors.values()])
            # ray.get([actor.enable_memory_tracing.remote(True) for actor in actors.values()])
            # ray.get([actor.reset_peak_memory.remote() for actor in actors.values()])
            # ray.get([actor.clear_trace_data.remote() for actor in actors.values()])
        
        start = time.perf_counter()
        if profile_iter is not None and iter_idx == profile_iter:
            with torch.profiler.record_function("piper_exec_iteration"):
                piper_exec(schedule, loss_fn, args.dp, args.naive_gradient_sync)
        else:
            piper_exec(schedule, loss_fn, args.dp, args.naive_gradient_sync)

        end = time.perf_counter()
        iter_times.append(end - start)

        # Collect profiling data and disable profiling after the specified iteration
        # if profile_iter is not None and iter_idx == profile_iter:
        #     print(f"\n=== Collecting profiling data for iteration {iter_idx} ===")
            
            # # Memory profiling
            # snapshot_paths = ray.get([actor.dump_memory_snapshot.remote() for actor in actors.values()])
            # print("\nMemory snapshots:")
            # for rank, path in enumerate(snapshot_paths):
            #     if path:
            #         print(f"Rank {rank}: {path}")
            
            # # Peak memory
            # mem_data_ret = ray.get([actor.get_peak_memory.remote() for actor in actors.values()])
            # print("\nPeak memory usage:")
            # for rank, peak_mem in mem_data_ret:
            #     print(f"  Rank {rank}: {peak_mem:.3f} GB")
            
            # # Timing data
            # trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
            # print("\nTiming data:")
            # for rank, trace_data in trace_data_ret:
            #     print(f"\n  Rank {rank}:")
            #     for key in trace_data:
            #         all_times = trace_data[key]
            #         if all_times:
            #             print(f"    {key}: {np.mean(all_times):.3f} ± {np.std(all_times):.3f} ms ({len(all_times)} samples)")
            
            # # Disable profiling
            # print("\n=== Disabling profiling ===")
            # ray.get([actor.enable_profiling.remote(False) for actor in actors.values()])
            # ray.get([actor.set_tracing.remote(False) for actor in actors.values()])
            # ray.get([actor.enable_memory_tracing.remote(False) for actor in actors.values()])
    
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    print(
        f"rank {dp_rank} iter time= {np.mean(iter_times):.2f} ± {np.std(iter_times):.2f} s ({len(iter_times)} samples)\n"
        f"rank {dp_rank} throughput= {(args.batch_size * args.mbs * seq_len)/np.mean(iter_times):.2f} tokens/s ({len(iter_times)} samples)"
    )

    if args.tracing:
        ray.get([actor.set_tracing.remote(args.tracing) for actor in actors.values()])

        print(f"Running {args.warmup} tracing iterations...")
        for _ in range(args.warmup):
            piper_exec(schedule, loss_fn, args.dp, args.naive_gradient_sync)

        trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        for rank, trace_data in trace_data_ret:
            for key in trace_data:
                all_times = trace_data[key]
                print(f"rank {rank} {key} time= {np.mean(all_times):.3f} ± {np.std(all_times):.3f} ms ({len(all_times)} samples)")



    if args.naive_gradient_sync:
        suffix = "-naive-sync"
    else:
        suffix = ""
    # Collect task_id -> label mappings from all actors
    all_task_labels = {}
    actors = piper_metadata.actors
    for labels in ray.get([actor.get_task_labels.remote() for actor in actors.values()]):
        all_task_labels.update(labels)


    os.makedirs("/m-coriander/coriander/shubham/moe-scheduling/piper_profiling/timeline", exist_ok=True)
    timeline_filename = f"/m-coriander/coriander/shubham/moe-scheduling/piper_profiling/timeline/qwen3-pp{args.pp}-dp{args.dp}-{args.schedule}{suffix}.json"
    ray.timeline(timeline_filename)
    labels_filename = timeline_filename.replace(".json", "-labels.json")
    with open(labels_filename, "w") as f:
        import json
        json.dump(all_task_labels, f)
    print(f"Ray timeline saved to: {timeline_filename}")
    print(f"Task labels saved to: {labels_filename}")

    piper_shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description='Run Qwen3 model with pipeline parallelism')
    parser.add_argument('--model', choices=['tiny', 'small', 'medium', 'large', '30B-A3B', '72B'], default='small',
                        help='Model configuration: tiny, small, medium, large, 30B-A3B, or 72B (default: tiny)')
    parser.add_argument('--schedule', choices=['gpipe', '1f1b', 'interleaved-1f1b', 'no-pp'], default='1f1b',
                        help='Schedule type: gpipe, 1f1b, or interleaved-1f1b (default: 1f1b)')
    parser.add_argument('--dp', type=int, default=1,
                        help='Number of data parallel degrees (default: 1)')
    parser.add_argument('--pp', type=int, default=2,
                        help='Number of pipeline parallel degrees (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--seq_len', type=int, default=1024,
                        help='Sequence length (default: 2048)')
    parser.add_argument('--mbs', type=int, default=4,
                        help='Number of microbatches (default: 4)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--iters', type=int, default=10,
                        help='Number of timing iterations (default: 10)')
    parser.add_argument('--tracing', action='store_true', default=False,
                        help='Enable tracing')
    parser.add_argument('--naive_gradient_sync', action='store_true', default=False,
                        help='Enable naive gradient sync')
    parser.add_argument('--profile_iter', type=int, default=5,
                        help='Iteration number to profile (0-indexed). If None, no profiling (default: None)')
    parser.add_argument('--activation_checkpointing', action='store_true', default=False,
                        help='Enable activation checkpointing')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ray.init(
        include_dashboard=False,
        log_to_driver=True,
        namespace="qwen3",
        _temp_dir="/dev/shm/ray"
    )
    pg = placement_group([{"CPU": args.pp, "GPU": args.pp}] * args.dp)
    ray.get(pg.ready(), timeout=10)
    print(placement_group_table(pg))
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, args, pg)
    ray.get(handles)
    time.sleep(3)
    ray.shutdown()