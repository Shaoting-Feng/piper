import logging
import ray
import torch
import torch.nn as nn
import argparse
import time
import os
import numpy as np
import itertools

logger = logging.getLogger(__name__)

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper_exec import piper_exec
from src.piper_utils import piper_metadata

from .models.mixtral import Transformer, ModelArgs
from .schedule_helpers import (
    build_1f1b_schedule,
    build_gpipe_schedule,
    print_schedule,
    NO_PP_SCHEDULE,
    INTERLEAVED_1F1B_PP2_MB4_SCHEDULE,
    INTERLEAVED_1F1B_PP4_MB8_SCHEDULE,
    DUALPIPEV_NOZB_MB6_SCHEDULE,
    DUALPIPEV_MB6_SCHEDULE,
)

def main(args):

    world_size = args.dp * args.pp
    mbs = args.mbs
    batch_size = args.batch_size

    match args.model:
        case 'tiny':
            config = ModelArgs.from_name("tiny")
        case 'small':
            config = ModelArgs.from_name("small")
        case 'medium':
            config = ModelArgs.from_name("medium")
        case 'large':
            config = ModelArgs.from_name("Mixtral-8x7B-v0.1")

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
        case "dualpipev":
            schedule = DUALPIPEV_MB6_SCHEDULE
        case "dualpipev-nozb":
            schedule = DUALPIPEV_NOZB_MB6_SCHEDULE
    print("Schedule:")
    print_schedule(schedule)

    x = torch.randint(0, config.vocab_size, (batch_size, config.block_size))
    input_pos = torch.arange(config.block_size)
    y = torch.randn(batch_size, config.block_size, config.vocab_size)

    loss_fn = torch.nn.CrossEntropyLoss()

    piper_setup(
        Transformer,
        model_args=(config,),
        optim_fn=torch.optim.Adam,
        example_inputs=[x, input_pos],
        example_outputs=y,
        schedule=schedule,
        naive_gradient_sync=args.naive_grad_sync,
        mode=args.mode,
    )

    print(f"Running {args.warmup} warmup iterations...")
    for _ in range(args.warmup):
        piper_exec(schedule, loss_fn, args.dp, args.naive_grad_sync)

    print(f"Running {args.iters} timed iterations...")
    iter_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        piper_exec(schedule, loss_fn, args.dp, args.naive_grad_sync)
        end = time.perf_counter()
        iter_times.append(end - start)
    
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    print(
        f"rank {dp_rank} iter time= {np.mean(iter_times):.5f} ± {np.std(iter_times):.5f} s ({len(iter_times)} samples)\n"
        f"rank {dp_rank} throughput= {(args.batch_size * args.mbs * config.block_size)/np.mean(iter_times):.3f} tokens/s ({len(iter_times)} samples)"
    )

    if args.tracing:
        actors = piper_metadata.actors
        ray.get([actor.set_tracing.remote(args.tracing) for actor in actors.values()])

        print(f"Running {args.trace_iters} tracing iterations...")
        for _ in range(args.trace_iters):
            piper_exec(schedule, loss_fn, args.dp, args.naive_grad_sync)

        trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        for rank, trace_data in trace_data_ret:
            for key in trace_data:
                all_times = trace_data[key]
                print(f"rank {rank} {key} time= {np.mean(all_times):.3f} ± {np.std(all_times):.3f} ms ({len(all_times)} samples)")

    timeline_filename = f"out/mixtral-pp{args.pp}-dp{args.dp}-{args.schedule}-{args.mode}"
    ray.timeline(timeline_filename)
    print(f"Ray timeline saved to: {timeline_filename}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['tiny', 'small', 'medium', 'large'], default='tiny',
                        help='Model configuration: tiny, small, medium, or large (default: tiny)')
    parser.add_argument('--schedule', choices=['gpipe', '1f1b', 'interleaved-1f1b', 'dualpipev', 'dualpipev-nozb', 'no-pp'], default='1f1b',
                        help='Schedule type: gpipe, 1f1b, interleaved-1f1b, dualpipev, dualpipev-nozb, or no-pp (default: 1f1b)')
    parser.add_argument('--dp', type=int, default=2,
                        help='Number of data parallel degrees (default: 2)')
    parser.add_argument('--pp', type=int, default=2,
                        help='Number of pipeline parallel degrees (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--mbs', type=int, default=4,
                        help='Number of microbatches (default: 4)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations (default: 3)')
    parser.add_argument('--iters', type=int, default=5,
                        help='Number of timing iterations (default: 5)')
    parser.add_argument('--trace-iters', type=int, default=3,
                        help='Number of trace iterations (default: 3)')
    parser.add_argument('--tracing', action='store_true', default=False,
                        help='Enable tracing')
    parser.add_argument('--naive-grad-sync', action='store_true', default=False,
                        help='Enable naive gradient sync')
    parser.add_argument('--mode', choices=['sequential', 'naive', 'overlapped'], default='overlapped',
                        help='Mode: sequential, naive, or overlapped (default: overlapped)')
    return parser.parse_args()

if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="mixtral")
    args = parse_args()
    print(args)
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, args)
    ray.get(handles)
    ray.shutdown()
