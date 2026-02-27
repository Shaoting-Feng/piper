import ray
import torch
import time
import argparse
import os
import numpy as np
import gc

from src.piper_exec import piper_exec
from src.piper_compile import piper_setup, piper_shutdown
from src.piper_coordinator import PiperProgramCoordinator

from .models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B
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


def main(args):
    
    # Set model configuration based on argument
    match args.model:
        case 'debug':
            llama_config = LLAMA_DEBUG
        case '1b':
            llama_config = LLAMA_1B
        case '3b':
            llama_config = LLAMA_3B
        case '8b':
            llama_config = LLAMA_8B
    print(args) 

    loss_fn = torch.nn.CrossEntropyLoss()
    
    model = Transformer(llama_config, args.seq_len)
    model.to('cuda')

    x = torch.randint(0, llama_config.vocab_size, (args.batch_size, args.seq_len)).to('cuda')
    y = torch.randn((args.batch_size, args.seq_len, llama_config.vocab_size)).to('cuda')

    schedule = None

    match args.schedule:
        case "no-pp":
            schedule = NO_PP_SCHEDULE
        case "interleaved-1f1b":
            if args.pp == 2:
                if args.mbs == 6:
                    schedule = INTERLEAVED_1F1B_PP2_MB6_SCHEDULE
                elif args.mbs == 4:
                    schedule = INTERLEAVED_1F1B_PP2_MB4_SCHEDULE
                else:
                    raise ValueError(f"Unsupported number of microbatches: for interleaved-1f1b with PP={args.pp}: {args.mbs}")
            elif args.pp == 4:
                assert args.mbs == 8
                schedule = INTERLEAVED_1F1B_PP4_MB8_SCHEDULE
        case "1f1b":
            schedule = build_1f1b_schedule(args.mbs, args.pp)
        case "gpipe":
            schedule = build_gpipe_schedule(args.mbs, args.pp)
        case "interleaved-gpipe":
            assert args.pp == 2 and args.mbs == 4
            schedule = INTERLEAVED_GPIPE_PP2_MB4_SCHEDULE
        case "dualpipev":
            assert args.pp == 2 and args.mbs == 6
            schedule = DUALPIPEV_MB6_SCHEDULE
        case "dualpipev-nozb":
            assert args.pp == 2 and args.mbs == 6
            schedule = DUALPIPEV_NOZB_MB6_SCHEDULE
        case "zerobubble":
            assert args.pp == 2 and args.mbs == 4
            schedule = ZEROBUBBLE_MB4_SCHEDULE
    print("Schedule:")
    print_schedule(schedule)

    piper_setup(
        model,
        torch.optim.Adam, 
        [x],
        y,
        schedule,
        args.naive_gradient_sync,
    )

    del model
    del x
    del y 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Warmup
    print(f"Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        piper_exec(schedule, loss_fn, args.dp, args.naive_gradient_sync)
        print(f"Warmup {i} completed")

    # Time training steps
    print(f"Running {args.iters} timed iterations...")
    iter_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        piper_exec(schedule, loss_fn, args.dp, args.naive_gradient_sync)
        end = time.perf_counter()
        iter_times.append(end - start)
    
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    print(
        f"rank {dp_rank} iter time= {np.mean(iter_times):.3f} ± {np.std(iter_times):.3f} s ({len(iter_times)} samples)\n"
        f"rank {dp_rank} throughput= {(args.batch_size * args.mbs * args.seq_len)/np.mean(iter_times):.3f} tokens/sec ({len(iter_times)} samples)"
    )

    if args.tracing:
        from src.piper_utils import piper_metadata
        actors = piper_metadata.actors
        ray.get([actor.set_tracing.remote(args.tracing) for actor in actors.values()])
        ray.get([actor.reset_peak_memory.remote() for actor in actors.values()])
        ray.get([actor.start_mem_tracing.remote() for actor in actors.values()])

        print(f"Running {args.warmup} tracing iterations...")
        for _ in range(args.warmup):
            piper_exec(schedule, loss_fn, args.dp, args.naive_gradient_sync)

        mem_data_ret = ray.get([actor.get_peak_memory.remote() for actor in actors.values()])
        for rank, peak_mem in mem_data_ret:
            print(f"rank {rank} peak memory= {peak_mem:.3f} GB")

        trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        for rank, trace_data in trace_data_ret:
            for key in trace_data:
                all_times = trace_data[key]
                print(f"rank {rank} {key} time= {np.mean(all_times):.3f} ± {np.std(all_times):.3f} ms ({len(all_times)} samples)")

    if args.naive_gradient_sync:
        suffix = "-naive-sync"
    else:
        suffix = ""
    timeline_filename = f"out/{args.model}-pp{args.pp}-dp{args.dp}-{args.schedule}{suffix}.json"
    ray.timeline(timeline_filename)
    print(f"Ray timeline saved to: {timeline_filename}")

    piper_shutdown()

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['debug', '1b', '3b', '8b'], default='debug',
                        help='Model configuration: debug, 1b, 3b, or 8b (default: debug)')
    parser.add_argument('--schedule', choices=['gpipe', '1f1b', 'interleaved-1f1b', 'interleaved-gpipe', 'dualpipev-nozb', 'dualpipev', 'zerobubble', 'no-pp'], default='1f1b',
                        help='Schedule type: gpipe, 1f1b, interleaved-1f1b, dualpipev, zerobubble, or no-pp (default: 1f1b)')
    parser.add_argument('--dp', type=int, default=1,
                        help='Number of data parallel degrees (default: 1)')
    parser.add_argument('--pp', type=int, default=2,
                        help='Number of pipeline parallel degrees (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--mbs', type=int, default=4,
                        help='Number of microbatches (default: 4)')
    parser.add_argument('--seq_len', type=int, default=256,
                        help='Sequence length (default: 256)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--iters', type=int, default=10,
                        help='Number of timing iterations (default: 10)')
    parser.add_argument('--tracing', action='store_true', default=False,
                        help='Enable tracing')
    parser.add_argument('--naive_gradient_sync', action='store_true', default=False,
                        help='Enable naive gradient sync')
    return parser.parse_args()


if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama")
    args = parse_args()
    piper_coordinator = PiperProgramCoordinator.remote(dp_degree=args.dp, pp_degree=args.pp)
    ray.get(piper_coordinator.run_program.remote(main, args))
    time.sleep(3)
    ray.shutdown()