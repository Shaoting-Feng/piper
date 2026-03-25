"""
test_dag.py – end-to-end test for the DAG-based execution path (piper_exec_dag).

Run exactly like test_mixtral.py:

    python -m test.test_dag [--model tiny] [--pp 2] [--dp 1] ...

The main difference is that each training step is dispatched through
piper_exec_dag() rather than the legacy piper_exec() scheduler.
"""
import logging
import ray
import torch
import argparse
import json
import time
import os
import numpy as np

logger = logging.getLogger(__name__)

from ray.util.placement_group import (
    placement_group,
    placement_group_table,
)

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper import piper_exec_dag
from src.piper_utils import piper_metadata

from .models.qwen3 import PiperQwen3Model, create_qwen3_config
from .schedule_helpers import (
    build_1f1b_schedule,
    build_gpipe_schedule,
    print_schedule,
    NO_PP_SCHEDULE,
    INTERLEAVED_1F1B_PP2_MB4_SCHEDULE,
    DUALPIPEV_MB6_SCHEDULE,
    ZEROBUBBLE_MB4_SCHEDULE,
)


def main(args, pg):
    batch_size = args.batch_size

    config = create_qwen3_config(args.model)

    match args.schedule:
        case "no-pp":
            schedule = NO_PP_SCHEDULE
        case "1f1b":
            schedule = build_1f1b_schedule(args.mbs, args.pp)
        case "gpipe":
            schedule = build_gpipe_schedule(args.mbs, args.pp)
        case "interleaved-1f1b":
            schedule = INTERLEAVED_1F1B_PP2_MB4_SCHEDULE
        case "dualpipev":
            schedule = DUALPIPEV_MB6_SCHEDULE
        case "zerobubble":
            schedule = ZEROBUBBLE_MB4_SCHEDULE

    print("Schedule:")
    print_schedule(schedule)

    x = torch.randint(0, config.vocab_size, (batch_size, args.seq_len))
    y = torch.randn(batch_size, args.seq_len, config.vocab_size)

    loss_fn = torch.nn.CrossEntropyLoss()

    piper_setup(
        PiperQwen3Model,
        model_args=(config, args.pp),
        optim_fn=torch.optim.Adam,
        example_inputs=[x],
        example_outputs=y,
        schedule=schedule,
        naive_gradient_sync=args.naive_grad_sync,
        activation_checkpointing=False,
        bucketing=args.bucketing,
        mode=args.mode,
        model_dtype=torch.bfloat16,
        pg=pg,
        nsight=args.nsight,
    )

    print(f"Running {args.warmup} warmup iterations (dag path)...")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn)
        time.sleep(1)

    actors = piper_metadata.actors

    print(f"Running {args.iters} timed iterations (dag path)...")
    iter_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        losses = piper_exec_dag(loss_fn)
        end = time.perf_counter()
        iter_times.append(end - start)
        time.sleep(1)

    dp_rank = int(os.environ["PIPER_DP_RANK"])
    print(
        f"rank {dp_rank} iter time= {np.mean(iter_times):.5f} ± {np.std(iter_times):.5f} s "
        f"({len(iter_times)} samples)\n"
        f"rank {dp_rank} throughput= "
        f"{(args.batch_size * args.mbs * config.block_size) / np.mean(iter_times):.3f} tokens/s"
    )

    if args.tracing:
        ray.get([actor.set_tracing.remote(True) for actor in actors.values()])
        print(f"Running {args.trace_iters} tracing iterations...")
        for _ in range(args.trace_iters):
            piper_exec_dag(loss_fn)
            time.sleep(1)
        trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        for rank, trace_data in trace_data_ret:
            for key in trace_data:
                all_times = trace_data[key]
                print(
                    f"rank {rank} {key} time= {np.mean(all_times):.3f} ± "
                    f"{np.std(all_times):.3f} ms ({len(all_times)} samples)"
                )

    all_task_labels = {}
    for labels in ray.get([actor.get_task_labels.remote() for actor in actors.values()]):
        all_task_labels.update(labels)

    os.makedirs("out", exist_ok=True)
    timeline_filename = f"out/mixtral-dag-pp{args.pp}-dp{args.dp}-{args.schedule}-{args.mode}"
    ray.timeline(timeline_filename)
    labels_filename = timeline_filename + "-labels.json"
    with open(labels_filename, "w") as f:
        json.dump(all_task_labels, f)
    print(f"Ray timeline saved to: {timeline_filename}")
    print(f"Task labels saved to: {labels_filename}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DAG-based piper execution with the Mixtral model"
    )
    parser.add_argument('--model', choices=['9M', '1B', '9B', '48B', '30B-A3B', '72B'], default='1B',
                        help='Model configuration: 9M, 1B, 9B, 48B, 30B-A3B, or 72B (default: 1B)')
    parser.add_argument("--schedule", choices=["gpipe", "1f1b", "no-pp", "interleaved-1f1b", "dualpipev", "zerobubble"], default="1f1b",)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--trace-iters", type=int, default=3)
    parser.add_argument("--tracing", action="store_true", default=False)
    parser.add_argument("--naive-grad-sync", action="store_true", default=False)
    parser.add_argument("--mode", choices=["sequential", "naive", "overlapped"], default="sequential")
    parser.add_argument("--bucketing", action="store_true", default=False,
                        help="Split stages into per-param-bucket sub-modules for overlapped all-reduce")
    parser.add_argument("--nsight", action="store_true", default=False,
                        help="Whether to use Nsight Systems for tracing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    ray.init(
        log_to_driver=True,
        namespace="mixtral_dag",
        include_dashboard=False,
        # _temp_dir="/m-coriander/coriander/mfris/piper/ray_tmp",
    )
    pg = placement_group([{"CPU": args.pp, "GPU": args.pp}] * args.dp, strategy="SPREAD")
    ray.get(pg.ready(), timeout=600)
    print(placement_group_table(pg))
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, args, pg)
    ray.get(handles)
    ray.shutdown()
