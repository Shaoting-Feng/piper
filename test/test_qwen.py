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
from src.piper_utils import piper_metadata, compute_transformer_flops_per_token

from .models.qwen3 import PiperQwen3Model, create_qwen3_config
from torchtitan.models.qwen3.model.model import precompute_rope_cache
from .schedule_helpers import (
    build_1f1b_schedule,
    build_gpipe_schedule,
    build_interleaved_1f1b_schedule,
    build_zerobubble_schedule,
    build_interleaved_zero_bubble,
    build_dualpipev_schedule,
    visualize_pipeline_schedule,
    NO_PP_SCHEDULE,
)


# from triton.backends.amd.driver import HIPDriver
# HIPDriver.is_active = staticmethod(lambda: False)

def main(args, pg):
    batch_size = args.batch_size

    config = create_qwen3_config(args.model)

    num_stages = args.pp
    v = 2
    match args.schedule:
        case "no-pp":
            schedule = NO_PP_SCHEDULE
        case "interleaved-1f1b":
            schedule = build_interleaved_1f1b_schedule(args.mbs, args.pp, v=v)
            num_stages *= v
        case "1f1b":
            schedule = build_1f1b_schedule(args.mbs, args.pp)
        case "gpipe":
            schedule = build_gpipe_schedule(args.mbs, args.pp)
        case "interleaved-gpipe":
            assert False
        case "dualpipev":
            schedule = build_dualpipev_schedule(args.mbs, args.pp)
            num_stages *= v
        case "dualpipev-nozb":
            assert False
        case "zerobubble":
            schedule = build_zerobubble_schedule(args.mbs, args.pp)
        case "interleaved-zerobubble":
            schedule = build_interleaved_zero_bubble(args.mbs, args.pp, v=v)
            num_stages *= v

    visualize_pipeline_schedule(schedule, f"out/{args.schedule}-pp{args.pp}-mb{args.mbs}")

    x = torch.randint(0, config.vocab_size, (batch_size, args.seq_len))
    y = torch.randn(batch_size, args.seq_len, config.vocab_size)

    loss_fn = torch.nn.CrossEntropyLoss()

    flops_per_token = compute_transformer_flops_per_token(
        hidden_dim=config.dim,
        n_layers=config.n_layers,
        ffn_dim=config.moe_inter_dim if config.moe_enabled else config.hidden_dim,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        activation_checkpointing=False,
        moe_top_k=config.moe_args.top_k if config.moe_enabled else 1,
    )

    rope_cache = precompute_rope_cache(
        config.head_dim,
        config.max_seq_len,
        config.rope_theta,
    )

    piper_setup(
        PiperQwen3Model,
        model_args=(config, num_stages),
        optim_fn=torch.optim.Adam,
        example_inputs=[x],
        example_outputs=y,
        loss_fn=loss_fn,
        schedule=schedule,
        naive_gradient_sync=args.naive_grad_sync,
        activation_checkpointing=False,
        bucketing=args.bucketing,
        model_dtype=torch.bfloat16,
        pg=pg,
        nsight=args.nsight,
        model_flops_per_token=flops_per_token,
        visualize_dag=not args.no_viz,
        const_attrs={"rope_cache": rope_cache},
    )

    print(f"Running {args.warmup} warmup iterations")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn)
        time.sleep(1)

    actors = piper_metadata.actors

    print(f"Running {args.iters} timed iterations")
    iter_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        losses = piper_exec_dag(loss_fn, log_stats=True, profiling=args.profiling)
        end = time.perf_counter()
        iter_times.append(end - start)
        time.sleep(1)

    dp_rank = int(os.environ["PIPER_DP_RANK"])
    print(
        f"rank {dp_rank} iter time= {np.mean(iter_times):.5f} ± {np.std(iter_times):.5f} s "
        f"({len(iter_times)} samples)\n"
        f"rank {dp_rank} throughput= "
        f"{(args.batch_size * args.mbs * args.seq_len) / np.mean(iter_times):.3f} tokens/s"
    )

    if args.tracing:
        ray.get([actor.set_tracing.remote(True) for actor in actors.values()])
        print(f"Running {args.trace_iters} tracing iterations...")
        for _ in range(args.trace_iters):
            piper_exec_dag(loss_fn)
            ray.get([actor.flush_timing_events.remote() for actor in actors.values()])
            time.sleep(1)
        trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        for rank, trace_data in trace_data_ret:
            for key in trace_data:
                all_times = trace_data[key]
                print(
                    f"rank {rank} {key} time= {np.mean(all_times):.3f} ± "
                    f"{np.std(all_times):.3f} ms ({len(all_times)} samples)"
                )

    os.makedirs("out", exist_ok=True)
    timeline_filename = f"out/qwen-dag-pp{args.pp}-dp{args.dp}-{args.schedule}"
    ray.timeline(timeline_filename)
    print(f"Ray timeline saved to: {timeline_filename}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DAG-based piper execution with the Mixtral model"
    )
    parser.add_argument('--model', choices=['9M', '1B', '9B', '48B', '30B-A3B', '30-A3B-half', '72B'], default='1B',
                        help='Model configuration: 9M, 1B, 9B, 48B, 30B-A3B, 30-A3B-half, or 72B (default: 1B)')
    parser.add_argument("--schedule", choices=["gpipe", "1f1b", "no-pp", "interleaved-1f1b", "dualpipev", "dualpipev-seq", "zerobubble"], default="1f1b",)
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
    parser.add_argument("--bucketing", action="store_true", default=False,
                        help="Split stages into per-param-bucket sub-modules for overlapped all-reduce")
    parser.add_argument("--profiling", action="store_true", default=False,
                        help="Profile each DAG task: log time and memory delta per task")
    parser.add_argument("--nsight", action="store_true", default=False,
                        help="Whether to use Nsight Systems for tracing")
    parser.add_argument("--no-viz", action="store_true", default=False,
                        help="Skip per-rank DAG visualization (speeds up startup for large models)")
    return parser.parse_args()


"""
CUDA_VISIBLE_DEVICES=4,5,6,7 /m-coriander/coriander/mfris/miniconda3/envs/piper/bin/python \
    /m-coriander/coriander/mfris/miniconda3/envs/piper/lib/python3.10/site-packages/ray/scripts/scripts.py \
        start --head \
        --port 4567 \
        --temp-dir=/m-coriander/coriander/mfris/piper/ray_tmp \
        --include-dashboard=false
"""
if __name__ == "__main__":
    args = parse_args()
    print(args)
    ray.init(
        address="10.158.48.71:4567",
        namespace="qwen",
        log_to_driver=True,
        include_dashboard=False,
        _temp_dir="/m-coriander/coriander/mfris/piper/ray_tmp",
    )
    pg = placement_group([{"CPU": args.pp, "GPU": args.pp}] * args.dp, strategy="PACK")
    ray.get(pg.ready(), timeout=600)
    print(placement_group_table(pg))
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, pg, args, pg)
    ray.get(handles)
    ray.shutdown()
