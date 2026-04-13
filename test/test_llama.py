import logging
import ray
import torch
import time
import argparse
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

from .models.llama import Transformer, precompute_freqs_cis, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B, LLAMA_70B
from .schedule_helpers import (
    visualize_pipeline_schedule,
    build_1f1b_schedule,
    build_gpipe_schedule,
    build_interleaved_1f1b_schedule,
    build_zerobubble_schedule,
    build_interleaved_zero_bubble,
    build_dualpipev_schedule,
    NO_PP_SCHEDULE,
)


def main(args, pg):
    match args.model:
        case 'debug':
            llama_config = LLAMA_DEBUG
        case '1b':
            llama_config = LLAMA_1B
        case '3b':
            llama_config = LLAMA_3B
        case '8b':
            llama_config = LLAMA_8B
        case '70b':
            llama_config = LLAMA_70B
    logger.info(args)

    _ce = torch.nn.CrossEntropyLoss()
    loss_fn = lambda output, labels: _ce(output.view(-1, output.size(-1)), labels.view(-1))

    x = torch.randint(0, llama_config.vocab_size, (args.batch_size, args.seq_len))
    y = torch.randint(0, llama_config.vocab_size, (args.batch_size, args.seq_len))

    match args.schedule:
        case "no-pp":
            schedule = NO_PP_SCHEDULE
        case "interleaved-1f1b":
            schedule = build_interleaved_1f1b_schedule(args.mbs, args.pp, v=2)
        case "1f1b":
            schedule = build_1f1b_schedule(args.mbs, args.pp)
        case "gpipe":
            schedule = build_gpipe_schedule(args.mbs, args.pp)
        case "interleaved-gpipe":
            assert False
        case "dualpipev":
            schedule = build_dualpipev_schedule(args.mbs, args.pp)
        case "dualpipev-nozb":
            assert False
        case "zerobubble":
            schedule = build_zerobubble_schedule(args.mbs, args.pp)
        case "interleaved-zerobubble":
            schedule = build_interleaved_zero_bubble(args.mbs, args.pp, v=2)
    
    visualize_pipeline_schedule(schedule, f"out/{args.schedule}-pp{args.pp}-mb{args.mbs}")

    # Llama FFN hidden dim (SwiGLU): 2/3 * 4 * dim, rounded up to multiple_of
    _ffn_base = int(2 * 4 * llama_config.dim / 3)
    if llama_config.ffn_dim_multiplier is not None:
        _ffn_base = int(llama_config.ffn_dim_multiplier * _ffn_base)
    _ffn_dim = llama_config.multiple_of * ((_ffn_base + llama_config.multiple_of - 1) // llama_config.multiple_of)
    n_kv_heads = llama_config.n_kv_heads if llama_config.n_kv_heads is not None else llama_config.n_heads
    flops_per_token = compute_transformer_flops_per_token(
        hidden_dim=llama_config.dim,
        n_layers=llama_config.n_layers,
        ffn_dim=_ffn_dim,
        n_heads=llama_config.n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=llama_config.dim // llama_config.n_heads,
        activation_checkpointing=args.activation_checkpointing,
    )

    freqs_cis = precompute_freqs_cis(
        llama_config.dim // llama_config.n_heads,
        args.seq_len,
        llama_config.rope_theta,
    )
    _mask = torch.full((args.seq_len, args.seq_len), float("-inf"))
    _mask = torch.triu(_mask, diagonal=1)
    mask = torch.hstack([torch.zeros((args.seq_len, 0)), _mask])

    piper_setup(
        Transformer,
        model_args=(llama_config, args.seq_len),
        optim_fn=torch.optim.Adam,
        example_inputs=[x],
        example_outputs=y,
        schedule=schedule,
        naive_gradient_sync=args.naive_grad_sync,
        activation_checkpointing=args.activation_checkpointing,
        num_checkpoints=args.num_checkpoints,
        bucketing=args.bucketing,
        model_dtype=torch.bfloat16,
        pg=pg,
        nsight=args.nsight,
        model_flops_per_token=flops_per_token,
        visualize_dag=not args.no_viz,
        const_attrs={"freqs_cis": freqs_cis, "mask": mask},
    )

    logger.info(f"Running {args.warmup} warmup iterations...")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn, log_stats=True, profiling=args.profiling)
        time.sleep(1)

    actors = piper_metadata.actors

    logger.info(f"Running {args.iters} timed iterations...")
    iter_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        piper_exec_dag(loss_fn, log_stats=True, profiling=args.profiling)
        end = time.perf_counter()
        iter_times.append(end - start)
        time.sleep(1)

    dp_rank = int(os.environ['PIPER_DP_RANK'])
    logger.info(
        f"rank {dp_rank} iter time= {np.mean(iter_times):.5f} ± {np.std(iter_times):.5f} s "
        f"({len(iter_times)} samples)\n"
        f"rank {dp_rank} throughput= "
        f"{(args.batch_size * args.mbs * args.seq_len) / np.mean(iter_times):.3f} tokens/s"
    )

    if args.tracing:
        ray.get([actor.set_tracing.remote(True) for actor in actors.values()])
        logger.info(f"Running {args.trace_iters} tracing iterations...")
        for _ in range(args.trace_iters):
            piper_exec_dag(loss_fn)
            ray.get([actor.flush_timing_events.remote() for actor in actors.values()])
            time.sleep(1)
        trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        for rank, trace_data in trace_data_ret:
            for key in trace_data:
                all_times = trace_data[key]
                logger.info(
                    f"rank {rank} {key} time= {np.mean(all_times):.3f} ± "
                    f"{np.std(all_times):.3f} ms ({len(all_times)} samples)"
                )

    os.makedirs("out", exist_ok=True)
    timeline_filename = f"out/llama-dag-pp{args.pp}-dp{args.dp}-{args.schedule}"
    ray.timeline(timeline_filename)
    logger.info(f"Ray timeline saved to: {timeline_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description='Run LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['debug', '1b', '3b', '8b', '70b'], default='debug')
    parser.add_argument(
        '--schedule',
        choices=['gpipe', '1f1b', 'interleaved-1f1b', 'interleaved-gpipe',
                 'dualpipev-nozb', 'dualpipev', 'zerobubble', 'interleaved-zerobubble', 'no-pp'],
        default='1f1b',
    )
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--pp', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--mbs', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--trace-iters', type=int, default=3)
    parser.add_argument('--tracing', action='store_true', default=False)
    parser.add_argument('--naive-grad-sync', action='store_true', default=False)
    parser.add_argument('--activation-checkpointing', action='store_true', default=False)
    parser.add_argument('--num-checkpoints', type=int, default=1,
                        help='Number of sequential activation-checkpoint regions per Piper bucket')
    parser.add_argument('--bucketing', action='store_true', default=False,
                        help='Split stages into per-param-bucket sub-modules for overlapped all-reduce')
    parser.add_argument('--nsight', action='store_true', default=False,
                        help='Whether to use Nsight Systems for tracing')
    parser.add_argument('--profiling', action='store_true', default=False,
                        help='Profile each DAG task: log time and memory delta per task')
    parser.add_argument('--no-viz', action='store_true', default=False,
                        help='Skip per-rank DAG visualization (speeds up startup for large models)')
    return parser.parse_args()

"""
/m-coriander/coriander/mfris/miniconda3/envs/piper/bin/python \
    /m-coriander/coriander/mfris/miniconda3/envs/piper/lib/python3.10/site-packages/ray/scripts/scripts.py \
        start --head \
        --port 3456 \
        --temp-dir=/m-coriander/coriander/mfris/piper/ray_tmp \
        --include-dashboard=false
"""
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    logger.info(args)
    ray.init(
        address="10.158.48.71:4567",
        namespace="llama",
        include_dashboard=False,
        _temp_dir="/m-coriander/coriander/mfris/piper/ray_tmp",
    )
    pg = placement_group([{"CPU": args.pp, "GPU": args.pp}] * args.dp, strategy="PACK")
    ray.get(pg.ready(), timeout=600)
    logger.info(placement_group_table(pg))
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, pg, args, pg)
    ray.get(handles)
    ray.shutdown()
