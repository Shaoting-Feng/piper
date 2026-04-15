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

from ray.util.placement_group import (
    placement_group,
    placement_group_table,
)

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper import piper_exec_dag
from src.piper_utils import piper_metadata, compute_transformer_flops_per_token, create_logger, LOG_LEVEL

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


logger = create_logger("test_qwen", LOG_LEVEL)

# from triton.backends.amd.driver import HIPDriver
# HIPDriver.is_active = staticmethod(lambda: False)


def _metrics_output_path(args):
    return os.path.join(
        args.output_dir,
        f"qwen{args.model}-pp{args.pp}-dp{args.dp}-{args.schedule}",
    )


def _build_benchmark_metrics(args, dp_rank, iter_times):
    mean_iter_time = float(np.mean(iter_times))
    std_iter_time = float(np.std(iter_times))
    throughput = float((args.batch_size * args.mbs * args.seq_len) / mean_iter_time)

    return {
        "dp_rank": dp_rank,
        "model": args.model,
        "schedule": args.schedule,
        "pp": args.pp,
        "dp": args.dp,
        "batch_size": args.batch_size,
        "mbs": args.mbs,
        "seq_len": args.seq_len,
        "samples": len(iter_times),
        "iter_time_mean_s": mean_iter_time,
        "iter_time_std_s": std_iter_time,
        "throughput_tokens_per_s": throughput,
        "iter_times_s": [float(iter_time) for iter_time in iter_times],
    }


def _format_benchmark_log_lines(metrics):
    rank = metrics["dp_rank"]
    lines = [
        (
            f"rank {rank} iter time= {metrics['iter_time_mean_s']:.5f} +/- "
            f"{metrics['iter_time_std_s']:.5f} s ({metrics['samples']} samples)"
        ),
        f"rank {rank} throughput= {metrics['throughput_tokens_per_s']:.3f} tokens/s",
    ]
    for trace in metrics.get("trace_times", []):
        lines.append(
            f"rank {trace['rank']} {trace['key']} time= {trace['mean_ms']:.3f} +/- "
            f"{trace['std_ms']:.3f} ms ({trace['samples']} samples)"
        )
    lines.append(f"rank {rank} metrics_json= {json.dumps(metrics, sort_keys=True)}")
    return lines


def _write_benchmark_metrics(args, metrics):
    metrics_path = _metrics_output_path(args)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as output_file:
        for line in _format_benchmark_log_lines(metrics):
            output_file.write(line + "\n")
        output_file.write("\n")

    return metrics_path


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

    # visualize_pipeline_schedule(schedule, f"out/{args.schedule}-pp{args.pp}-mb{args.mbs}")

    x = torch.randint(0, config.vocab_size, (batch_size, args.seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, args.seq_len))

    _ce = torch.nn.CrossEntropyLoss()
    loss_fn = lambda output, labels: _ce(output.view(-1, output.size(-1)), labels.view(-1))

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
        activation_checkpointing=args.activation_checkpointing,
        a2a_ar_no_overlap=True,
        bucket_size=int(args.bucket_size * 1024 * 1024) if args.bucket_size is not None else None,
        zero_stage=args.zero_stage,
        model_dtype=torch.bfloat16,
        pg=pg,
        nsight=args.nsight,
        temp_dir=args.temp_dir,
        model_flops_per_token=flops_per_token,
        visualize_dag=args.save_viz,
        const_attrs={"rope_cache": rope_cache},
    )

    del x, y

    logger.info(f"Running {args.warmup} warmup iterations")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn)
        time.sleep(1)

    actors = piper_metadata.actors

    logger.info(f"Running {args.iters} timed iterations")
    iter_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        losses = piper_exec_dag(loss_fn, log_stats=True, profiling=args.profiling)
        end = time.perf_counter()
        iter_times.append(end - start)
        time.sleep(1)

    dp_rank = int(os.environ["PIPER_DP_RANK"])
    metrics = _build_benchmark_metrics(
        args,
        dp_rank,
        iter_times,
    )
    iter_time_msg, throughput_msg, _ = _format_benchmark_log_lines(metrics)
    metrics_msg = f"rank {dp_rank} benchmark metrics will be saved by the driver to {_metrics_output_path(args)}"
    logger.info(iter_time_msg)
    logger.info(throughput_msg)
    logger.info(metrics_msg)
    print(iter_time_msg, flush=True)
    print(throughput_msg, flush=True)
    print(metrics_msg, flush=True)

    if args.tracing:
        ray.get([actor.set_tracing.remote(True) for actor in actors.values()])
        logger.info(f"Running {args.trace_iters} tracing iterations...")
        for _ in range(args.trace_iters):
            piper_exec_dag(loss_fn)
            ray.get([actor.flush_timing_events.remote() for actor in actors.values()])
            time.sleep(1)
        trace_data_ret = ray.get([actor.get_trace_data.remote() for actor in actors.values()])
        trace_times = []
        for rank, trace_data in trace_data_ret:
            for key in trace_data:
                all_times = trace_data[key]
                trace = {
                    "rank": rank,
                    "key": key,
                    "mean_ms": float(np.mean(all_times)),
                    "std_ms": float(np.std(all_times)),
                    "samples": len(all_times),
                    "times_ms": [float(trace_time) for trace_time in all_times],
                }
                trace_times.append(trace)
                trace_msg = (
                    f"rank {trace['rank']} {trace['key']} time= {trace['mean_ms']:.3f} +/- "
                    f"{trace['std_ms']:.3f} ms ({trace['samples']} samples)"
                )
                logger.info(trace_msg)
                print(trace_msg, flush=True)
        metrics["trace_times"] = trace_times

    # os.makedirs("out", exist_ok=True)
    # timeline_filename = f"out/qwen-dag-pp{args.pp}-dp{args.dp}-{args.schedule}"
    # ray.timeline(timeline_filename)
    # print(f"Ray timeline saved to: {timeline_filename}")
    if args.nsight:
        logger.info("Stopping Piper actors so Nsight Systems reports are flushed")
        try:
            ray.get([actor.__ray_terminate__.remote() for actor in actors.values()])
        except ray.exceptions.ActorDiedError as exc:
            logger.info(f"Piper actors stopped for Nsight flush: {exc}")
    return metrics


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
    parser.add_argument('--activation-checkpointing', action='store_true', default=False)
    parser.add_argument("--zero-stage", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Apply ZeRO memory optimizations")
    parser.add_argument("--bucket-size", type=float, help="Bucket size in MB")
    parser.add_argument("--profiling", action="store_true", default=False,
                        help="Profile each DAG task: log time and memory delta per task")
    parser.add_argument("--nsight", action="store_true", default=False,
                        help="Whether to use Nsight Systems for tracing")
    parser.add_argument("--save-viz", action="store_true", default=False,
                        help="Save per-rank DAG visualization (slow for models > 1B)")
    parser.add_argument("--address", default="",
                        help="Ray head address to connect to")
    parser.add_argument("--port", type=int, default=4567,
                        help="Ray head port to connect to (default: 4567)")
    parser.add_argument("--temp-dir", default="/tmp/piper/ray_tmp",
                        help="Ray temp directory (default: /tmp/piper/ray_tmp)")
    parser.add_argument("--output-dir", default="out",
                        help="Directory for benchmark metrics output (default: out)")
    return parser.parse_args()


"""
python /m-coriander/coriander/mfris/miniconda3/envs/piper/lib/python3.10/site-packages/ray/scripts/scripts.py \
        start --head \
        --port 4567 \
        --temp-dir=/m-coriander/coriander/mfris/piper/ray_tmp \
        --include-dashboard=false
"""
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    metrics_path = _metrics_output_path(args)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    open(metrics_path, "w", encoding="utf-8").close()
    logger.info(args)
    if args.address != "":
        ray.init(
            address=f"{args.address}:{args.port}",
            namespace="qwen",
            log_to_driver=True,
            include_dashboard=False,
            _temp_dir=args.temp_dir,
        )
    else:
        ray.init(
            namespace="qwen",
            log_to_driver=True,
            include_dashboard=False,
            _temp_dir=args.temp_dir,
        )
    pg = placement_group([{"CPU": args.pp, "GPU": args.pp}] * args.dp, strategy="SPREAD")
    ray.get(pg.ready(), timeout=600)
    logger.info(placement_group_table(pg))
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, pg, args, pg)
    dp_metrics = ray.get(handles)
    open(metrics_path, "w", encoding="utf-8").close()
    for metrics in sorted(dp_metrics, key=lambda item: item["dp_rank"]):
        _write_benchmark_metrics(args, metrics)
    logger.info(f"Benchmark metrics saved to {metrics_path}")
    ray.shutdown()
