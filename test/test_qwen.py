import ray
import torch
import torch.nn as nn
import argparse
import time
import os
import numpy as np

from src.piper_coordinator import PiperProgramCoordinator
from src.piper_compile import piper_setup
from src.piper_exec import piper_exec
from src.piper_utils import piper_metadata

from .models.qwen3 import PiperQwen3Model
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


def create_qwen3_config(name: str) -> Qwen3Model.Config:
    """Create Qwen3 model config based on name."""
    match name:
        case 'tiny':
            return Qwen3Model.Config(
                vocab_size=2048,
                dim=256,
                n_layers=8,
                norm_eps=1e-6,
                enable_weight_tying=False,
                layer=Qwen3TransformerBlock.Config(
                    depth_init=True,
                    moe_enabled=True,
                    norm_eps=1e-6,
                    moe=MoE.Config(
                        hidden_dim=128,
                        num_experts=8,
                        top_k=2,
                        use_grouped_mm=True,
                        num_expert_groups=None,
                        num_limited_groups=None,
                        score_func="softmax",
                        route_norm=False,
                        route_scale=1.0,
                        gate_bias=False,
                        score_before_experts=False,
                        num_shared_experts=0,
                        load_balance_coeff=None,
                        _debug_force_load_balance=False,
                    ),
                    feed_forward=FeedForward.Config(hidden_dim=512),
                    attention=GQAttention.Config(
                        n_heads=8,
                        n_kv_heads=4,
                        head_dim=32,
                        qk_norm=True,
                        norm_eps=1e-6,
                        attn_backend="sdpa",
                        rope_backend="cos_sin",
                    ),
                ),
                rope=RoPE.Config(
                    dim=32,
                    max_seq_len=2048,
                    theta=1000000.0,
                    backend="cos_sin",
                ),
            )
        case 'small':
            return Qwen3Model.Config(
                vocab_size=151936,
                dim=1024,
                n_layers=16,
                norm_eps=1e-6,
                enable_weight_tying=False,
                layer=Qwen3TransformerBlock.Config(
                    depth_init=True,
                    moe_enabled=True,
                    norm_eps=1e-6,
                    moe=MoE.Config(
                        hidden_dim=3584,
                        num_experts=2,
                        top_k=2,
                        use_grouped_mm=True,
                        num_expert_groups=None,
                        num_limited_groups=None,
                        score_func="softmax",
                        route_norm=False,
                        route_scale=1.0,
                        gate_bias=False,
                        score_before_experts=False,
                        num_shared_experts=0,
                        load_balance_coeff=None,
                        _debug_force_load_balance=False,
                    ),
                    attention=GQAttention.Config(
                        n_heads=16,
                        n_kv_heads=8,
                        head_dim=64,
                        qk_norm=True,
                        norm_eps=1e-6,
                        attn_backend="sdpa",
                        rope_backend="cos_sin",
                    ),
                ),
                rope=RoPE.Config(
                    dim=64,
                    max_seq_len=2048,
                    theta=1000000.0,
                    backend="cos_sin",
                ),
            )
        case 'medium':
            return Qwen3Model.Config(
                vocab_size=151936,
                dim=2048,
                n_layers=24,
                norm_eps=1e-6,
                enable_weight_tying=False,
                layer=Qwen3TransformerBlock.Config(
                    depth_init=True,
                    moe_enabled=True,
                    norm_eps=1e-6,
                    moe=MoE.Config(
                        hidden_dim=7168,
                        num_experts=8,
                        top_k=2,
                        use_grouped_mm=True,
                        num_expert_groups=None,
                        num_limited_groups=None,
                        score_func="softmax",
                        route_norm=False,
                        route_scale=1.0,
                        gate_bias=False,
                        score_before_experts=False,
                        num_shared_experts=0,
                        load_balance_coeff=None,
                        _debug_force_load_balance=False,
                    ),
                    attention=GQAttention.Config(
                        n_heads=32,
                        n_kv_heads=8,
                        head_dim=64,
                        qk_norm=True,
                        norm_eps=1e-6,
                        attn_backend="sdpa",
                        rope_backend="cos_sin",
                    ),
                ),
                rope=RoPE.Config(
                    dim=64,
                    max_seq_len=2048,
                    theta=1000000.0,
                    backend="cos_sin",
                ),
            )
        case 'large':
            return Qwen3Model.Config(
                vocab_size=151936,
                dim=4096,
                n_layers=32,
                norm_eps=1e-6,
                enable_weight_tying=False,
                layer=Qwen3TransformerBlock.Config(
                    depth_init=True,
                    moe_enabled=True,
                    norm_eps=1e-6,
                    moe=MoE.Config(
                        hidden_dim=14336,
                        num_experts=8,
                        top_k=2,
                        use_grouped_mm=True,
                        num_expert_groups=None,
                        num_limited_groups=None,
                        score_func="softmax",
                        route_norm=False,
                        route_scale=1.0,
                        gate_bias=False,
                        score_before_experts=False,
                        num_shared_experts=0,
                        load_balance_coeff=None,
                        _debug_force_load_balance=False,
                    ),
                    attention=GQAttention.Config(
                        n_heads=32,
                        n_kv_heads=8,
                        head_dim=128,
                        qk_norm=True,
                        norm_eps=1e-6,
                        attn_backend="sdpa",
                        rope_backend="cos_sin",
                    ),
                ),
                rope=RoPE.Config(
                    dim=128,
                    max_seq_len=2048,
                    theta=1000000.0,
                    backend="cos_sin",
                ),
            )
        case _:
            raise ValueError(f"Unknown model config: {name}")


def main(args):
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
    
    # Initialize weights using TorchTitan's method
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device("cuda:0"))
    
    # Create input tensors
    # Qwen3 forward signature: forward(tokens, attention_masks=None, positions=None)
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to('cuda')
    # For Qwen3, we don't need input_pos like Mixtral - positions can be None
    # attention_masks can also be None for causal attention
    
    # Create labels for loss (vocab_size)
    y = torch.randn(batch_size, seq_len, config.vocab_size).to('cuda')

    # Setup Piper
    piper_setup(
        model, 
        torch.optim.Adam, 
        [x],  # Only tokens, not [x, input_pos] like Mixtral
        y,
        schedule,
        args.naive_gradient_sync,
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
    for _ in range(args.iters):
        start = time.perf_counter()
        piper_exec(schedule, loss_fn, args.dp, args.naive_gradient_sync)
        end = time.perf_counter()
        iter_times.append(end - start)
    
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

    timeline_filename = f"out/qwen3-pp{args.pp}-dp{args.dp}"
    ray.timeline(timeline_filename)
    print(f"Ray timeline saved to: {timeline_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description='Run Qwen3 model with pipeline parallelism')
    parser.add_argument('--model', choices=['tiny', 'small', 'medium', 'large'], default='small',
                        help='Model configuration: tiny, small, medium, or large (default: tiny)')
    parser.add_argument('--schedule', choices=['gpipe', '1f1b', 'interleaved-1f1b', 'no-pp'], default='1f1b',
                        help='Schedule type: gpipe, 1f1b, or interleaved-1f1b (default: 1f1b)')
    parser.add_argument('--dp', type=int, default=1,
                        help='Number of data parallel degrees (default: 1)')
    parser.add_argument('--pp', type=int, default=2,
                        help='Number of pipeline parallel degrees (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--seq_len', type=int, default=2048,
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
    return parser.parse_args()


if __name__ == "__main__":
    ray.init(include_dashboard=False, log_to_driver=True, namespace="qwen3", _temp_dir="/m-coriander/coriander/mfris/tmp/ray")
    args = parse_args()
    piper_coordinator = PiperProgramCoordinator.remote(pp_degree=args.pp, dp_degree=args.dp)
    handles = piper_coordinator.run_program.remote(main, args)
    ray.get(handles)
    ray.shutdown()