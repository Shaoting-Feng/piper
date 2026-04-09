import ray
import torch
import torch.fx as fx
import os
import gc
import pickle
import time
from torch._dynamo.backends.registry import register_backend
from .piper_utils import _serialize_graphmodule, piper_metadata, create_logger, LOG_LEVEL, get_gpu_peak_flops_bf16


def _collect_triton_constant_args(gm):
    """Return {constant_args_idx: args_dict} for every triton_kernel_wrapper node in gm."""
    try:
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
    except ImportError:
        return {}

    result = {}

    def scan(module):
        if isinstance(module, fx.GraphModule):
            for node in module.graph.nodes:
                if node.op == "call_function":
                    target_str = getattr(node.target, "__name__", "") or str(node.target)
                    if "triton_kernel_wrapper" in target_str:
                        idx = node.kwargs.get("constant_args_idx")
                        if idx is not None and idx in kernel_side_table.constant_args:
                            result[idx] = kernel_side_table.constant_args[idx]
        for child in module.children():
            scan(child)

    scan(gm)
    return result
from .piper_graph_transform import (
    _split_gm_by_stages,
    _profile_and_split_gm,
    expand_chunks_to_dags,
    add_temporal_dependencies,
    split_dag_by_rank,
    assign_time_steps,
    find_overlappable_tasks,
    overlap_a2a_tasks,
    visualize_dag,
    compute_critical_path,
    print_dag_order,
    bucket_stage,
    split_by_a2a,
)
from .piper_exec import Chunk, TaskType, BatchMeta
from .piper_actor import _get_actor

logger = create_logger("piper_backend", LOG_LEVEL)


# ---------------------------------------------------------------------------
# Piper torch.compile backend
# ---------------------------------------------------------------------------

@register_backend
def piper(gm, example_inputs, **kwargs):

    # gm.print_readable()

    original_gm = gm
    num_stages = len(piper_metadata.stage_to_device.keys())

    # Check if the graph has stage annotations
    has_annotations = any(
        isinstance(node.meta.get('custom'), dict) and node.meta['custom'].get('stage') is not None
        for node in gm.graph.nodes
    )

    logger.debug(f"Graph has stage annotations: {has_annotations}")

    if has_annotations:
        top_level_gm, submodules = _split_gm_by_stages(gm)
    else:
        logger.info(f"No stage annotations found, profiling graph to split into {num_stages} stages")
        top_level_gm, submodules = _profile_and_split_gm(gm, num_stages)

    dp_degree = int(os.environ['PIPER_DP_DEGREE'])

    del top_level_gm

    refs = []
    actor_stages = []
    # stage_id -> {boundary_bucket_id -> tensor_idx} collected for DAG expansion
    stage_a2a_boundaries: dict = {}
    stage_bucket_counts: dict = {}
    trainable_bucket_keys: set = set()
    # Collected for DP broadcasting: stage_id -> (modules_data, a2a_boundaries)
    all_stage_compiled: dict = {}
    # Running counter for assigning globally unique bucket IDs (same ordering as
    # expand_chunks_to_dags Pass 1.5, which sorts by stage_id).  Populated after
    # all stages are compiled so we can compute offsets in a second pass.
    _stage_ubid_offsets: dict = {}  # stage_id -> first unique_bucket_id for that stage

    for (stage_id, stage_gm, input_idxs, param_idxs, graphargs, placeholders) in submodules:
        actor_id = piper_metadata.stage_to_device[stage_id]
        actor = _get_actor(actor_id)
        actor_stages.append((actor, stage_id))

        # Split at A2A annotation boundaries (expert-parallel only, requires dp_degree > 1).
        if dp_degree > 1:
            a2a_segments, boundary_infos = split_by_a2a(stage_gm, graphargs, input_idxs, param_idxs)
            if boundary_infos:
                logger.debug(
                    f"Stage {stage_id} split into {len(a2a_segments)} A2A segments "
                    f"with {len(boundary_infos)} boundaries"
                )
        else:
            a2a_segments = [(stage_gm, input_idxs, param_idxs, graphargs)]
            boundary_infos = []

        # Apply parameter bucketing only to even-indexed segments (segments outside A2A pairs).
        all_modules: list = []
        a2a_boundaries: dict = {}  # boundary_bucket_id -> tensor_idx

        for seg_idx, (seg_gm, seg_in, seg_param, seg_args) in enumerate(a2a_segments):
            if piper_metadata.bucketing and seg_idx % 2 == 0:
                buckets = bucket_stage(seg_gm, seg_args, seg_in, seg_param)
                if len(buckets) > 1:
                    logger.debug(
                        f"Stage {stage_id} segment {seg_idx} bucketed into {len(buckets)} buckets"
                    )
            else:
                buckets = [(seg_gm, seg_in, seg_param, seg_args)]

            all_modules.extend(buckets)
            seg_end_bucket = len(all_modules) - 1  # inclusive index of last bucket in this segment

            if seg_idx < len(boundary_infos):
                binfo = boundary_infos[seg_idx]
                a2a_boundaries[seg_end_bucket] = binfo["tensor_idx"]

        if a2a_boundaries:
            stage_a2a_boundaries[stage_id] = a2a_boundaries

        stage_bucket_counts[stage_id] = len(all_modules)
        for b_idx, (bgm, b_in, b_param, bargs) in enumerate(all_modules):
            has_trainable = any(
                bargs[i] is not None and getattr(bargs[i], "requires_grad", False)
                for i in b_param
            )
            if has_trainable:
                trainable_bucket_keys.add((stage_id, b_idx))
            else:
                logger.warning(
                    f"Stage {stage_id} bucket {b_idx} has no trainable parameters."
                )

        modules_data = [
            {
                "gm_data": _serialize_graphmodule(bgm),
                "graphargs": bargs,
                "input_idxs": b_in,
                "param_idxs": b_param,
                "triton_constant_args": _collect_triton_constant_args(bgm),
            }
            for bgm, b_in, b_param, bargs in all_modules
        ]

        all_stage_compiled[stage_id] = (modules_data, a2a_boundaries, actor)

        del stage_gm
        del graphargs
        del placeholders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute globally unique bucket IDs for each (stage_id, stage_bucket_id) pair.
    # Stages are sorted by stage_id to match the ordering in expand_chunks_to_dags
    # Pass 1.5, ensuring DAG task.unique_bucket_id matches actor data structure keys.
    _ubid = 0
    for s in sorted(stage_bucket_counts.keys()):
        _stage_ubid_offsets[s] = _ubid
        _ubid += stage_bucket_counts[s]

    # Now dispatch _load_stage with unique_bucket_ids embedded in modules_data.
    refs = []
    for stage_id, (modules_data, a2a_boundaries, actor) in all_stage_compiled.items():
        K = stage_bucket_counts[stage_id]
        ubid_offset = _stage_ubid_offsets[stage_id]
        for b_idx, md in enumerate(modules_data):
            md["unique_bucket_id"] = ubid_offset + b_idx
        refs.append(actor._load_stage.remote(
            stage_id,
            modules_data,
            a2a_boundaries,
            use_activation_checkpointing=piper_metadata.use_activation_checkpointing,
        ))
        # Remove actor from stored compiled data (not needed downstream)
        all_stage_compiled[stage_id] = (modules_data, a2a_boundaries)

    ray.get(refs)

    piper_metadata.stage_bucket_counts = stage_bucket_counts
    piper_metadata.trainable_bucket_keys = trainable_bucket_keys

    # Build task DAG from the schedule stored by piper_setup.
    if piper_metadata.schedule is not None:
        dag = expand_chunks_to_dags(
            piper_metadata.schedule,
            piper_metadata.stage_bucket_counts,
            stage_a2a_boundaries,
            dp_degree,
        )
        logger.debug(
            f"Built task DAG: {len(dag.nodes)} nodes, "
            f"{sum(len(n.data_succs) for n in dag.nodes)} data edges"
        )

        add_temporal_dependencies(dag, piper_metadata.schedule)
        logger.debug(
            f"Added temporal dependencies: "
            f"{sum(len(n.temporal_succs) for n in dag.nodes)} temporal edges"
        )

        if dp_degree > 1 and stage_a2a_boundaries:
            overlappable = find_overlappable_tasks(piper_metadata.schedule)
            for t1, t2 in overlappable:
                logger.info(
                    f"Found adjacent task pair for A2A/compute overlap: "
                    f"{t1} -> {t2}"
                )
            # if overlappable:
            #     dag = overlap_a2a_tasks(dag, overlappable)
            #     logger.debug(
            #         f"Overlapped {len(overlappable)} adjacent task pair(s) for A2A/compute overlap"
            #     )

        assign_time_steps(dag)

        # Save a deep copy of the full DAG before split_dag_by_rank removes
        # cross-rank data edges.  Used later for profiling and critical-path analysis.
        piper_metadata.full_dag_no_overlap = pickle.loads(pickle.dumps(dag))

        # Split into per-rank DAGs.
        per_rank_dags = split_dag_by_rank(dag)

        piper_metadata.per_rank_dags = per_rank_dags
        actors = piper_metadata.actors
        if piper_metadata.visualize_dag:
            for pp_rank, per_rank_dag in enumerate(piper_metadata.per_rank_dags):
                try:
                    visualize_dag(per_rank_dag, output_path=f"out/rank{pp_rank}_dag")
                except Exception as e:
                    logger.warning(f"DAG visualization failed for rank {pp_rank} (DAG may be too large for dot): {e}")
            # uncomment for debugging DAG construction:
            # print_dag_order(per_rank_dag, label=f"rank {pp_rank}")
        ray.get([
            actors[pp_rank].load_dag.remote(per_rank_dag)
            for pp_rank, per_rank_dag in enumerate(piper_metadata.per_rank_dags)
        ])
        logger.debug(
            f"Loaded per-rank DAGs onto {len(piper_metadata.per_rank_dags)} actors"
        )

        # Publish compiled data so dp_rank > 0 can skip torch.compile.
        piper_metadata.compiled_stage_data = {
            "stages": all_stage_compiled,           # {stage_id: (modules_data, a2a_boundaries)}
            "stage_bucket_counts": stage_bucket_counts,
            "trainable_bucket_keys": trainable_bucket_keys,
            "per_rank_dags": per_rank_dags,
            "full_dag_no_overlap": piper_metadata.full_dag_no_overlap,
        }

    example_outputs = original_gm(*example_inputs)

    def callback(*args):
        # assert False
        logger.warning("Should not directly call compiled module, running non-distributed execution")
        return example_outputs

    del original_gm
    del example_inputs

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return callback


# ---------------------------------------------------------------------------
# DAG-based execution entry point
# ---------------------------------------------------------------------------

def piper_exec_dag(loss_fn, profiling: bool = False, log_stats: bool = False) -> list:
    """Execute one training step using the per-rank TaskDAG and :meth:`run_dag`.

    Per-rank DAGs are built and loaded onto actors by the piper backend during
    compilation.  This function just invokes ``run_dag`` in parallel on all actors.

    Args:
        loss_fn: Loss function passed through to each actor's run_dag.
        profiling: If True, each actor times every node and accumulates the
            measurement into ``node.profiling_measurements``.
        log_stats: If True, log step time, throughput, MFU (when
            ``model_flops_per_token`` was set in :func:`piper_setup`), and
            peak GPU memory for each rank.

    Returns:
        Flat list of per-microbatch losses collected from all actors.
    """
    assert piper_metadata.per_rank_dags is not None, (
        "per_rank_dags is None — ensure the model was compiled with the piper backend "
        "before calling piper_exec_dag()"
    )
    actors = piper_metadata.actors
    run_refs = [
        actors[pp_rank].run_dag.remote(loss_fn=loss_fn, profiling=profiling)
        for pp_rank in range(len(piper_metadata.per_rank_dags))
    ]
    t0 = time.perf_counter()
    results = ray.get(run_refs)
    step_time = time.perf_counter() - t0

    if log_stats or piper_metadata.model_flops_per_token is not None:
        _log_step_stats(step_time, log_stats, actors)

    return [item for sublist in results if sublist for item in sublist]


def _log_step_stats(step_time: float, log_memory: bool, actors: dict) -> None:
    """Log throughput, MFU, and optionally per-rank peak GPU memory."""
    stats = [f"step_time={step_time:.3f}s"]

    tokens = piper_metadata.tokens_per_step
    if tokens is not None:
        stats.append(f"throughput={tokens / step_time:.1f} tok/s")

    flops_per_token = piper_metadata.model_flops_per_token
    if tokens is not None and flops_per_token is not None:
        peak_flops = 1979e12
        if peak_flops is not None:
            dp_degree = int(os.environ.get("PIPER_DP_DEGREE", "1"))
            total_gpus = len(actors) * dp_degree
            achieved_flops = flops_per_token * tokens / step_time
            mfu = achieved_flops / (peak_flops * total_gpus)
            stats.append(f"MFU={mfu:.2%}")
        else:
            logger.warning(
                "MFU not computed: GPU peak FLOPs unknown for this device. "
                "Add it to _GPU_PEAK_FLOPS_BF16 in piper_utils.py."
            )

    if log_memory:
        mem_refs = [
            actors[pp_rank].get_and_reset_peak_memory_stats.remote()
            for pp_rank in range(len(piper_metadata.per_rank_dags))
        ]
        for global_rank, max_alloc_bytes in ray.get(mem_refs):
            stats.append(f"rank{global_rank}_peak_mem={max_alloc_bytes / 1e9:.2f}GB")

    logger.info("  ".join(stats))
