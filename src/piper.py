import ray
import torch
import torch.fx as fx
import os
import gc
import pickle
import time
import threading
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
    overlap_zero_ops,
    find_overlappable_tasks,
    insert_zero_ops,
    overlap_a2a_tasks,
    visualize_dag,
    compute_critical_path,
    print_dag_order,
    bucket_stage,
    apply_activation_checkpointing,
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
    zero_bucket_keys: set = set()
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
            # Hack: only bucket even-indexed segments to avoid bucketing MoE subgraphs
            if piper_metadata.bucket_size is not None:
                if seg_idx % 2 == 0:
                    buckets = bucket_stage(
                        seg_gm,
                        seg_args,
                        seg_in,
                        seg_param,
                        bucket_size_bytes=piper_metadata.bucket_size,
                    )
                    bucket_zero_flags = [True] * len(buckets)
                else:
                    bucket_zero_flags = [False]
            else:
                buckets = [(seg_gm, seg_in, seg_param, seg_args)]
                if seg_idx % 2 == 0:
                    bucket_zero_flags = [True]
                else:
                    bucket_zero_flags = [False]

            logger.debug(
                f"Stage {stage_id} segment {seg_idx} bucketed into {len(buckets)} buckets"
            )
            for b_idx, (_, _, bp, ba) in enumerate(buckets):
                for i in bp:
                    arg = ba[i] if i < len(ba) else None
                    if arg is not None and hasattr(arg, "numel"):
                        mb = arg.numel() * arg.element_size() / (1024 * 1024)
                        logger.debug(
                            f"  bucket {b_idx} tensor idx={i} shape={tuple(arg.shape)} size={mb:.3f}MB"
                        )

            for (bucket_gm, bucket_in, bucket_param, bucket_args), apply_zero in zip(
                buckets, bucket_zero_flags
            ):
                original_bucket_gm = bucket_gm
                shared_placeholder_names = [
                    n.name for n in original_bucket_gm.graph.nodes if n.op == "placeholder"
                ]
                requested_ac_subgraphs = 1
                if piper_metadata.use_activation_checkpointing:
                    requested_ac_subgraphs = piper_metadata.activation_num_checkpoints
                    ac_gms = apply_activation_checkpointing(
                        bucket_gm,
                        bucket_args,
                        bucket_in,
                        bucket_param,
                        piper_metadata.activation_num_checkpoints,
                    )
                    if len(ac_gms) > 1:
                        bucket_gm = ac_gms
                actual_ac_subgraphs = len(bucket_gm) if isinstance(bucket_gm, list) else 1
                if piper_metadata.use_activation_checkpointing:
                    logger.log(VERBOSE,
                        f"[ac_compile_bucket] stage={stage_id} segment={seg_idx} "
                        f"requested_subgraphs={requested_ac_subgraphs} "
                        f"actual_subgraphs={actual_ac_subgraphs} "
                        f"inputs={len(bucket_in)} params={len(bucket_param)}"
                    )
                all_modules.append(
                    (
                        bucket_gm,
                        bucket_in,
                        bucket_param,
                        bucket_args,
                        original_bucket_gm,
                        shared_placeholder_names,
                        apply_zero,
                    )
                )
            seg_end_bucket = len(all_modules) - 1  # inclusive index of last bucket in this segment

            if seg_idx < len(boundary_infos):
                binfo = boundary_infos[seg_idx]
                a2a_boundaries[seg_end_bucket] = binfo["tensor_idx"]

        if a2a_boundaries:
            stage_a2a_boundaries[stage_id] = a2a_boundaries

        stage_bucket_counts[stage_id] = len(all_modules)
        for b_idx, (
            bgm,
            b_in,
            b_param,
            bargs,
            _original_gm,
            _shared_placeholder_names,
            apply_zero,
        ) in enumerate(all_modules):
            has_trainable = any(
                bargs[i] is not None and getattr(bargs[i], "requires_grad", False)
                for i in b_param
            )
            logger.debug(
                f"[zero_bucket_classify] stage={stage_id} bucket={b_idx} "
                f"has_trainable={has_trainable} apply_zero={apply_zero} "
                f"param_idxs={b_param}"
            )
            if has_trainable:
                trainable_bucket_keys.add((stage_id, b_idx))
                if apply_zero:
                    zero_bucket_keys.add((stage_id, b_idx))
            else:
                logger.warning(
                    f"Stage {stage_id} bucket {b_idx} has no trainable parameters."
                )

        modules_data = []
        for bgm, b_in, b_param, bargs, original_gm, shared_placeholder_names, apply_zero in all_modules:
            if isinstance(bgm, list):
                modules_data.append(
                    {
                        "gm_data_list": [_serialize_graphmodule(sgm) for sgm in bgm],
                        "ac_num_subgraphs": len(bgm),
                        "ac_requested_subgraphs": piper_metadata.activation_num_checkpoints,
                        "graphargs": bargs,
                        "input_idxs": b_in,
                        "param_idxs": b_param,
                        "shared_placeholder_names": shared_placeholder_names,
                        "apply_zero": apply_zero,
                        "triton_constant_args_list": [
                            _collect_triton_constant_args(sgm) for sgm in bgm
                        ],
                    }
                )
            else:
                modules_data.append(
                    {
                        "gm_data": _serialize_graphmodule(bgm),
                        "ac_num_subgraphs": 1,
                        "ac_requested_subgraphs": (
                            piper_metadata.activation_num_checkpoints
                            if piper_metadata.use_activation_checkpointing
                            else 1
                        ),
                        "graphargs": bargs,
                        "input_idxs": b_in,
                        "param_idxs": b_param,
                        "shared_placeholder_names": shared_placeholder_names,
                        "apply_zero": apply_zero,
                        "triton_constant_args": _collect_triton_constant_args(bgm),
                    }
                )

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
    for stage_id, (modules_data, a2a_boundaries, actor) in all_stage_compiled.items():
        ubid_offset = _stage_ubid_offsets[stage_id]
        for b_idx, md in enumerate(modules_data):
            md["unique_bucket_id"] = ubid_offset + b_idx
        actor._load_stage.remote(
            stage_id,
            modules_data,
            a2a_boundaries,
            use_activation_checkpointing=piper_metadata.use_activation_checkpointing,
        )
        # Remove actor from stored compiled data (not needed downstream)
        all_stage_compiled[stage_id] = (modules_data, a2a_boundaries)

    logger.debug("Sent stages to actors for loading")

    piper_metadata.stage_bucket_counts = stage_bucket_counts
    piper_metadata.trainable_bucket_keys = trainable_bucket_keys
    piper_metadata.zero_bucket_keys = zero_bucket_keys
    stage_data = {
        "stages": all_stage_compiled,
        "stage_bucket_counts": stage_bucket_counts,
        "trainable_bucket_keys": trainable_bucket_keys,
        "zero_bucket_keys": zero_bucket_keys,
    }
    compiled_store = getattr(piper_metadata, "compiled_data_store", None)
    if compiled_store is not None:
        ray.get(compiled_store.publish_stage_data.remote(stage_data))
        logger.debug("Published compiled stage data")

    # Build task DAG from the schedule stored by piper_setup.
    if piper_metadata.schedule is not None:
        dag = expand_chunks_to_dags(
            piper_metadata.schedule,
            piper_metadata.stage_bucket_counts,
            stage_a2a_boundaries,
        )
        logger.debug(f"Built task DAG: {len(dag.nodes)} nodes")

        add_temporal_dependencies(dag, piper_metadata.schedule)
        logger.debug(f"Added temporal dependencies")

        # Save a deep copy of the full DAG before split_dag_by_rank removes
        # cross-rank data edges.  Used later for profiling and critical-path analysis.
        piper_metadata.full_dag_no_overlap = pickle.loads(pickle.dumps(dag))

        # Split into per-rank DAGs.
        per_rank_dags = split_dag_by_rank(dag)

        # Insert ZeRO/DP communication tasks.
        if dp_degree > 1:
            logger.debug(
                f"ZeRO bucket filter: {sorted(piper_metadata.zero_bucket_keys)}"
            )
            per_rank_dags = insert_zero_ops(
                per_rank_dags,
                piper_metadata.zero_stage,
                zero_bucket_keys=piper_metadata.zero_bucket_keys,
                gradient_accumulation=piper_metadata.gradient_accumulation,
            )
            logger.debug(
                f"Inserted ZeRO-{piper_metadata.zero_stage} collective nodes "
                f"(gradient_accumulation={piper_metadata.gradient_accumulation})"
            )

            overlappable = find_overlappable_tasks(piper_metadata.schedule)
            for t1, t2 in overlappable:
                logger.debug(
                    f"Found adjacent task pair for A2A/compute overlap: "
                    f"{t1} -> {t2}"
                )

        # Assign time steps
        for rank_dag in per_rank_dags:
            assign_time_steps(rank_dag)
            if piper_metadata.overlap_zero_ops:
                overlap_zero_ops(rank_dag)
        logger.debug("Assigned time steps")

        piper_metadata.per_rank_dags = per_rank_dags

        # Visualize DAG
        actors = piper_metadata.actors
        for pp_rank, per_rank_dag in enumerate(per_rank_dags):
            if piper_metadata.visualize_dag:
                try:
                    visualize_dag(per_rank_dag, output_path=f"out/rank{pp_rank}_dag")
                except Exception as e:
                    logger.warning(f"DAG visualization failed for rank {pp_rank} (DAG may be too large for dot): {e}")
            print_dag_order(per_rank_dag, label=f"rank {pp_rank}", rank=pp_rank)


        # Send DAG to actors
        refs = [actors[pp_rank].load_dag.remote(per_rank_dag)
            for pp_rank, per_rank_dag in enumerate(piper_metadata.per_rank_dags)]

        logger.debug(
            f"Sent per-rank DAGs onto {len(piper_metadata.per_rank_dags)} actors"
        )
        dag_data = {
            "per_rank_dags": per_rank_dags,
            "full_dag_no_overlap": piper_metadata.full_dag_no_overlap,
        }
        if compiled_store is not None:
            ray.get(compiled_store.publish_dag_data.remote(dag_data))
            logger.debug("Published compiled DAG data")

        # Publish compiled data so dp_rank > 0 can skip torch.compile.
        piper_metadata.compiled_stage_data = {
            "stages": all_stage_compiled,           # {stage_id: (modules_data, a2a_boundaries)}
            "stage_bucket_counts": stage_bucket_counts,
            "trainable_bucket_keys": trainable_bucket_keys,
            "zero_bucket_keys": zero_bucket_keys,
            "per_rank_dags": per_rank_dags,
            "full_dag_no_overlap": piper_metadata.full_dag_no_overlap,
        }

    def callback(*args, _gm=gm):
        logger.warning("Should not directly call compiled module, running non-distributed execution")
        return _gm(*args)

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
        _log_step_stats(step_time, log_stats, actors, results)

    losses = []
    for result in results:
        if isinstance(result, dict):
            losses.extend(result.get("losses", []))
        elif result:
            losses.extend(result)
    return losses


def _log_step_stats(step_time: float, log_memory: bool, actors: dict, results: list | None = None) -> None:
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
