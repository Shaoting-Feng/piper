import ray
import torch
import torch.fx as fx
import os
import gc
from torch._dynamo.backends.registry import register_backend
from .piper_utils import _serialize_graphmodule, piper_metadata, create_logger, LOG_LEVEL


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
    schedule_to_dag,
    assign_time_steps,
    insert_p2p_ops,
    insert_ar_ops,
    expand_bucket_tasks,
    expand_a2a_tasks,
    overlap_a2a_tasks,
    visualize_dag,
    print_dag_order,
    bucket_stage,
    split_by_a2a,
)
from .piper_exec import Task, TaskType, BatchMeta
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
        refs.append(actor._load_stage.remote(
            stage_id,
            modules_data,
            a2a_boundaries,
            use_activation_checkpointing=piper_metadata.use_activation_checkpointing,
        ))

        all_stage_compiled[stage_id] = (modules_data, a2a_boundaries)

        del stage_gm
        del graphargs
        del placeholders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ray.get(refs)

    piper_metadata.stage_bucket_counts = stage_bucket_counts
    piper_metadata.trainable_bucket_keys = trainable_bucket_keys

    # Build task DAG from the schedule stored by piper_setup.
    if piper_metadata.schedule is not None:
        piper_metadata.task_dag = schedule_to_dag(piper_metadata.schedule)
        logger.debug(
            f"Built task DAG: {len(piper_metadata.task_dag.nodes)} nodes, "
            f"{sum(len(n.data_succs) for n in piper_metadata.task_dag.nodes)} data edges, "
            f"{sum(len(n.temporal_succs) for n in piper_metadata.task_dag.nodes)} temporal edges"
        )

        dag = piper_metadata.task_dag

        # Expand multi-segment stages into per-segment FWD/BWD task chains.
        # This covers both param-bucketed stages (--bucketing) and A2A-split stages
        # (--dp > 1 with MoE), both of which load multiple modules per stage.
        # Only expand stages that actually have more than one segment.
        bucket_counts = {
            sid: n for sid, n in piper_metadata.stage_bucket_counts.items() if n > 1
        }
        if bucket_counts:
            dag = expand_bucket_tasks(dag, bucket_counts)
            logger.debug(
                f"Expanded stages {list(bucket_counts.keys())} into "
                f"per-segment FWD/BWD nodes"
            )

        # Insert FWD_A2A / BWD_A2A nodes for expert-parallel all-to-all boundaries.
        if dp_degree > 1 and stage_a2a_boundaries:
            dag = expand_a2a_tasks(dag, stage_a2a_boundaries)
            logger.debug(
                f"Inserted A2A task nodes for stages {list(stage_a2a_boundaries.keys())}"
            )

            # Overlap compute and A2A tasks across FWD_BWD task pairs so that
            # one task's A2A communication runs concurrently with the other's compute.
            fwdbwd_pairs: list[tuple[Task, Task]] = []
            for row in piper_metadata.schedule.grid:
                for cell in row:
                    if cell is not None and cell.type == TaskType.FWD_BWD:
                        fwd_task = Task(
                            pp_rank=cell.pp_rank,
                            batches=[cell.batches[0]],
                            type=TaskType.FWD,
                        )
                        bwd_task = Task(
                            pp_rank=cell.pp_rank,
                            batches=[cell.batches[1]],
                            type=TaskType.BWD,
                        )
                        fwdbwd_pairs.append((fwd_task, bwd_task))
            # fwdbwd_pairs.append((
            #     Task(pp_rank=0, batches=[BatchMeta(0, 1)], type=TaskType.FWD), 
            #     Task(pp_rank=0, batches=[BatchMeta(0, 2)], type=TaskType.FWD)))
            if fwdbwd_pairs:
                dag = overlap_a2a_tasks(dag, fwdbwd_pairs)
                logger.debug(
                    f"Overlapped A2A and compute tasks for {len(fwdbwd_pairs)} FWD_BWD pairs"
                )

        # Assign final time_step values from the temporal dependency graph.
        assign_time_steps(dag)

        # Split into per-rank DAGs.
        per_rank_dags = insert_p2p_ops(dag)

        # Insert all-reduce nodes when using data parallelism.
        if dp_degree > 1:
            per_rank_dags = insert_ar_ops(per_rank_dags, piper_metadata.trainable_bucket_keys)
            logger.debug("Inserted ALL_REDUCE nodes for DP gradient sync")
            # ZeRO-1/2/3: insert AG/RS into the task DAG before/after the appropriate FWD/BWD tasks.


        piper_metadata.per_rank_dags = per_rank_dags
        actors = piper_metadata.actors
        for pp_rank, per_rank_dag in enumerate(piper_metadata.per_rank_dags):
            try:
                visualize_dag(per_rank_dag, output_path=f"figs/rank{pp_rank}_dag")
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

def piper_exec_dag(loss_fn) -> list:
    """Execute one training step using the per-rank TaskDAG and :meth:`run_dag`.

    Per-rank DAGs are built and loaded onto actors by the piper backend during
    compilation.  This function just invokes ``run_dag`` in parallel on all actors.

    Returns:
        Flat list of per-microbatch losses collected from all actors.
    """
    assert piper_metadata.per_rank_dags is not None, (
        "per_rank_dags is None — ensure the model was compiled with the piper backend "
        "before calling piper_exec_dag()"
    )
    actors = piper_metadata.actors
    run_refs = [
        actors[pp_rank].run_dag.remote(loss_fn=loss_fn)
        for pp_rank in range(len(piper_metadata.per_rank_dags))
    ]
    results = ray.get(run_refs)
    return [item for sublist in results if sublist for item in sublist]
