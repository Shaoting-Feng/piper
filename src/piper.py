import ray
import torch
import os
import gc
from torch._dynamo.backends.registry import register_backend
from .piper_utils import _serialize_graphmodule, piper_metadata, create_logger, LOG_LEVEL
from .piper_graph_transform import (
    _get_dp_comm_ops,
    _split_gm_by_stages,
    _profile_and_split_gm,
    _insert_a2a_ops,
    _insert_p2p_ops,
    schedule_to_dag,
    insert_p2p_ops,
    insert_ar_ops,
    expand_bucket_tasks,
    expand_a2a_tasks,
    visualize_dag,
    visualize_schedule,
    print_dag_order,
    bucket_stage,
    split_by_a2a,
)
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

    if has_annotations:
        top_level_gm, submodules = _split_gm_by_stages(gm)
    else:
        logger.info(f"No stage annotations found, profiling graph to split into {num_stages} stages")
        top_level_gm, submodules = _profile_and_split_gm(gm, num_stages)

    dp_rank = int(os.environ['PIPER_DP_RANK'])
    pp_degree = int(os.environ['PIPER_PP_DEGREE'])
    dp_degree = int(os.environ['PIPER_DP_DEGREE'])

    del top_level_gm

    refs = []
    actor_stages = []
    # stage_id -> {boundary_bucket_id -> tensor_idx} collected for DAG expansion
    stage_a2a_boundaries: dict = {}

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

        modules_data = [
            {
                "gm_data": _serialize_graphmodule(bgm),
                "graphargs": bargs,
                "input_idxs": b_in,
                "param_idxs": b_param,
            }
            for bgm, b_in, b_param, bargs in all_modules
        ]
        refs.append(actor._load_stage.remote(
            stage_id,
            modules_data,
            a2a_boundaries,
            use_activation_checkpointing=piper_metadata.use_activation_checkpointing,
        ))

        del stage_gm
        del graphargs
        del placeholders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ray.get(refs)

    # TODO: build dag by analyzing stage dependencies
    # this only supports sequential pipelines
    for stage_id in piper_metadata.stage_to_device.keys():
        if stage_id < len(piper_metadata.stage_to_device.keys()) - 1:
            piper_metadata.dag.add((stage_id, stage_id+1))

    # Build task DAG from the schedule stored by piper_setup.
    if piper_metadata.schedule is not None:
        piper_metadata.task_dag = schedule_to_dag(piper_metadata.schedule)
        logger.info(
            f"Built task DAG: {len(piper_metadata.task_dag.nodes)} nodes, "
            f"{sum(len(n.data_succs) for n in piper_metadata.task_dag.nodes)} data edges, "
            f"{sum(1 for n in piper_metadata.task_dag.nodes if n.temporal_succ is not None)} temporal edges"
        )

        dag = piper_metadata.task_dag

        # Expand multi-segment stages into per-segment FWD/BWD task chains.
        # This covers both param-bucketed stages (--bucketing) and A2A-split stages
        # (--dp > 1 with MoE), both of which load multiple modules per stage.
        bucket_counts: dict = {}
        for actor in piper_metadata.actors.values():
            counts = ray.get(actor.get_bucket_fwd_counts.remote())
            bucket_counts.update(counts)
        # Only expand stages that actually have more than one segment.
        bucket_counts = {sid: n for sid, n in bucket_counts.items() if n > 1}
        if bucket_counts:
            dag = expand_bucket_tasks(dag, bucket_counts)
            logger.info(
                f"Expanded stages {list(bucket_counts.keys())} into "
                f"per-segment FWD/BWD nodes"
            )

        # Insert FWD_A2A / BWD_A2A nodes for expert-parallel all-to-all boundaries.
        if dp_degree > 1 and stage_a2a_boundaries:
            dag = expand_a2a_tasks(dag, stage_a2a_boundaries)
            logger.info(
                f"Inserted A2A task nodes for stages {list(stage_a2a_boundaries.keys())}"
            )

        # Split into per-rank DAGs.
        per_rank_dags = insert_p2p_ops(dag)

        # Insert all-reduce nodes when using data parallelism.
        if dp_degree > 1:
            per_rank_dags = insert_ar_ops(per_rank_dags)
            logger.info("Inserted ALL_REDUCE nodes for DP gradient sync")

        piper_metadata.per_rank_dags = per_rank_dags
        actors = piper_metadata.actors
        for pp_rank, per_rank_dag in enumerate(piper_metadata.per_rank_dags):
            visualize_dag(per_rank_dag, output_path=f"figs/rank{pp_rank}_dag")
            print_dag_order(per_rank_dag, label=f"rank {pp_rank}")
        ray.get([
            actors[pp_rank].load_dag.remote(per_rank_dag)
            for pp_rank, per_rank_dag in enumerate(piper_metadata.per_rank_dags)
        ])
        logger.info(
            f"Loaded per-rank DAGs onto {len(piper_metadata.per_rank_dags)} actors"
        )

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
