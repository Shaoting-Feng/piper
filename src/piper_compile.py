import ray
import torch
from torch._dynamo.backends.debugging import eager
import torch.distributed as dist
from torch._dynamo.backends.debugging import eager
import torch.distributed as dist
import threading
import os
import gc
import copy
import itertools

from torch._dynamo.backends.debugging import eager

from .piper_actor import _create_actors
from .piper_utils import piper_metadata, create_logger, LOG_LEVEL
from .piper_exec import DAGEdge, Schedule2D, CompType, Task, BatchMeta
from .piper import piper

logger = create_logger("piper_compile", LOG_LEVEL)

_RANK0_ADDR_ACTOR = "piper_rank0_addr"
_COMPILED_DATA_ACTOR = "piper_compiled_data"

@ray.remote
class _Rank0AddrStore:
    """Named actor used to share global rank-0's IP+port across independent dp_rank workers."""
    def __init__(self, addr: str, port: int):
        self._addr = addr
        self._port = port
    def get(self):
        return self._addr, self._port


@ray.remote
class _CompiledDataStore:
    """Named actor used to share compiled stage/DAG data from dp_rank=0 to dp_rank>0.

    dp_rank=0 calls publish() once after compilation.  All other dp_ranks poll
    is_ready() and then call get() to retrieve the data, avoiding a redundant
    torch.compile call for large models.
    """
    def __init__(self):
        self._data = None
        self._ready = False

    def publish(self, data: dict):
        self._data = data
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get(self) -> dict:
        return self._data



def _validate_schedule(schedule: list[list[Task | None]], dag_edges: list[DAGEdge], num_mbs: int) -> None:
    """
    Validate that the schedule respects well-formedness rules and DAG dependencies.
    
    Args:
        schedule: 2D array with one row per device and one column per time step
        dag_edges: List of DAG edges defining stage dependencies
        num_mbs: Number of microbatches in the schedule
        
    Raises:
        ValueError: If the schedule violates any validation rules
    """
    num_devices, num_steps = len(schedule), len(schedule[0]) if schedule else 0
    
    for row in schedule:
        assert len(row) == num_steps, "Each row must have the same number of time steps"
    
    # Check well-formedness: no duplicates, pp_rank matches row, and all stages present
    all_tasks = set()
    microbatch_tasks = {}  # mb_idx -> set of (stage_id, type)
    
    for pp_rank in range(num_devices):
        for time_step in range(num_steps):
            task = schedule[pp_rank][time_step]
            if task is not None:
                # Check pp_rank matches row
                if task.pp_rank != pp_rank:
                    raise ValueError(
                        f"Task pp_rank {task.pp_rank} does not match row {pp_rank} "
                        f"at time step {time_step}"
                    )
                
                # Check for duplicates and track by (stage_id, mb_idx, type) per batch
                for batch in task.batches:
                    comp_type = (
                        task.type
                        if task.type != CompType.FWD_BWD
                        else (CompType.FWD if batch is task.batches[0] else CompType.BWD)
                    )
                    task_key = (batch.stage_id, batch.mb_idx, comp_type)
                    if task_key in all_tasks:
                        raise ValueError(
                            f"Duplicate task found: stage_id={batch.stage_id}, "
                            f"mb_idx={batch.mb_idx}, type={comp_type}"
                        )
                    all_tasks.add(task_key)
                    if batch.mb_idx not in microbatch_tasks:
                        microbatch_tasks[batch.mb_idx] = set()
                    microbatch_tasks[batch.mb_idx].add((batch.stage_id, comp_type))
    
    # Get all required stages from DAG edges
    all_required_stages = set()
    for edge in dag_edges:
        all_required_stages.add(edge.from_stage)
        all_required_stages.add(edge.to_stage)
    
    # Check that each microbatch has all required forward and backward stages
    for mb_idx, tasks in microbatch_tasks.items():
        # Find all stages that have forward/backward tasks for this microbatch
        fwd_stages = {stage_id for stage_id, task_type in tasks if task_type == CompType.FWD}
        bwd_stages = {
            stage_id
            for stage_id, task_type in tasks
            if task_type in (CompType.BWD, CompType.BWD_I)
        }
        
        # Check that all required stages have forward tasks
        missing_fwd = all_required_stages - fwd_stages
        if missing_fwd:
            raise ValueError(f"Microbatch {mb_idx} missing forward stages: {missing_fwd}")

        # Check that all required stages have backward tasks (BWD or BWD_I)
        missing_bwd = all_required_stages - bwd_stages
        if missing_bwd:
            raise ValueError(
                f"Microbatch {mb_idx} missing backward stages: {missing_bwd} "
                f"(need BWD, BWD_I, or FWD_BWD backward for each stage)"
            )
    
    # Check pipeline stage dependencies
    for mb_idx in range(num_mbs):
        # Find all tasks for this microbatch
        fwd_times: dict[int, int] = {}  # stage_id -> time_step
        bwd_times: dict[int, int] = {}  # stage_id -> time_step (BWD or BWD_I)
        bwd_w_times: dict[int, int] = {}  # stage_id -> time_step (BWD_W)
        
        for pp_rank in range(num_devices):
            for time_step in range(num_steps):
                task = schedule[pp_rank][time_step]
                if task is None:
                    continue
                for batch in task.batches:
                    if batch.mb_idx != mb_idx:
                        continue
                    comp_type = (
                        task.type
                        if task.type != CompType.FWD_BWD
                        else (CompType.FWD if batch is task.batches[0] else CompType.BWD)
                    )
                    if comp_type == CompType.FWD:
                        fwd_times[batch.stage_id] = time_step
                    elif comp_type in (CompType.BWD, CompType.BWD_I):
                        bwd_times[batch.stage_id] = time_step
                    elif comp_type == CompType.BWD_W:
                        bwd_w_times[batch.stage_id] = time_step
        
        # Check forward stage ordering: if A -> B, then fwd(A) < fwd(B)
        for edge in dag_edges:
            from_stage, to_stage = edge.from_stage, edge.to_stage
            if from_stage in fwd_times and to_stage in fwd_times:
                if fwd_times[from_stage] >= fwd_times[to_stage]:
                    raise ValueError(
                        f"Forward stage ordering violation for microbatch {mb_idx}: "
                        f"forward stage {from_stage} (time {fwd_times[from_stage]}) must come "
                        f"before forward stage {to_stage} (time {fwd_times[to_stage]})"
                    )
        
        # Check forward-backward ordering: fwd(A) < bwd(A)
        for stage_id in fwd_times:
            if stage_id in bwd_times:
                if fwd_times[stage_id] >= bwd_times[stage_id]:
                    raise ValueError(
                        f"Forward-backward ordering violation for microbatch {mb_idx}, "
                        f"stage {stage_id}: forward (time {fwd_times[stage_id]}) must come "
                        f"before backward (time {bwd_times[stage_id]})"
                    )
        
        # Check backward stage ordering: if A -> B, then bwd(B) < bwd(A)
        for edge in dag_edges:
            from_stage, to_stage = edge.from_stage, edge.to_stage
            if from_stage in bwd_times and to_stage in bwd_times:
                if bwd_times[to_stage] >= bwd_times[from_stage]:
                    raise ValueError(
                        f"Backward stage ordering violation for microbatch {mb_idx}: "
                        f"backward stage {to_stage} (time {bwd_times[to_stage]}) must come "
                        f"before backward stage {from_stage} (time {bwd_times[from_stage]})"
                    )
        
        # Check BWD_W must come after BWD_I for same (stage_id, mb_idx)
        for stage_id in bwd_w_times:
            if stage_id not in bwd_times:
                raise ValueError(
                    f"BWD_W for microbatch {mb_idx} stage {stage_id} has no corresponding "
                    f"BWD_I or BWD task"
                )
            if bwd_w_times[stage_id] <= bwd_times[stage_id]:
                raise ValueError(
                    f"BWD_W ordering violation for microbatch {mb_idx}, stage {stage_id}: "
                    f"BWD_W (time {bwd_w_times[stage_id]}) must come after "
                    f"BWD_I/BWD (time {bwd_times[stage_id]})"
                )
            


def piper_setup(
    model_class,
    model_args=(),
    model_kwargs={},
    model_dtype=torch.bfloat16,
    optim_fn=None,
    example_inputs=None,
    example_outputs=None,
    schedule: Schedule2D=None,
    naive_gradient_sync=False,
    activation_checkpointing=False,
    bucketing=False,
    pg=None,
    nsight=False,
):
    """
    Compile a model with the piper backend.

    Args:
        model: The model to compile.
        optim_fn: Callable ``(params) -> Optimizer`` used to create optimizers.
        example_inputs: Example inputs for tracing.
        example_outputs: Example outputs (labels) for tracing.
        schedule: 2D schedule grid (rank x time_step).
        naive_gradient_sync: Use a simple blocking all-reduce instead of
            pipelined per-param hooks.
    """

    stage_to_device = schedule.stage_to_device()
    assert len(stage_to_device) > 0
    piper_metadata.stage_to_device = stage_to_device
    piper_metadata.use_activation_checkpointing = activation_checkpointing
    piper_metadata.bucketing = bucketing
    piper_metadata.schedule = schedule

    dag_edges = []
    # TODO: build dag by analyzing stage dependencies rather than assuming a linear chain
    for stage_id in piper_metadata.stage_to_device.keys():
        if stage_id < len(piper_metadata.stage_to_device.keys()) - 1:
            dag_edges.append(DAGEdge(stage_id, stage_id+1))
    _validate_schedule(schedule.grid, dag_edges=dag_edges, num_mbs=schedule.num_mbs())

    num_mbs = schedule.num_mbs()
    num_stages = schedule.num_stages()
    num_devices = schedule.num_ranks()
    
    _create_actors(
        num_devices, optim_fn, num_mbs, num_stages,
        naive_gradient_sync, profile=nsight, stage_to_device=stage_to_device, pg=pg,
    )

    # All dp_ranks must agree on a single master_addr: the IP of the actor with
    # global_rank=0 (pp_rank=0, dp_rank=0).  dp_rank=0 publishes it via a named
    # Ray actor; other dp_ranks wait until it's available.
    import time
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    if dp_rank == 0:
        # Create the compiled-data store early so dp_rank>0 can poll for it.
        _CompiledDataStore.options(
            name=_COMPILED_DATA_ACTOR, lifetime="detached", get_if_exists=True
        ).remote()

        # Get IP and a free port from actor 0's node — it will be the TCPStore server.
        master_addr, master_port = ray.get(piper_metadata.actors[0].get_node_ip_and_free_port.remote())
        _Rank0AddrStore.options(
            name=_RANK0_ADDR_ACTOR, lifetime="detached", get_if_exists=True
        ).remote(master_addr, master_port)
    else:
        master_addr = master_port = None
        while master_addr is None:
            try:
                store = ray.get_actor(_RANK0_ADDR_ACTOR)
                master_addr, master_port = ray.get(store.get.remote())
            except Exception:
                time.sleep(0.05)
    logger.debug(f"Master address for process groups: {master_addr}:{master_port}")
    ray.get(
        [
            actor._join_process_groups.remote(master_addr, master_port)
            for actor in piper_metadata.actors.values()
        ]
    )
    if dp_rank == 0:
        try:
            ray.kill(ray.get_actor(_RANK0_ADDR_ACTOR))
        except Exception:
            pass

    if dp_rank == 0:
        # --- dp_rank=0: run torch.compile, build DAGs, publish compiled data ---

        # Build the model directly on meta device
        with torch.device("meta"):
            model = model_class(*model_args, **model_kwargs)
            if model_dtype is not None:
                model = model.to(model_dtype)

        num_params = sum(p.numel() for p in model.parameters())
        if model_dtype == torch.bfloat16:
            param_size_gb = num_params * 2 / (1024**3)
        elif model_dtype == torch.float32:
            param_size_gb = num_params * 4 / (1024**3)
        else:
            raise ValueError(f"Unsupported model dtype: {model_dtype}")
        print(f"Model size: {num_params/(1e6):.0f} M parameters ({param_size_gb:.2f} GB), dtype: {model_dtype}")

        compiled = torch.compile(model, backend=piper, fullgraph=True)
        meta_inputs = [x.to(device="meta") for x in example_inputs]
        _ = compiled(*meta_inputs)
        logger.info(f"DP rank 0 stage graphs loaded onto actors.")

        # Publish compiled stage data for dp_rank > 0.
        compiled_store = ray.get_actor(_COMPILED_DATA_ACTOR)
        ray.get(compiled_store.publish.remote(piper_metadata.compiled_stage_data))
        logger.info(f"DP rank 0 published compiled stage data.")

    else:
        # --- dp_rank>0: wait for dp_rank=0 to finish, then load directly ---
        logger.info(f"DP rank {dp_rank} waiting for dp_rank=0 to finish compiling...")
        compiled_data = None
        while compiled_data is None:
            try:
                store = ray.get_actor(_COMPILED_DATA_ACTOR)
                if ray.get(store.is_ready.remote()):
                    compiled_data = ray.get(store.get.remote())
            except Exception:
                pass
            if compiled_data is None:
                time.sleep(0.2)
        logger.info(f"DP rank {dp_rank} received compiled stage data, loading onto actors...")

        # Load stages onto this dp_rank's actors (same serialized graphs, different actor refs).
        refs = []
        for stage_id, (modules_data, a2a_boundaries) in compiled_data["stages"].items():
            actor_id = piper_metadata.stage_to_device[stage_id]
            actor = piper_metadata.actors[actor_id]
            refs.append(actor._load_stage.remote(
                stage_id,
                modules_data,
                a2a_boundaries,
                use_activation_checkpointing=activation_checkpointing,
            ))
        ray.get(refs)

        # Restore piper_metadata fields that the piper backend sets on dp_rank=0.
        piper_metadata.stage_bucket_counts = compiled_data["stage_bucket_counts"]
        piper_metadata.trainable_bucket_keys = compiled_data["trainable_bucket_keys"]
        piper_metadata.per_rank_dags = compiled_data["per_rank_dags"]

        # Load the same per-rank DAGs onto this dp_rank's actors.
        ray.get([
            piper_metadata.actors[pp_rank].load_dag.remote(per_rank_dag)
            for pp_rank, per_rank_dag in enumerate(piper_metadata.per_rank_dags)
        ])
        logger.info(f"DP rank {dp_rank} stage graphs and DAGs loaded onto actors.")

    last_stage_rank = stage_to_device[num_stages - 1]
    ray.get(piper_metadata.actors[0].load_input.remote(example_inputs))
    ray.get(piper_metadata.actors[last_stage_rank].load_labels.remote(example_outputs))
    logger.info(f"DP rank {dp_rank} real inputs/labels loaded onto actors.")

    logger.info(f"DP rank {dp_rank} done.")
    

def piper_shutdown():
    ray.get([actor.shutdown.remote() for actor in piper_metadata.actors.values()])
