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
from .piper_exec import DAGEdge, PipelineSchedule, CompType, Chunk, BatchMeta, _validate_schedule
from .piper import piper, piper_exec_dag
from .piper_graph_transform import compute_critical_path, visualize_dag

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







def piper_setup(
    model_class,
    model_args=(),
    model_kwargs={},
    model_dtype=torch.bfloat16,
    optim_fn=None,
    example_inputs=None,
    example_outputs=None,
    loss_fn=None,
    schedule: PipelineSchedule=None,
    naive_gradient_sync=False,
    activation_checkpointing=False,
    num_checkpoints=1,
    bucketing=False,
    a2a_ar_no_overlap=False,
    pg=None,
    nsight=False,
    model_flops_per_token: float = None,
    visualize_dag: bool = True,
    const_attrs: dict = None,
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
        a2a_ar_no_overlap: When true, schedule gradient all-reduces after
            same-rank A2A operations to avoid NCCL interference.
    """

    # Clear Dynamo's global compilation cache so that a previous piper_setup
    # call (e.g. with a different model size or schedule) in the same Python
    # worker process cannot contaminate this compilation via a stale cache hit.
    torch._dynamo.reset()

    stage_to_device = schedule.stage_to_device()
    assert len(stage_to_device) > 0
    piper_metadata.stage_to_device = stage_to_device
    piper_metadata.use_activation_checkpointing = activation_checkpointing
    piper_metadata.activation_num_checkpoints = max(1, int(num_checkpoints))
    piper_metadata.bucketing = bucketing
    piper_metadata.a2a_ar_no_overlap = a2a_ar_no_overlap
    piper_metadata.schedule = schedule
    piper_metadata.visualize_dag = visualize_dag

    # Reset DAG/compile fields so stale data from a prior run never leaks into
    # this run if the piper backend is somehow not re-invoked.
    piper_metadata.per_rank_dags = None
    piper_metadata.full_dag_no_overlap = None
    piper_metadata.stage_bucket_counts = {}
    piper_metadata.trainable_bucket_keys = set()
    piper_metadata.compiled_stage_data = None

    # MFU tracking
    piper_metadata.model_flops_per_token = model_flops_per_token
    if model_flops_per_token is not None and example_inputs is not None:
        dp_degree = int(os.environ.get("PIPER_DP_DEGREE", "1"))
        # example_inputs[0] is the token index tensor of shape (batch_size, seq_len)
        piper_metadata.tokens_per_step = example_inputs[0].numel() * dp_degree

    dag_edges = []
    # TODO: build dag by analyzing stage dependencies rather than assuming a linear chain
    for stage_id in piper_metadata.stage_to_device.keys():
        if stage_id < len(piper_metadata.stage_to_device.keys()) - 1:
            dag_edges.append(DAGEdge(stage_id, stage_id+1))
    _validate_schedule(schedule, dag_edges=dag_edges, num_mbs=schedule.num_mbs())

    num_mbs = schedule.num_mbs()
    num_stages = schedule.num_stages()
    num_devices = schedule.num_ranks()

    # All dp_ranks must agree on a single master_addr: the IP of the actor with
    # global_rank=0 (pp_rank=0, dp_rank=0).  dp_rank=0 publishes it via a named
    # Ray actor; other dp_ranks wait until it's available.
    import time
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    _create_actors(
        num_devices, optim_fn, num_mbs, num_stages,
        naive_gradient_sync, profile=nsight, stage_to_device=stage_to_device, pg=pg,
    )

    if dp_rank == 0:
        # Create the compiled-data store early so dp_rank>0 can poll for it.
        _CompiledDataStore.options(
            name=_COMPILED_DATA_ACTOR, lifetime="detached", get_if_exists=True
        ).remote()

        # Get IP and a free port from actor 0's node — it will be the TCPStore server.
        master_addr, master_port = ray.get(piper_metadata.actors[0].get_node_ip_and_free_port.remote())
        _Rank0AddrStore.options(
            name=_RANK0_ADDR_ACTOR, lifetime="detached"
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

    # Push non-trainable constant tensor attributes (e.g. freqs_cis, rope_cache, mask)
    # to all actors so _load_stage can initialize them correctly instead of zero-filling.
    # Must happen on every dp_rank since each dp_rank owns its own actors.
    _const_attrs = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in (const_attrs or {}).items()
    }
    if _const_attrs:
        ray.get([
            actor.load_const_attrs.remote(_const_attrs)
            for actor in piper_metadata.actors.values()
        ])
        logger.debug(f"Pushed {len(_const_attrs)} const attrs to actors: {list(_const_attrs.keys())}")

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
        piper_metadata.full_dag_no_overlap = compiled_data.get("full_dag_no_overlap")

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

    # if loss_fn is not None:
    #     _NUM_WARMUP = 3
    #     _NUM_PROFILE = 5

    #     logger.info(f"DP rank {dp_rank}: running {_NUM_WARMUP} warmup iterations before profiling.")
    #     for _ in range(_NUM_WARMUP):
    #         piper_exec_dag(loss_fn)

    #     logger.info(f"DP rank {dp_rank}: running {_NUM_PROFILE} profiling iterations.")
    #     for _ in range(_NUM_PROFILE):
    #         piper_exec_dag(loss_fn, profiling=True)

    #     # Only dp_rank=0 aggregates timing data and renders the critical-path DAG.
    #     if dp_rank == 0:
    #         uid_to_measurements: dict = {}
    #         for pp_rank in range(len(piper_metadata.per_rank_dags)):
    #             actor = piper_metadata.actors[pp_rank]
    #             node_runtimes = ray.get(actor.get_node_runtimes.remote())
    #             uid_to_measurements.update(node_runtimes)

    #         full_dag = piper_metadata.full_dag_no_overlap
    #         if full_dag is not None:
    #             for node in full_dag.nodes:
    #                 measurements = uid_to_measurements.get(node.uid, [])
    #                 if measurements:
    #                     node.profiling_measurements = measurements
    #                     node.runtime = sum(measurements) / len(measurements)

    #             critical_nodes = compute_critical_path(full_dag)
    #             logger.info(
    #                 f"Critical path has {len(critical_nodes)} nodes "
    #                 f"(of {len(full_dag.nodes)} total)."
    #             )
    #             try:
    #                 visualize_dag(
    #                     full_dag,
    #                     output_path="figs/profiled_full_dag",
    #                     critical_path_nodes=critical_nodes,
    #                 )
    #             except Exception as e:
    #                 logger.warning(f"Profiled DAG visualisation failed: {e}")

    logger.info(f"DP rank {dp_rank} done.")
    

def piper_shutdown():
    ray.get([actor.shutdown.remote() for actor in piper_metadata.actors.values()])
