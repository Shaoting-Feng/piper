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
from .piper_utils import piper_metadata, create_logger, LOG_LEVEL, VERBOSE
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

    Stage data is published as soon as it is serialized so nonzero DP ranks can
    start _load_stage while dp_rank=0 continues DAG construction.
    """
    def __init__(self):
        self._stage_data = None
        self._dag_data = None
        self._stage_ready = False
        self._dag_ready = False

    def publish(self, data: dict):
        self._stage_data = {
            "stages": data["stages"],
            "stage_bucket_counts": data["stage_bucket_counts"],
            "trainable_bucket_keys": data["trainable_bucket_keys"],
            "zero_bucket_keys": data.get("zero_bucket_keys", set()),
        }
        self._dag_data = {
            "per_rank_dags": data["per_rank_dags"],
            "full_dag_no_overlap": data.get("full_dag_no_overlap"),
        }
        self._stage_ready = True
        self._dag_ready = True

    def publish_stage_data(self, data: dict):
        self._stage_data = data
        self._stage_ready = True

    def publish_dag_data(self, data: dict):
        self._dag_data = data
        self._dag_ready = True

    def is_ready(self) -> bool:
        return self._stage_ready and self._dag_ready

    def is_stage_ready(self) -> bool:
        return self._stage_ready

    def is_dag_ready(self) -> bool:
        return self._dag_ready

    def get(self) -> dict:
        if not self.is_ready():
            return None
        result = dict(self._stage_data)
        result.update(self._dag_data)
        return result

    def get_stage_data(self) -> dict:
        return self._stage_data

    def get_dag_data(self) -> dict:
        return self._dag_data







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
    bucket_size: int = 25 * 1024 * 1024,
    zero_stage: int = 0,
    gradient_accumulation: bool = True,
    use_inductor: bool = False,
    enable_ep: bool = False,
    ar_a2a_same_stream: bool = False,
    overlap_zero_ops: bool = False,
    overlap_chunks: bool = False,
    schedule_name: str = "",
    visualize_dag_render: bool = True,
    no_nvtx: bool = False,
    a2a_ar_no_overlap=False,
    pg=None,
    nsight=False,
    temp_dir: str = None,
    model_flops_per_token: float = None,
    visualize_dag: bool = True,
    output_dir: str = "out",
    const_attrs: dict = None,
    memory_profile: bool = False,
    pp_outer: bool = False,
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
        zero_stage: ZeRO stage: 0 disabled, 1 optimizer-state sharding,
            2 gradient sharding, 3 parameter and gradient sharding.
        gradient_accumulation: When true, delay bucket gradient reductions
            until the last occurrence of each bucket.
        use_inductor: When true, actors torch.compile stage GraphModules
            during _load_stage with the default inductor backend.
        enable_ep: When true, honor expert-parallel A2A annotations by
            splitting stage graphs and inserting DAG-level A2A tasks.
        ar_a2a_same_stream: When true, actors run A2A ops on the same CUDA
            stream as AR ops.
        overlap_zero_ops: When true, run overlap_zero_ops on per-rank DAGs
            after ZeRO collective insertion.
        overlap_chunks: When true, apply the chunk-overlap transform to
            per-rank DAGs before assigning time steps.
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
    piper_metadata.bucket_size = bucket_size
    piper_metadata.zero_stage = int(zero_stage)
    piper_metadata.gradient_accumulation = bool(gradient_accumulation)
    piper_metadata.use_inductor = bool(use_inductor)
    piper_metadata.enable_ep = bool(enable_ep)
    piper_metadata.ar_a2a_same_stream = bool(ar_a2a_same_stream)
    piper_metadata.overlap_zero_ops = bool(overlap_zero_ops)
    piper_metadata.overlap_chunks = bool(overlap_chunks)
    piper_metadata.schedule_name = schedule_name
    piper_metadata.visualize_dag_render = visualize_dag_render
    piper_metadata.a2a_ar_no_overlap = a2a_ar_no_overlap
    piper_metadata.schedule = schedule
    piper_metadata.visualize_dag = visualize_dag
    piper_metadata.output_dir = output_dir

    # Reset DAG/compile fields so stale data from a prior run never leaks into
    # this run if the piper backend is somehow not re-invoked.
    piper_metadata.per_rank_dags = None
    piper_metadata.full_dag_no_overlap = None
    piper_metadata.stage_bucket_counts = {}
    piper_metadata.trainable_bucket_keys = set()
    piper_metadata.compiled_stage_data = None
    piper_metadata.compiled_data_store = None

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
        naive_gradient_sync, profile=nsight, stage_to_device=stage_to_device,
        zero_stage=zero_stage, no_nvtx=no_nvtx, pg=pg, temp_dir=temp_dir,
        use_inductor=use_inductor, ar_a2a_same_stream=ar_a2a_same_stream,
        memory_profile=memory_profile, pp_outer=pp_outer,
    )

    if dp_rank == 0:
        # Create the compiled-data store early so dp_rank>0 can poll for it.
        piper_metadata.compiled_data_store = _CompiledDataStore.options(
            name=_COMPILED_DATA_ACTOR,
            lifetime="detached",
            num_cpus=0,
            get_if_exists=True,
        ).remote()
        ray.get(piper_metadata.compiled_data_store.is_ready.remote())

        # Get IP and a free port from actor 0's node — it will be the TCPStore server.
        master_addr, master_port = ray.get(piper_metadata.actors[0].get_node_ip_and_free_port.remote())
        addr_store = _Rank0AddrStore.options(
            name=_RANK0_ADDR_ACTOR,
            lifetime="detached",
            num_cpus=0,
        ).remote(master_addr, master_port)
        ray.get(addr_store.get.remote())
    else:
        master_addr = master_port = None
        while master_addr is None:
            try:
                store = ray.get_actor(_RANK0_ADDR_ACTOR)
                master_addr, master_port = ray.get(store.get.remote())
            except Exception:
                time.sleep(0.05)

    logger.debug(f"DP rank {dp_rank}: Master address for process groups: {master_addr}:{master_port}")

    handles = [actor._join_process_groups.remote(master_addr, master_port) for actor in piper_metadata.actors.values()]

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
        logger.log(VERBOSE, f"Pushed {len(_const_attrs)} const attrs to actors: {list(_const_attrs.keys())}")

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
        logger.info(f"Model size: {num_params/(1e6):.0f} M parameters ({param_size_gb:.2f} GB), dtype: {model_dtype}")

        compiled = torch.compile(model, backend=piper, fullgraph=True)
        meta_inputs = [x.to(device="meta") for x in example_inputs]
        _ = compiled(*meta_inputs)

    else:
        # --- dp_rank>0: wait for stage data first, then load while dp_rank=0 builds DAGs. ---
        logger.debug(f"DP rank {dp_rank} waiting for dp_rank=0 stage data...")
        stage_data = None
        while stage_data is None:
            try:
                store = ray.get_actor(_COMPILED_DATA_ACTOR)
                if ray.get(store.is_stage_ready.remote()):
                    stage_data = ray.get(store.get_stage_data.remote())
            except Exception:
                pass
            if stage_data is None:
                time.sleep(0.2)
        logger.debug(f"DP rank {dp_rank} received compiled stage data, loading onto actors...")

        # Load stages onto this dp_rank's actors (same serialized graphs, different actor refs).
        refs = []
        for stage_id, (modules_data, a2a_boundaries) in stage_data["stages"].items():
            actor_id = piper_metadata.stage_to_device[stage_id]
            actor = piper_metadata.actors[actor_id]
            refs.append(actor._load_stage.remote(
                stage_id,
                modules_data,
                a2a_boundaries,
                use_activation_checkpointing=activation_checkpointing,
            ))
        ray.get(refs)

        # Restore stage metadata that the piper backend sets on dp_rank=0.
        piper_metadata.stage_bucket_counts = stage_data["stage_bucket_counts"]
        piper_metadata.trainable_bucket_keys = stage_data["trainable_bucket_keys"]
        piper_metadata.zero_bucket_keys = stage_data.get("zero_bucket_keys", set())

        logger.debug(f"DP rank {dp_rank} waiting for dp_rank=0 DAG data...")
        dag_data = None
        while dag_data is None:
            try:
                store = ray.get_actor(_COMPILED_DATA_ACTOR)
                if ray.get(store.is_dag_ready.remote()):
                    dag_data = ray.get(store.get_dag_data.remote())
            except Exception:
                pass
            if dag_data is None:
                time.sleep(0.2)

        piper_metadata.per_rank_dags = dag_data["per_rank_dags"]
        piper_metadata.full_dag_no_overlap = dag_data.get("full_dag_no_overlap")

        # Load the same per-rank DAGs onto this dp_rank's actors.
        ray.get([
            piper_metadata.actors[pp_rank].load_dag.remote(per_rank_dag)
            for pp_rank, per_rank_dag in enumerate(piper_metadata.per_rank_dags)
        ])

    last_stage_rank = stage_to_device[num_stages - 1]
    ray.get(piper_metadata.actors[0].load_input.remote(example_inputs))
    ray.get(piper_metadata.actors[last_stage_rank].load_labels.remote(example_outputs))

    logger.info(f"DP rank {dp_rank} done.")


def piper_shutdown():
    ray.get([actor.shutdown.remote() for actor in piper_metadata.actors.values()])
