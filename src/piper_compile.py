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
from .piper_exec import Schedule2D
from .piper import piper

logger = create_logger("piper_compile", LOG_LEVEL)

_RANK0_ADDR_ACTOR = "piper_rank0_addr"

@ray.remote
class _Rank0AddrStore:
    """Named actor used to share global rank-0's IP+port across independent dp_rank workers."""
    def __init__(self, addr: str, port: int):
        self._addr = addr
        self._port = port
    def get(self):
        return self._addr, self._port

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
    print("Stage to device mapping:", stage_to_device)
    assert len(stage_to_device) > 0
    piper_metadata.stage_to_device = stage_to_device
    piper_metadata.use_activation_checkpointing = activation_checkpointing
    piper_metadata.bucketing = bucketing
    piper_metadata.schedule = schedule

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

    # Build the model directly on meta device
    with torch.device("meta"):
        model = model_class(*model_args, **model_kwargs).to(model_dtype)
        # if model_dtype is not None:
        #     model = model.to(model_dtype)

    num_params = sum(p.numel() for p in model.parameters())

    if model_dtype == torch.bfloat16:
        param_size_mb = num_params * 2 / (1024**3)  # bfloat16
    elif model_dtype == torch.float32:
        param_size_mb = num_params * 4 / (1024**3)  # float32
    else:
        raise ValueError(f"Unsupported model dtype: {model_dtype}")
    print(f"Model size: {num_params/(1e6):.0f} M parameters ({param_size_mb:.2f} GB), dtype: {model_dtype}")
    

    compiled = torch.compile(model, backend=piper, fullgraph=True)

    dp_rank = int(os.environ["PIPER_DP_RANK"])
    logger.info(f"DP rank {dp_rank+1} compiling (meta)...")

    # Create meta tensors from our example_inputs to pass to the compiled graph
    
    meta_inputs = [x.to(device="meta") for x in example_inputs]
    _ = compiled(*meta_inputs)

    logger.info(f"DP rank {dp_rank+1} stage graphs loaded onto actors.")


    last_stage_rank = stage_to_device[num_stages - 1]
    ray.get(piper_metadata.actors[0].load_input.remote(example_inputs))
    ray.get(piper_metadata.actors[last_stage_rank].load_labels.remote(example_outputs))
    logger.info(f"DP rank {dp_rank+1} real inputs/labels loaded onto actors.")

    logger.info(f"DP rank {dp_rank+1} done.")
    

def piper_shutdown():
    ray.get([actor.shutdown.remote() for actor in piper_metadata.actors.values()])
