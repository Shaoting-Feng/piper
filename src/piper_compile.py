import ray
import torch
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

def piper_setup(
    model_class,
    model_args=(),
    model_kwargs={},
    optim_fn=None,
    example_inputs=None,
    example_outputs=None,
    schedule: Schedule2D=None,
    naive_gradient_sync=False,
    mode="sequential",
):
    """
    Compile a model with the piper backend.
    schedule: 2D schedule grid (rank x time_step).
    """

    stage_to_device = schedule.stage_to_device()
    assert len(stage_to_device) > 0
    piper_metadata.stage_to_device = stage_to_device

    num_mbs = schedule.num_mbs()
    num_stages = schedule.num_stages()
    num_devices = schedule.num_ranks()

    _create_actors(
        num_devices, optim_fn, num_mbs, num_stages, naive_gradient_sync,
        profile=True, mode=mode, stage_to_device=stage_to_device,
    )

    ray.get(
        [
            actor._join_process_groups.remote()
            for actor in piper_metadata.actors.values()
        ]
    )

    # Build the model directly on meta device
    with torch.device("meta"):
        model = model_class(*model_args, **model_kwargs)

    num_params = sum(p.numel() for p in model.parameters())
    param_size_mb = num_params * 4 / (1024**3)  # float32
    print(f"Model size: {num_params/(1e6):.0f} M parameters ({param_size_mb:.2f} GB)")
    
    compiled = torch.compile(model, backend=piper)

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
