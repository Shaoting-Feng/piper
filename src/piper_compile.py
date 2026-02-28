import ray
import torch
import threading
import os
import gc
import copy
import itertools

from torch._dynamo.backends.debugging import eager

from .piper_actor import _create_actors
from .piper_utils import piper_metadata, create_logger, LOG_LEVEL
from .piper_exec import CompType, Schedule2D, Task
from .piper import piper

logger = create_logger("piper_compile", LOG_LEVEL)


def order_p2p_comms(schedule: Schedule2D, num_devices: int, num_stages: int, stage_to_device: dict[int, int]):
    schedule = schedule.grid
    # Filter to fwd/bwd tasks and split FWD_BWD into two consecutive tasks (FWD then BWD).
    schedule = [[task for task in row if task is not None and task.type != CompType.UPD] for row in schedule]

    assert num_devices == len(schedule), "Some rank has no fwd/bwd tasks"
    # Tasks are guaranteed to be fwd or bwd
    num_fwd_bwd_tasks = sum(len(rank_tasks) for rank_tasks in schedule)

    num_executed_tasks = 0
    # (src_stage, dst_stage, mb_idx) for each p2p comm in execution order
    # is_fwd if src_stage < dst_stage else bwd
    p2p_comms = []
    rank_to_next_task = {rank: 0 for rank in range(len(schedule))}
    # P2P comm is enqueued when sender is ready
    # Pick the next executable task in a round-robin fashion
    while True:
        executable_tasks = []
        for rank, time_step in rank_to_next_task.items():
            if time_step < len(schedule[rank]):
                task = schedule[rank][time_step]
                # Check the task's dependency is satisfied
                # i.e., the sender has enqueued the comm
                dependee = []
                if task.type == CompType.FWD:
                    batch = task.batches[0]
                    if batch.stage_id > 0:
                        dependee.append((batch.stage_id - 1, batch.stage_id, batch.mb_idx))
                elif task.type == CompType.BWD:
                    batch = task.batches[0]
                    if batch.stage_id < num_stages - 1:
                        dependee.append((batch.stage_id + 1, batch.stage_id, batch.mb_idx))
                elif task.type == CompType.FWD_BWD:
                    fwd_batch, bwd_batch = task.batches
                    if fwd_batch.stage_id > 0:
                        dependee.append((fwd_batch.stage_id - 1, fwd_batch.stage_id, fwd_batch.mb_idx))
                    if bwd_batch.stage_id < num_stages - 1:
                        dependee.append((bwd_batch.stage_id + 1, bwd_batch.stage_id, bwd_batch.mb_idx))

                can_exec = dependee is [] or all(dependee in p2p_comms for dependee in dependee)
                if can_exec:
                    executable_tasks.append((rank, task))

        if not executable_tasks:
            break

        for rank, task in executable_tasks:
            if task.type == CompType.FWD:
                batch = task.batches[0]
                if batch.stage_id < num_stages - 1:
                    p2p_comms.append((batch.stage_id, batch.stage_id + 1, batch.mb_idx))
            elif task.type == CompType.BWD:
                batch = task.batches[0]
                if batch.stage_id > 0:
                    p2p_comms.append((batch.stage_id, batch.stage_id - 1, batch.mb_idx))
            elif task.type == CompType.FWD_BWD:
                fwd_batch, bwd_batch = task.batches
                if fwd_batch.stage_id < num_stages - 1:
                    p2p_comms.append((fwd_batch.stage_id, fwd_batch.stage_id + 1, fwd_batch.mb_idx))
                if bwd_batch.stage_id > 0:
                    p2p_comms.append((bwd_batch.stage_id, bwd_batch.stage_id - 1, bwd_batch.mb_idx))
            rank_to_next_task[rank] += 1

        num_executed_tasks += len(executable_tasks)

    if num_executed_tasks != num_fwd_bwd_tasks:
        raise ValueError(
            f"Only {num_executed_tasks} out of {num_fwd_bwd_tasks} "
            f"fwd/bwd tasks can be executed"
        )

    # (src_stage, dst_stage, mb_idx, is_sender)
    p2p_schedules = {rank: [] for rank in range(num_devices)}
    for src_stage, dst_stage, mb_idx in p2p_comms:
        src_rank = stage_to_device[src_stage]
        dst_rank = stage_to_device[dst_stage]
        p2p_schedules[src_rank].append((src_stage, dst_stage, mb_idx, True))
        p2p_schedules[dst_rank].append((src_stage, dst_stage, mb_idx, False))
    
    for rank, schedule in p2p_schedules.items():
        logger.info(f"P2P schedule for rank {rank}: {schedule}")

    return p2p_schedules



def piper_setup(
    model_class,
    model_args=(),
    model_kwargs={},
    optim_fn=None,
    example_inputs=None,
    example_outputs=None,
    schedule: Schedule2D=None,
    naive_gradient_sync=False,
):
    """
    Compile a model with the piper backend.
    schedule: 2D schedule grid (rank x time_step).
    """

    stage_to_device = schedule.stage_to_device()
    piper_metadata.stage_to_device = stage_to_device

    num_mbs = schedule.num_mbs()
    num_stages = schedule.num_stages()
    num_devices = schedule.num_ranks()

    p2p_schedules = order_p2p_comms(
        schedule=schedule,
        num_devices=num_devices,
        num_stages=num_stages,
        stage_to_device=stage_to_device
    )
    _create_actors(
        num_devices, optim_fn, num_mbs, num_stages, p2p_schedules, naive_gradient_sync
    )

    # last_stage_rank = stage_to_device[num_stages - 1]
    # ray.get(piper_metadata.actors[0].load_input.remote(example_inputs))
    # ray.get(piper_metadata.actors[last_stage_rank].load_labels.remote(example_outputs))

    ray.get(
        [
            actor._join_process_groups.remote()
            for actor in piper_metadata.actors.values()
        ]
    )

    # Build the model on meta device
    with torch.device("meta"):
        model = model_class(*model_args, **model_kwargs)

    compiled = torch.compile(model, backend=piper)

    dp_rank = int(os.environ["PIPER_DP_RANK"])
    logger.info(f"DP rank {dp_rank+1} compiling (meta)...")

    # Create meta tensors from our example_inputs to pass to the compiled graph
    # meta_inputs = [torch.empty_like(t, device="meta") for t in example_inputs]
    meta_inputs = [x.to(device="meta") for x in example_inputs]

    # # 1) Verify meta model params are actually meta
    # for n, p in model.named_parameters():
    #     assert p.device.type == "meta", (n, p.device)

    # for n, b in model.named_buffers():
    #     assert b.device.type == "meta", (n, b.device)

    # # 2) Verify meta inputs
    # for i, x in enumerate(meta_inputs):
    #     assert isinstance(x, torch.Tensor)
    #     assert x.device.type == "meta", (i, x.device)

    _ = compiled(*meta_inputs)

    logger.info(f"DP rank {dp_rank+1} stage graphs loaded onto actors.")

    ray.get([
        actor.materialize_all_stages.remote()
        for actor in piper_metadata.actors.values()
    ])
    logger.info(f"DP rank {dp_rank+1} stages materialized on actors.")

    last_stage_rank = stage_to_device[num_stages - 1]
    ray.get(piper_metadata.actors[0].load_input.remote(example_inputs))
    ray.get(piper_metadata.actors[last_stage_rank].load_labels.remote(example_outputs))
    logger.info(f"DP rank {dp_rank+1} real inputs/labels loaded onto actors.")

    logger.info(f"DP rank {dp_rank+1} done.")

    del compiled
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    

def piper_shutdown():
    ray.get([actor.shutdown.remote() for actor in piper_metadata.actors.values()])
