import os
import torch
import threading
import time
import ray
import torch.distributed as dist
from typing import NamedTuple, Optional
from enum import Enum
from dataclasses import dataclass, field
import threading
import time
from collections import defaultdict
import itertools

from .piper_utils import piper_metadata, create_logger, LOG_LEVEL

logger = create_logger("piper_exec", LOG_LEVEL)

class TaskType(Enum):
    FWD = "forward"
    BWD = "backward"
    UPD = "update"
    BWD_I = "backward_input"
    BWD_W = "backward_weight"
    FWD_BWD = "forward_backward"
    SEND = "send"
    RECV = "recv"
    ALL_REDUCE = "all_reduce"
    FWD_A2A = "forward_a2a"
    BWD_A2A = "backward_a2a"

# Backwards-compatible alias
CompType = TaskType


class BatchMeta(NamedTuple):
    """Metadata for one microbatch executed as part of a task."""
    stage_id: int
    mb_idx: int

# Task for 2D schedule grid: one cell (rank, time_step).
class Task(NamedTuple):
    pp_rank: int
    batches: list[BatchMeta]
    type: TaskType
    bucket_id: int = 0
    a2a_tensor_idx: Optional[int] = None

    def __repr__(self) -> str:
        bucket_str = f", bucket_id={self.bucket_id}" if self.bucket_id else ""
        a2a_str = f", a2a_tensor_idx={self.a2a_tensor_idx}" if self.a2a_tensor_idx is not None else ""
        return f"Task(pp_rank={self.pp_rank}, batches={[(batch.stage_id, batch.mb_idx) for batch in self.batches]}, type={self.type}{bucket_str}{a2a_str})"


@dataclass
class Schedule2D:
    """2D schedule grid: grid[rank][time_step] = Task | None. Execution order: by time_step, then rank descending."""
    grid: list[list["Task | None"]]

    def stage_to_device(self) -> dict[int, int]:
        stage_to_device = {}
        for row in self.grid:
            for task in row:
                if task is not None:
                    for batch in task.batches:
                        stage_to_device[batch.stage_id] = task.pp_rank
        return stage_to_device
    
    def num_mbs(self) -> int:
        mbs = set()
        for row in self.grid:
            for task in row:
                if task is not None:
                    for batch in task.batches:
                        mbs.add(batch.mb_idx)
        return len(mbs)
    
    def num_stages(self) -> int:
        stages = set()
        for row in self.grid:
            for task in row:
                if task is not None:
                    for batch in task.batches:
                        stages.add(batch.stage_id)
        return len(stages)
    
    def num_ranks(self) -> int:
        return len(self.grid)

@dataclass
class TaskNode:
    """A single node in the task DAG, corresponding to one non-None cell of a
    Schedule2D grid.

    Edges are of two kinds:

    * **Data-dependency** (``data_preds`` / ``data_succs``): drawn between tasks
      that share a ``mb_idx`` and are *adjacent* in time – i.e. no other task with
      the same microbatch index sits between them in the time-step ordering.  These
      edges may cross rows (actors).

    * **Temporal** (``temporal_pred`` / ``temporal_succ``): drawn between
      consecutive non-None tasks within the same row, but **only** when those
      tasks do not already share a data-dependency edge.  Each task has at most
      one temporal predecessor and one successor.
    """
    task: Task
    pp_rank: int    # row index in Schedule2D.grid
    time_step: int  # column index in Schedule2D.grid

    # Filled in by schedule_to_dag()
    data_preds: list["TaskNode"] = field(default_factory=list)
    data_succs: list["TaskNode"] = field(default_factory=list)
    temporal_pred: Optional["TaskNode"] = None
    temporal_succ: Optional["TaskNode"] = None

    # For SEND/RECV nodes: the peer pipeline rank
    peer_pp_rank: Optional[int] = None

    def node_id(self) -> str:
        """Unique string identifier for use as a graph node key."""
        ttype = self.task.type.value if self.task.type is not None else "none"
        mb = self.task.batches[0].mb_idx if self.task.batches else "x"
        return f"r{self.pp_rank}_t{self.time_step}_{ttype}_mb{mb}"


@dataclass
class TaskDAG:
    """DAG of :class:`TaskNode` objects for one training iteration.

    Produced by :func:`~src.piper.schedule_to_dag`.
    """
    nodes: list[TaskNode]

    def roots(self) -> list[TaskNode]:
        """Return nodes that have no predecessors of any kind."""
        return [n for n in self.nodes if not n.data_preds and n.temporal_pred is None]


class DAGEdge(NamedTuple):
    from_stage: int
    to_stage: int


def _get_backward_targets(stage_id: int, dag_edges: list[DAGEdge]):
    return [edge for edge in dag_edges if edge.to_stage == stage_id]

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


def _build_p2p_schedule(schedule, num_stages, num_devices, num_steps, stage_to_device):
    """
    Build a global p2p schedule by iterating the compute schedule in execution order.

    Returns:
        rank_p2p: dict mapping pp_rank -> list of (op_type, stage_from, stage_to, mb_idx, is_fwd)
        p2p_idx_map: dict mapping (pp_rank, stage_from, stage_to, mb_idx, is_fwd, op_type) -> rank_local_idx
    """
    # Global p2p ops in schedule iteration order. Each entry: (stage_from, stage_to, mb_idx, is_fwd).
    # Only cross-rank ops are included. Each communication is added exactly once,
    # from the SENDER's compute task, to avoid duplicates.
    p2p_ops = []

    for i in range(num_steps):
        for j in range(num_devices - 1, -1, -1):
            if i >= len(schedule[j]):
                continue
            task = schedule[j][i]
            if task is None:
                continue
            pp_rank, batches, task_type, *_ = task

            # Forward send (added after forward compute produces output)
            if task_type in (CompType.FWD, CompType.FWD_BWD):
                stage_id, mb_idx = batches[0]
                if stage_id < num_stages - 1 and stage_to_device[stage_id] != stage_to_device[stage_id + 1]:
                    p2p_ops.append((stage_id, stage_id + 1, mb_idx, True))

            # Backward send (added after backward compute produces gradients)
            if task_type in (CompType.BWD, CompType.BWD_I, CompType.FWD_BWD):
                if task_type == CompType.FWD_BWD:
                    stage_id, mb_idx = batches[1]
                else:
                    stage_id, mb_idx = batches[0]
                if stage_id > 0 and stage_to_device[stage_id] != stage_to_device[stage_id - 1]:
                    p2p_ops.append((stage_id, stage_id - 1, mb_idx, False))

    # Build per-rank schedules and index map
    rank_p2p = defaultdict(list)
    p2p_idx_map = {}

    for stage_from, stage_to, mb_idx, is_fwd in p2p_ops:
        sender_pp = stage_to_device[stage_from]
        receiver_pp = stage_to_device[stage_to]

        send_idx = len(rank_p2p[sender_pp])
        rank_p2p[sender_pp].append(("send", stage_from, stage_to, mb_idx, is_fwd))
        p2p_idx_map[(sender_pp, stage_from, stage_to, mb_idx, is_fwd, "send")] = send_idx

        recv_idx = len(rank_p2p[receiver_pp])
        rank_p2p[receiver_pp].append(("recv", stage_from, stage_to, mb_idx, is_fwd))
        p2p_idx_map[(receiver_pp, stage_from, stage_to, mb_idx, is_fwd, "recv")] = recv_idx

    return rank_p2p, p2p_idx_map


def piper_exec(
    schedule: Schedule2D,
    loss_fn,
    dp_degree=1,
    naive_gradient_sync=False,
):
    """
    Execute one step of the pipeline schedule on the distributed model.

    Args:
        schedule: 2D schedule grid (rank x time_step).
        inputs: Inputs to the model.
        truth: Ground-truth labels or targets.
        loss_fn: Loss function to be used for training.

    Returns:
        List of losses per microbatch (from UPD tasks).
    """
    num_mbs = schedule.num_mbs()
    num_devices = schedule.num_ranks()
    num_stages = schedule.num_stages()
    num_steps = len(schedule.grid[0])

    schedule = schedule.grid

    dag_edges = piper_metadata.dag
    dag_edges = list(map(lambda e: (DAGEdge(e[0], e[1])), list(piper_metadata.dag)))
    _validate_schedule(schedule, dag_edges, num_mbs)

    actors = piper_metadata.actors
    stage_to_device = piper_metadata.stage_to_device

    # Build p2p schedule and send to actors
    rank_p2p, p2p_idx_map = _build_p2p_schedule(
        schedule, num_stages, num_devices, num_steps, stage_to_device
    )

    def _p2p_recv(actor, pp_rank, stage_from, stage_to, mb_idx, is_fwd, dep):
        """Dispatch a recv: _exec_p2p_op for cross-rank, old method for local."""
        key = (pp_rank, stage_from, stage_to, mb_idx, is_fwd, "recv")
        if key in p2p_idx_map:
            return actor._exec_p2p_op.remote(p2p_idx_map[key], dep)
        if is_fwd:
            return actor._exec_fwd_recv.remote(stage_to, mb_idx, dep)
        return actor._exec_bwd_recv.remote(stage_to, mb_idx, dep)

    def _p2p_send(actor, pp_rank, stage_from, stage_to, mb_idx, is_fwd, dep):
        """Dispatch a send: _exec_p2p_op for cross-rank, old method for local."""
        key = (pp_rank, stage_from, stage_to, mb_idx, is_fwd, "send")
        if key in p2p_idx_map:
            return actor._exec_p2p_op.remote(p2p_idx_map[key], dep)
        if is_fwd:
            return actor._exec_fwd_send.remote(stage_from, mb_idx, dep)
        return actor._exec_bwd_send.remote(stage_from, mb_idx, dep)

    ret = []
    # map pp_rank -> latest dependency
    deps = {}

    # Send p2p schedules to actors before dispatch
    for pp_rank in range(num_devices):
        deps[pp_rank] = actors[pp_rank].set_p2p_schedule.remote(
            rank_p2p.get(pp_rank, [])
        )

    for i in range(num_steps):
        for j in range(num_devices-1, -1, -1):
            if not i < len(schedule[j]):
                continue
            task = schedule[j][i]
            if task:
                pp_rank, batches, task_type, *_ = task
                actor = actors[j]
                match task_type:
                    case CompType.UPD:
                        loss = actor._update.remote(deps[pp_rank])
                        ret.append(loss)
                    case CompType.FWD:
                        stage_id, mb_idx = batches[0]
                        dep = deps[pp_rank]
                        if stage_id > 0:
                            dep = _p2p_recv(actor, pp_rank, stage_id - 1, stage_id, mb_idx, True, dep)
                        dep = actor._forward.remote(stage_id, mb_idx, dep)
                        if stage_id < num_stages - 1:
                            dep = _p2p_send(actor, pp_rank, stage_id, stage_id + 1, mb_idx, True, dep)
                        deps[pp_rank] = dep
                    case CompType.BWD:
                        stage_id, mb_idx = batches[0]
                        dep = deps[pp_rank]
                        if stage_id < num_stages - 1:
                            dep = _p2p_recv(actor, pp_rank, stage_id + 1, stage_id, mb_idx, False, dep)
                        dep = actor._backward.remote(stage_id, mb_idx, dep, loss_fn=loss_fn)
                        if stage_id > 0:
                            dep = _p2p_send(actor, pp_rank, stage_id, stage_id - 1, mb_idx, False, dep)
                        deps[pp_rank] = dep
                    case CompType.BWD_I:
                        stage_id, mb_idx = batches[0]
                        dep = deps[pp_rank]
                        if stage_id < num_stages - 1:
                            dep = _p2p_recv(actor, pp_rank, stage_id + 1, stage_id, mb_idx, False, dep)
                        dep = actor._backward_input.remote(stage_id, mb_idx, dep, loss_fn=loss_fn)
                        if stage_id > 0:
                            dep = _p2p_send(actor, pp_rank, stage_id, stage_id - 1, mb_idx, False, dep)
                        deps[pp_rank] = dep
                    case CompType.BWD_W:
                        stage_id, mb_idx = batches[0]
                        dep = deps[pp_rank]
                        dep = actor._backward_weight.remote(stage_id, mb_idx, dep)
                        deps[pp_rank] = dep
                    case CompType.FWD_BWD:
                        fwd_stage_id, fwd_mb_idx = batches[0]
                        bwd_stage_id, bwd_mb_idx = batches[1]
                        dep = deps[pp_rank]
                        if fwd_stage_id > 0:
                            dep = _p2p_recv(actor, pp_rank, fwd_stage_id - 1, fwd_stage_id, fwd_mb_idx, True, dep)
                        if bwd_stage_id < num_stages - 1:
                            dep = _p2p_recv(actor, pp_rank, bwd_stage_id + 1, bwd_stage_id, bwd_mb_idx, False, dep)
                        dep = actor._forward_backward.remote(fwd_stage_id, fwd_mb_idx, bwd_stage_id, bwd_mb_idx, dep, loss_fn=loss_fn)
                        if fwd_stage_id < num_stages - 1:
                            dep = _p2p_send(actor, pp_rank, fwd_stage_id, fwd_stage_id + 1, fwd_mb_idx, True, dep)
                        if bwd_stage_id > 0:
                            dep = _p2p_send(actor, pp_rank, bwd_stage_id, bwd_stage_id - 1, bwd_mb_idx, False, dep)
                        deps[pp_rank] = dep
    return ray.get(ret)