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

_uid_counter = itertools.count()


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
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    ALLOC_FULL_GRADS = "alloc_full_grads"
    FREE_FULL_GRADS = "free_full_grads"
    ALLOC_FULL_PARAMS = "alloc_full_params"
    FREE_FULL_PARAMS = "free_full_params"
    FWD_A2A = "forward_a2a"
    BWD_A2A = "backward_a2a"

# Backwards-compatible alias
CompType = TaskType


class BatchMeta(NamedTuple):
    """Metadata for one microbatch executed as part of a task."""
    stage_id: int
    mb_idx: int

# Chunk for a schedule: one unit of work on one rank.
class Chunk(NamedTuple):
    pp_rank: int
    batches: list[BatchMeta]
    type: TaskType

    def __repr__(self) -> str:
        return f"Chunk(pp_rank={self.pp_rank}, batches={[(batch.stage_id, batch.mb_idx) for batch in self.batches]}, type={self.type})"


def logical_time(chunk: "Chunk") -> int:
    """Logical duration of a chunk: BWD = 2 (fused I+W), all others = 1."""
    time = 1
    match chunk.type:
        case TaskType.BWD:
            time = 2
        case TaskType.FWD_BWD:
            time = 3
    return time


@dataclass
class PipelineSchedule:
    """Per-rank ordered chunk lists.

    ``grid[rank]`` is the ordered sequence of :class:`Chunk` objects that rank
    executes.  There are no ``None`` sentinels — bubbles are implied by data
    dependencies and are made explicit only when rendering via :meth:`_compute_start_times`.
    """
    grid: list[list["Chunk"]]

    def stage_to_device(self) -> dict[int, int]:
        stage_to_device = {}
        for row in self.grid:
            for chunk in row:
                for batch in chunk.batches:
                    stage_to_device[batch.stage_id] = chunk.pp_rank
        return stage_to_device

    def num_mbs(self) -> int:
        mbs = set()
        for row in self.grid:
            for chunk in row:
                for batch in chunk.batches:
                    mbs.add(batch.mb_idx)
        return len(mbs)

    def num_stages(self) -> int:
        stages = set()
        for row in self.grid:
            for chunk in row:
                for batch in chunk.batches:
                    stages.add(batch.stage_id)
        return len(stages)

    def num_ranks(self) -> int:
        return len(self.grid)

    def _compute_start_times(self) -> list[list[int]]:
        """Return start_times[rank][i] = logical start time of chunk i on rank.

        Uses a repeated-pass list-scheduling simulation that respects:
        * within-rank sequentiality (each chunk starts after the previous finishes)
        * cross-rank data deps: FWD(s,m) needs FWD(s-1,m); BWD/BWD_I(s,m) needs
          BWD(s+1,m); BWD_W(s,m) needs BWD_I(s,m) on the same rank.
        """
        n_ranks = len(self.grid)
        num_stages = self.num_stages()
        rank_free_at = [0] * n_ranks
        next_idx = [0] * n_ranks
        start_times: list[list[Optional[int]]] = [[None] * len(row) for row in self.grid]
        finish: dict = {}   # keyed by ('FWD', s, m), ('BWD', s, m), ('BWDI', rank, s, m)
        total = sum(len(row) for row in self.grid)
        scheduled = 0

        while scheduled < total:
            progress = False
            for rank in range(n_ranks):
                if next_idx[rank] >= len(self.grid[rank]):
                    continue
                i = next_idx[rank]
                chunk = self.grid[rank][i]
                earliest = rank_free_at[rank]
                ready = True

                for batch in chunk.batches:
                    s, m = batch.stage_id, batch.mb_idx
                    t = chunk.type
                    if t == TaskType.FWD:
                        if s > 0:
                            k = ('FWD', s - 1, m)
                            if k not in finish:
                                ready = False; break
                            earliest = max(earliest, finish[k])
                    elif t in (TaskType.BWD, TaskType.BWD_I):
                        if s < num_stages - 1:
                            k = ('BWD', s + 1, m)
                            if k not in finish:
                                ready = False; break
                            earliest = max(earliest, finish[k])
                    elif t == TaskType.BWD_W:
                        k = ('BWDI', rank, s, m)
                        if k not in finish:
                            ready = False; break
                        earliest = max(earliest, finish[k])
                    elif t == TaskType.FWD_BWD:
                        s0, m0 = chunk.batches[0].stage_id, chunk.batches[0].mb_idx
                        s1, m1 = chunk.batches[1].stage_id, chunk.batches[1].mb_idx
                        if s0 > 0:
                            k = ('FWD', s0 - 1, m0)
                            if k not in finish:
                                ready = False; break
                            earliest = max(earliest, finish[k])
                        if s1 < num_stages - 1:
                            k = ('BWD', s1 + 1, m1)
                            if k not in finish:
                                ready = False; break
                            earliest = max(earliest, finish[k])
                        break  # FWD_BWD has exactly 2 batches, handled above

                if not ready:
                    continue

                lt = logical_time(chunk)
                start_times[rank][i] = earliest
                fin = earliest + lt
                rank_free_at[rank] = fin

                for batch in chunk.batches:
                    s, m = batch.stage_id, batch.mb_idx
                    t = chunk.type
                    if t == TaskType.FWD:
                        finish[('FWD', s, m)] = fin
                    elif t == TaskType.BWD:
                        finish[('BWD', s, m)] = fin
                    elif t == TaskType.BWD_I:
                        finish[('BWD', s, m)] = fin
                        finish[('BWDI', rank, s, m)] = fin
                    elif t == TaskType.FWD_BWD:
                        s0, m0 = chunk.batches[0].stage_id, chunk.batches[0].mb_idx
                        s1, m1 = chunk.batches[1].stage_id, chunk.batches[1].mb_idx
                        finish[('FWD', s0, m0)] = earliest + 1
                        finish[('BWD', s1, m1)] = fin
                        break

                next_idx[rank] += 1
                scheduled += 1
                progress = True

            if not progress:
                raise ValueError("PipelineSchedule simulation deadlock — check data dependencies")

        return start_times  # type: ignore[return-value]


@dataclass
class Task:
    """A single node in the task DAG, corresponding to one non-None cell of a
    PipelineSchedule grid.

    Edges are of two kinds:

    * **Data-dependency** (``data_preds`` / ``data_succs``): drawn between tasks
      that share a ``mb_idx`` and are *adjacent* in time – i.e. no other task with
      the same microbatch index sits between them in the time-step ordering.  These
      edges may cross rows (actors).

    * **Temporal** (``temporal_preds`` / ``temporal_succs``): drawn between
      consecutive non-None tasks within the same row.  Normally each task has
      at most one temporal predecessor and one successor (a linear chain), but
      :func:`overlap_a2a_tasks` can introduce forks and joins so the fields
      are lists, mirroring ``data_preds`` / ``data_succs``.
    """
    task_type: TaskType
    batches: list[BatchMeta]
    task_pp_rank: int
    pp_rank: int    # row index in PipelineSchedule.grid
    time_step: int  # column index in PipelineSchedule.grid

    # Filled in by expand_chunks_to_dags()
    data_preds: list["Task"] = field(default_factory=list)
    data_succs: list["Task"] = field(default_factory=list)
    temporal_preds: list["Task"] = field(default_factory=list)
    temporal_succs: list["Task"] = field(default_factory=list)

    # For SEND/RECV nodes: the peer pipeline rank
    peer_pp_rank: Optional[int] = None

    # Set by graph-transform passes (not meaningful for raw schedule tasks)
    bucket_id: int = 0           # stage-local bucket index (stage_bucket_id); actor must not use this
    resource: str = "compute_stream"
    # Task-type-specific metadata (e.g. {"a2a_tensor_idx": int} for A2A tasks).
    custom_metadata: dict = field(default_factory=dict)

    # The original Chunk from the PipelineSchedule that this task was derived from.
    # Set by expand_chunks_to_dags; preserved through all subsequent transforms.
    source_chunk: Optional[Chunk] = None

    # The specific logical chunk that this task belongs to. For FWD_BWD cells,
    # this distinguishes the derived FWD half from the derived BWD half.
    associated_chunk: Optional[Chunk] = None

    # Unique integer ID assigned at construction; survives pickle round-trips.
    uid: int = field(default_factory=lambda: next(_uid_counter))

    # Profiling data: raw per-iteration GPU times (ms) accumulated by run_dag
    # when profiling=True.  Set by piper_compile after profiling.
    profiling_measurements: list = field(default_factory=list)

    # Average of profiling_measurements (ms).  Set by piper_compile.
    runtime: Optional[float] = None

    # Globally unique bucket index assigned by expand_chunks_to_dags.
    # The actor uses this to look up bucket_fwd_fns, bucket_fwd_args, etc.
    # unique_bucket_id is None for non-compute tasks (SEND, RECV, ALL_REDUCE, UPD).
    unique_bucket_id: Optional[int] = None

    # True on the first BWD/BWD_I bucket of the last pipeline stage.
    # When set, run_dag applies loss_fn before dispatching the backward method.
    compute_loss: bool = False

    def node_id(self) -> str:
        """Unique string identifier for use as a graph node key."""
        ttype = self.task_type.value if self.task_type is not None else "none"
        mb = self.batches[0].mb_idx if self.batches else "x"
        return f"r{self.pp_rank}_t{self.time_step}_{ttype}_mb{mb}"


def runtime_sort_key(node: Task) -> tuple:
    """Return the actor dispatch ordering key for a task.

    Runtime dispatch order is driven by ``Task.time_step`` first. For tasks that
    intentionally share a time step, dispatch uses a fixed priority:
    SEND, A2A, alloc, AG, compute, RS/AR, free, RECV.
    """
    priority = {
        TaskType.SEND: 0,
        TaskType.FREE_FULL_GRADS: 1,
        TaskType.FREE_FULL_PARAMS: 1,
        TaskType.FWD_A2A: 2,
        TaskType.BWD_A2A: 2,
        TaskType.REDUCE_SCATTER: 3,
        TaskType.ALL_REDUCE: 3,
        TaskType.ALL_GATHER: 4,
        TaskType.FWD: 5,
        TaskType.BWD: 5,
        TaskType.BWD_I: 5,
        TaskType.BWD_W: 5,
        TaskType.FWD_BWD: 5,
        TaskType.ALLOC_FULL_PARAMS: 6,
        TaskType.ALLOC_FULL_GRADS: 6,
        TaskType.RECV: 7,
    }.get(node.task_type, 4)
    return (node.time_step, priority, node.uid)


def _rebuild_task_dag(node_data):
    """Reconstruct a TaskDAG from the flat index-based representation produced
    by TaskDAG.__reduce__.  Must be a module-level function so pickle can find
    it by name."""
    nodes = [
        Task(task_type=d[0], batches=d[1], task_pp_rank=d[2], pp_rank=d[3], time_step=d[4], peer_pp_rank=d[5],
                 bucket_id=d[10], resource=d[11], custom_metadata=d[12], source_chunk=d[14],
                 associated_chunk=d[15], unique_bucket_id=d[16], compute_loss=d[17])
        for d in node_data
    ]
    for node, d in zip(nodes, node_data):
        node.data_preds     = [nodes[j] for j in d[6]]
        node.data_succs     = [nodes[j] for j in d[7]]
        node.temporal_preds = [nodes[j] for j in d[8]]
        node.temporal_succs = [nodes[j] for j in d[9]]
        node.uid            = d[13]
    return TaskDAG(nodes=nodes)


@dataclass
class TaskDAG:
    """DAG of :class:`Task` objects for one training iteration.

    Produced by :func:`~src.piper_graph_transform.expand_chunks_to_dags`.
    """
    nodes: list[Task]

    def roots(self) -> list[Task]:
        """Return nodes that have no predecessors of any kind."""
        return [n for n in self.nodes if not n.data_preds and not n.temporal_preds]

    def __reduce__(self):
        """Serialize as a flat list of index-referenced tuples to avoid the
        deep recursion that cloudpickle would need when following temporal_succ
        chains through the full node graph."""
        idx = {id(n): i for i, n in enumerate(self.nodes)}
        node_data = [
            (
                n.task_type, n.batches, n.task_pp_rank, n.pp_rank, n.time_step, n.peer_pp_rank,
                [idx[id(p)] for p in n.data_preds],
                [idx[id(s)] for s in n.data_succs],
                [idx[id(p)] for p in n.temporal_preds],
                [idx[id(s)] for s in n.temporal_succs],
                n.bucket_id, n.resource, n.custom_metadata, n.uid, n.source_chunk,
                n.associated_chunk, n.unique_bucket_id, n.compute_loss,
            )
            for n in self.nodes
        ]
        return (_rebuild_task_dag, (node_data,))


class DAGEdge(NamedTuple):
    from_stage: int
    to_stage: int


def _get_backward_targets(stage_id: int, dag_edges: list[DAGEdge]):
    return [edge for edge in dag_edges if edge.to_stage == stage_id]

def _validate_schedule(schedule: "PipelineSchedule", dag_edges: list[DAGEdge], num_mbs: int) -> None:
    """
    Validate that the schedule respects well-formedness rules and DAG dependencies.

    Args:
        schedule: PipelineSchedule (list[list[Chunk]], no Nones)
        dag_edges: List of DAG edges defining stage dependencies
        num_mbs: Number of microbatches in the schedule

    Raises:
        ValueError: If the schedule violates any validation rules
    """
    start_times_per_rank = schedule._compute_start_times()

    # chunk id -> logical start time for ordering comparisons
    chunk_start: dict[int, int] = {}
    for row, starts in zip(schedule.grid, start_times_per_rank):
        for chunk, start in zip(row, starts):
            chunk_start[id(chunk)] = start

    # Check well-formedness: no duplicates, pp_rank matches row, all stages present
    all_tasks: set = set()
    microbatch_tasks: dict[int, set] = {}

    for pp_rank, row in enumerate(schedule.grid):
        for chunk in row:
            if chunk.pp_rank != pp_rank:
                raise ValueError(
                    f"Chunk pp_rank {chunk.pp_rank} does not match row {pp_rank}"
                )
            for batch in chunk.batches:
                comp_type = (
                    chunk.type
                    if chunk.type != CompType.FWD_BWD
                    else (CompType.FWD if batch is chunk.batches[0] else CompType.BWD)
                )
                task_key = (batch.stage_id, batch.mb_idx, comp_type)
                if task_key in all_tasks:
                    raise ValueError(
                        f"Duplicate chunk found: stage_id={batch.stage_id}, "
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
        fwd_stages = {stage_id for stage_id, task_type in tasks if task_type == CompType.FWD}
        bwd_stages = {
            stage_id
            for stage_id, task_type in tasks
            if task_type in (CompType.BWD, CompType.BWD_I)
        }
        missing_fwd = all_required_stages - fwd_stages
        if missing_fwd:
            raise ValueError(f"Microbatch {mb_idx} missing forward stages: {missing_fwd}")
        missing_bwd = all_required_stages - bwd_stages
        if missing_bwd:
            raise ValueError(
                f"Microbatch {mb_idx} missing backward stages: {missing_bwd} "
                f"(need BWD, BWD_I, or FWD_BWD backward for each stage)"
            )

    # Check pipeline stage dependencies using logical start times
    for mb_idx in range(num_mbs):
        fwd_times: dict[int, int] = {}   # stage_id -> logical start
        bwd_times: dict[int, int] = {}   # stage_id -> logical start (BWD or BWD_I)
        bwd_w_times: dict[int, int] = {} # stage_id -> logical start (BWD_W)

        for row in schedule.grid:
            for chunk in row:
                for batch in chunk.batches:
                    if batch.mb_idx != mb_idx:
                        continue
                    comp_type = (
                        chunk.type
                        if chunk.type != CompType.FWD_BWD
                        else (CompType.FWD if batch is chunk.batches[0] else CompType.BWD)
                    )
                    t = chunk_start[id(chunk)]
                    if comp_type == CompType.FWD:
                        fwd_times[batch.stage_id] = t
                    elif comp_type in (CompType.BWD, CompType.BWD_I):
                        bwd_times[batch.stage_id] = t
                    elif comp_type == CompType.BWD_W:
                        bwd_w_times[batch.stage_id] = t

        # Forward ordering: if A -> B, then fwd(A) < fwd(B)
        for edge in dag_edges:
            from_stage, to_stage = edge.from_stage, edge.to_stage
            if from_stage in fwd_times and to_stage in fwd_times:
                if fwd_times[from_stage] >= fwd_times[to_stage]:
                    raise ValueError(
                        f"Forward stage ordering violation for microbatch {mb_idx}: "
                        f"forward stage {from_stage} (time {fwd_times[from_stage]}) must come "
                        f"before forward stage {to_stage} (time {fwd_times[to_stage]})"
                    )

        # Forward-backward ordering: fwd(A) < bwd(A)
        for stage_id in fwd_times:
            if stage_id in bwd_times:
                if fwd_times[stage_id] >= bwd_times[stage_id]:
                    raise ValueError(
                        f"Forward-backward ordering violation for microbatch {mb_idx}, "
                        f"stage {stage_id}: forward (time {fwd_times[stage_id]}) must come "
                        f"before backward (time {bwd_times[stage_id]})"
                    )

        # Backward ordering: if A -> B, then bwd(B) < bwd(A)
        for edge in dag_edges:
            from_stage, to_stage = edge.from_stage, edge.to_stage
            if from_stage in bwd_times and to_stage in bwd_times:
                if bwd_times[to_stage] >= bwd_times[from_stage]:
                    raise ValueError(
                        f"Backward stage ordering violation for microbatch {mb_idx}: "
                        f"backward stage {to_stage} (time {bwd_times[to_stage]}) must come "
                        f"before backward stage {from_stage} (time {bwd_times[from_stage]})"
                    )

        # BWD_W must come after BWD_I for same (stage_id, mb_idx)
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


def _build_p2p_schedule(schedule: "PipelineSchedule", num_stages, num_devices, stage_to_device):
    """
    Build a global p2p schedule by iterating the compute schedule in logical execution order.

    Returns:
        rank_p2p: dict mapping pp_rank -> list of (op_type, stage_from, stage_to, mb_idx, is_fwd)
        p2p_idx_map: dict mapping (pp_rank, stage_from, stage_to, mb_idx, is_fwd, op_type) -> rank_local_idx
    """
    start_times_per_rank = schedule._compute_start_times()

    # Sort events by (logical_start_time, reverse_rank) to match the original
    # time-step-first, reverse-rank-second iteration order.
    events: list[tuple[int, int, int, "Chunk"]] = []
    for rank, (row, starts) in enumerate(zip(schedule.grid, start_times_per_rank)):
        for chunk, start in zip(row, starts):
            events.append((start, -rank, rank, chunk))
    events.sort(key=lambda e: (e[0], e[1]))

    # Global p2p ops in execution order. Each communication is added exactly once,
    # from the SENDER's compute task, to avoid duplicates.
    p2p_ops = []
    for _, _, pp_rank, chunk in events:
        task_type = chunk.type
        batches = chunk.batches

        # Forward send
        if task_type in (CompType.FWD, CompType.FWD_BWD):
            stage_id, mb_idx = batches[0].stage_id, batches[0].mb_idx
            if stage_id < num_stages - 1 and stage_to_device[stage_id] != stage_to_device[stage_id + 1]:
                p2p_ops.append((stage_id, stage_id + 1, mb_idx, True))

        # Backward send
        if task_type in (CompType.BWD, CompType.BWD_I, CompType.FWD_BWD):
            b = batches[1] if task_type == CompType.FWD_BWD else batches[0]
            stage_id, mb_idx = b.stage_id, b.mb_idx
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
    schedule: PipelineSchedule,
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

    dag_edges = list(map(lambda e: DAGEdge(e[0], e[1]), list(piper_metadata.dag)))
    _validate_schedule(schedule, dag_edges, num_mbs)

    actors = piper_metadata.actors
    stage_to_device = piper_metadata.stage_to_device

    # Build p2p schedule and send to actors
    rank_p2p, p2p_idx_map = _build_p2p_schedule(schedule, num_stages, num_devices, stage_to_device)

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
    deps = {}

    # Send p2p schedules to actors before dispatch
    for pp_rank in range(num_devices):
        deps[pp_rank] = actors[pp_rank].set_p2p_schedule.remote(rank_p2p.get(pp_rank, []))

    # Iterate chunks in logical execution order: sorted by (start_time, reverse_rank)
    start_times_per_rank = schedule._compute_start_times()
    events: list[tuple[int, int, int, Chunk]] = []
    for rank, (row, starts) in enumerate(zip(schedule.grid, start_times_per_rank)):
        for chunk, start in zip(row, starts):
            events.append((start, -rank, rank, chunk))
    events.sort(key=lambda e: (e[0], e[1]))

    for _, _, j, chunk in events:
        pp_rank, batches, task_type, *_ = chunk
        actor = actors[j]
        match task_type:
            case CompType.UPD:
                loss = actor._update.remote(deps[pp_rank])
                ret.append(loss)
            case CompType.FWD:
                stage_id, mb_idx = batches[0].stage_id, batches[0].mb_idx
                dep = deps[pp_rank]
                if stage_id > 0:
                    dep = _p2p_recv(actor, pp_rank, stage_id - 1, stage_id, mb_idx, True, dep)
                dep = actor._forward.remote(stage_id, mb_idx, dep)
                if stage_id < num_stages - 1:
                    dep = _p2p_send(actor, pp_rank, stage_id, stage_id + 1, mb_idx, True, dep)
                deps[pp_rank] = dep
            case CompType.BWD:
                stage_id, mb_idx = batches[0].stage_id, batches[0].mb_idx
                dep = deps[pp_rank]
                if stage_id < num_stages - 1:
                    dep = _p2p_recv(actor, pp_rank, stage_id + 1, stage_id, mb_idx, False, dep)
                dep = actor._backward.remote(stage_id, mb_idx, dep, loss_fn=loss_fn)
                if stage_id > 0:
                    dep = _p2p_send(actor, pp_rank, stage_id, stage_id - 1, mb_idx, False, dep)
                deps[pp_rank] = dep
            case CompType.BWD_I:
                stage_id, mb_idx = batches[0].stage_id, batches[0].mb_idx
                dep = deps[pp_rank]
                if stage_id < num_stages - 1:
                    dep = _p2p_recv(actor, pp_rank, stage_id + 1, stage_id, mb_idx, False, dep)
                dep = actor._backward_input.remote(stage_id, mb_idx, dep, loss_fn=loss_fn)
                if stage_id > 0:
                    dep = _p2p_send(actor, pp_rank, stage_id, stage_id - 1, mb_idx, False, dep)
                deps[pp_rank] = dep
            case CompType.BWD_W:
                stage_id, mb_idx = batches[0].stage_id, batches[0].mb_idx
                dep = deps[pp_rank]
                dep = actor._backward_weight.remote(stage_id, mb_idx, dep)
                deps[pp_rank] = dep
            case CompType.FWD_BWD:
                fwd_stage_id, fwd_mb_idx = batches[0].stage_id, batches[0].mb_idx
                bwd_stage_id, bwd_mb_idx = batches[1].stage_id, batches[1].mb_idx
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
