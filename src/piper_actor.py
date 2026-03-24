import ray
import torch
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import gc
import threading
from torch._guards import CompileId
from torch.nn import Parameter
from torch.autograd.graph import GradientEdge, Node
import torch.distributed as dist
from collections import defaultdict

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .piper_utils import (
    _deserialize_graphmodule,
    create_logger,
    LOG_LEVEL,
    piper_metadata,
    NcclOverlapDetector,
)
from .backward_utils import get_param_groups, construct_reverse_graph, _get_grad_fn_or_grad_acc
from .piper_exec import TaskType, TaskDAG, TaskNode

CLEANUP_MEMORY = False

logger = create_logger("piper_actor", LOG_LEVEL)


def _get_rank(pp_rank, dp_rank, pp_degree):
    return pp_rank + dp_rank * pp_degree


def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    return port

def _create_actors(
    num_actors,
    optim_class,
    num_mbs,
    num_stages,
    naive_gradient_sync=False,
    profile=False,
    mode="sequential",
    stage_to_device=None,
    pg=None,
):
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    world_size = int(os.environ["PIPER_WORLD_SIZE"])
    dp_degree = int(os.environ["PIPER_DP_DEGREE"])
    pp_degree = int(os.environ["PIPER_PP_DEGREE"])

    from .piper_utils import piper_metadata

    for pp_rank in range(num_actors):
        global_rank = _get_rank(pp_rank, dp_rank, pp_degree)
        nsight_env = {"nsight": {
            "t": "cuda,cudnn,cublas,nvtx",
            "cuda-event-trace": "false",
            "stop-on-exit": "true",
        }} if profile else {}
        master_env = {
            "env_vars": {
                "PIPER_MASTER_ADDR": os.environ.get("PIPER_MASTER_ADDR", "127.0.0.1"),
                "PIPER_MASTER_PORT": os.environ.get("PIPER_MASTER_PORT", "10000"),
            }
        }
        actor = PiperActor.options(
            num_gpus=0.7,
            runtime_env={**nsight_env, **master_env},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=dp_rank
            ),
        ).remote(
            pp_rank,
            optim_class,
            world_size,
            num_mbs,
            num_stages,
            naive_gradient_sync,
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            pp_degree=pp_degree,
            mode=mode,
            stage_to_device=stage_to_device,
        )
        piper_metadata.actors[pp_rank] = actor
        logger.debug(
            f"DP rank {dp_rank} created actor {actor} global rank {global_rank}"
        )


def _get_actor(pp_rank):
    from .piper_utils import piper_metadata

    return piper_metadata.actors[pp_rank]

@ray.remote
class PiperActor:
    def __init__(
        self,
        pp_rank,
        optim_class,
        world_size,
        num_mbs,
        num_stages,
        naive_gradient_sync=False,
        dp_rank=0,
        dp_degree=1,
        pp_degree=1,
        mode="sequential",
        stage_to_device=None,
    ):
        self.logger = create_logger("piper_actor", LOG_LEVEL)
        self.mode = mode

        self.pp_rank = pp_rank
        self.optim_class = optim_class
        self.naive_gradient_sync = naive_gradient_sync

        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.pp_degree = pp_degree
        self.world_size = world_size

        self.num_mbs = num_mbs
        self.num_stages = num_stages
        self.stage_to_device = stage_to_device or {}
        self.dp_group = None
        self.ep_group = None  # separate NCCL communicator for expert-parallel (all2all) ops
        # Per-direction communicators: (src_global_rank, dst_global_rank) -> ProcessGroup
        self.pp_groups = {}
        self.device = "cuda"

        self.global_rank = _get_rank(pp_rank, dp_rank, pp_degree)

        self.logger.info(
            f"Initializing Ray actor {self.global_rank} GPU {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

        self.input = None
        self.labels = None

        self.comp_stream = torch.cuda.Stream()
        self.comm_stream = torch.cuda.Stream()
        self.a2a_stream = torch.cuda.Stream()
        self.p2p_stream = torch.cuda.Stream()
        self.p2p_send_stream = torch.cuda.Stream()
        self.p2p_recv_stream = torch.cuda.Stream()
        self.overlapped_comp_stream = torch.cuda.Stream()
        self.overlapped_p2p_stream = torch.cuda.Stream()
        # Aliases for FWD and BWD compute streams.  Setting these to separate
        # torch.cuda.Stream() instances enables compute-A2A overlap for FWD_BWD
        # schedule cells; keeping them as aliases of comp_stream is correct but
        # does not overlap FWD and BWD compute.
        self.fwd_comp_stream = self.comp_stream
        self.bwd_comp_stream = self.comp_stream
        self.overlap_detector = NcclOverlapDetector()
        if mode == "naive":
            self.per_mb_streams = [torch.cuda.Stream() for _ in range(2)]
        self.n_a2a_ops = dict()
        self.overlap_a2a_ops = False
        # A2A boundaries per stage: stage_id -> {boundary_bucket_id -> tensor_idx}
        self.a2a_boundaries: dict = {}

        # map stage id -> compiled fx.Graph function
        self.forward_fns = dict()
        # map stage id -> original GraphModule (for hook registration)
        self.graph_modules = dict()
        # map stage id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.forward_args = dict()
        # map stage id -> input idx -> input tensor metadata
        self.forward_input_meta = defaultdict(dict)
        # map stage id -> indices of the input tensors (as opposed to model parameters) used by the fx.Graph
        self.input_idxs = dict()
        # map stage id -> indices of the model parameters used by the fx.Graph
        self.param_idxs = dict()
        # map stage id -> indices of the model parameters used by the fx.Graph
        self.trainable_param_idxs = dict()
        # map stage id -> optimizer for the fx.Graph (used in legacy non-DAG path)
        self.optims = dict()
        # map stage id -> mb_idx -> previous activation (if this stage is not first)
        self.inp_activation = defaultdict(dict)
        # map stage id -> mb_idx -> current activation
        self.out_activation = defaultdict(dict)
        # map stage id -> (shape, dtype) of the output activation tensor (for pre-FWD BWD recv)
        self.output_activation_shape: dict = {}
        # map (src_stage, dst_stage, mb_idx, is_sender) -> tensor for p2p communication
        self.p2p_cache = dict()
        # map recv p2p_op -> CUDA event recorded after the recv completes on p2p_stream
        self.p2p_events = dict()
        # accumuate loss for each microbatch
        self.loss = []
        # map stage id -> data parallel communication operations
        self.comm_ops = dict()
        # map stage id -> tensor id -> comm op status
        self.comm_op_status = defaultdict(lambda: defaultdict(int))
        # map stage id -> tensor id -> comm op handle
        self.comm_op_handles = defaultdict(dict)
        # map stage id -> list of tensor ids that require communciation
        self.comm_op_tensor_ids = dict()

        self.tracing = False
        self.trace_events = dict()
        self.trace_data = defaultdict(list)
        self.task_labels = dict()

        # DAG execution state
        self.dag = None
        self.send_buffer: dict = {}  # ((stage_id, bucket_id_or_None), mb_idx) -> tensor(s); or (None, mb_idx) for BWD
        self.send_buffer_ready: dict = {}  # same key format as send_buffer -> threading.Event
        self.recv_buffer: dict = {}  # same key format as send_buffer -> tensor(s)
        self.recv_events: dict = {}  # same key format as send_buffer -> cuda.Event
        self.bucket_buffer: dict = {}  # (stage_id, mb_idx, bucket_id) -> (pre_detach_outs, detached_outs)
        # A2A boundary state: (stage_id, mb_idx, boundary_bucket_id) -> (x_detached, x_a2a)
        self.a2a_buffer: dict = {}
        # CUDA events recorded on a2a_stream after each FWD_A2A / BWD_A2A op
        self.a2a_events: dict = {}  # (stage_id, mb_idx, type, bucket_id) -> cuda.Event
        # CUDA events recorded on comm_stream after each all-reduce launches
        self.ar_events: dict = {}  # (stage_id, bucket_id) -> cuda.Event

        # Per-bucket stage data (populated by _load_stage)
        self.bucket_fwd_fns: dict[int, list] = {}   # stage_id -> [fwd_fn per bucket]
        self.bucket_fwd_args: dict[int, list] = {}  # stage_id -> [args_list per bucket]
        self.bucket_param_idxs: dict[int, list] = {}  # stage_id -> [param_idxs per bucket]
        self.bucket_optims: dict[int, list] = {}      # stage_id -> [optimizer per bucket]
        # Contiguous flat tensors for param data and gradients, keyed (stage_id, bucket_id)
        self.bucket_flat_params: dict = {}   # (stage_id, bucket_id) -> flat param tensor
        self.bucket_flat_grads: dict = {}    # (stage_id, bucket_id) -> flat grad tensor
        self.bucket_trainable_param_idxs: dict = {}  # (stage_id, bucket_id) -> list of indices into bucket_fwd_args
        # stage_id -> mb_idx -> list of (pre_detach_out, detached_input_to_next) per boundary
        self.bucket_boundaries: dict = defaultdict(dict)

        # map stage id -> mb_idx -> parameter groups for backward pass
        self.bw_param_groups = defaultdict(dict)
        # map stage id -> mb_idx -> gradient cache for backward pass
        self.bw_grad_cache = defaultdict(dict)
        # map stage id -> mb_idx -> upstream gradient cache for backward pass
        self.upstream_grad_cache = defaultdict(dict)

        from .piper_utils import piper_metadata

        piper_metadata.actor_self = self

    def _label_task(self, label: str):
        task_id = ray.get_runtime_context().get_task_id()
        if task_id:
            self.task_labels[task_id] = label

    def get_task_labels(self) -> dict:
        return self.task_labels

    def reset_p2p_states(self):
        self.p2p_cache = dict()
        self.p2p_events = dict()
        self.p2p_cursor = 0

    def set_p2p_schedule(self, schedule):
        """Set the per-rank p2p schedule. Each entry: (op_type, stage_from, stage_to, mb_idx, is_fwd)."""
        self.p2p_rank_schedule = schedule
        self.p2p_cursor = 0
        self.logger.debug(
            f"Global rank {self.global_rank} set p2p schedule with {len(schedule)} ops"
        )

    def _exec_p2p_op(self, target_rank_idx, *deps):
        """Execute all p2p ops from cursor up to and including target_rank_idx."""
        while self.p2p_cursor <= target_rank_idx:
            op_type, stage_from, stage_to, mb_idx, is_fwd = self.p2p_rank_schedule[self.p2p_cursor]
            if is_fwd:
                if op_type == "recv":
                    self._exec_fwd_recv(stage_to, mb_idx)
                else:
                    self._exec_fwd_send(stage_from, mb_idx)
            else:
                if op_type == "recv":
                    self._exec_bwd_recv(stage_to, mb_idx)
                else:
                    self._exec_bwd_send(stage_from, mb_idx)
            self.p2p_cursor += 1

    def enable_nccl_monitoring(self) -> None:
        """Enable NCCL overlap monitoring for the next training iteration.

        Call this before the iteration you want to profile.  After the iteration
        completes, call print_nccl_overlaps() to analyse and display results.
        Monitoring is automatically disabled after print_nccl_overlaps() returns.
        """
        self.overlap_detector.enable()
        self.logger.info("Actor %d: NCCL monitoring enabled", self.global_rank)

    def print_nccl_overlaps(self) -> None:
        """Synchronise the GPU, analyse recorded NCCL events, and log all overlapping
        kernel pairs.  Disables monitoring afterwards so subsequent iterations are
        not affected.

        Call this after the monitored training iteration has completed.
        """
        torch.cuda.synchronize()
        overlaps = self.overlap_detector.find_overlaps(self.logger)
        n_kernels = len(self.overlap_detector._records)
        if not overlaps:
            self.logger.info(
                "Actor %d: NCCL overlap monitor — no overlaps detected "
                "(%d kernels recorded)", self.global_rank, n_kernels,
            )
        else:
            self.logger.warning(
                "Actor %d: NCCL overlap monitor — %d overlapping pair(s) "
                "across %d kernels:", self.global_rank, len(overlaps), n_kernels,
            )
            for a, b, overlap_ms in overlaps:
                self.logger.warning(
                    "  '%s' on '%s' [%.3f–%.3f ms]  <->  '%s' on '%s' [%.3f–%.3f ms]"
                    "  (overlap %.3f ms)",
                    a.kernel_name, a.stream_label, a.t_start_ms, a.t_end_ms,
                    b.kernel_name, b.stream_label, b.t_start_ms, b.t_end_ms,
                    overlap_ms,
                )
        self.overlap_detector.build_enforcement(overlaps, self.logger)
        self.overlap_detector.disable()

    def get_trace_data(self) -> dict:
        return self.global_rank, self.trace_data

    def clear_trace_data(self) -> None:
        self.trace_data.clear()

    def set_tracing(self, enabled: bool) -> None:
        self.tracing = enabled
        self.logger.info(
            f"Actor {self.global_rank}: Tracing {'enabled' if enabled else 'disabled'}"
        )

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()

    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(
            f"actor{self.global_rank}_memory_snapshot_mb4_gpipe.pickle"
        )
        self.logger.info(
            f"Saved memory snapshot to actor{self.global_rank}_memory_snapshot_mb4_gpipe.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self):
        return self.global_rank, torch.cuda.max_memory_allocated() / (1024**3)

    def load_input(self, inputs):
        self.inputs = [inp.to(self.device) for inp in inputs]
        self.logger.debug(f"Actor {self.global_rank} loaded inputs {len(self.inputs)}")

    def load_labels(self, labels):
        self.labels = labels.to(self.device)
        self.logger.debug(f"Actor {self.global_rank} loaded labels {self.labels.shape}")

    def _start_timing(self, stream, label):
        if self.tracing:
            if label not in self.trace_events:
                self.trace_events[label] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
            start, _ = self.trace_events[label]
            start.record(stream)

    def _stop_timing(self, stream, label):
        if self.tracing:
            if label in self.trace_events:
                start, stop = self.trace_events[label]
                stop.record(stream)
                stop.synchronize()
                self.trace_data[label].append(start.elapsed_time(stop))

    def get_node_ip_and_free_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]
        return ray.util.get_node_ip_address(), port

    def _join_process_groups(self, master_addr, master_port):
        init_method = f"tcp://{master_addr}:{master_port}"

        self.logger.info(f"Actor {self.global_rank} initializing process groups with master address {master_addr}:{master_port}")

        self.device = f"cuda:{self.global_rank % torch.cuda.device_count()}"
        torch.cuda.set_device(self.device)

        dist.init_process_group(
            "nccl",
            init_method=init_method,
            rank=self.global_rank,
            world_size=self.world_size,
        )
        self.logger.debug(
            f"Actor {self.global_rank} has GPU {os.environ['CUDA_VISIBLE_DEVICES']}, joined the global process group"
        )

        if self.dp_degree > 1:
            self._join_dp_process_group()
        if self.pp_degree > 1:
            self._join_pp_process_group()

    def _join_dp_process_group(self):
        num_dp_groups = self.world_size // self.dp_degree
        for dp_group_id in range(num_dp_groups):
            group_ranks = [
                (dp_group_id + num_dp_groups * i) for i in range(self.dp_degree)
            ]
            # Two separate NCCL communicators over the same ranks: one for allreduce,
            # one for all2all.  Sharing a communicator causes both op types to run on
            # the same internal NCCL proxy stream, which prevents true overlap.
            process_group = dist.new_group(ranks=group_ranks, backend="nccl")
            ep_process_group = dist.new_group(ranks=group_ranks, backend="nccl")
            if self.global_rank % num_dp_groups == dp_group_id:
                self.dp_group = process_group
                self.ep_group = ep_process_group
                self.logger.debug(
                    f"Global rank {self.global_rank} joined its dp group {dp_group_id} along with ranks {group_ranks}"
                )

    def _join_pp_process_group(self):
        num_pp_groups = self.world_size // self.pp_degree
        my_pp_group_id = self.global_rank // self.pp_degree

        # One communicator per direction per rank pair. The global p2p schedule
        # (built in piper_exec) ensures both ranks issue ops in the same FIFO order.
        for pp_group_id in range(num_pp_groups):
            group_ranks = [
                (pp_group_id * self.pp_degree + i) for i in range(self.pp_degree)
            ]
            for i in range(len(group_ranks)):
                for j in range(i + 1, len(group_ranks)):
                    rank_lo, rank_hi = group_ranks[i], group_ranks[j]
                    pg_lo_to_hi = dist.new_group(ranks=[rank_lo, rank_hi], backend="nccl")
                    pg_hi_to_lo = dist.new_group(ranks=[rank_lo, rank_hi], backend="nccl")
                    if self.global_rank in (rank_lo, rank_hi):
                        self.logger.debug(f"Global rank {self.global_rank} saving pp communicators for ranks {[rank_lo, rank_hi]}")
                        self.pp_groups[(rank_lo, rank_hi)] = pg_lo_to_hi
                        self.pp_groups[(rank_hi, rank_lo)] = pg_hi_to_lo
            # if pp_group_id == my_pp_group_id:
            #     self.logger.debug(
            #         f"Global rank {self.global_rank} joined pp group {pp_group_id} "
            #         f"with communicators for ranks {group_ranks}"
            #     )

        # Warm up all communicators to force eager NCCL initialization.
        dummy = torch.zeros(1, device=self.device)
        for key, pg in self.pp_groups.items():
            self.logger.debug(f"Global rank {self.global_rank} warming up pp communicator {key}")
            dist.all_reduce(dummy, group=pg)
        torch.cuda.synchronize()
        self.logger.info(f"Global rank {self.global_rank} warmed up {len(self.pp_groups)} pp communicators")

    def shutdown(self):
        dist.destroy_process_group()

    def _maybe_trigger_grad_allreduce(
        self, stage_id: int, param: torch.nn.Parameter
    ) -> None:
        """
        Preserve `_prepare_dp_comm_ops` semantics for split backward (BWD_I/BWD_W).

        `_prepare_dp_comm_ops` attaches `register_post_accumulate_grad_hook` to params.
        Those hooks fire only when autograd's AccumulateGrad runs. In ZeroBubble's
        split backward, we compute grads via `torch.autograd.grad` and manually
        accumulate into `param.grad`, so AccumulateGrad (and thus the hooks) never run.

        This helper mirrors the hook behavior: once a parameter has had gradients
        accumulated for all microbatches (`num_mbs`), launch an async all-reduce on
        the accumulated `param.grad` and store the handle so `_wait_for_comm_ops()`
        can wait on it before the optimizer step.
        """
        if self.naive_gradient_sync or self.dp_degree <= 1:
            return
        if param.grad is None:
            return

        tid = id(param)
        self.logger.debug(f"Split bwd: Updating status on actor {self.global_rank}, tensor={tid}")
        if stage_id not in self.comm_op_tensor_ids:
            return
        if tid not in self.comm_op_tensor_ids[stage_id]:
            return

        self.comm_op_status[stage_id][tid] += 1
        self.logger.debug(
            f"Split bwd: Updating status on actor {self.global_rank}, tensor={tid}, "
            f"status={self.comm_op_status[stage_id][tid]}/{self.num_mbs}"
        )

        if self.comm_op_status[stage_id][tid] == self.num_mbs:
            # Same ordering fix as in _prepare_dp_comm_ops: the backward_weight call
            # accumulates into param.grad on comp_stream.  Establish a GPU-side
            # dependency so comm_stream doesn't start the all-reduce until the
            # gradient write has completed.  This does not block the CPU.
            dep_event = torch.cuda.Event()
            dep_event.record(self.comp_stream)
            self.comm_stream.wait_event(dep_event)
            _ov_token = self.overlap_detector.before_kernel(self.comm_stream, "grad_allreduce", "comm_stream")
            with torch.cuda.stream(self.comm_stream):
                handle = dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.AVG,
                    group=self.dp_group,
                )
            self.overlap_detector.after_kernel(self.comm_stream, _ov_token)
            self.logger.debug(
                f"Split bwd: Allreduce launched on actor {self.global_rank}, tensor={tid}"
            )
            self.comm_op_status[stage_id][tid] = 0
            self.comm_op_handles[stage_id][tid] = handle

    def _prepare_dp_comm_ops(self, stage_id):
        def hook_maker(tensor_id):
            def post_backward_hook(grad):
                self.comm_op_status[stage_id][tensor_id] += 1
                self.logger.debug(
                    f"Updating status on actor: {self.global_rank}, tensor={tensor_id}, status={self.comm_op_status[stage_id][tensor_id]}/{self.num_mbs}"
                )
                if self.comm_op_status[stage_id][tensor_id] == self.num_mbs:
                    # The AccumulateGrad CUDA kernel that writes the final gradient into
                    # param.grad is in-flight on comp_stream.  The hook fires on the CPU
                    # as soon as the kernel is *submitted*, not when it *completes*.
                    # Record an event on comp_stream now so that comm_stream can wait for
                    # the gradient to be fully written before starting the all-reduce.
                    # comm_stream.wait_event() is a GPU-side barrier — it does not block
                    # the CPU, so backward compute on comp_stream continues to overlap.
                    dep_event = torch.cuda.Event()
                    dep_event.record(self.comp_stream)
                    self.comm_stream.wait_event(dep_event)
                    _ov_token = self.overlap_detector.before_kernel(self.comm_stream, "grad_allreduce", "comm_stream")
                    with torch.cuda.stream(self.comm_stream):
                        handle = dist.all_reduce(
                            grad,
                            op=dist.ReduceOp.AVG,
                            group=self.dp_group,
                        )
                    self.overlap_detector.after_kernel(self.comm_stream, _ov_token)
                    self.logger.debug(
                        f"Allreduce on actor: {self.global_rank}, tensor={tensor_id}"
                    )
                    self.comm_op_status[stage_id][tensor_id] = 0
                    self.comm_op_handles[stage_id][tensor_id] = handle
                return grad
            return post_backward_hook

        ids = []
        for t in self.forward_args[stage_id]:
            if t is not None and t.requires_grad:
                tid = id(t)
                ids.append(tid)
                t.register_post_accumulate_grad_hook(hook_maker(tid))
        self.comm_op_tensor_ids[stage_id] = ids

    def _wait_for_comm_ops(self):
        self.logger.debug(f"Actor {self.global_rank} waiting for comm ops")
        for stage_id, tids in self.comm_op_tensor_ids.items():
            for tid in tids:
                self.logger.debug(
                    f"Waiting for comm op to be launched dp_rank: {self.dp_rank}, tensor={tid}"
                )
                done = False
                while not done:
                    if tid in self.comm_op_handles[stage_id]:
                        self.logger.debug(
                            f"Waiting for comm op to be finished dp_rank: {self.dp_rank}, tensor={tid}"
                        )
                        # self.comm_op_handles[stage_id][tid].wait()
                        done = True

    def _load_stage(
        self,
        stage_id: int,
        modules_data: list,
        a2a_boundaries: dict = None,
        use_activation_checkpointing: bool = False,
    ) -> None:
        """Load a (possibly bucketed) stage.

        *modules_data* is a list of dicts, one per module/bucket, each with keys:
        ``gm_data``, ``graphargs``, ``input_idxs``, ``param_idxs``.

        A non-bucketed stage is represented as a single-element list.
        Bucket 0 handles the activation input from the previous stage (or
        ``self.inputs`` for stage 0).  Subsequent modules receive the
        cross-segment activation output of the previous module.
        """
        self.logger.debug(
            f"Loading stage {stage_id} ({len(modules_data)} module(s)) on actor {self.global_rank}"
        )

        self.bucket_fwd_fns[stage_id] = []
        self.bucket_fwd_args[stage_id] = []
        self.bucket_param_idxs[stage_id] = []
        self.bucket_optims[stage_id] = []
        self.a2a_boundaries[stage_id] = a2a_boundaries or {}

        g = torch.Generator(device=self.device)
        g.manual_seed(1000 * self.global_rank + stage_id)

        first_gm = None
        last_gm = None

        for b_idx, bd in enumerate(modules_data):
            gm = _deserialize_graphmodule(bd["gm_data"])
            if b_idx == 0:
                first_gm = gm
            last_gm = gm

            forward_args = list(bd["graphargs"])
            b_input_idxs = list(bd["input_idxs"])
            b_param_idxs = list(bd["param_idxs"])

            self.logger.debug(
                f"Stage {stage_id} module {b_idx} input indices: {b_input_idxs}"
            )

            # Module 0: save activation-input metadata for the stage interface
            # (used by _exec_recv to pre-allocate recv buffers).
            if b_idx == 0:
                self.input_idxs[(stage_id, None)] = b_input_idxs  # legacy path alias
                for i in b_input_idxs:
                    meta = forward_args[i]
                    if meta is not None:
                        self.forward_input_meta[stage_id][i] = (
                            tuple(meta.shape),
                            meta.dtype,
                            bool(getattr(meta, "requires_grad", False)),
                        )
                    forward_args[i] = None
            else:
                for i in b_input_idxs:
                    forward_args[i] = None

            self.input_idxs[(stage_id, b_idx)] = b_input_idxs

            # Realize parameter tensors.
            realized = [None] * len(forward_args)
            for i, arg in enumerate(forward_args):
                if arg is None:
                    continue
                t = torch.empty(arg.shape, dtype=arg.dtype, device=self.device)
                if arg.requires_grad:
                    t.requires_grad_(True)
                    torch.nn.init.normal_(t, mean=0.0, std=0.02, generator=g)
                else:
                    t.zero_()
                realized[i] = t

            # Forward function (with optional activation checkpointing).
            if use_activation_checkpointing:
                fwd = gm.forward
                forward_fn = lambda *args, _fn=fwd: torch.utils.checkpoint.checkpoint(
                    _fn, *args, use_reentrant=False
                )
            else:
                forward_fn = gm.forward

            self.bucket_fwd_fns[stage_id].append(forward_fn)
            self.bucket_fwd_args[stage_id].append(realized)
            self.bucket_param_idxs[stage_id].append(b_param_idxs)

            # Collect trainable parameters and build a contiguous flat tensor so
            # a single all-reduce call can sync the entire module's gradients.
            trainable_idxs = [
                i for i in b_param_idxs
                if realized[i] is not None and realized[i].requires_grad
            ]
            trainable = [realized[i] for i in trainable_idxs]
            self.bucket_trainable_param_idxs[(stage_id, b_idx)] = trainable_idxs

            if trainable:
                flat_params = torch.cat([p.detach().view(-1) for p in trainable]).contiguous()
                flat_params.requires_grad_(False)
                flat_grads = torch.zeros_like(flat_params)
                offset = 0
                for idx, p in zip(trainable_idxs, trainable):
                    numel = p.numel()
                    realized[idx] = realized[idx].detach()
                    realized[idx].data = flat_params[offset:offset + numel].view(p.shape)
                    realized[idx].requires_grad_(True)
                    offset += numel
                self.bucket_flat_params[(stage_id, b_idx)] = flat_params
                self.bucket_flat_grads[(stage_id, b_idx)] = flat_grads
            else:
                self.bucket_flat_params[(stage_id, b_idx)] = None
                self.bucket_flat_grads[(stage_id, b_idx)] = None

            # Optimizer for this module's trainable parameters.
            trainable_for_optim = [realized[i] for i in trainable_idxs]
            optim = self.optim_class(trainable_for_optim) if trainable_for_optim else None
            self.bucket_optims[stage_id].append(optim)

            # Legacy non-DAG DDP: register per-param allreduce hooks for module 0.
            # if b_idx == 0 and self.dp_degree > 1 and not self.naive_gradient_sync:
            #     self.forward_args[stage_id] = realized
                # self._prepare_dp_comm_ops(stage_id)

        # Set legacy fields from first module for ZeroBubble BWD_I/BWD_W compatibility.
        self.forward_args[stage_id] = self.bucket_fwd_args[stage_id][0]
        self.param_idxs[stage_id] = self.bucket_param_idxs[stage_id][0]
        self.graph_modules[stage_id] = first_gm
        self.stage_id = stage_id

        # Compute output activation shape from the last module for BWD recv buffer
        # pre-allocation (e.g. interleaved 1F1B recv before forward has run).
        last_realized = self.bucket_fwd_args[stage_id][-1]
        last_b_input_idxs = self.input_idxs[(stage_id, len(modules_data) - 1)]
        output_shapes = []

        # Try FX graph node metadata first.
        for node in last_gm.graph.nodes:
            if node.op == "output":
                out_args = node.args[0]
                if not isinstance(out_args, (tuple, list)):
                    out_args = [out_args]
                for out_node in out_args:
                    if isinstance(out_node, torch.fx.Node):
                        meta_val = out_node.meta.get("val", None)
                        if meta_val is not None and isinstance(meta_val, torch.Tensor) and meta_val.requires_grad:
                            output_shapes.append((tuple(meta_val.shape), meta_val.dtype))
                break

        if not output_shapes:
            # Fallback: probe forward pass with dummy inputs.
            probe_args = list(last_realized)
            g_probe = torch.Generator(device=self.device)
            g_probe.manual_seed(42 + stage_id)
            for i in last_b_input_idxs:
                meta = modules_data[-1]["graphargs"][i]
                if meta is not None:
                    t = torch.zeros(meta.shape, dtype=meta.dtype, device=self.device)
                    t.requires_grad_(bool(getattr(meta, "requires_grad", True)))
                    probe_args[i] = t
            with torch.no_grad():
                probe_out = last_gm.forward(*probe_args)
            for out in (probe_out if isinstance(probe_out, (tuple, list)) else [probe_out]):
                if isinstance(out, torch.Tensor) and out.requires_grad:
                    output_shapes.append((tuple(out.shape), out.dtype))

        if output_shapes:
            self.output_activation_shape[stage_id] = output_shapes
            self.logger.debug(f"Stage {stage_id} output activation shapes: {output_shapes}")
        else:
            self.logger.warning(
                f"Could not determine output activation shape for stage {stage_id}"
            )

    def _exec_fwd_recv(self, stage_id: int, mb_idx: int, *deps, p2p_stream=None, comp_stream=None):
        if stage_id == 0:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_recv_stream
        if comp_stream is None:
            comp_stream = self.comp_stream

        p2p_op = (stage_id - 1, stage_id, mb_idx, False)
        # assert p2p_op not in self.p2p_cache

        inputs_to_recv = []
        for i in self.input_idxs[(stage_id, None)]:
            shape, dtype, requires_grad = self.forward_input_meta[stage_id][i]
            inputs_to_recv.append(
                torch.empty(
                    shape, dtype=dtype, requires_grad=requires_grad, device=self.device
                )
            )

        # For non-first stages, receive input tensors from the previous stage
        pp_rank = self.stage_to_device[stage_id - 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        if self.global_rank == global_src_rank:
            for i in self.input_idxs[(stage_id, None)]:
                inputs_to_recv[i] = self.p2p_cache.pop((stage_id-1, stage_id, mb_idx, True))
        else:
            self.logger.debug(
                f"Dispatch fwd p2p recv on communicator ({global_src_rank}, {self.global_rank}) to {self.global_rank} from {global_src_rank}, op: ({stage_id-1} -> {stage_id}, mb {mb_idx})"
            )
            self._start_timing(p2p_stream, "fwd_p2p_recv")
            _ov_token = self.overlap_detector.before_kernel(p2p_stream, "fwd_p2p_recv", "p2p_recv_stream")
            with torch.cuda.stream(p2p_stream):
                for i in self.input_idxs[(stage_id, None)]:
                    dist.recv(
                        inputs_to_recv[i],
                        src=global_src_rank,
                        group=self.pp_groups[(global_src_rank, self.global_rank)],
                    )
            recv_event = torch.cuda.Event()
            recv_event.record(p2p_stream)
            self.overlap_detector.after_kernel(p2p_stream, _ov_token)
            self.p2p_events[p2p_op] = recv_event
            self._stop_timing(p2p_stream, "fwd_p2p_recv")
            self.logger.debug(
                f"Completed fwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id-1} -> {stage_id}, mb {mb_idx})"
            )

        self.p2p_cache[p2p_op] = inputs_to_recv

    def _exec_fwd_send(self, stage_id: int, mb_idx: int, *deps, p2p_stream=None, comp_stream=None):
        if stage_id == self.num_stages - 1:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_send_stream
        if comp_stream is None:
            comp_stream = self.comp_stream

        p2p_op = (stage_id, stage_id + 1, mb_idx, True)
        output = self.p2p_cache.pop(p2p_op)
        # For non-final stages, send output tensors to the next stage
        pp_rank = self.stage_to_device[stage_id + 1]
        global_dst_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        if self.global_rank == global_dst_rank:
            self.p2p_cache[p2p_op] = output
        else:
            # Ensure send sees the latest writes from the default stream.
            self.logger.debug(
                f"Dispatch fwd p2p send on communicator ({self.global_rank}, {global_dst_rank}) from {self.global_rank} to {global_dst_rank}, op: ({stage_id} -> {stage_id+1}, mb {mb_idx})"
            )
            self._start_timing(p2p_stream, "fwd_p2p_send")
            p2p_stream.wait_stream(comp_stream)
            _ov_token = self.overlap_detector.before_kernel(p2p_stream, "fwd_p2p_send", "p2p_send_stream")
            with torch.cuda.stream(p2p_stream):
                for i in range(len(output)):
                    dist.send(
                        output[i],
                        dst=global_dst_rank,
                        group=self.pp_groups[(self.global_rank, global_dst_rank)],
                    )
            self.overlap_detector.after_kernel(p2p_stream, _ov_token)
            self._stop_timing(p2p_stream, "fwd_p2p_send")
            self.logger.debug(
                f"Completed fwd p2p send on {self.global_rank} to {global_dst_rank}, op: ({stage_id} -> {stage_id+1}, mb {mb_idx})"
            )

    def _exec_bwd_recv(self, stage_id: int, mb_idx: int, *deps, p2p_stream=None, comp_stream=None):
        if stage_id >= self.num_stages - 1:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_recv_stream
        if comp_stream is None:
            comp_stream = self.comp_stream

        out_activation = self.out_activation[stage_id][mb_idx]
        # For non-final stages, recieve input gradients from the subsequent backward pass
        input_grad = torch.empty_like(out_activation)
        pp_rank = self.stage_to_device[stage_id + 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        p2p_op = (stage_id + 1, stage_id, mb_idx, False)

        if self.global_rank == global_src_rank:
            input_grad = self.p2p_cache.pop((stage_id + 1, stage_id, mb_idx, True))
        else:
            self.logger.debug(
                f"Dispatch bwd p2p recv on communicator ({global_src_rank}, {self.global_rank}) to {self.global_rank} from {global_src_rank}, op: ({stage_id+1} -> {stage_id}, mb {mb_idx})"
            )
            self._start_timing(p2p_stream, "bwd_p2p_recv")
            _ov_token = self.overlap_detector.before_kernel(p2p_stream, "bwd_p2p_recv", "p2p_recv_stream")
            with torch.cuda.stream(p2p_stream):
                dist.recv(
                    input_grad, src=global_src_rank, group=self.pp_groups[(global_src_rank, self.global_rank)]
                )
            recv_event = torch.cuda.Event()
            recv_event.record(p2p_stream)
            self.overlap_detector.after_kernel(p2p_stream, _ov_token)
            self.p2p_events[p2p_op] = recv_event
            self._stop_timing(p2p_stream, "bwd_p2p_recv")
            self.logger.debug(
                f"Completed bwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id+1} -> {stage_id}, mb {mb_idx})"
            )
        
        self.p2p_cache[p2p_op] = input_grad

    def _exec_bwd_send(self, stage_id: int, mb_idx: int, *deps, p2p_stream=None, comp_stream=None):
        if stage_id <= 0:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_send_stream
        if comp_stream is None:
            comp_stream = self.comp_stream

        # For non-first stages, send output gradients to the previous backward stage
        output_grad = self.inp_activation[stage_id][mb_idx].grad
        if output_grad is None:
            self.logger.warning(f"No output gradient found for stage {stage_id} mb {mb_idx} on actor {self.global_rank}")
            assert False
        pp_rank = self.stage_to_device[stage_id - 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        if self.global_rank == global_src_rank:
            p2p_op = (stage_id, stage_id - 1, mb_idx, True)
            self.p2p_cache[p2p_op] = output_grad
        else:
            self.logger.debug(
                f"Dispatch bwd p2p send on communicator ({self.global_rank}, {global_src_rank}) from {self.global_rank} to {global_src_rank}, op: ({stage_id} -> {stage_id-1}, mb {mb_idx})"
            )
            self._start_timing(p2p_stream, "bwd_p2p_send")
            p2p_stream.wait_stream(comp_stream)
            _ov_token = self.overlap_detector.before_kernel(p2p_stream, "bwd_p2p_send", "p2p_send_stream")
            with torch.cuda.stream(p2p_stream):
                dist.send(
                    output_grad, dst=global_src_rank, group=self.pp_groups[(self.global_rank, global_src_rank)]
                )
            self.overlap_detector.after_kernel(p2p_stream, _ov_token)
            self._stop_timing(p2p_stream, "bwd_p2p_send")
            self.logger.debug(
                f"Completed bwd p2p send on {self.global_rank} to {global_src_rank}, op: ({stage_id} -> {stage_id-1}, mb {mb_idx})"
            )

        self.inp_activation[stage_id][mb_idx] = None
    
    def _forward(self, stage_id: int, mb_idx: int, *deps):
        torch.cuda.nvtx.range_push(f"forward_stage_{stage_id}_mb_{mb_idx}")
        ret = self._forward_impl(stage_id, mb_idx, *deps)
        torch.cuda.nvtx.range_pop()
        return ret

    def _forward_impl(self, stage_id: int, mb_idx: int, *deps):
        self._label_task(f"{stage_id}:{mb_idx}")
        if self.mode == "sequential":
            comp_stream = self.comp_stream
        elif self.mode == "naive":
            comp_stream = self.per_mb_streams[(stage_id + mb_idx) % 2]
        elif self.mode == "overlapped":
            comp_stream = self.comp_stream
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.overlap_a2a_ops = False

        self.logger.debug(
            f"Calling forward {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        if stage_id == 0:
            # For the first stage, load input tensors from self.inputs
            for i, inp in zip(self.input_idxs[(stage_id, None)], self.inputs):
                self.forward_args[stage_id][i] = inp
        else:
            # Wait for recv to complete if this stage receives from the previous stage
            self._start_timing(comp_stream, "fwd_recv_wait")
            recv_key = (stage_id - 1, stage_id, mb_idx, False)
            recv_event = self.p2p_events.pop(recv_key, None)
            if recv_event is not None:
                comp_stream.wait_event(recv_event)
                # self.nccl_monitor.notify_ordered_after(comp_stream, recv_event)
            inputs_from_prev_stage = self.p2p_cache.pop(recv_key)
            self._stop_timing(comp_stream, "fwd_recv_wait")

            # Detach to avoid double-backprop if the previous stage is co-located
            for i, tensor in zip(self.input_idxs[(stage_id, None)], inputs_from_prev_stage):
                if isinstance(tensor, (tuple, list)):
                    assert len(tensor) == 1
                    tensor = tensor[0]
                if stage_id > 0 and self.stage_to_device[stage_id] == self.stage_to_device[stage_id - 1] and tensor.requires_grad:
                    tensor = tensor.detach().requires_grad_(True)
                self.forward_args[stage_id][i] = tensor

            # save first input that requires grad as input activation
            inp_with_grad = [
                self.forward_args[stage_id][i]
                for i in self.input_idxs[(stage_id, None)]
                if self.forward_args[stage_id][i].requires_grad
            ]
            assert (
                len(inp_with_grad) == 1
            ), "Exactly one input per stage should require a gradient"
            self.inp_activation[stage_id][mb_idx] = inp_with_grad[0]

        # Run the forward pass
        self._start_timing(comp_stream, "forward_comp")
        with torch.cuda.stream(comp_stream):
            # torch.cuda.nvtx.range_push(f"forward_stage_{stage_id}_mb_{mb_idx}")
            output = self.forward_fns[stage_id](*self.forward_args[stage_id])
            # torch.cuda.nvtx.range_pop()
        self._stop_timing(comp_stream, "forward_comp")

        # Save first output that requires grad as output activation
        # TODO: support multiple outputs
        out_with_grad = [out for out in output if out.requires_grad]
        if len(out_with_grad) != 1:
            self.logger.warning(
                f"Expected exactly one output with requires_grad in stage {stage_id} mb {mb_idx} on actor {self.global_rank}, but found {len(out_with_grad)}. Output shapes: {[out.shape for out in output]}"
            )
            assert False
        self.out_activation[stage_id][mb_idx] = out_with_grad[0]

        # clear the input tensors
        for i in self.input_idxs[(stage_id, None)]:
            self.forward_args[stage_id][i] = None

        if stage_id < self.num_stages - 1:
            send_p2p_op = (stage_id, stage_id + 1, mb_idx, True)
            assert send_p2p_op not in self.p2p_cache
            self.p2p_cache[send_p2p_op] = output

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comp_stream.synchronize()

        return 0

    def _backward(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        torch.cuda.nvtx.range_push(f"backward_stage_{stage_id}_mb_{mb_idx}")
        ret = self._backward_impl(stage_id, mb_idx, *deps, loss_fn=loss_fn)
        torch.cuda.nvtx.range_pop()
        return ret

    def _backward_impl(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        self._label_task(f"{stage_id}:{mb_idx}")
        if self.mode == "sequential":
            comp_stream = self.comp_stream
        elif self.mode == "naive":
            comp_stream = self.per_mb_streams[(stage_id + mb_idx) % 2]
        elif self.mode == "overlapped":
            comp_stream = self.comp_stream
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.overlap_a2a_ops = False

        self.logger.debug(
            f"Calling backward {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        out_activation = self.out_activation[stage_id][mb_idx]

        # torch.cuda.nvtx.range_push(f"backward_stage_{stage_id}_mb_{mb_idx}")
        if stage_id < self.num_stages - 1:
            # Wait for recv to complete if this stage receives from the next stage
            self._start_timing(comp_stream, "bwd_recv_wait")
            recv_key = (stage_id + 1, stage_id, mb_idx, False)
            recv_event = self.p2p_events.pop(recv_key, None)
            if recv_event is not None:
                comp_stream.wait_event(recv_event)
                # self.nccl_monitor.notify_ordered_after(comp_stream, recv_event)
            input_grad = self.p2p_cache.pop(recv_key)
            self._stop_timing(comp_stream, "bwd_recv_wait")

            self._start_timing(comp_stream, "backward_comp")
            try:
                with torch.cuda.stream(comp_stream):
                    out_activation.backward(gradient=input_grad)
            except RuntimeError:
                raise
            self._stop_timing(comp_stream, "backward_comp")
        else:
            assert loss_fn is not None
            labels = self.labels
            assert out_activation.shape == labels.shape

            self._start_timing(comp_stream, "backward_comp")
            try:
                with torch.cuda.stream(comp_stream):
                    loss = loss_fn(out_activation, labels)
                    loss.backward()
            except RuntimeError:
                raise
            self._stop_timing(comp_stream, "backward_comp")

            # self.loss.append(loss.item())

        # Clear output activation after backward pass
        self.out_activation[stage_id][mb_idx] = None
        del out_activation

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comp_stream.synchronize()

        return 0


    def _backward_input(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        torch.cuda.nvtx.range_push(f"backward_input_stage_{stage_id}_mb_{mb_idx}")
        ret = self._backward_input_impl(stage_id, mb_idx, *deps, loss_fn=loss_fn)
        torch.cuda.nvtx.range_pop()
        return ret

    def _backward_input_impl(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        self._label_task(f"{stage_id}:{mb_idx}")
        comp_stream = self.comp_stream

        self.logger.debug(
            f"Calling backward I {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        out_activation = self.out_activation[stage_id][mb_idx]

        activation_or_loss = None
        upstream_grad = None
        if stage_id < self.num_stages - 1:
            # Wait for recv to complete if this stage receives from the next stage
            recv_key = (stage_id + 1, stage_id, mb_idx, False)
            recv_event = self.p2p_events.pop(recv_key, None)
            if recv_event is not None:
                comp_stream.wait_event(recv_event)
                # self.nccl_monitor.notify_ordered_after(comp_stream, recv_event)
            upstream_grad = self.p2p_cache.pop(recv_key)
            activation_or_loss = out_activation
        else:
            assert loss_fn is not None
            labels = self.labels
            assert out_activation.shape == labels.shape
            with torch.cuda.stream(comp_stream):
                # torch.cuda.nvtx.range_push(f"backward_loss_stage_{stage_id}_mb_{mb_idx}")
                loss = loss_fn(out_activation, labels)
                # self.loss.append(loss.item())
                # torch.cuda.nvtx.range_pop()
            activation_or_loss = loss
            upstream_grad = torch.ones_like(loss)

        # no-op for the first stage
        if stage_id == 0:
            self.upstream_grad_cache[stage_id][mb_idx] = upstream_grad
            self.logger.debug(
                f"Saving upstream gradient {upstream_grad.shape} for stage {stage_id} mb {mb_idx}"
            )
            return 0

        stage_input = self.inp_activation[stage_id][mb_idx]
        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]

        output_nodes = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [activation_or_loss]) if n is not None]
        input_nodes  = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [stage_input]) if n is not None]
        param_nodes   = [n for n in (_get_grad_fn_or_grad_acc(p) for p in stage_params) if n is not None]

        # Use the autograd graph with edges reversed to compute parameter groups, which are groups 
        # of parameters that share the same intermediate nodes. Intermediate nodes are the nodes that
        # lie on both (1) a backward path from the output node(s) to the stage input nodes and 
        # (2) in a path from the output node(s) a parameter node/gradient accumulator
        reverse_edges = construct_reverse_graph(output_nodes)
        param_groups = get_param_groups(input_nodes, param_nodes, reverse_edges)

        # Hooks to capture grads at intermediate nodes. In backward_weight,
        # we'll backprop from these intermediate values
        handles = []
        for pg in param_groups:
            intermediates = pg["intermediates"]
            if not intermediates:
                continue

            pg["grads"] = [None] * len(intermediates)

            for i, intermediate_node in enumerate(intermediates):
                def make_hook(group: Dict[str, Any], idx: int):
                    def hook(grad_inputs):
                        group["grads"][idx] = grad_inputs
                    return hook
                handles.append(intermediate_node.register_prehook(make_hook(pg, i)))

        self._start_timing(comp_stream, "backward_input")
        with torch.cuda.stream(comp_stream):
            # torch.cuda.nvtx.range_push(f"backward_input_stage_{stage_id}_mb_{mb_idx}")
            gx = torch.autograd.grad(
                outputs=activation_or_loss,
                inputs=stage_input,
                grad_outputs=upstream_grad,
                retain_graph=True,
                allow_unused=True,
            )
            # torch.cuda.nvtx.range_pop()
        self._stop_timing(comp_stream, "backward_input")

        gx = gx[0] # Take the gx gradient out of the tuple returned by autograd.grad
        if gx is not None and stage_input.requires_grad:
            if stage_input.grad is None:
                stage_input.grad = gx
            else:
                stage_input.grad.add_(gx)

        # Free output tensors between the output nodes and the intermediate nodes
        if not isinstance(activation_or_loss, list):
            activation_or_loss = [activation_or_loss]

        for t in activation_or_loss:
            t.detach_()

        for h in handles:
            h.remove()

        del activation_or_loss
        del stage_input

        # Save parameter groups for use in backward_weight
        self.bw_param_groups[stage_id][mb_idx] = param_groups

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comp_stream.synchronize()

        return 0

    def _backward_weight(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
        ret = self._backward_weight_impl(stage_id, mb_idx, *deps, loss_fn=loss_fn)
        torch.cuda.nvtx.range_pop()
        return ret

    def _backward_weight_impl(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        self._label_task(f"{stage_id}:{mb_idx}")
        comp_stream = self.comp_stream

        self.logger.debug(
            f"Calling backward W {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]
        updated_params: dict[int, torch.nn.Parameter] = {}

        # Special case to handle stage 0 since backward_input is a NOOP, 
        # meaning no parameter groups are created
        if stage_id == 0:
            upstream_grad = self.upstream_grad_cache[stage_id][mb_idx]
            out_activation = self.out_activation[stage_id][mb_idx]
            self._start_timing(comp_stream, "backward_weight")
            if stage_id < self.num_stages - 1:
                with torch.cuda.stream(comp_stream):
                    # torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
                    gparams = torch.autograd.grad(
                        outputs=out_activation,
                        inputs=stage_params,
                        grad_outputs=upstream_grad,
                        retain_graph=False,
                    )
                    # torch.cuda.nvtx.range_pop()
            else:
                assert loss_fn is not None
                labels = self.labels
                assert out_activation.shape == labels.shape
                with torch.cuda.stream(comp_stream):
                    # torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
                    loss = loss_fn(out_activation, labels)
                    gparams = torch.autograd.grad(
                        outputs=loss,
                        inputs=stage_params,
                        retain_graph=False,
                    )
                    # torch.cuda.nvtx.range_pop()
            self._stop_timing(comp_stream, "backward_weight")

            assert len(gparams) == len(stage_params), (
                f"Stage {stage_id}: mismatch #param grads {len(gparams)} vs params {len(stage_params)}"
            )
            
            for p, pg in zip(stage_params, gparams):
                if pg is None:
                    continue
                if p.grad is None:
                    p.grad = pg.clone()
                else:
                    p.grad.add_(pg)
                updated_params[id(p)] = p
        else:
            # Create mapping from autograd nodes -> parameters
            grad_acc_to_weight: Dict[Node, Tuple[Parameter, int]] = {}
            for param in stage_params:
                node = _get_grad_fn_or_grad_acc(param)
                grad_acc_to_weight[node] = param

            param_groups = self.bw_param_groups[stage_id][mb_idx]

            # Perform the weight updates separately for each param_group, beginning
            # backprop from each the intermediate node(s) of each group
            for pg in param_groups:
                intermediates: List[Node] = pg.get("intermediates", [])
                intermediate_grads = pg.get("grads", None) # List of intermediate node gradients, captured by the hooks

                # Skip groups without intermediate nodes (could happen in weird cases
                # where one node is disconnected from the rest of the autograd graph for some reason)
                if not intermediates or intermediate_grads is None:
                    continue

                intermediate_edges: List[GradientEdge] = []
                intermediate_edge_grads: List[torch.Tensor] = []

                for intermediate_node, grad_inputs in zip(intermediates, intermediate_grads):
                    if grad_inputs is None:
                        continue

                    gs = [x for x in grad_inputs if x is not None]
                    if not gs:
                        continue
                    
                    # Sum all gradients arriving at the current intermediate node
                    # in case the node has multiple source of gradients
                    summed = sum(gs)

                    # Create a GradientEdge for each intermediate node (we can backprop with respect to these)
                    # and store the summed gradient for that node
                    intermediate_edges.append(GradientEdge(intermediate_node, 0))
                    intermediate_edge_grads.append(summed)

                del pg["intermediates"]

                if not intermediate_edges:
                    continue

                # Grab params for the param_nodes in this param group using our grad_acc_to_weight map from earlier
                mapped_param_nodes = [p for p in pg["params"] if p in grad_acc_to_weight]
                if not mapped_param_nodes:
                    continue

                # Use these parameters to create a GradientEdge that we'll use as our input to autograd.grad
                weight_edges = tuple(GradientEdge(p, 0) for p in mapped_param_nodes)

                self._start_timing(comp_stream, "backward_weight")
                with torch.cuda.stream(comp_stream):
                    # torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
                    gparams = torch.autograd.grad(
                        outputs=intermediate_edges,
                        inputs=weight_edges,
                        grad_outputs=intermediate_edge_grads,
                        retain_graph=False,
                    )
                    # torch.cuda.nvtx.range_pop()
                self._stop_timing(comp_stream, "backward_weight")

                del pg["grads"]

                assert len(gparams) == len(mapped_param_nodes), (
                    f"Stage {stage_id}: mismatch #param grads {len(gparams)} vs params {len(mapped_param_nodes)}"
                )
                
                # Finally, update gradients for the params in this param_group
                for param_node, dw in zip(mapped_param_nodes, gparams):
                    if dw is None:
                        continue

                    weight = grad_acc_to_weight[param_node]

                    if weight.grad is None:
                        weight.grad = dw
                    else:
                        weight.grad.add_(dw)
                    updated_params[id(weight)] = weight

        # Mirror post-accumulate hook behavior for split backward.
        for p in updated_params.values():
            self._maybe_trigger_grad_allreduce(stage_id, p)

        self.bw_grad_cache[stage_id][mb_idx] = None
        self.upstream_grad_cache[stage_id][mb_idx] = None
        self.bw_param_groups[stage_id][mb_idx] = None
        self.out_activation[stage_id][mb_idx] = None

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comp_stream.synchronize()

        return 0


    def _backward_input(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        comp_stream = self.comp_stream

        self.logger.debug(
            f"Calling backward I {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        out_activation = self.out_activation[stage_id][mb_idx]

        activation_or_loss = None
        upstream_grad = None
        if stage_id < self.num_stages - 1:
            activation_or_loss = out_activation
            upstream_grad = self.p2p_cache.pop(
                (stage_id + 1, stage_id, mb_idx, False)
            )
        else:
            assert loss_fn is not None
            labels = self.labels
            assert out_activation.shape == labels.shape
            with torch.cuda.stream(comp_stream):
                loss = loss_fn(out_activation, labels)
                self.loss.append(loss.item())
            activation_or_loss = loss
            upstream_grad = torch.ones_like(loss)

        # no-op for the first stage
        if stage_id == 0:
            self.upstream_grad_cache[stage_id][mb_idx] = upstream_grad
            self.logger.debug(
                f"Saving upstream gradient {upstream_grad.shape} for stage {stage_id} mb {mb_idx}"
            )
            return 1

        stage_input = self.inp_activation[stage_id][mb_idx]
        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]

        output_nodes = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [activation_or_loss]) if n is not None]
        input_nodes  = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [stage_input]) if n is not None]
        param_nodes   = [n for n in (_get_grad_fn_or_grad_acc(p) for p in stage_params) if n is not None]

        # Use the autograd graph with edges reversed to compute parameter groups, which are groups 
        # of parameters that share the same intermediate nodes. Intermediate nodes are the nodes that
        # lie on both (1) a backward path from the output node(s) to the stage input nodes and 
        # (2) in a path from the output node(s) a parameter node/gradient accumulator
        reverse_edges = construct_reverse_graph(output_nodes)
        param_groups = get_param_groups(input_nodes, param_nodes, reverse_edges)

        # Hooks to capture grads at intermediate nodes. In backward_weight,
        # we'll backprop from these intermediate values
        handles = []
        for pg in param_groups:
            intermediates = pg["intermediates"]
            if not intermediates:
                continue

            pg["grads"] = [None] * len(intermediates)

            for i, intermediate_node in enumerate(intermediates):
                def make_hook(group: Dict[str, Any], idx: int):
                    def hook(grad_inputs):
                        group["grads"][idx] = grad_inputs
                    return hook
                handles.append(intermediate_node.register_prehook(make_hook(pg, i)))

        gx = torch.autograd.grad(
            outputs=activation_or_loss,
            inputs=stage_input,
            grad_outputs=upstream_grad,
            retain_graph=True,
            allow_unused=True,
        )

        gx = gx[0] # Take the gx gradient out of the tuple returned by autograd.grad
        if gx is not None and stage_input.requires_grad:
            if stage_input.grad is None:
                stage_input.grad = gx
            else:
                stage_input.grad.add_(gx)

        # Free output tensors between the output nodes and the intermediate nodes
        if not isinstance(activation_or_loss, list):
            activation_or_loss = [activation_or_loss]

        for t in activation_or_loss:
            t.detach_()

        for h in handles:
            h.remove()

        del activation_or_loss
        del stage_input

        # Save parameter groups for use in backward_weight
        self.bw_param_groups[stage_id][mb_idx] = param_groups

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comp_stream.synchronize()

        return 0


    def _backward_input(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        torch.cuda.nvtx.range_push(f"backward_input_stage_{stage_id}_mb_{mb_idx}")
        ret = self._backward_input_impl(stage_id, mb_idx, *deps, loss_fn=loss_fn)
        torch.cuda.nvtx.range_pop()
        return ret

    def _backward_input_impl(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        self._label_task(f"{stage_id}:{mb_idx}")
        comp_stream = self.comp_stream

        self.logger.debug(
            f"Calling backward I {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        out_activation = self.out_activation[stage_id][mb_idx]

        activation_or_loss = None
        upstream_grad = None
        if stage_id < self.num_stages - 1:
            # Wait for recv to complete if this stage receives from the next stage
            recv_key = (stage_id + 1, stage_id, mb_idx, False)
            recv_event = self.p2p_events.pop(recv_key, None)
            if recv_event is not None:
                comp_stream.wait_event(recv_event)
                # self.nccl_monitor.notify_ordered_after(comp_stream, recv_event)
            upstream_grad = self.p2p_cache.pop(recv_key)
            activation_or_loss = out_activation
        else:
            assert loss_fn is not None
            labels = self.labels
            assert out_activation.shape == labels.shape
            with torch.cuda.stream(comp_stream):
                # torch.cuda.nvtx.range_push(f"backward_loss_stage_{stage_id}_mb_{mb_idx}")
                loss = loss_fn(out_activation, labels)
                # self.loss.append(loss.item())
                # torch.cuda.nvtx.range_pop()
            activation_or_loss = loss
            upstream_grad = torch.ones_like(loss)

        # no-op for the first stage
        if stage_id == 0:
            self.upstream_grad_cache[stage_id][mb_idx] = upstream_grad
            self.logger.debug(
                f"Saving upstream gradient {upstream_grad.shape} for stage {stage_id} mb {mb_idx}"
            )
            return 0

        stage_input = self.inp_activation[stage_id][mb_idx]
        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]

        output_nodes = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [activation_or_loss]) if n is not None]
        input_nodes  = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [stage_input]) if n is not None]
        param_nodes   = [n for n in (_get_grad_fn_or_grad_acc(p) for p in stage_params) if n is not None]

        # Use the autograd graph with edges reversed to compute parameter groups, which are groups 
        # of parameters that share the same intermediate nodes. Intermediate nodes are the nodes that
        # lie on both (1) a backward path from the output node(s) to the stage input nodes and 
        # (2) in a path from the output node(s) a parameter node/gradient accumulator
        reverse_edges = construct_reverse_graph(output_nodes)
        param_groups = get_param_groups(input_nodes, param_nodes, reverse_edges)

        # Hooks to capture grads at intermediate nodes. In backward_weight,
        # we'll backprop from these intermediate values
        handles = []
        for pg in param_groups:
            intermediates = pg["intermediates"]
            if not intermediates:
                continue

            pg["grads"] = [None] * len(intermediates)

            for i, intermediate_node in enumerate(intermediates):
                def make_hook(group: Dict[str, Any], idx: int):
                    def hook(grad_inputs):
                        group["grads"][idx] = grad_inputs
                    return hook
                handles.append(intermediate_node.register_prehook(make_hook(pg, i)))

        self._start_timing(comp_stream, "backward_input")
        with torch.cuda.stream(comp_stream):
            # torch.cuda.nvtx.range_push(f"backward_input_stage_{stage_id}_mb_{mb_idx}")
            gx = torch.autograd.grad(
                outputs=activation_or_loss,
                inputs=stage_input,
                grad_outputs=upstream_grad,
                retain_graph=True,
                allow_unused=True,
            )
            # torch.cuda.nvtx.range_pop()
        self._stop_timing(comp_stream, "backward_input")

        gx = gx[0] # Take the gx gradient out of the tuple returned by autograd.grad
        if gx is not None and stage_input.requires_grad:
            if stage_input.grad is None:
                stage_input.grad = gx
            else:
                stage_input.grad.add_(gx)

        # Free output tensors between the output nodes and the intermediate nodes
        if not isinstance(activation_or_loss, list):
            activation_or_loss = [activation_or_loss]

        for t in activation_or_loss:
            t.detach_()

        for h in handles:
            h.remove()

        del activation_or_loss
        del stage_input

        # Save parameter groups for use in backward_weight
        self.bw_param_groups[stage_id][mb_idx] = param_groups

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comp_stream.synchronize()

        return 0

    def _backward_weight(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
        ret = self._backward_weight_impl(stage_id, mb_idx, *deps, loss_fn=loss_fn)
        torch.cuda.nvtx.range_pop()
        return ret

    def _backward_weight_impl(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        self._label_task(f"{stage_id}:{mb_idx}")
        comp_stream = self.comp_stream

        self.logger.debug(
            f"Calling backward W {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]
        updated_params: dict[int, torch.nn.Parameter] = {}

        # Special case to handle stage 0 since backward_input is a NOOP, 
        # meaning no parameter groups are created
        if stage_id == 0:
            upstream_grad = self.upstream_grad_cache[stage_id][mb_idx]
            out_activation = self.out_activation[stage_id][mb_idx]
            self._start_timing(comp_stream, "backward_weight")
            if stage_id < self.num_stages - 1:
                with torch.cuda.stream(comp_stream):
                    # torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
                    gparams = torch.autograd.grad(
                        outputs=out_activation,
                        inputs=stage_params,
                        grad_outputs=upstream_grad,
                        retain_graph=False,
                    )
                    # torch.cuda.nvtx.range_pop()
            else:
                assert loss_fn is not None
                labels = self.labels
                assert out_activation.shape == labels.shape
                with torch.cuda.stream(comp_stream):
                    # torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
                    loss = loss_fn(out_activation, labels)
                    gparams = torch.autograd.grad(
                        outputs=loss,
                        inputs=stage_params,
                        retain_graph=False,
                    )
                    # torch.cuda.nvtx.range_pop()
            self._stop_timing(comp_stream, "backward_weight")

            assert len(gparams) == len(stage_params), (
                f"Stage {stage_id}: mismatch #param grads {len(gparams)} vs params {len(stage_params)}"
            )
            
            for p, pg in zip(stage_params, gparams):
                if pg is None:
                    continue
                if p.grad is None:
                    p.grad = pg.clone()
                else:
                    p.grad.add_(pg)
                updated_params[id(p)] = p
        else:
            # Create mapping from autograd nodes -> parameters
            grad_acc_to_weight: Dict[Node, Tuple[Parameter, int]] = {}
            for param in stage_params:
                node = _get_grad_fn_or_grad_acc(param)
                grad_acc_to_weight[node] = param

            param_groups = self.bw_param_groups[stage_id][mb_idx]

            # Perform the weight updates separately for each param_group, beginning
            # backprop from each the intermediate node(s) of each group
            for pg in param_groups:
                intermediates: List[Node] = pg.get("intermediates", [])
                intermediate_grads = pg.get("grads", None) # List of intermediate node gradients, captured by the hooks

                # Skip groups without intermediate nodes (could happen in weird cases
                # where one node is disconnected from the rest of the autograd graph for some reason)
                if not intermediates or intermediate_grads is None:
                    continue

                intermediate_edges: List[GradientEdge] = []
                intermediate_edge_grads: List[torch.Tensor] = []

                for intermediate_node, grad_inputs in zip(intermediates, intermediate_grads):
                    if grad_inputs is None:
                        continue

                    gs = [x for x in grad_inputs if x is not None]
                    if not gs:
                        continue
                    
                    # Sum all gradients arriving at the current intermediate node
                    # in case the node has multiple source of gradients
                    summed = sum(gs)

                    # Create a GradientEdge for each intermediate node (we can backprop with respect to these)
                    # and store the summed gradient for that node
                    intermediate_edges.append(GradientEdge(intermediate_node, 0))
                    intermediate_edge_grads.append(summed)

                del pg["intermediates"]

                if not intermediate_edges:
                    continue

                # Grab params for the param_nodes in this param group using our grad_acc_to_weight map from earlier
                mapped_param_nodes = [p for p in pg["params"] if p in grad_acc_to_weight]
                if not mapped_param_nodes:
                    continue

                # Use these parameters to create a GradientEdge that we'll use as our input to autograd.grad
                weight_edges = tuple(GradientEdge(p, 0) for p in mapped_param_nodes)

                self._start_timing(comp_stream, "backward_weight")
                with torch.cuda.stream(comp_stream):
                    # torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
                    gparams = torch.autograd.grad(
                        outputs=intermediate_edges,
                        inputs=weight_edges,
                        grad_outputs=intermediate_edge_grads,
                        retain_graph=False,
                    )
                    # torch.cuda.nvtx.range_pop()
                self._stop_timing(comp_stream, "backward_weight")

                del pg["grads"]

                assert len(gparams) == len(mapped_param_nodes), (
                    f"Stage {stage_id}: mismatch #param grads {len(gparams)} vs params {len(mapped_param_nodes)}"
                )
                
                # Finally, update gradients for the params in this param_group
                for param_node, dw in zip(mapped_param_nodes, gparams):
                    if dw is None:
                        continue

                    weight = grad_acc_to_weight[param_node]

                    if weight.grad is None:
                        weight.grad = dw
                    else:
                        weight.grad.add_(dw)
                    updated_params[id(weight)] = weight

        # Mirror post-accumulate hook behavior for split backward.
        for p in updated_params.values():
            self._maybe_trigger_grad_allreduce(stage_id, p)

        self.bw_grad_cache[stage_id][mb_idx] = None
        self.upstream_grad_cache[stage_id][mb_idx] = None
        self.bw_param_groups[stage_id][mb_idx] = None
        self.out_activation[stage_id][mb_idx] = None

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comp_stream.synchronize()

        return 0

    def _backward_weight(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
        comp_stream = self.comp_stream

        self.logger.debug(
            f"Calling backward W {stage_id} mb {mb_idx} on actor {self.global_rank}"
        )

        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]

        # Special case to handle stage 0 since backward_input is a NOOP, 
        # meaning no parameter groups are created
        if stage_id == 0:
            upstream_grad = self.upstream_grad_cache[stage_id][mb_idx]
            out_activation = self.out_activation[stage_id][mb_idx]
            if stage_id < self.num_stages - 1:
                gparams = torch.autograd.grad(
                    outputs=out_activation,
                    inputs=stage_params,
                    grad_outputs=upstream_grad,
                    retain_graph=False,
                )
            else:
                assert loss_fn is not None
                labels = self.labels
                assert out_activation.shape == labels.shape
                with torch.cuda.stream(comp_stream):
                    loss = loss_fn(out_activation, labels)
                    gparams = torch.autograd.grad(
                        outputs=loss,
                        inputs=stage_params,
                        retain_graph=False,
                    )

            assert len(gparams) == len(stage_params), (
                f"Stage {stage_id}: mismatch #param grads {len(gparams)} vs params {len(stage_params)}"
            )
            
            for p, pg in zip(stage_params, gparams):
                if p.grad is None:
                    p.grad = pg.clone()
                else:
                    p.grad.add_(pg)
        else:
            # Create mapping from autograd nodes -> parameters
            grad_acc_to_weight: Dict[Node, Tuple[Parameter, int]] = {}
            for param in stage_params:
                node = _get_grad_fn_or_grad_acc(param)
                grad_acc_to_weight[node] = param

            param_groups = self.bw_param_groups[stage_id][mb_idx]

            # Perform the weight updates separately for each param_group, beginning
            # backprop from each the intermediate node(s) of each group
            for pg in param_groups:
                intermediates: List[Node] = pg.get("intermediates", [])
                intermediate_grads = pg.get("grads", None) # List of intermediate node gradients, captured by the hooks

                # Skip groups without intermediate nodes (could happen in weird cases
                # where one node is disconnected from the rest of the autograd graph for some reason)
                if not intermediates or intermediate_grads is None:
                    continue

                intermediate_edges: List[GradientEdge] = []
                intermediate_edge_grads: List[torch.Tensor] = []

                for intermediate_node, grad_inputs in zip(intermediates, intermediate_grads):
                    if grad_inputs is None:
                        continue

                    gs = [x for x in grad_inputs if x is not None]
                    if not gs:
                        continue
                    
                    # Sum all gradients arriving at the current intermediate node
                    # in case the node has multiple source of gradients
                    summed = sum(gs)

                    # Create a GradientEdge for each intermediate node (we can backprop with respect to these)
                    # and store the summed gradient for that node
                    intermediate_edges.append(GradientEdge(intermediate_node, 0))
                    intermediate_edge_grads.append(summed)

                del pg["intermediates"]

                if not intermediate_edges:
                    continue

                # Grab params for the param_nodes in this param group using our grad_acc_to_weight map from earlier
                mapped_param_nodes = [p for p in pg["params"] if p in grad_acc_to_weight]
                if not mapped_param_nodes:
                    continue

                # Use these parameters to create a GradientEdge that we'll use as our input to autograd.grad
                weight_edges = tuple(GradientEdge(p, 0) for p in mapped_param_nodes)

                gparams = torch.autograd.grad(
                    outputs=intermediate_edges,
                    inputs=weight_edges,
                    grad_outputs=intermediate_edge_grads,
                    retain_graph=False,
                )

                del pg["grads"]

                assert len(gparams) == len(mapped_param_nodes), (
                    f"Stage {stage_id}: mismatch #param grads {len(gparams)} vs params {len(mapped_param_nodes)}"
                )
                
                # Finally, update gradients for the params in this param_group
                for param_node, dw in zip(pg["params"], gparams):
                    if dw is None:
                        continue

                    weight = grad_acc_to_weight[param_node]

                    if weight.grad is None:
                        weight.grad = dw
                    else:
                        weight.grad.add_(dw)

        self.bw_grad_cache[stage_id][mb_idx] = None
        self.upstream_grad_cache[stage_id][mb_idx] = None
        self.bw_param_groups[stage_id][mb_idx] = None
        self.out_activation[stage_id][mb_idx] = None

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.cuda.synchronize()

        return 1

    def _forward_backward(self, fwd_stage_id: int, fwd_mb_idx: int, bwd_stage_id: int, bwd_mb_idx: int, *deps, loss_fn=None):
        torch.cuda.nvtx.range_push(f"forward_stage_{fwd_stage_id}_mb_{fwd_mb_idx}_backward_stage_{bwd_stage_id}_mb_{bwd_mb_idx}")
        ret = self._forward_backward_impl(fwd_stage_id, fwd_mb_idx, bwd_stage_id, bwd_mb_idx, *deps, loss_fn=loss_fn)
        torch.cuda.nvtx.range_pop()
        return ret

    def _forward_backward_impl(self, fwd_stage_id: int, fwd_mb_idx: int, bwd_stage_id: int, bwd_mb_idx: int, *deps, loss_fn=None):
        self._label_task(f"{fwd_stage_id}:{fwd_mb_idx}|{bwd_stage_id}:{bwd_mb_idx}")
        if self.mode == "sequential":
            fwd_comp_stream = self.comp_stream
            bwd_comp_stream = self.comp_stream
            self.overlap_a2a_ops = False
        elif self.mode == "naive":
            fwd_comp_stream = self.per_mb_streams[(fwd_stage_id + fwd_mb_idx) % 2]
            bwd_comp_stream = self.per_mb_streams[(bwd_stage_id + bwd_mb_idx) % 2]
            self.overlap_a2a_ops = False
        elif self.mode == "overlapped":
            fwd_comp_stream = self.comp_stream
            bwd_comp_stream = self.comp_stream
            self.overlap_a2a_ops = True
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.logger.debug(
            f"Calling forward {fwd_stage_id} mb {fwd_mb_idx} backward {bwd_stage_id} mb {bwd_mb_idx} on actor {self.global_rank}"
        )
        
        # PREPARE FORWARD PASS

        if fwd_stage_id == 0:
            # For the first stage, load input tensors from self.inputs
            for i, inp in zip(self.input_idxs[(fwd_stage_id, None)], self.inputs):
                self.forward_args[fwd_stage_id][i] = inp
        else:
            fwd_recv_key = (fwd_stage_id - 1, fwd_stage_id, fwd_mb_idx, False)

            # Wait for recv to complete if this stage receives from the previous stage
            fwd_recv_event = self.p2p_events.pop(fwd_recv_key, None)
            if fwd_recv_event is not None:
                fwd_comp_stream.wait_event(fwd_recv_event)
                # self.nccl_monitor.notify_ordered_after(fwd_comp_stream, fwd_recv_event)
            inputs_from_prev_stage = self.p2p_cache.pop(fwd_recv_key)

            # Detach to avoid double-backprop if the previous stage is co-located
            for i, tensor in zip(self.input_idxs[(fwd_stage_id, None)], inputs_from_prev_stage):
                if isinstance(tensor, (tuple, list)):
                    assert len(tensor) == 1
                    tensor = tensor[0]
                if fwd_stage_id > 0 and self.stage_to_device[fwd_stage_id] == self.stage_to_device[fwd_stage_id - 1] and tensor.requires_grad:
                    tensor = tensor.detach().requires_grad_(True)
                self.forward_args[fwd_stage_id][i] = tensor

            # save first input that requires grad as input activation
            inp_with_grad = [
                self.forward_args[fwd_stage_id][i]
                for i in self.input_idxs[(fwd_stage_id, None)]
                if self.forward_args[fwd_stage_id][i].requires_grad
            ]
            assert (
                len(inp_with_grad) == 1
            ), "Exactly one input per stage should require a gradient"
            self.logger.debug(
                f"Saving input activation {inp_with_grad[0].shape} for stage {fwd_stage_id} mb {fwd_mb_idx}"
            )
            self.inp_activation[fwd_stage_id][fwd_mb_idx] = inp_with_grad[0]

        # PREPARE BACKWARD PASS

        out_activation = self.out_activation[bwd_stage_id][bwd_mb_idx]
        if bwd_stage_id < self.num_stages - 1:
            bwd_recv_key = (bwd_stage_id + 1, bwd_stage_id, bwd_mb_idx, False)

            # Wait for recv to complete if this stage receives from the next stage
            bwd_recv_event = self.p2p_events.pop(bwd_recv_key, None)
            if bwd_recv_event is not None:
                bwd_comp_stream.wait_event(bwd_recv_event)
                # self.nccl_monitor.notify_ordered_after(bwd_comp_stream, bwd_recv_event)
            input_grad = self.p2p_cache.pop(bwd_recv_key)
        else:
            assert loss_fn is not None
            labels = self.labels
            assert out_activation.shape == labels.shape
            
        # RUN FORWARD AND BACKWARD PASSES
        if self.overlap_a2a_ops:
            output = None
            loss = None
            # threading.Events for CPU-side thread coordination
            n_a2a = self.n_a2a_ops[fwd_stage_id]
            self.fwd_a2a_submitted = [threading.Event() for _ in range(n_a2a)]
            self.bwd_a2a_submitted = [threading.Event() for _ in range(n_a2a)]
            self.fwd_a2a_counter = 0
            self.bwd_a2a_counter = 0

            def run_fwd():
                nonlocal output
                with torch.cuda.stream(fwd_comp_stream):
                    output = self.forward_fns[fwd_stage_id](*self.forward_args[fwd_stage_id])

            def run_bwd():
                nonlocal loss
                self.fwd_a2a_submitted[0].wait()
                if bwd_stage_id < self.num_stages - 1:
                    with torch.cuda.stream(bwd_comp_stream):
                        out_activation.backward(gradient=input_grad)
                else:
                    with torch.cuda.stream(bwd_comp_stream):
                        loss = loss_fn(out_activation, labels)
                        loss.backward()

            fwd_thread = threading.Thread(target=run_fwd)
            bwd_thread = threading.Thread(target=run_bwd)
            fwd_thread.start()
            bwd_thread.start()
            fwd_thread.join()
            bwd_thread.join()
        else:
            with torch.cuda.stream(fwd_comp_stream):
                output = self.forward_fns[fwd_stage_id](*self.forward_args[fwd_stage_id])

            if bwd_stage_id < self.num_stages - 1:
                with torch.cuda.stream(bwd_comp_stream):
                    out_activation.backward(gradient=input_grad)
            else:
                with torch.cuda.stream(bwd_comp_stream):
                    loss = loss_fn(out_activation, labels)
                    loss.backward()

        # POST PROCESS FORWARD PASS

        out_with_grad = [out for out in output if out.requires_grad]
        assert (
            len(out_with_grad) == 1
        ), "Piper only supports one output per subgraph with requires_grad"
        self.logger.debug(
            f"Saving output activation {out_with_grad[0].shape} for stage {fwd_stage_id} mb {fwd_mb_idx}"
        )
        self.out_activation[fwd_stage_id][fwd_mb_idx] = out_with_grad[0]

        # clear the input tensors
        for i in self.input_idxs[(fwd_stage_id, None)]:
            self.forward_args[fwd_stage_id][i] = None

        if fwd_stage_id < self.num_stages - 1:
            send_p2p_op = (fwd_stage_id, fwd_stage_id + 1, fwd_mb_idx, True)
            assert send_p2p_op not in self.p2p_cache
            self.p2p_cache[send_p2p_op] = output

        # POST PROCESS BACKWARD PASS
        # if bwd_stage_id == self.num_stages - 1:
        #     self.loss.append(loss.item())

        # Clear output activation after backward pass
        self.out_activation[bwd_stage_id][bwd_mb_idx] = None
        del out_activation

        # POST FORWARD P2P OPERATIONS
        if fwd_stage_id < self.num_stages - 1:
            send_p2p_op = (fwd_stage_id, fwd_stage_id + 1, fwd_mb_idx, True)
            assert send_p2p_op not in self.p2p_cache
            self.p2p_cache[send_p2p_op] = output

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fwd_comp_stream.synchronize()
        if bwd_comp_stream is not fwd_comp_stream:
            bwd_comp_stream.synchronize()

        # clear A2A states
        if self.overlap_a2a_ops:
            self.overlap_a2a_ops = False
            self.fwd_a2a_submitted.clear()
            self.bwd_a2a_submitted.clear()
            self.fwd_a2a_counter = 0
            self.bwd_a2a_counter = 0

        return 0

    def _synchronize_gradients(self):
        self.logger.info(f"Actor {self.global_rank} synchronizing gradients")
        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.forward_args.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    _ov_token = self.overlap_detector.before_kernel(self.comm_stream, "grad_allreduce_naive", "comm_stream")
                    with torch.cuda.stream(self.comm_stream):
                        dist.all_reduce(
                            param.grad, op=dist.ReduceOp.AVG, group=self.dp_group
                        )
                    self.overlap_detector.after_kernel(self.comm_stream, _ov_token)

    def _update(self, *deps):
        ret = self._update_impl(*deps)
        return ret

    def _update_impl(self, *deps):
        self._label_task("update")

        if self.ar_events:
            # DAG execution path: wait for in-flight all-reduces then step optimizers.
            self._start_timing(self.comp_stream, "optim_step")
            for ar_evt in self.ar_events.values():
                self.comp_stream.wait_event(ar_evt)
            for s_id, optim_list in self.bucket_optims.items():
                for optim in optim_list:
                    if optim is not None:
                        optim.step()
                        optim.zero_grad(set_to_none=False)
            self._stop_timing(self.comp_stream, "optim_step")
        elif self.dp_degree > 1:
            # Legacy non-DAG DDP path (no ALL_REDUCE nodes in the DAG).
            self._start_timing(self.comm_stream, "backward_sync")
            if self.naive_gradient_sync:
                self._synchronize_gradients()
            else:
                self._wait_for_comm_ops()
            self._stop_timing(self.comm_stream, "backward_sync")

            self._start_timing(self.comp_stream, "optim_step")
            for s_id, optim_list in self.bucket_optims.items():
                for optim in optim_list:
                    if optim is not None:
                        optim.step()
                        optim.zero_grad(set_to_none=False)
            self._stop_timing(self.comp_stream, "optim_step")
        else:
            # Single-device path.
            self._start_timing(self.comp_stream, "optim_step")
            for s_id, optim_list in self.bucket_optims.items():
                for optim in optim_list:
                    if optim is not None:
                        optim.step()
                        optim.zero_grad(set_to_none=False)
            self._stop_timing(self.comp_stream, "optim_step")

        torch.cuda.synchronize()

        losses = self.loss
        self.loss.clear()

        self.reset_p2p_states()
        self.overlap_detector.reset_iteration()

        return losses

    # -----------------------------------------------------------------------
    # DAG-based execution
    # -----------------------------------------------------------------------

    def load_dag(self, dag: TaskDAG) -> None:
        """Store the per-rank TaskDAG for subsequent run_dag() calls."""
        self.dag = dag

    def get_bucket_fwd_counts(self) -> dict:
        """Return the number of forward buckets for each bucketed stage on this actor."""
        return {stage_id: len(fns) for stage_id, fns in self.bucket_fwd_fns.items()}

    def get_a2a_boundaries(self) -> dict:
        """Return A2A boundary info: stage_id -> {boundary_bucket_id -> tensor_idx}."""
        return dict(self.a2a_boundaries)

    def run_dag(self, loss_fn=None):
        """Execute the loaded TaskDAG in topological order.

        Synchronisation contract:
        - Before a SEND node: p2p_send_stream waits on the compute event so the
          send buffer is fully written before the transfer begins.
        - Before a compute node that has a RECV predecessor: comp_stream waits
          on the recv_event so the recv buffer is populated before compute reads it.
        - RECV records a recv_event after the dist.recv completes.
        """
        from collections import deque

        assert self.dag is not None, "load_dag() must be called before run_dag()"
        dag = self.dag

        # Reset per-iteration DAG buffers
        self.send_buffer = {}
        self.bucket_buffer = {}
        self.a2a_buffer = {}
        self.a2a_events = {}
        self.ar_events = {}

        # Point each trainable param's .grad at the appropriate slice of its
        # bucket's flat_grads tensor so backward accumulates contiguously.
        for (s_id, b_idx), trainable_idxs in self.bucket_trainable_param_idxs.items():
            flat_grads = self.bucket_flat_grads.get((s_id, b_idx))
            if flat_grads is None:
                continue
            flat_grads.zero_()
            args = self.bucket_fwd_args[s_id][b_idx]
            offset = 0
            for i in trainable_idxs:
                p = args[i]
                if p is not None:
                    numel = p.numel()
                    p.grad = flat_grads[offset:offset + numel].view(p.shape)
                    offset += numel
        # Pre-create a threading.Event for every expected send key so that
        # _exec_send can wait on it if the compute function hasn't written the
        # buffer yet (guards against out-of-order CPU dispatch).
        self.send_buffer_ready = {}
        for _node in dag.nodes:
            if _node.task.type == TaskType.SEND:
                _compute = _node.data_preds[0]
                _b = _compute.task.batches[0]
                _sid, _mid, _ct = _b.stage_id, _b.mb_idx, _compute.task.type
                if _ct == TaskType.FWD:
                    _k = (_sid, _compute.task.bucket_id)
                else:
                    _k = None
                self.send_buffer_ready[(_k, _mid)] = threading.Event()
        self.recv_buffer = {}
        self.recv_events = {}
        comp_events: dict = {}  # (stage_id, mb_idx, task_type, bucket_id) -> cuda.Event

        # Kahn's algorithm: combined data + temporal predecessor counts
        in_degree = {
            id(n): len(n.data_preds) + (1 if n.temporal_pred is not None else 0)
            for n in dag.nodes
        }
        ready: deque = deque(n for n in dag.nodes if in_degree[id(n)] == 0)

        while ready:
            node = ready.popleft()
            task = node.task
            # batches always has exactly one entry for single-stage tasks
            batch = task.batches[0]
            stage_id = batch.stage_id
            mb_idx = batch.mb_idx

            self.logger.debug(
                f"run_dag dispatch: {task.type.value} s{stage_id} mb{mb_idx} bkt={task.bucket_id}"
            )

            match task.type:

                case TaskType.SEND:
                    compute_node = node.data_preds[0]
                    _cs = compute_node.task.batches[0].stage_id
                    _cm = compute_node.task.batches[0].mb_idx
                    _ct = compute_node.task.type
                    send_key = (_cs, compute_node.task.bucket_id) if _ct == TaskType.FWD else None
                    comp_key = (_cs, _cm, _ct, compute_node.task.bucket_id)
                    self.p2p_send_stream.wait_event(comp_events[comp_key])
                    self._exec_send(stage_id, mb_idx, send_key, node.peer_pp_rank)

                case TaskType.RECV:
                    compute_node = node.data_succs[0]
                    recv_key = (stage_id, compute_node.task.bucket_id) \
                        if compute_node.task.type == TaskType.FWD else None
                    self._exec_recv(stage_id, mb_idx, recv_key, node.peer_pp_rank)

                case TaskType.FWD:
                    bucket_id = task.bucket_id
                    recv_key = ((stage_id, bucket_id), mb_idx)
                    if recv_key in self.recv_events:
                        self.fwd_comp_stream.wait_event(self.recv_events.pop(recv_key))
                    torch.cuda.nvtx.range_push(f"forward_s{stage_id}_b{bucket_id}_mb{mb_idx}")
                    self._forward_dag(stage_id, bucket_id, mb_idx)
                    torch.cuda.nvtx.range_pop()
                    evt = torch.cuda.Event()
                    evt.record(self.fwd_comp_stream)
                    comp_events[(stage_id, mb_idx, TaskType.FWD, bucket_id)] = evt

                case TaskType.BWD:
                    bucket_id = task.bucket_id
                    recv_key = (None, mb_idx)
                    if recv_key in self.recv_events:
                        self.bwd_comp_stream.wait_event(self.recv_events.pop(recv_key))
                    torch.cuda.nvtx.range_push(f"backward_s{stage_id}_b{bucket_id}_mb{mb_idx}")
                    self._backward_dag(stage_id, bucket_id, mb_idx, loss_fn=loss_fn)
                    torch.cuda.nvtx.range_pop()
                    evt = torch.cuda.Event()
                    evt.record(self.bwd_comp_stream)
                    comp_events[(stage_id, mb_idx, TaskType.BWD, bucket_id)] = evt

                case TaskType.BWD_I:
                    bucket_id = task.bucket_id
                    recv_key = (None, mb_idx)
                    if recv_key in self.recv_events:
                        self.bwd_comp_stream.wait_event(self.recv_events.pop(recv_key))
                    torch.cuda.nvtx.range_push(f"backward_input_stage_{stage_id}_b{bucket_id}_mb_{mb_idx}")
                    if len(self.bucket_fwd_fns.get(stage_id, [])) > 1:
                        self._backward_input_dag_bucket(stage_id, bucket_id, mb_idx, loss_fn=loss_fn)
                    else:
                        self._backward_input_dag(stage_id, mb_idx, loss_fn=loss_fn)
                    torch.cuda.nvtx.range_pop()
                    evt = torch.cuda.Event()
                    evt.record(self.bwd_comp_stream)
                    comp_events[(stage_id, mb_idx, TaskType.BWD_I, bucket_id)] = evt

                case TaskType.BWD_W:
                    torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
                    self._backward_weight_dag(stage_id, mb_idx, loss_fn=loss_fn)
                    torch.cuda.nvtx.range_pop()

                case TaskType.FWD_A2A:
                    bucket_id = task.bucket_id
                    torch.cuda.nvtx.range_push(f"fwd_a2a_s{stage_id}_b{bucket_id}_mb{mb_idx}")
                    self.a2a_stream.wait_event(
                        comp_events[(stage_id, mb_idx, TaskType.FWD, bucket_id)]
                    )
                    self._exec_fwd_a2a(stage_id, bucket_id, mb_idx)
                    torch.cuda.nvtx.range_pop()
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(self.a2a_stream)
                    self.a2a_events[(stage_id, mb_idx, TaskType.FWD_A2A, bucket_id)] = a2a_evt
                    self.fwd_comp_stream.wait_event(a2a_evt)

                case TaskType.BWD_A2A:
                    bucket_id = task.bucket_id
                    torch.cuda.nvtx.range_push(f"bwd_a2a_s{stage_id}_b{bucket_id}_mb{mb_idx}")
                    bwd_evt = (
                        comp_events.get((stage_id, mb_idx, TaskType.BWD, bucket_id + 1))
                        or comp_events.get((stage_id, mb_idx, TaskType.BWD_I, bucket_id + 1))
                    )
                    self.a2a_stream.wait_event(bwd_evt)
                    self._exec_bwd_a2a(stage_id, bucket_id, mb_idx)
                    torch.cuda.nvtx.range_pop()
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(self.a2a_stream)
                    self.a2a_events[(stage_id, mb_idx, TaskType.BWD_A2A, bucket_id)] = a2a_evt
                    self.bwd_comp_stream.wait_event(a2a_evt)

                case TaskType.ALL_REDUCE:
                    bucket_id = task.bucket_id
                    bwd_node = node.data_preds[0]
                    bwd_key = (
                        bwd_node.task.batches[0].stage_id,
                        bwd_node.task.batches[0].mb_idx,
                        bwd_node.task.type,
                        bwd_node.task.bucket_id,
                    )
                    torch.cuda.nvtx.range_push(f"all_reduce_s{stage_id}_b{bucket_id}")
                    self._exec_all_reduce(stage_id, bucket_id, comp_events[bwd_key])
                    torch.cuda.nvtx.range_pop()

                case TaskType.UPD:
                    torch.cuda.nvtx.range_push("update")
                    self._update_impl()
                    torch.cuda.nvtx.range_pop()

            # Decrement in-degree for all successors; execute in topological order.
            all_succs = list(node.data_succs)
            if node.temporal_succ is not None:
                all_succs.append(node.temporal_succ)
            for succ in all_succs:
                in_degree[id(succ)] -= 1
                if in_degree[id(succ)] == 0:
                    ready.append(succ)

    def _exec_send(
        self, stage_id: int, mb_idx: int, key, peer_pp_rank: int
    ) -> None:
        """Send the contents of send_buffer[(key, mb_idx)] to peer_pp_rank.

        key is (stage_id, None) for non-bucket compute dependencies and
        (stage_id, bucket_id) for bucket compute dependencies; None for BWD/BWD_I.

        The caller (run_dag) must have already made p2p_send_stream wait on the
        compute event before calling this method.
        """
        self.logger.debug(f"exec_send key {(key, mb_idx)} to peer pp rank {peer_pp_rank}")

        buf = self.send_buffer.pop((key, mb_idx))
        global_dst_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)

        if self.global_rank == global_dst_rank:
            # Co-located stages: write to recv_buffer under the peer's expected key.
            if key is not None:  # FWD send: peer receives into its stage's input slot
                recv_stage_id = stage_id + 1
                peer_recv_key = (recv_stage_id, 0)
            else:  # BWD send
                peer_recv_key = None
            self.recv_buffer[(peer_recv_key, mb_idx)] = buf
            return

        with torch.cuda.stream(self.p2p_send_stream):
            tensors = buf if isinstance(buf, (list, tuple)) else [buf]
            for tensor in tensors:
                dist.send(
                    tensor,
                    dst=global_dst_rank,
                    group=self.pp_groups[(self.global_rank, global_dst_rank)],
                )

    def _exec_recv(
        self, stage_id: int, mb_idx: int, key, peer_pp_rank: int
    ) -> None:
        """Receive data from peer_pp_rank into recv_buffer[(key, mb_idx)].

        key is (stage_id, None) for non-bucket compute dependencies and
        (stage_id, bucket_id) for bucket compute dependencies; None for BWD/BWD_I.

        Records a cuda.Event in recv_events[(key, mb_idx)] once the recv
        completes on p2p_recv_stream.
        """
        self.logger.debug(f"exec_recv key {(key, mb_idx)} from peer pp rank {peer_pp_rank}")

        buf_key = (key, mb_idx)
        global_src_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)

        if self.global_rank == global_src_rank:
            # Co-located: _exec_send already wrote to recv_buffer[(key, mb_idx)].
            return

        # Allocate receive buffer
        if key is not None:  # FWD recv
            buf = []
            for i in self.input_idxs[key]:
                shape, dtype, requires_grad = self.forward_input_meta[stage_id][i]
                buf.append(
                    torch.empty(shape, dtype=dtype, requires_grad=requires_grad, device=self.device)
                )
        else:
            # BWD / BWD_I: gradient shaped like the saved output activation.
            # In interleaved schedules the BWD recv may be issued before the
            # corresponding FWD has run, so fall back to the pre-computed shape.
            act_list = self.out_activation[stage_id].get(mb_idx)
            if act_list is not None:
                buf = [torch.empty_like(a) for a in act_list]
            else:
                shapes = self.output_activation_shape.get(stage_id, [])
                buf = [torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in shapes]

        with torch.cuda.stream(self.p2p_recv_stream):
            tensors = buf if isinstance(buf, list) else [buf]
            for tensor in tensors:
                dist.recv(
                    tensor,
                    src=global_src_rank,
                    group=self.pp_groups[(global_src_rank, self.global_rank)],
                )

        recv_event = torch.cuda.Event()
        recv_event.record(self.p2p_recv_stream)
        self.recv_events[buf_key] = recv_event
        self.recv_buffer[buf_key] = buf

    def _forward_dag(self, stage_id: int, bucket_id: int, mb_idx: int) -> None:
        """Forward step for DAG execution (unified for bucketed and non-bucketed stages).

        Non-bucketed stages are loaded as single-bucket stages (bucket_id=0 only).
        Bucketed stages have multiple bucket_ids dispatched individually by run_dag.
        """
        comp_stream = self.comp_stream
        bucket_fns = self.bucket_fwd_fns[stage_id]
        bucket_args = self.bucket_fwd_args[stage_id]
        n_buckets = len(bucket_fns)

        # --- First bucket of stage: load activation inputs ---
        if bucket_id == 0:
            if stage_id == 0:
                for i, inp in zip(self.input_idxs[(stage_id, 0)], self.inputs):
                    bucket_args[0][i] = inp
            else:
                recv_key = ((stage_id, 0), mb_idx)
                inputs_from_prev = self.recv_buffer.pop(recv_key)
                if not isinstance(inputs_from_prev, (list, tuple)):
                    inputs_from_prev = [inputs_from_prev]
                for i, tensor in zip(self.input_idxs[(stage_id, 0)], inputs_from_prev):
                    if isinstance(tensor, (list, tuple)):
                        tensor = tensor[0]
                    if (
                        self.stage_to_device[stage_id] == self.stage_to_device[stage_id - 1]
                        and tensor.requires_grad
                    ):
                        tensor = tensor.detach().requires_grad_(True)
                    bucket_args[0][i] = tensor

                inp_with_grad = [
                    bucket_args[0][i] for i in self.input_idxs[(stage_id, 0)]
                    if bucket_args[0][i] is not None and bucket_args[0][i].requires_grad
                ]
                self.inp_activation[stage_id][mb_idx] = inp_with_grad  # list

        # --- Run this bucket ---
        with torch.cuda.stream(comp_stream):
            output = bucket_fns[bucket_id](*bucket_args[bucket_id])

        # Clear activation input slots
        for i in self.input_idxs[(stage_id, bucket_id)]:
            bucket_args[bucket_id][i] = None

        out_list = list(output) if isinstance(output, tuple) else [output]

        # --- Store boundary or finalize ---
        if bucket_id < n_buckets - 1:
            possibly_detached = [
                t.detach().requires_grad_(True) if t.requires_grad else t
                for t in out_list
            ]
            self.bucket_buffer[(stage_id, mb_idx, bucket_id)] = (out_list, possibly_detached)
            # Skip direct feed at A2A boundaries — _exec_fwd_a2a will feed after applying A2A.
            stage_a2a = self.a2a_boundaries.get(stage_id, {})
            if bucket_id not in stage_a2a:
                for i, t in zip(self.input_idxs[(stage_id, bucket_id + 1)], possibly_detached):
                    bucket_args[bucket_id + 1][i] = t
        else:
            # Last bucket: save requires_grad outputs for backward
            out_with_grad = [t for t in out_list if isinstance(t, torch.Tensor) and t.requires_grad]
            self.out_activation[stage_id][mb_idx] = out_with_grad  # list
            if stage_id < self.num_stages - 1:
                self.send_buffer[((stage_id, bucket_id), mb_idx)] = output
                # Co-located next stage: no SEND/RECV DAG nodes are created by insert_p2p_ops
                # for same-rank edges, so populate recv_buffer directly.
                next_stage = stage_id + 1
                if (next_stage in self.stage_to_device and
                        self.stage_to_device[next_stage] == self.stage_to_device[stage_id]):
                    self.recv_buffer[((next_stage, 0), mb_idx)] = output

    def _backward_dag(self, stage_id: int, bucket_id: int, mb_idx: int, *, loss_fn=None) -> None:
        """Backward step for DAG execution (unified for bucketed and non-bucketed stages).

        The backward chain runs in reverse bucket order: the highest bucket_id
        runs first (receives upstream gradient or computes loss) and bucket 0
        runs last (sends input gradient to the previous stage).
        """
        comp_stream = self.comp_stream
        n_buckets = len(self.bucket_fwd_fns[stage_id])

        if bucket_id == n_buckets - 1:
            # First to backward: receive upstream gradient or compute loss
            out_with_grad = self.out_activation[stage_id][mb_idx]  # list
            if stage_id < self.num_stages - 1:
                recv_key = (None, mb_idx)
                upstream_grads = self.recv_buffer.pop(recv_key)
                if not isinstance(upstream_grads, (list, tuple)):
                    upstream_grads = [upstream_grads]
                with torch.cuda.stream(comp_stream):
                    torch.autograd.backward(out_with_grad, upstream_grads)
            else:
                assert loss_fn is not None
                with torch.cuda.stream(comp_stream):
                    loss = loss_fn(out_with_grad[0], self.labels)
                    loss.backward()
            self.out_activation[stage_id][mb_idx] = None
        else:
            # Middle / earlier bucket: propagate backward through the boundary
            pre_detach_outs, detached_outs = self.bucket_buffer.pop((stage_id, mb_idx, bucket_id))
            outputs_bwd = [p for p, d in zip(pre_detach_outs, detached_outs) if d.requires_grad]
            grads_bwd = [d.grad for p, d in zip(pre_detach_outs, detached_outs) if d.requires_grad]
            assert all(g is not None for g in grads_bwd), (
                f"Stage {stage_id} bucket {bucket_id}: detached boundary has no .grad"
            )
            with torch.cuda.stream(comp_stream):
                torch.autograd.backward(outputs_bwd, grads_bwd)

        # Bucket 0 is last to run: send input gradients to previous stage
        if bucket_id == 0 and stage_id > 0:
            inp_list = self.inp_activation[stage_id][mb_idx]  # list
            output_grads = [t.grad for t in inp_list if t.grad is not None]
            self.send_buffer[(None, mb_idx)] = output_grads
            # Co-located prev stage: no BWD SEND/RECV DAG nodes for same-rank edges.
            prev_stage = stage_id - 1
            if (prev_stage in self.stage_to_device and
                    self.stage_to_device[prev_stage] == self.stage_to_device[stage_id]):
                self.recv_buffer[(None, mb_idx)] = output_grads
            self.inp_activation[stage_id][mb_idx] = None

    def _exec_fwd_a2a(self, stage_id: int, boundary_bucket_id: int, mb_idx: int) -> None:
        """Apply a forward all-to-all at the given A2A boundary.

        Called after FWD(bucket_id=boundary_bucket_id) completes.  Reads the
        boundary tensor from bucket_buffer, applies dist.all_to_all_single
        on the a2a_stream, replaces the entry in detached_outs with the
        communicated tensor, then feeds all detached_outs to the next bucket.
        """
        tensor_idx = self.a2a_boundaries[stage_id][boundary_bucket_id]
        pre_detach_outs, detached_outs = self.bucket_buffer[(stage_id, mb_idx, boundary_bucket_id)]

        x_detached = detached_outs[tensor_idx]
        output_buf = torch.empty_like(x_detached)
        with torch.cuda.stream(self.a2a_stream):
            dist.all_to_all_single(output_buf, x_detached, group=self.ep_group)
        x_a2a = output_buf.requires_grad_(True)

        # Store for BWD_A2A to reverse
        self.a2a_buffer[(stage_id, mb_idx, boundary_bucket_id)] = (x_detached, x_a2a)

        # Replace entry so that _backward_dag will read x_a2a.grad (set by seg+1 backward)
        detached_outs[tensor_idx] = x_a2a

        # Feed all detached_outs to next bucket's input slots
        next_bucket_id = boundary_bucket_id + 1
        bucket_args = self.bucket_fwd_args[stage_id]
        for i, t in zip(self.input_idxs[(stage_id, next_bucket_id)], detached_outs):
            bucket_args[next_bucket_id][i] = t

    def _exec_bwd_a2a(self, stage_id: int, boundary_bucket_id: int, mb_idx: int) -> None:
        """Apply the reverse all-to-all for the backward pass at an A2A boundary.

        Called after BWD(bucket_id=boundary_bucket_id+1) completes (which has set
        x_a2a.grad).  Applies the reverse A2A to obtain the gradient in the
        pre-A2A space, sets it on x_detached.grad, and restores x_detached in
        detached_outs so that _backward_dag for boundary_bucket_id uses the
        correct gradient.
        """
        tensor_idx = self.a2a_boundaries[stage_id][boundary_bucket_id]
        x_detached, x_a2a = self.a2a_buffer.pop((stage_id, mb_idx, boundary_bucket_id))

        grad_a2a_out = x_a2a.grad
        assert grad_a2a_out is not None, (
            f"Stage {stage_id} A2A boundary {boundary_bucket_id} mb {mb_idx}: "
            f"x_a2a.grad is None after backward through next segment"
        )
        if not grad_a2a_out.is_contiguous():
            grad_a2a_out = grad_a2a_out.contiguous()
        reversed_grad = torch.empty_like(grad_a2a_out)
        with torch.cuda.stream(self.a2a_stream):
            dist.all_to_all_single(reversed_grad, grad_a2a_out, group=self.ep_group)
        x_detached.grad = reversed_grad

        # Restore x_detached in bucket_buffer so _backward_dag reads x_detached.grad
        _, detached_outs = self.bucket_buffer[(stage_id, mb_idx, boundary_bucket_id)]
        detached_outs[tensor_idx] = x_detached

    def _backward_input_dag(self, stage_id: int, mb_idx: int, *, loss_fn=None) -> None:
        """Backward-input pass for DAG execution (ZeroBubble split backward).

        Reads upstream gradient from recv_buffer and writes input gradient to
        send_buffer.  Does NOT call comp_stream.synchronize().
        """
        comp_stream = self.comp_stream
        _out = self.out_activation[stage_id][mb_idx]
        out_activation = _out[0] if isinstance(_out, list) else _out

        activation_or_loss = None
        upstream_grad = None
        if stage_id < self.num_stages - 1:
            recv_key = (None, mb_idx)
            upstream_grad = self.recv_buffer.pop(recv_key)
            activation_or_loss = out_activation
        else:
            assert loss_fn is not None
            labels = self.labels
            with torch.cuda.stream(comp_stream):
                loss = loss_fn(out_activation, labels)
            activation_or_loss = loss
            upstream_grad = torch.ones_like(loss)

        if stage_id == 0:
            self.upstream_grad_cache[stage_id][mb_idx] = upstream_grad
            return

        _inp = self.inp_activation[stage_id][mb_idx]
        stage_input = _inp[0] if isinstance(_inp, list) else _inp
        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]

        output_nodes = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [activation_or_loss]) if n is not None]
        input_nodes  = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [stage_input]) if n is not None]
        param_nodes  = [n for n in (_get_grad_fn_or_grad_acc(p) for p in stage_params) if n is not None]

        reverse_edges = construct_reverse_graph(output_nodes)
        param_groups = get_param_groups(input_nodes, param_nodes, reverse_edges)

        handles = []
        for pg in param_groups:
            intermediates = pg["intermediates"]
            if not intermediates:
                continue
            pg["grads"] = [None] * len(intermediates)
            for i, intermediate_node in enumerate(intermediates):
                def make_hook(group: Dict[str, Any], idx: int):
                    def hook(grad_inputs):
                        group["grads"][idx] = grad_inputs
                    return hook
                handles.append(intermediate_node.register_prehook(make_hook(pg, i)))

        with torch.cuda.stream(comp_stream):
            gx = torch.autograd.grad(
                outputs=activation_or_loss,
                inputs=stage_input,
                grad_outputs=upstream_grad,
                retain_graph=True,
                allow_unused=True,
            )

        gx = gx[0]
        if gx is not None and stage_input.requires_grad:
            if stage_input.grad is None:
                stage_input.grad = gx
            else:
                stage_input.grad.add_(gx)

        if not isinstance(activation_or_loss, list):
            activation_or_loss = [activation_or_loss]
        for t in activation_or_loss:
            t.detach_()

        for h in handles:
            h.remove()

        # Write input gradient to send_buffer for the upstream stage to recv
        # stage_input is the unwrapped tensor set above; read .grad before deleting.
        output_grad = stage_input.grad
        if output_grad is not None:
            self.send_buffer[(None, mb_idx)] = output_grad
            # Co-located prev stage: no BWD_I SEND/RECV DAG nodes for same-rank edges.
            prev_stage = stage_id - 1
            if (prev_stage in self.stage_to_device and
                    self.stage_to_device[prev_stage] == self.stage_to_device[stage_id]):
                self.recv_buffer[(None, mb_idx)] = output_grad
        else:
            self.inp_activation[stage_id][mb_idx] = None

        self.bw_param_groups[stage_id][mb_idx] = param_groups
        del activation_or_loss
        del stage_input

    def _backward_input_dag_bucket(self, stage_id: int, bucket_id: int, mb_idx: int, *, loss_fn=None) -> None:
        """Per-bucket backward-input pass for ZeroBubble with bucketed / A2A stages.

        Mirrors _backward_dag but uses autograd.grad(..., retain_graph=True) instead of
        backward(), so weight gradients are NOT accumulated during BWD_I (they are
        deferred to BWD_W via bw_param_groups).

        Must be called in reverse bucket order: n_buckets-1, …, 1, 0.
        BWD_A2A must run between consecutive BWD_I calls at A2A boundaries.
        """
        comp_stream = self.comp_stream
        n_buckets = len(self.bucket_fwd_fns[stage_id])
        a2a_bnd = self.a2a_boundaries.get(stage_id, {})

        # ---- Step 1: outputs and upstream grads for this bucket ----
        if bucket_id == n_buckets - 1:
            _out = self.out_activation[stage_id][mb_idx]
            out_list = _out if isinstance(_out, list) else [_out]
            if stage_id < self.num_stages - 1:
                upstream_grads_raw = self.recv_buffer.pop((None, mb_idx))
                if not isinstance(upstream_grads_raw, (list, tuple)):
                    upstream_grads_raw = [upstream_grads_raw]
                outputs = out_list
                upstream_grads = list(upstream_grads_raw)
            else:
                assert loss_fn is not None
                with torch.cuda.stream(comp_stream):
                    loss = loss_fn(out_list[0], self.labels)
                outputs = [loss]
                upstream_grads = [torch.ones_like(loss)]
        else:
            # BWD_A2A has restored x_detached (with .grad set) in detached_outs
            pre_detach_outs, detached_outs = self.bucket_buffer[(stage_id, mb_idx, bucket_id)]
            outputs = [p for p, d in zip(pre_detach_outs, detached_outs) if d.requires_grad]
            upstream_grads = [d.grad for p, d in zip(pre_detach_outs, detached_outs) if d.requires_grad]
            assert all(g is not None for g in upstream_grads), (
                f"Stage {stage_id} BWD_I bucket {bucket_id} mb {mb_idx}: boundary .grad is None"
            )

        # Stage 0 special case: no prev stage to send grad to.  Cache upstream_grad for
        # BWD_W and return early (BWD_W will use out_activation directly for stage 0).
        if stage_id == 0:
            if bucket_id == n_buckets - 1:
                ug = upstream_grads[0] if len(upstream_grads) == 1 else upstream_grads
                self.upstream_grad_cache[stage_id][mb_idx] = ug
            return

        # ---- Step 2: activation inputs (what we compute grad w.r.t.) ----
        if bucket_id > 0:
            prev_bkt = bucket_id - 1
            _, detached_outs_prev = self.bucket_buffer[(stage_id, mb_idx, prev_bkt)]
            if prev_bkt in a2a_bnd:
                tensor_idx = a2a_bnd[prev_bkt]
                activation_inputs = [detached_outs_prev[tensor_idx]]  # x_a2a leaf
            else:
                activation_inputs = [d for d in detached_outs_prev if d.requires_grad]
        else:
            # Bucket 0: grad to stage_input
            _inp = self.inp_activation[stage_id][mb_idx]
            stage_input = _inp[0] if isinstance(_inp, list) else _inp
            activation_inputs = [stage_input] if (stage_input is not None and stage_input.requires_grad) else []

        # ---- Step 3: ZeroBubble hooks so BWD_W can compute weight grads ----
        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]
        output_nodes = [n for n in (_get_grad_fn_or_grad_acc(t) for t in outputs) if n is not None]
        input_nodes = [n for n in (_get_grad_fn_or_grad_acc(t) for t in activation_inputs) if n is not None]
        param_nodes = [n for n in (_get_grad_fn_or_grad_acc(p) for p in stage_params) if n is not None]
        reverse_edges = construct_reverse_graph(output_nodes)
        param_groups = get_param_groups(input_nodes, param_nodes, reverse_edges)

        handles = []
        for pg in param_groups:
            intermediates = pg["intermediates"]
            if not intermediates:
                continue
            pg["grads"] = [None] * len(intermediates)
            for i, intermediate_node in enumerate(intermediates):
                def make_hook(group, idx):
                    def hook(grad_inputs):
                        group["grads"][idx] = grad_inputs
                    return hook
                handles.append(intermediate_node.register_prehook(make_hook(pg, i)))

        # ---- Step 4: compute grad w.r.t. activation inputs only ----
        if activation_inputs:
            with torch.cuda.stream(comp_stream):
                grads = torch.autograd.grad(
                    outputs=outputs,
                    inputs=activation_inputs,
                    grad_outputs=upstream_grads,
                    retain_graph=True,
                    allow_unused=True,
                )
            for inp, g in zip(activation_inputs, grads):
                if g is not None:
                    if inp.grad is None:
                        inp.grad = g
                    else:
                        inp.grad.add_(g)

        for h in handles:
            h.remove()

        # ---- Step 5: accumulate bw_param_groups for BWD_W ----
        existing = self.bw_param_groups[stage_id].get(mb_idx)
        if existing is None:
            self.bw_param_groups[stage_id][mb_idx] = param_groups
        else:
            existing.extend(param_groups)

        # ---- Step 6: bucket 0 sends input grad to prev stage ----
        if bucket_id == 0:
            _inp = self.inp_activation[stage_id][mb_idx]
            stage_input = _inp[0] if isinstance(_inp, list) else _inp
            output_grad = stage_input.grad if stage_input is not None else None
            if output_grad is not None and stage_id > 0:
                self.send_buffer[(None, mb_idx)] = output_grad
                prev_stage = stage_id - 1
                if (prev_stage in self.stage_to_device and
                        self.stage_to_device[prev_stage] == self.stage_to_device[stage_id]):
                    self.recv_buffer[(None, mb_idx)] = output_grad
            self.inp_activation[stage_id][mb_idx] = None

        # Free output reference after processing the first-to-bwd bucket
        if bucket_id == n_buckets - 1:
            for t in outputs:
                t.detach_()
            self.out_activation[stage_id][mb_idx] = None

    def _exec_all_reduce(self, stage_id: int, bucket_id: int, bwd_event: torch.cuda.Event) -> None:
        """Launch an all-reduce for a stage/bucket's gradients.

        All-reduces the pre-allocated flat_grads tensor for the given
        (stage_id, bucket_id).

        comm_stream waits on *bwd_event* first (ensuring all gradient
        accumulation is complete) then records an event so _update_impl can
        wait on it before stepping the optimizer.
        """
        lookup_key = (stage_id, bucket_id)
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(bwd_event)
            flat_grads = self.bucket_flat_grads.get(lookup_key)
            if flat_grads is not None:
                dist.all_reduce(flat_grads, group=self.dp_group)
            evt = torch.cuda.Event()
            evt.record(self.comm_stream)
        self.ar_events[(stage_id, bucket_id)] = evt

    def _backward_weight_dag(self, stage_id: int, mb_idx: int, *, loss_fn=None) -> None:
        """Backward-weight pass for DAG execution (ZeroBubble split backward).

        Does NOT call comp_stream.synchronize().
        """
        comp_stream = self.comp_stream
        stage_params = [self.forward_args[stage_id][i] for i in self.param_idxs[stage_id]]
        updated_params: dict[int, torch.nn.Parameter] = {}

        if stage_id == 0:
            upstream_grad = self.upstream_grad_cache[stage_id][mb_idx]
            _out = self.out_activation[stage_id][mb_idx]
            out_activation = _out[0] if isinstance(_out, list) else _out
            if stage_id < self.num_stages - 1:
                with torch.cuda.stream(comp_stream):
                    gparams = torch.autograd.grad(
                        outputs=out_activation,
                        inputs=stage_params,
                        grad_outputs=upstream_grad,
                        retain_graph=False,
                    )
            else:
                assert loss_fn is not None
                labels = self.labels
                with torch.cuda.stream(comp_stream):
                    loss = loss_fn(out_activation, labels)
                    gparams = torch.autograd.grad(
                        outputs=loss,
                        inputs=stage_params,
                        retain_graph=False,
                    )

            assert len(gparams) == len(stage_params)
            for p, pg in zip(stage_params, gparams):
                if pg is None:
                    continue
                if p.grad is None:
                    p.grad = pg.clone()
                else:
                    p.grad.add_(pg)
                updated_params[id(p)] = p
        else:
            grad_acc_to_weight: Dict[Node, Tuple[Parameter, int]] = {}
            for param in stage_params:
                node = _get_grad_fn_or_grad_acc(param)
                grad_acc_to_weight[node] = param

            param_groups = self.bw_param_groups[stage_id][mb_idx]

            for pg in param_groups:
                intermediates: List[Node] = pg.get("intermediates", [])
                intermediate_grads = pg.get("grads", None)

                if not intermediates or intermediate_grads is None:
                    continue

                intermediate_edges: List[GradientEdge] = []
                intermediate_edge_grads: List[torch.Tensor] = []

                for intermediate_node, grad_inputs in zip(intermediates, intermediate_grads):
                    if grad_inputs is None:
                        continue
                    gs = [x for x in grad_inputs if x is not None]
                    if not gs:
                        continue
                    summed = sum(gs)
                    intermediate_edges.append(GradientEdge(intermediate_node, 0))
                    intermediate_edge_grads.append(summed)

                del pg["intermediates"]

                if not intermediate_edges:
                    continue

                mapped_param_nodes = [p for p in pg["params"] if p in grad_acc_to_weight]
                if not mapped_param_nodes:
                    continue

                weight_edges = tuple(GradientEdge(p, 0) for p in mapped_param_nodes)

                with torch.cuda.stream(comp_stream):
                    gparams = torch.autograd.grad(
                        outputs=intermediate_edges,
                        inputs=weight_edges,
                        grad_outputs=intermediate_edge_grads,
                        retain_graph=False,
                    )

                del pg["grads"]

                assert len(gparams) == len(mapped_param_nodes)
                for param_node, dw in zip(mapped_param_nodes, gparams):
                    if dw is None:
                        continue
                    weight = grad_acc_to_weight[param_node]
                    if weight.grad is None:
                        weight.grad = dw
                    else:
                        weight.grad.add_(dw)
                    updated_params[id(weight)] = weight

        for p in updated_params.values():
            self._maybe_trigger_grad_allreduce(stage_id, p)

        self.bw_grad_cache[stage_id][mb_idx] = None
        self.upstream_grad_cache[stage_id][mb_idx] = None
        self.bw_param_groups[stage_id][mb_idx] = None
        self.out_activation[stage_id][mb_idx] = None
