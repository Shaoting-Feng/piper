import ray
import torch
import logging
import os
import time
from typing import Any, Dict, List, Set, Tuple
import gc
from torch._guards import CompileId
from torch.nn import Parameter
from torch.autograd.graph import GradientEdge, Node
import torch.distributed as dist
from collections import defaultdict

from .piper_utils import (
    _deserialize_graphmodule,
    create_logger,
    LOG_LEVEL,
    piper_metadata,
)
from .backward_utils import get_param_groups, construct_reverse_graph, _get_grad_fn_or_grad_acc

CLEANUP_MEMORY = False

logger = create_logger("piper_actor", LOG_LEVEL)


def _get_rank(pp_rank, dp_rank, pp_degree):
    return pp_rank + dp_rank * pp_degree


def _create_actors(
    num_actors,
    optim_class,
    num_mbs,
    num_stages,
    p2p_schedules,
    naive_gradient_sync=False,
    profile=False,
    mode="sequential",
):
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    world_size = int(os.environ["PIPER_WORLD_SIZE"])
    dp_degree = int(os.environ["PIPER_DP_DEGREE"])
    pp_degree = int(os.environ["PIPER_PP_DEGREE"])

    from .piper_utils import piper_metadata

    for pp_rank in range(num_actors):
        global_rank = _get_rank(pp_rank, dp_rank, pp_degree)
        p2p_schedule = p2p_schedules[pp_rank]
        actor = PiperActor.options(num_gpus=1).remote(
            pp_rank,
            optim_class,
            world_size,
            num_mbs,
            num_stages,
            p2p_schedule,
            naive_gradient_sync,
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            pp_degree=pp_degree,
            profile=profile,
            mode=mode,
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
        p2p_schedule,
        naive_gradient_sync=False,
        dp_rank=0,
        dp_degree=1,
        pp_degree=1,
        profile=False,
        mode="sequential",
    ):
        self.logger = create_logger("piper_actor", LOG_LEVEL)
        self.profile = profile
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
        # list of (src_stage, dst_stage, mb_idx, is_sender)
        self.p2p_schedule = p2p_schedule
        self.next_p2p_idx = 0
        # set of (src_stage, dst_stage, mb_idx, is_sender) for completed p2p ops
        self.executed_p2ps = set()
        self.dp_group = None
        self.pp_group = None
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
        self.overlapped_comp_stream = torch.cuda.Stream()
        self.overlapped_p2p_stream = torch.cuda.Stream()
        if mode == "naive":
            self.per_mb_streams = [torch.cuda.Stream() for _ in range(2)]
        self.n_a2a_ops = dict()
        self.overlap_a2a_ops = False

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
        # map stage id -> optimizer for the fx.Graph
        self.optims = dict()
        # map stage id -> mb_idx -> previous activation (if this stage is not first)
        self.inp_activation = defaultdict(dict)
        # map stage id -> mb_idx -> current activation
        self.out_activation = defaultdict(dict)
        # map (src_stage, dst_stage, mb_idx, is_sender) -> tensor for p2p communication
        self.p2p_cache = dict()
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

        # map stage id -> mb_idx -> parameter groups for backward pass
        self.bw_param_groups = defaultdict(dict)
        # map stage id -> mb_idx -> gradient cache for backward pass
        self.bw_grad_cache = defaultdict(dict)
        # map stage id -> mb_idx -> upstream gradient cache for backward pass
        self.upstream_grad_cache = defaultdict(dict)

        from .piper_utils import piper_metadata

        piper_metadata.actor_self = self

    def reset_p2p_states(self):
        self.next_p2p_idx = 0
        self.executed_p2ps = set()
        self.p2p_cache = dict()

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
        return torch.cuda.max_memory_allocated() / (1024**3)

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

    def _join_process_groups(self):
        master_addr = os.environ.get("PIPER_MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("PIPER_MASTER_PORT", "10000")
        init_method = f"tcp://{master_addr}:{master_port}"

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
            process_group = dist.new_group(ranks=group_ranks, backend="nccl")
            if self.global_rank % num_dp_groups == dp_group_id:
                self.dp_group = process_group
                self.logger.debug(
                    f"Global rank {self.global_rank} joined its dp group {dp_group_id} along with ranks {group_ranks}"
                )

    def _join_pp_process_group(self):
        num_pp_groups = self.world_size // self.pp_degree
        for pp_group_id in range(num_pp_groups):
            group_ranks = [
                (pp_group_id * self.pp_degree + i) for i in range(self.pp_degree)
            ]
            process_group = dist.new_group(ranks=group_ranks, backend="nccl")
            if self.global_rank // self.pp_degree == pp_group_id:
                self.pp_group = process_group
                self.logger.debug(f"Global rank {self.global_rank} joined its pp group {pp_group_id} along with ranks {group_ranks}")

    def shutdown(self):
        dist.destroy_process_group()

    def _prepare_dp_comm_ops(self, stage_id):
        def hook_maker(tensor_id):
            def post_backward_hook(grad):
                self.comm_op_status[stage_id][tensor_id] += 1
                self.logger.debug(
                    f"Updating status on actor: {self.global_rank}, tensor={tensor_id}, status={self.comm_op_status[stage_id][tensor_id]}/{self.num_mbs}"
                )
                if self.comm_op_status[stage_id][tensor_id] == self.num_mbs:
                    with torch.cuda.stream(self.comm_stream):
                        handle = dist.all_reduce(
                            grad,
                            op=dist.ReduceOp.AVG,
                            group=self.dp_group,
                            async_op=True,
                        )
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
                        self.comm_op_handles[stage_id][tid].wait()
                        done = True

    def _load_stage(
        self, stage_id: int, gm_data, forward_args, input_idxs, param_idxs, n_a2a_ops
    ):
        self.logger.debug(f"Loading stage {stage_id} graph on actor {self.global_rank}")

        # compile the graph with the given graphargs
        gm = _deserialize_graphmodule(gm_data)

        # Store GraphModule reference
        self.graph_modules[stage_id] = gm
        self.forward_fns[stage_id] = gm.forward

        # initialize A2A states
        self.n_a2a_ops[stage_id] = n_a2a_ops

        # place parameters on the device
        def move_to_device(idx, arg):
            if arg.requires_grad:
                return arg.to(self.device).detach().requires_grad_(True)
            else:
                return arg.to(self.device)
        forward_args = list(map(move_to_device, range(len(forward_args)), forward_args))

        # save parameters
        self.forward_args[stage_id] = forward_args
        self.stage_id = stage_id
        self.input_idxs[stage_id] = input_idxs
        self.param_idxs[stage_id] = param_idxs

        for i in self.input_idxs[stage_id]:
            self.forward_input_meta[stage_id][i] = (
                forward_args[i].shape,
                forward_args[i].dtype,
                forward_args[i].requires_grad,
            )
            self.forward_args[stage_id][i] = None

        # prepare tensors with DP comm ops
        if not self.naive_gradient_sync and self.dp_degree > 1:
            self._prepare_dp_comm_ops(stage_id)

        # add the parameters to the optimizer for this stage
        params = [param for param in forward_args if param is not None and param.requires_grad]
        if stage_id not in self.optims:
            self.optims[stage_id] = self.optim_class(params)
        else:
            self.optims[stage_id].add_param_group({"params": params})

        del gm_data

    def _exec_p2p_op(
        self, src_stage: int, dst_stage: int, mb_idx: int, is_sender: bool, dep, p2p_stream=None
    ):
        p2p_op = (src_stage, dst_stage, mb_idx, is_sender)
        if p2p_op in self.executed_p2ps:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_stream

        op_idx = self.next_p2p_idx
        while op_idx < len(self.p2p_schedule):
            op = self.p2p_schedule[op_idx]
            if op == p2p_op:
                break
            op_idx += 1
        assert op_idx < len(
            self.p2p_schedule
        ), f"P2P op {p2p_op} not found in schedule for actor {self.global_rank}"

        # Execute everything before and including the given p2p op
        for idx in range(self.next_p2p_idx, op_idx + 1):
            op = self.p2p_schedule[idx]
            src_stage, dst_stage, mb_idx, is_sender = op
            is_fwd = src_stage == dst_stage - 1
            is_bwd = src_stage == dst_stage + 1
            assert is_fwd or is_bwd
            is_recver = not is_sender

            op_name = "P2P(unknown)"
            if is_fwd and is_sender:
                self._exec_fwd_send(src_stage, mb_idx, p2p_stream=p2p_stream)
                op_name = (
                    f"P2P(fwd_send, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_fwd and is_recver:
                self._exec_fwd_recv(dst_stage, mb_idx, p2p_stream=p2p_stream)
                op_name = (
                    f"P2P(fwd_recv, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_bwd and is_sender:
                self._exec_bwd_send(src_stage, mb_idx, p2p_stream=p2p_stream)
                op_name = (
                    f"P2P(bwd_send, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_bwd and is_recver:
                self._exec_bwd_recv(dst_stage, mb_idx, p2p_stream=p2p_stream)
                op_name = (
                    f"P2P(bwd_recv, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            else:
                raise ValueError(f"Invalid p2p op: {op}")

            self.executed_p2ps.add(op)
            logger.debug(f"Executed {op_name} on actor {self.global_rank}")

        self.next_p2p_idx = op_idx + 1
        
    def _exec_p2p_op(
        self, src_stage: int, dst_stage: int, mb_idx: int, is_sender: bool, dep, p2p_stream=None
    ):
        p2p_op = (src_stage, dst_stage, mb_idx, is_sender)
        if p2p_op in self.executed_p2ps:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_stream

        op_idx = self.next_p2p_idx
        while op_idx < len(self.p2p_schedule):
            op = self.p2p_schedule[op_idx]
            if op == p2p_op:
                break
            op_idx += 1
        assert op_idx < len(
            self.p2p_schedule
        ), f"P2P op {p2p_op} not found in schedule for actor {self.global_rank}"

        # Execute everything before and including the given p2p op
        for idx in range(self.next_p2p_idx, op_idx + 1):
            op = self.p2p_schedule[idx]
            src_stage, dst_stage, mb_idx, is_sender = op
            is_fwd = src_stage == dst_stage - 1
            is_bwd = src_stage == dst_stage + 1
            assert is_fwd or is_bwd
            is_recver = not is_sender

            op_name = "P2P(unknown)"
            if is_fwd and is_sender:
                self._exec_fwd_send(src_stage, mb_idx, p2p_stream=p2p_stream)
                op_name = (
                    f"P2P(fwd_send, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_fwd and is_recver:
                self._exec_fwd_recv(dst_stage, mb_idx, p2p_stream=p2p_stream)
                op_name = (
                    f"P2P(fwd_recv, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_bwd and is_sender:
                self._exec_bwd_send(src_stage, mb_idx, p2p_stream=p2p_stream)
                op_name = (
                    f"P2P(bwd_send, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            elif is_bwd and is_recver:
                self._exec_bwd_recv(dst_stage, mb_idx, p2p_stream=p2p_stream)
                op_name = (
                    f"P2P(bwd_recv, stage {src_stage} -> {dst_stage}, mb {mb_idx})"
                )
            else:
                raise ValueError(f"Invalid p2p op: {op}")

            self.executed_p2ps.add(op)
            logger.debug(f"Executed {op_name} on actor {self.global_rank}")

        self.next_p2p_idx = op_idx + 1

    def _exec_fwd_recv(self, stage_id: int, mb_idx: int, p2p_stream=None, comp_stream=None):
        if stage_id == 0:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_stream
        if comp_stream is None:
            comp_stream = self.comp_stream

        p2p_op = (stage_id - 1, stage_id, mb_idx, False)
        assert p2p_op not in self.p2p_cache

        inputs_to_recv = []
        for i in self.input_idxs[stage_id]:
            shape, dtype, requires_grad = self.forward_input_meta[stage_id][i]
            inputs_to_recv.append(
                torch.empty(
                    shape, dtype=dtype, requires_grad=requires_grad, device=self.device
                )
            )

        # For non-first stages, receive input tensors from the previous stage
        pp_rank = piper_metadata.stage_to_device[stage_id - 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        if self.global_rank == global_src_rank:
            for i in self.input_idxs[stage_id]:
                inputs_to_recv[i] = self.p2p_cache.pop((stage_id-1, stage_id, mb_idx, True))
        else:
            self.logger.debug(
                f"Dispatch fwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id-1} -> {stage_id}, mb {mb_idx})"
            )
            self._start_timing(p2p_stream, "fwd_p2p_recv")
            with torch.cuda.stream(p2p_stream):
                for i in self.input_idxs[stage_id]:
                    dist.recv(
                        inputs_to_recv[i],
                        src=global_src_rank,
                        group=self.pp_group,
                    )
            # Ensure the default stream only consumes tensors after recv completes.
            comp_stream.wait_stream(p2p_stream)
            torch.cuda.synchronize()
            self._stop_timing(p2p_stream, "fwd_p2p_recv")
            self.logger.debug(
                f"Completed fwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id-1} -> {stage_id}, mb {mb_idx})"
            )

        self.p2p_cache[p2p_op] = inputs_to_recv

    def _exec_fwd_send(self, stage_id: int, mb_idx: int, p2p_stream=None, comp_stream=None):
        if stage_id == self.num_stages - 1:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_stream
        if comp_stream is None:
            comp_stream = self.comp_stream

        p2p_op = (stage_id, stage_id + 1, mb_idx, True)
        output = self.p2p_cache.pop(p2p_op)
        # For non-final stages, send output tensors to the next stage
        pp_rank = piper_metadata.stage_to_device[stage_id + 1]
        global_dst_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        if self.global_rank == global_dst_rank:
            self.p2p_cache[p2p_op] = output
        else:
            # Ensure send sees the latest writes from the default stream.
            self.logger.debug(
                f"Dispatch fwd p2p send on {self.global_rank} to {global_dst_rank}, op: ({stage_id} -> {stage_id+1}, mb {mb_idx})"
            )
            self._start_timing(p2p_stream, "fwd_p2p_send")
            p2p_stream.wait_stream(comp_stream)
            with torch.cuda.stream(p2p_stream):
                for i in range(len(output)):
                    dist.send(
                        output[i],
                        dst=global_dst_rank,
                        group=self.pp_group,
                    )
            self._stop_timing(p2p_stream, "fwd_p2p_send")
            self.logger.debug(
                f"Completed fwd p2p send on {self.global_rank} to {global_dst_rank}, op: ({stage_id} -> {stage_id+1}, mb {mb_idx})"
            )

    def _exec_bwd_recv(self, stage_id: int, mb_idx: int, p2p_stream=None, comp_stream=None):
        if stage_id >= self.num_stages - 1:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_stream
        if comp_stream is None:
            comp_stream = self.comp_stream

        out_activation = self.out_activation[stage_id][mb_idx]
        # For non-final stages, recieve input gradients from the subsequent backward pass
        input_grad = torch.empty_like(out_activation)
        pp_rank = piper_metadata.stage_to_device[stage_id + 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        p2p_op = (stage_id + 1, stage_id, mb_idx, False)

        if self.global_rank == global_src_rank:
            input_grad = self.p2p_cache.pop((stage_id + 1, stage_id, mb_idx, True))
        else:
            self.logger.debug(
                f"Dispatch bwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id+1} -> {stage_id}, mb {mb_idx})"
            )
            self._start_timing(p2p_stream, "bwd_p2p_recv")
            with torch.cuda.stream(p2p_stream):
                dist.recv(
                    input_grad, src=global_src_rank, group=self.pp_group
                )
            # Ensure the default stream only consumes gradients after recv completes.
            comp_stream.wait_stream(p2p_stream)
            torch.cuda.synchronize()
            self._stop_timing(p2p_stream, "bwd_p2p_recv")
            self.logger.debug(
                f"Completed bwd p2p recv on {self.global_rank} from {global_src_rank}, op: ({stage_id+1} -> {stage_id}, mb {mb_idx})"
            )
        
        self.p2p_cache[p2p_op] = input_grad

    def _exec_bwd_send(self, stage_id: int, mb_idx: int, p2p_stream=None, comp_stream=None):
        if stage_id <= 0:
            return

        if p2p_stream is None:
            p2p_stream = self.p2p_stream
        if comp_stream is None:
            comp_stream = self.comp_stream

        # For non-first stages, send output gradients to the previous backward stage
        output_grad = self.inp_activation[stage_id][mb_idx].grad
        if output_grad is None:
            self.logger.warning(f"No output gradient found for stage {stage_id} mb {mb_idx} on actor {self.global_rank}")
            assert False
        pp_rank = piper_metadata.stage_to_device[stage_id - 1]
        global_src_rank = _get_rank(pp_rank, self.dp_rank, self.pp_degree)

        if self.global_rank == global_src_rank:
            p2p_op = (stage_id, stage_id - 1, mb_idx, True)
            self.p2p_cache[p2p_op] = output_grad
        else:
            self.logger.debug(
                f"Dispatch bwd p2p send on {self.global_rank} to {global_src_rank}, op: ({stage_id} -> {stage_id-1}, mb {mb_idx})"
            )
            self._start_timing(p2p_stream, "bwd_p2p_send")
            # Ensure send sees the latest gradient writes from the default stream.
            p2p_stream.wait_stream(comp_stream)
            with torch.cuda.stream(p2p_stream):
                dist.send(
                    output_grad, dst=global_src_rank, group=self.pp_group
                )
            self._stop_timing(p2p_stream, "bwd_p2p_send")
            self.logger.debug(
                f"Completed bwd p2p send on {self.global_rank} to {global_src_rank}, op: ({stage_id} -> {stage_id-1}, mb {mb_idx})"
            )

        self.inp_activation[stage_id][mb_idx] = None
    
    def _forward(self, stage_id: int, mb_idx: int, *deps):
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

        if self.profile:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
            )
            profiler.__enter__()
            forward_ctx_manager = torch.profiler.record_function(f"forward_stage_{stage_id}_mb_{mb_idx}")
            forward_ctx_manager.__enter__()

        if stage_id == 0:
            # For the first stage, load input tensors from self.inputs
            for i, inp in zip(self.input_idxs[stage_id], self.inputs):
                self.forward_args[stage_id][i] = inp
        else:
            inputs_from_prev_stage = self.p2p_cache.pop(
                (stage_id - 1, stage_id, mb_idx, False)
            )
            for i, tensor in zip(self.input_idxs[stage_id], inputs_from_prev_stage):
                if isinstance(tensor, (tuple, list)):
                    assert len(tensor) == 1
                    tensor = tensor[0]
                if stage_id > 0 and piper_metadata.stage_to_device[stage_id] == piper_metadata.stage_to_device[stage_id - 1] and tensor.requires_grad:
                    tensor = tensor.detach().requires_grad_(True)
                self.forward_args[stage_id][i] = tensor

            # save first input that requires grad as input activation
            inp_with_grad = [
                self.forward_args[stage_id][i]
                for i in self.input_idxs[stage_id]
                if self.forward_args[stage_id][i].requires_grad
            ]
            assert (
                len(inp_with_grad) == 1
            ), "Exactly one input per stage should require a gradient"
            self.logger.debug(
                f"Saving input activation {inp_with_grad[0].shape} for stage {stage_id} mb {mb_idx}"
            )
            self.inp_activation[stage_id][mb_idx] = inp_with_grad[0]

        # Run the forward pass
        self._start_timing(comp_stream, "forward_comp")
        with torch.cuda.stream(comp_stream):
            output = self.forward_fns[stage_id](*self.forward_args[stage_id])
        self._stop_timing(comp_stream, "forward_comp")

        # Save first output that requires grad as output activation
        # TODO: support multiple outputs
        out_with_grad = [out for out in output if out.requires_grad]
        assert (
            len(out_with_grad) == 1
        ), "Piper only supports one output per subgraph with requires_grad"
        self.logger.debug(
            f"Saving output activation {out_with_grad[0].shape} for stage {stage_id} mb {mb_idx}"
        )
        self.out_activation[stage_id][mb_idx] = out_with_grad[0]

        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.forward_args[stage_id][i] = None

        if stage_id < self.num_stages - 1:
            send_p2p_op = (stage_id, stage_id + 1, mb_idx, True)
            assert send_p2p_op not in self.p2p_cache
            self.p2p_cache[send_p2p_op] = output

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.debug(
            f"Forward {stage_id} mb {mb_idx} on actor {self.global_rank} returning {[out.shape for out in output]}"
        )

        torch.cuda.synchronize()

        if self.profile:
            forward_ctx_manager.__exit__(None, None, None)
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(f"out/{self.mode}/actor{self.global_rank}_fwd{stage_id}mb{mb_idx}_trace.json")

        return 1

    def _backward(self, stage_id: int, mb_idx: int, *deps, loss_fn=None):
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

        if self.profile:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
            )
            profiler.__enter__()
            backward_ctx_manager = torch.profiler.record_function(f"backward_stage_{stage_id}_mb_{mb_idx}")
            backward_ctx_manager.__enter__()

        out_activation = self.out_activation[stage_id][mb_idx]

        if stage_id < self.num_stages - 1:
            input_grad = self.p2p_cache.pop(
                (stage_id + 1, stage_id, mb_idx, False)
            )

            self._start_timing(comp_stream, "backward_comp")
            with torch.cuda.stream(comp_stream):
                out_activation.backward(gradient=input_grad)
            self._stop_timing(comp_stream, "backward_comp")
        else:
            assert loss_fn is not None
            labels = self.labels
            assert out_activation.shape == labels.shape

            self._start_timing(comp_stream, "backward_comp")
            with torch.cuda.stream(comp_stream):
                loss = loss_fn(out_activation, labels)
                loss.backward()
            self._stop_timing(comp_stream, "backward_comp")

            self.loss.append(loss.item())

        # Clear output activation after backward pass
        self.out_activation[stage_id][mb_idx] = None
        del out_activation

        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.cuda.synchronize()

        if self.profile:
            backward_ctx_manager.__exit__(None, None, None)
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(f"out/{self.mode}/actor{self.global_rank}_bwd{stage_id}mb{mb_idx}_trace.json")

        return 1


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

        self._start_timing(comp_stream, "backward_input")
        with torch.cuda.stream(comp_stream):
            gx = torch.autograd.grad(
                outputs=activation_or_loss,
                inputs=stage_input,
                grad_outputs=upstream_grad,
                retain_graph=True,
                allow_unused=True,
            )
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

        torch.cuda.synchronize()

        return 1

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
            self._start_timing(comp_stream, "backward_weight")
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
                assert out_activation.shape == labels.shape
                with torch.cuda.stream(comp_stream):
                    loss = loss_fn(out_activation, labels)
                    gparams = torch.autograd.grad(
                        outputs=loss,
                        inputs=stage_params,
                        retain_graph=False,
                    )
            self._stop_timing(comp_stream, "backward_weight")

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

                self._start_timing(comp_stream, "backward_weight")
                with torch.cuda.stream(comp_stream):
                    gparams = torch.autograd.grad(
                        outputs=intermediate_edges,
                        inputs=weight_edges,
                        grad_outputs=intermediate_edge_grads,
                        retain_graph=False,
                    )
                self._stop_timing(comp_stream, "backward_weight")

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
            bwd_comp_stream = self.overlapped_comp_stream
            self.overlap_a2a_ops = True
            # initialize A2A states
            self.fwd_a2a_pre_events = [torch.cuda.Event() for _ in range(self.n_a2a_ops[fwd_stage_id])]
            self.bwd_a2a_pre_events = [torch.cuda.Event() for _ in range(self.n_a2a_ops[bwd_stage_id])]
            self.fwd_a2a_counter = 0
            self.bwd_a2a_counter = 0
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        fwd_p2p_stream = self.p2p_stream
        bwd_p2p_stream = self.p2p_stream

        self.logger.debug(
            f"Calling forward {fwd_stage_id} mb {fwd_mb_idx} backward {bwd_stage_id} mb {bwd_mb_idx} on actor {self.global_rank}"
        )
        
        if self.profile:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
            )
            profiler.__enter__()

        # PREPARE FORWARD PASS
        if fwd_stage_id == 0:
            # For the first stage, load input tensors from self.inputs
            for i, inp in zip(self.input_idxs[fwd_stage_id], self.inputs):
                self.forward_args[fwd_stage_id][i] = inp
        else:
            inputs_from_prev_stage = self.p2p_cache.pop(
                (fwd_stage_id - 1, fwd_stage_id, fwd_mb_idx, False)
            )
            for i, tensor in zip(self.input_idxs[fwd_stage_id], inputs_from_prev_stage):
                if isinstance(tensor, (tuple, list)):
                    assert len(tensor) == 1
                    tensor = tensor[0]
                if fwd_stage_id > 0 and piper_metadata.stage_to_device[fwd_stage_id] == piper_metadata.stage_to_device[fwd_stage_id - 1] and tensor.requires_grad:
                    tensor = tensor.detach().requires_grad_(True)
                self.forward_args[fwd_stage_id][i] = tensor

            # save first input that requires grad as input activation
            inp_with_grad = [
                self.forward_args[fwd_stage_id][i]
                for i in self.input_idxs[fwd_stage_id]
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
            input_grad = self.p2p_cache.pop(
                (bwd_stage_id + 1, bwd_stage_id, bwd_mb_idx, False)
            )
        else:
            assert loss_fn is not None
            labels = self.labels
            assert out_activation.shape == labels.shape

        # RUN FORWARD PASS
        if self.profile:
            forward_ctx_manager = torch.profiler.record_function(f"forward_stage_{fwd_stage_id}_mb_{fwd_mb_idx}")
            forward_ctx_manager.__enter__()

        with torch.cuda.stream(fwd_comp_stream):
            output = self.forward_fns[fwd_stage_id](*self.forward_args[fwd_stage_id])

        # RUN BACKWARD PASS
        if self.profile:
            forward_ctx_manager.__exit__(None, None, None)
            backward_ctx_manager = torch.profiler.record_function(f"backward_stage_{bwd_stage_id}_mb_{bwd_mb_idx}")
            backward_ctx_manager.__enter__()
        if self.overlap_a2a_ops and self.dp_degree > 1:
            bwd_comp_stream.wait_event(self.fwd_a2a_pre_events[0])
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
        for i in self.input_idxs[fwd_stage_id]:
            self.forward_args[fwd_stage_id][i] = None
        
        # POST PROCESS BACKWARD PASS
        if bwd_stage_id == self.num_stages - 1:
            self.loss.append(loss.item())

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

        torch.cuda.synchronize()

        # clear A2A states
        if self.overlap_a2a_ops and self.dp_degree > 1:
            self.fwd_a2a_pre_events.clear()
            self.bwd_a2a_pre_events.clear()
            self.fwd_a2a_counter = 0
            self.bwd_a2a_counter = 0

        if self.profile:
            backward_ctx_manager.__exit__(None, None, None)
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(
                f"out/{self.mode}/actor{self.global_rank}_"
                f"fwd{fwd_stage_id}mb{fwd_mb_idx}_"
                f"bwd{bwd_stage_id}mb{bwd_mb_idx}_trace.json")

        return 1

    def _synchronize_gradients(self):
        self.logger.info(f"Actor {self.global_rank} synchronizing gradients")
        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.forward_args.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    with torch.cuda.stream(self.comm_stream):
                        dist.all_reduce(
                            param.grad, op=dist.ReduceOp.AVG, group=self.dp_group
                        )

    def _update(self, *deps):
        self.logger.debug(f"Actor {self.global_rank} waiting for backward sync events")

        # if dp degree > 1, make sure all gradients are synchronized before optimizer step
        # TODO: this does not allow overlapping with the optimizer step
        if self.dp_degree > 1:
            self._start_timing(self.comm_stream, "backward_sync")
            if self.naive_gradient_sync:
                self._synchronize_gradients()
            else:
                self._wait_for_comm_ops()
            self._stop_timing(self.comm_stream, "backward_sync")

        # step the optimizer for each stage
        self._start_timing(self.comp_stream, "optim_step")
        for _, optim in self.optims.items():
            optim.step()
            optim.zero_grad()
        self._stop_timing(self.comp_stream, "optim_step")

        losses = self.loss
        self.loss.clear()

        torch.cuda.synchronize()

        self.reset_p2p_states()

        return losses
