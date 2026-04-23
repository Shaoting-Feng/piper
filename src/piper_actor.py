import ray
import torch
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set
import gc
from torch.nn import Parameter
from torch.autograd.graph import GradientEdge, Node, set_warn_on_accumulate_grad_stream_mismatch
import torch.distributed as dist
from collections import defaultdict

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .piper_utils import (
    _deserialize_graphmodule,
    create_logger,
    LOG_LEVEL,
    VERBOSE,
    NcclOverlapDetector,
)
from .backward_utils import get_param_groups, construct_reverse_graph, _get_grad_fn_or_grad_acc
from .piper_exec import TaskType, TaskDAG, Task, runtime_sort_key

CLEANUP_MEMORY = False

# Disable AOT autograd's donated-buffer optimization so that retain_graph=True
# works during the split BWD_I pass (which must keep the graph alive for BWD_W).
# try:
#     import torch._functorch.config as _tmp_fc
#     _tmp_fc.donated_buffer = False
#     del _tmp_fc  # remove from module globals so Ray/cloudpickle never sees it
# except (ImportError, AttributeError):
#     pass

logger = create_logger("piper_actor", LOG_LEVEL)

def _get_rank(pp_rank, dp_rank, pp_degree):
    return pp_rank + dp_rank * pp_degree


def _create_actors(
    num_actors,
    optim_class,
    num_mbs,
    num_stages,
    naive_gradient_sync=False,
    profile=False,
    stage_to_device=None,
    zero_stage: int = 0,
    no_nvtx: bool = False,
    pg=None,
    temp_dir: str = None,
    use_inductor: bool = False,
    ar_a2a_same_stream: bool = False,
    memory_profile: bool = False,
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
            "sample": "process-tree",
            "backtrace": "dwarf",
            "cudabacktrace": "sync:0,memory:0",
            "python-backtrace": "cuda",
            "stop-on-exit": "true",
        }} if profile else {}
        nccl_env = {
            "env_vars": {
                "NCCL_SOCKET_IFNAME": "ens32",
                "GLOO_SOCKET_IFNAME": "ens32",
                "NCCL_DEBUG": "WARN",
                **({"TMPDIR": temp_dir} if (profile and temp_dir) else {}),
            }
        }
        actor = PiperActor.options(
            num_gpus=0.6,
            runtime_env={**nsight_env, **nccl_env},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=dp_rank,
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
            stage_to_device=stage_to_device,
            zero_stage=zero_stage,
            no_nvtx=no_nvtx,
            use_inductor=use_inductor,
            ar_a2a_same_stream=ar_a2a_same_stream,
            memory_profile=memory_profile,
        )
        piper_metadata.actors[pp_rank] = actor


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
        stage_to_device=None,
        zero_stage: int = 0,
        no_nvtx: bool = False,
        use_inductor: bool = False,
        ar_a2a_same_stream: bool = False,
        memory_profile: bool = False,
    ):
        self.logger = create_logger("piper_actor", LOG_LEVEL)

        # BWD_I uses torch.autograd.grad (not .backward), so AccumulateGrad nodes are
        # traversed for stream-sync bookkeeping but never accumulate to p.grad.
        # Suppress the spurious stream-mismatch warning.
        set_warn_on_accumulate_grad_stream_mismatch(False)


        if memory_profile:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="all", max_entries=5_000_000,
            )

        self.pp_rank = pp_rank
        self.optim_class = optim_class
        self.naive_gradient_sync = naive_gradient_sync
        self.zero_stage = int(zero_stage)
        self.no_nvtx = no_nvtx
        self.use_inductor = bool(use_inductor)
        self.ar_a2a_same_stream = bool(ar_a2a_same_stream)

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
        self.pp_lo_hi = None
        self.pp_hi_lo = None
        self.device = "cuda"

        self.global_rank = _get_rank(pp_rank, dp_rank, pp_degree)

        self.logger.debug(
            f"Initializing Ray actor {self.global_rank} GPU {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

        self.input = None
        self.labels = None

        self.comp_stream = torch.cuda.Stream()
        self.overlapped_comp_stream = torch.cuda.Stream()
        self.ar_stream = torch.cuda.Stream()
        if self.ar_a2a_same_stream:
            self.a2a_stream = self.ar_stream
        else:
            self.a2a_stream = torch.cuda.Stream()
        self.p2p_send_stream = torch.cuda.Stream()
        self.p2p_recv_stream = torch.cuda.Stream()

        # map stage id -> original GraphModule (for hook registration)
        self.graph_modules = dict()
        # map stage id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.forward_args = dict()
        # map stage id -> input idx -> input tensor metadata
        self.forward_input_meta = defaultdict(dict)
        # accumuate loss for each microbatch
        self.loss = []

        self.tracing = False
        self._pending_timing_events: list = []  # (label, start_event, stop_event)
        self.trace_data = defaultdict(list)
        self.memory_tracing_enabled = False
        # DAG execution state
        self.dag = None
        self.sorted_dag_nodes = None

        # Unified inter-task buffer.  Keyed by task.uid (int), storing whatever
        # data the task produced for downstream consumers.
        # Cleared at the start of each run_dag() call.
        self.task_buffer: dict = {}
        self.task_buffer_refcounts: dict[int, int] = {}

        # CUDA events for synchronisation between streams
        self.recv_events: dict = {}   # task_uid -> cuda.Event (for RECV tasks)
        self.a2a_events: dict = {}    # task_uid -> cuda.Event (for FWD_A2A/BWD_A2A tasks)
        self.ar_events: dict = {}     # unique_bucket_id -> cuda.Event (for ALL_REDUCE tasks)
        self.rs_events: dict = {}     # unique_bucket_id -> cuda.Event (for REDUCE_SCATTER tasks)
        self.ag_events: dict = {}     # ALL_GATHER task_uid -> cuda.Event
        self.bwd_events: dict = {}    # unique_bucket_id -> cuda.Event (for BWD/BWD_I/BWD_W tasks)

        # Per-bucket data keyed by unique_bucket_id (populated by _load_stage).
        # "Bucket" here is a globally unique unit of computation; each stage may
        # contribute one or more buckets.  The actor is agnostic to which stage
        # owns a bucket.
        self.bucket_fwd_fns: dict = {}          # ubid -> bucket forward runner
        self.bucket_fwd_args: dict = {}         # ubid -> args list (with None holes for inputs)
        self.bucket_input_idxs: dict = {}       # ubid -> list of input slot indices
        self.bucket_param_idxs: dict = {}       # ubid -> list of param slot indices
        self.bucket_param_names: dict = {}      # ubid -> [FX placeholder name per param]
        self.bucket_optims: dict = {}           # ubid -> optimizer (or None)
        self.bucket_trainable_param_idxs: dict = {}  # ubid -> list of trainable param slot indices
        self.bucket_ac_subgraph_counts: dict = {}    # ubid -> number of sequential AC subgraphs
        self.bucket_flat_params: dict = {}       # ubid -> flat param tensor for ZeRO
        self.bucket_flat_grads: dict = {}        # ubid -> flat grad tensor for ZeRO
        self.bucket_shard_params: dict = {}      # ubid -> shard param tensor for ZeRO
        self.bucket_shard_optims: dict = {}      # ubid -> shard optimizer for ZeRO
        self.bucket_rs_grads: dict = {}          # ubid -> reduced shard grad tensor for ZeRO-2/3
        self.zero_grad_dtype = torch.float32     # ZeRO gradient buffers stay fp32
        self.param_shard_info: dict = {}         # ubid -> (shard_start, shard_size, orig_numel)
        self.bucket_param_view_specs: dict = {}  # ubid -> [(param, offset, numel, shape)]
        self.zero3_full_params_fresh: dict = {}  # ubid -> bool
        self.bucket_param_locks: dict = {}       # ubid -> threading.Lock for full-param alloc/free
        self.bucket_grad_locks: dict = {}        # ubid -> threading.Lock for full-grad alloc/free
        self._logged_bucket_forward_info: set = set()

        # Non-trainable constant tensor attributes (e.g. freqs_cis, mask) pushed
        # from the coordinator before compilation so _load_stage can fill them in
        # instead of zero-initializing.  Keyed by bare attribute name (e.g. "freqs_cis").
        self.model_const_attrs: dict = {}

        from .piper_utils import piper_metadata
        piper_metadata.actor_self = self

    def get_trace_data(self) -> dict:
        return self.global_rank, self.trace_data

    def get_and_reset_peak_memory_stats(self) -> tuple:
        """Return (global_rank, max_memory_allocated_bytes) and reset peak stats."""
        max_alloc = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return self.global_rank, max_alloc

    def clear_trace_data(self) -> None:
        self.trace_data.clear()
        self._pending_timing_events.clear()

    def set_tracing(self, enabled: bool) -> None:
        self.tracing = enabled
        self.logger.info(
            f"Actor {self.global_rank}: Tracing {'enabled' if enabled else 'disabled'}"
        )

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self):
        return self.global_rank, torch.cuda.max_memory_allocated() / (1024**3)


    def start_memory_recording(self, max_entries: int = 100000) -> None:
        """Begin recording CUDA memory allocation history on this actor's GPU."""
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=max_entries,
        )
        self.logger.info(
            f"Actor {self.global_rank}: started CUDA memory recording "
            f"(max_entries={max_entries})"
        )

    def dump_memory_snapshot(self, path: str) -> str:
        """Dump the recorded CUDA memory snapshot to a pickle file.

        Returns the path to the written file.
        """
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"rank{self.global_rank}_memory.pickle")
        torch.cuda.memory._dump_snapshot(filepath)
        torch.cuda.memory._record_memory_history(enabled=None)
        self.logger.info(
            f"Actor {self.global_rank}: dumped memory snapshot to {filepath}"
        )
        return filepath

    def set_oom_snapshot_dir(self, path: str) -> None:
        """Tell this actor where to drop a memory snapshot if run_dag hits a CUDA OOM."""
        os.makedirs(path, exist_ok=True)
        self._oom_snapshot_dir = path

    def _log_mem(self, label: str, extra: str = "") -> None:
        """Emit a single-line memory trajectory marker to the actor log.

        Captures current/reserved/peak allocator stats plus retry/OOM counters.
        Cheap: no CUDA sync, just reads allocator counters.
        """
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            reserv = torch.cuda.memory_reserved() / (1024 ** 3)
            max_a = torch.cuda.max_memory_allocated() / (1024 ** 3)
            max_r = torch.cuda.max_memory_reserved() / (1024 ** 3)
            stats = torch.cuda.memory_stats()
            active = stats.get("active_bytes.all.current", 0) / (1024 ** 3)
            retries = stats.get("num_alloc_retries", 0)
            ooms = stats.get("num_ooms", 0)
            self.logger.debug(
                f"[mem] rank={self.global_rank} stage={getattr(self, 'pp_rank', '?')} "
                f"label={label} allocated={alloc:.3f}GB reserved={reserv:.3f}GB "
                f"max_alloc={max_a:.3f}GB max_res={max_r:.3f}GB active={active:.3f}GB "
                f"retries={retries} ooms={ooms} {extra}"
            )
        except Exception as e:
            self.logger.warning(f"[mem] log failed (label={label}): {e}")

    def _nvtx_push(self, label: str) -> None:
        if not self.no_nvtx:
            torch.cuda.nvtx.range_push(label)

    def _nvtx_pop(self) -> None:
        if not self.no_nvtx:
            torch.cuda.nvtx.range_pop()

    def load_input(self, inputs):
        self.inputs = [inp.to(self.device) for inp in inputs]
        self.logger.debug(f"Actor {self.global_rank} loaded inputs {len(self.inputs)}")

    def load_labels(self, labels):
        self.labels = labels.to(self.device)
        self.logger.debug(f"Actor {self.global_rank} loaded labels {self.labels.shape}")

    def load_const_attrs(self, const_attrs: dict) -> None:
        """Store non-trainable constant tensor attributes (e.g. freqs_cis, mask).

        *const_attrs* maps bare attribute name → CPU tensor.  Values are moved to
        the actor's device so ``_load_stage`` can copy them directly.
        """
        self.model_const_attrs = {k: v.to(self.device) for k, v in const_attrs.items()}
        self.logger.log(VERBOSE,
            f"Actor {self.global_rank} loaded {len(self.model_const_attrs)} "
            f"constant attrs: {list(self.model_const_attrs.keys())}"
        )

    def _start_timing(self, stream, label):
        if self.tracing:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            self._pending_timing_events.append((label, start, stop))

    def _stop_timing(self, stream, label):
        if self.tracing:
            for i in range(len(self._pending_timing_events) - 1, -1, -1):
                lbl, _start, stop = self._pending_timing_events[i]
                if lbl == label:
                    stop.record(stream)
                    break

    def flush_timing_events(self) -> None:
        """Synchronize all pending timing events and accumulate elapsed times.
        Call this once per iteration after GPU work completes, instead of
        synchronizing in _stop_timing which would serialize GPU execution."""
        if not self.tracing or not self._pending_timing_events:
            return
        torch.cuda.synchronize()
        for label, start, stop in self._pending_timing_events:
            self.trace_data[label].append(start.elapsed_time(stop))
        self._pending_timing_events.clear()

    def get_node_ip_and_free_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]
        return ray.util.get_node_ip_address(), port

    def _join_process_groups(self, master_addr, master_port):

        self.logger.info(f"Actor {self.global_rank} using GPU {os.environ['CUDA_VISIBLE_DEVICES']}, master addr {master_addr}:{master_port}")

        init_method = f"tcp://{master_addr}:{master_port}"

        self.device = f"cuda:{self.global_rank % torch.cuda.device_count()}"
        torch.cuda.set_device(self.device)

        if self.pp_degree > 1 or self.dp_degree > 1:
            dist.init_process_group(
                "nccl",
                init_method=init_method,
                rank=self.global_rank,
                world_size=self.world_size,
            )
            self.logger.log(VERBOSE,
                f"Actor {self.global_rank} has GPU {os.environ['CUDA_VISIBLE_DEVICES']}, joined the global process group"
            )

            if self.dp_degree > 1:
                self._join_dp_process_group()
            if self.pp_degree > 1:
                self._join_pp_process_group()

            # Force cuBLAS context initialization on both compute streams so the
            # first backward pass does not encounter a "no current CUDA context" warning.
            for compute_stream in (self.comp_stream, self.overlapped_comp_stream):
                with torch.cuda.stream(compute_stream):
                    _w = torch.zeros(4, 4, device=self.device)
                    torch.mm(_w, _w)

            self.logger.info(f"Actor {self.global_rank} joined process groups")

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
                self.logger.log(VERBOSE,
                    f"Global rank {self.global_rank} joined its dp group {dp_group_id} along with ranks {group_ranks}"
                )

    def _join_pp_process_group(self):
        num_pp_groups = self.world_size // self.pp_degree

        for pp_group_id in range(num_pp_groups):
            group_ranks = [
                (pp_group_id * self.pp_degree + i) for i in range(self.pp_degree)
            ]
            lo_hi_group = dist.new_group(ranks=group_ranks, backend="nccl")
            hi_lo_group = dist.new_group(ranks=group_ranks, backend="nccl")

            if self.global_rank in group_ranks:
                self.pp_lo_hi = lo_hi_group
                self.pp_hi_lo = hi_lo_group

        self.logger.log(VERBOSE, f"Actor {self.global_rank} created PP communicators")

    def shutdown(self):
        dist.destroy_process_group()


    @staticmethod
    def _replace_meta_constants(gm, device):
        """Replace any constants on meta device with tensors on the actual device."""
        import torch.fx as fx
        from .piper_utils import create_logger, LOG_LEVEL
        from torch._subclasses.fake_tensor import FakeTensor
        logger = create_logger("piper_actor", LOG_LEVEL)

        replaced_count = 0
        params_count = 0
        buffers_count = 0
        get_attr_count = 0
        constants_count = 0
        attrs_count = 0
        node_args_count = 0
        node_meta_count = 0
        submodule_count = 0

        def is_meta_or_fake(t):
            """Check if tensor is on meta device or is a FakeTensor."""
            if isinstance(t, FakeTensor):
                return True
            if isinstance(t, torch.Tensor):
                return t.device.type == 'meta'
            return False

        def replace_meta_tensor(t):
            if is_meta_or_fake(t):
                return torch.empty_like(t, device=device)
            return t

        def replace_in_module(module, module_name=""):
            """Recursively replace meta tensors in a module and all its submodules."""
            nonlocal replaced_count, params_count, buffers_count, get_attr_count, constants_count, attrs_count, node_args_count, node_meta_count, submodule_count

            # replace all parameters that are on meta device or FakeTensor
            for name, param in module.named_parameters(recurse=False):
                if is_meta_or_fake(param):
                    full_name = f"{module_name}.{name}" if module_name else name
                    logger.log(VERBOSE, f"Replacing parameter '{full_name}' from meta/fake to {device} (shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad})")
                    new_param = torch.empty_like(param, device=device)
                    new_param.requires_grad_(param.requires_grad)
                    setattr(module, name, torch.nn.Parameter(new_param, requires_grad=param.requires_grad))
                    params_count += 1
                    replaced_count += 1

            # replace all buffers that are on meta device or FakeTensor
            for name, buffer in module.named_buffers(recurse=False):
                if is_meta_or_fake(buffer):
                    full_name = f"{module_name}.{name}" if module_name else name
                    logger.log(VERBOSE, f"Replacing buffer '{full_name}' from meta/fake to {device} (shape={buffer.shape}, dtype={buffer.dtype})")
                    new_buffer = torch.empty_like(buffer, device=device)
                    setattr(module, name, new_buffer)
                    buffers_count += 1
                    replaced_count += 1

            # check gm
            if isinstance(module, fx.GraphModule):
                # replace in get_attr nodes
                for node in module.graph.nodes:
                    if node.op == 'get_attr':
                        try:
                            attr_value = getattr(module, node.target)
                            if is_meta_or_fake(attr_value):
                                full_name = f"{module_name}.{node.target}" if module_name else node.target
                                logger.log(VERBOSE, f"Replacing get_attr '{full_name}' from meta/fake to {device} (shape={attr_value.shape}, dtype={attr_value.dtype})")
                                new_tensor = torch.empty_like(attr_value, device=device)
                                setattr(module, node.target, new_tensor)
                                get_attr_count += 1
                                replaced_count += 1
                        except AttributeError:
                            pass

                    # check node.meta for tensor values
                    if hasattr(node, 'meta') and isinstance(node.meta, dict):
                        for key, value in node.meta.items():
                            if is_meta_or_fake(value):
                                full_name = f"{module_name}.{node.name}.meta['{key}']" if module_name else f"{node.name}.meta['{key}']"
                                logger.log(VERBOSE, f"Replacing node.meta['{key}'] for node '{node.name}' from meta/fake to {device} (shape={value.shape}, dtype={value.dtype})")
                                node.meta[key] = torch.empty_like(value, device=device)
                                node_meta_count += 1
                                replaced_count += 1

                # replace in constants dict if it exists
                if hasattr(module, '_constants'):
                    for key, value in module._constants.items():
                        if is_meta_or_fake(value):
                            full_name = f"{module_name}._constants['{key}']" if module_name else f"_constants['{key}']"
                            logger.log(VERBOSE, f"Replacing constant '{full_name}' from meta/fake to {device} (shape={value.shape}, dtype={value.dtype})")
                            module._constants[key] = torch.empty_like(value, device=device)
                            constants_count += 1
                            replaced_count += 1

                # traverse all nodes and replace meta tensors in args/kwargs
                for node in module.graph.nodes:
                    # replace in args
                    new_args = []
                    for arg in node.args:
                        if is_meta_or_fake(arg):
                            full_name = f"{module_name}.{node.name}" if module_name else node.name
                            logger.log(VERBOSE, f"Replacing meta/fake tensor in node '{full_name}' args (shape={arg.shape}, dtype={arg.dtype})")
                            new_args.append(torch.empty_like(arg, device=device))
                            node_args_count += 1
                            replaced_count += 1
                        elif isinstance(arg, torch.device) and arg.type == 'meta':
                            # replace meta device objects with the actual device
                            full_name = f"{module_name}.{node.name}" if module_name else node.name
                            logger.log(VERBOSE, f"Replacing meta device object in node '{full_name}' args with {device}")
                            new_args.append(torch.device(device))
                            node_args_count += 1
                            replaced_count += 1
                        elif isinstance(arg, (list, tuple)):
                            new_args.append(type(arg)(replace_meta_tensor(x) for x in arg))
                        else:
                            new_args.append(arg)
                    node.args = tuple(new_args)

                    # replace in kwargs
                    new_kwargs = {}
                    for key, value in node.kwargs.items():
                        if is_meta_or_fake(value):
                            full_name = f"{module_name}.{node.name}" if module_name else node.name
                            logger.log(VERBOSE, f"Replacing meta/fake tensor in node '{full_name}' kwargs['{key}'] (shape={value.shape}, dtype={value.dtype})")
                            new_kwargs[key] = torch.empty_like(value, device=device)
                            node_args_count += 1
                            replaced_count += 1
                        elif isinstance(value, torch.device) and value.type == 'meta':
                            # replace meta device objects with the actual device
                            full_name = f"{module_name}.{node.name}" if module_name else node.name
                            logger.log(VERBOSE, f"Replacing meta device object in node '{full_name}' kwargs['{key}'] with {device}")
                            new_kwargs[key] = torch.device(device)
                            node_args_count += 1
                            replaced_count += 1
                        elif isinstance(value, (list, tuple)):
                            new_kwargs[key] = type(value)(replace_meta_tensor(x) for x in value)
                        else:
                            new_kwargs[key] = value
                    node.kwargs = new_kwargs

                # recompile the GraphModule after modifications
                module.recompile()

            # recursively process all submodules
            for name, submodule in module.named_children():
                submodule_name = f"{module_name}.{name}" if module_name else name
                replace_in_module(submodule, submodule_name)
                submodule_count += 1

        replace_in_module(gm)

        if replaced_count > 0:
            logger.log(VERBOSE,
                f"_replace_meta_constants: Replaced {replaced_count} meta device tensors "
                f"(params: {params_count}, buffers: {buffers_count}, get_attr: {get_attr_count}, "
                f"constants: {constants_count}, attrs: {attrs_count}, node_args: {node_args_count}, "
                f"node_meta: {node_meta_count}, checked {submodule_count} submodules)"
            )
        else:
            logger.log(VERBOSE, f"_replace_meta_constants: No meta device tensors found to replace! (checked {submodule_count} submodules)")

        return gm

    def _load_stage_kernels(self):
        try:
            try:
                import torchtitan.models.common.moe.kernels
                import torchtitan.models.common.moe.utils
                from torchtitan.models.common.moe.kernels import generate_permute_indices
                from torchtitan.models.common.moe.kernels import fill_indices_wrapper
            except ModuleNotFoundError:
                import torchtitan.models.moe.kernels
                import torchtitan.models.moe.utils
                from torchtitan.models.moe.kernels import generate_permute_indices
                from torchtitan.models.moe.kernels import fill_indices_wrapper
            import torch._higher_order_ops.triton_kernel_wrap
            self.logger.log(VERBOSE, f"Actor {self.global_rank} imported and initialized Triton kernel modules")
        except ImportError as e:
            self.logger.warning(
                f"Actor {self.global_rank} failed to import Triton kernel modules: {e}. "
                f"This may cause issues with Triton kernels in the graph."
            )

        # HACK: manually register Triton kernel in the side table
        # the graph expects kernel_idx=0, so we need to register it at that index
        self.logger.log(VERBOSE, f"Manually registering Triton kernel at index 0 on actor {self.global_rank}...")
        try:
            # Import the kernel wrapper
            try:
                from torchtitan.models.common.moe.kernels import fill_indices_wrapper
            except ModuleNotFoundError:
                from torchtitan.models.moe.kernels import fill_indices_wrapper

            # create a dummy function that uses the Triton kernel
            # This needs to be compiled with torch.compile to register the kernel
            def dummy_kernel_user(tokens, starts, offsets):
                return fill_indices_wrapper(
                    tokens, starts, offsets,
                    experts_per_rank=1, num_ranks=1, max_len=8, block_size=128
                )

            # Compile the function to trigger kernel registration
            device_idx = int(self.device.split(":")[1]) if ":" in self.device else 0
            with torch.cuda.device(device_idx):
                # Create dummy inputs
                dummy_tokens = torch.zeros(8, dtype=torch.int32, device=self.device)
                dummy_start = torch.zeros(8, dtype=torch.int64, device=self.device)
                dummy_offsets = torch.zeros(1, dtype=torch.int64, device=self.device)

                # Compile and run to register the kernel
                compiled_dummy = torch.compile(dummy_kernel_user, mode="reduce-overhead")
                _ = compiled_dummy(dummy_tokens, dummy_start, dummy_offsets)

            # Register the kernel at index 0 in id_to_kernel (kernel_idx=0 is baked into
            # compiled graphs; constant_args are restored separately in _load_stage).
            from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
            if hasattr(kernel_side_table, 'id_to_kernel'):
                registered_kernels = list(kernel_side_table.id_to_kernel.items())
                if registered_kernels:
                    max_idx, kernel = max(registered_kernels, key=lambda x: x[0])
                    kernel_side_table.id_to_kernel[0] = kernel
                    self.logger.log(VERBOSE,
                        f"Actor {self.global_rank} registered kernel at id_to_kernel[0] "
                        f"(copied from index {max_idx})"
                    )
                else:
                    self.logger.warning(
                        f"Actor {self.global_rank} no kernels found in side table after dummy compile"
                    )
        except Exception as e:
            self.logger.warning(
                f"Actor {self.global_rank} failed to manually register kernel: {e}. "
                f"This may cause Triton kernel errors."
            )
            import traceback
            self.logger.info(f"Traceback: {traceback.format_exc()}")

    def _load_stage(
        self,
        stage_id: int,
        modules_data: list,
        a2a_boundaries: dict = None,
        use_activation_checkpointing: bool = False,
    ) -> None:
        """Load a (possibly bucketed) stage.

        *modules_data* is a list of dicts, one per module/bucket, each with keys:
        ``gm_data``, ``graphargs``, ``input_idxs``, ``param_idxs``, ``unique_bucket_id``.

        A non-bucketed stage is represented as a single-element list.
        All per-bucket data structures are keyed by ``unique_bucket_id`` — the
        globally unique bucket index assigned by piper.py — so that run_dag can
        look up bucket data without knowing which stage owns a bucket.
        """
        self.logger.debug(
            f"Loading stage {stage_id} ({len(modules_data)} module(s)) on actor {self.global_rank}"
        )

        g = torch.Generator(device=self.device)
        g.manual_seed(1000 * self.global_rank + stage_id)

        self._load_stage_kernels()

        # Restore triton kernel constant_args that were captured at compile time.
        try:
            from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
            for bd in modules_data:
                if "triton_constant_args_list" in bd:
                    const_arg_sets = bd.get("triton_constant_args_list", [])
                else:
                    const_arg_sets = [bd.get("triton_constant_args", {})]
                for arg_set in const_arg_sets:
                    for idx, args in arg_set.items():
                        kernel_side_table.constant_args[int(idx)] = args
        except Exception as e:
            self.logger.warning(f"Actor {self.global_rank} failed to restore triton constant_args: {e}")

        first_gm = None

        for b_idx, bd in enumerate(modules_data):
            ubid: int = bd["unique_bucket_id"]
            ac_num_subgraphs = int(bd.get("ac_num_subgraphs", 1))
            ac_requested_subgraphs = int(bd.get("ac_requested_subgraphs", ac_num_subgraphs))
            gms = [_deserialize_graphmodule(gm_data) for gm_data in bd["gm_data_list"]] if "gm_data_list" in bd else [_deserialize_graphmodule(bd["gm_data"])]
            gms = [self._replace_meta_constants(gm, self.device) for gm in gms]
            if self.use_inductor:
                compiled_gms = []
                for subgraph_idx, gm in enumerate(gms):
                    try:
                        compiled_gm = torch.compile(gm)
                        compiled_gms.append(compiled_gm)
                        self.logger.debug(
                            f"[load_stage_compile] rank={self.global_rank} stage={stage_id} "
                            f"ubid={ubid} subgraph={subgraph_idx} compiled=True"
                        )
                    except Exception as exc:
                        self.logger.warning(
                            f"[load_stage_compile] rank={self.global_rank} stage={stage_id} "
                            f"ubid={ubid} subgraph={subgraph_idx} compiled=False error={exc}"
                        )
                        compiled_gms.append(gm)
                gms = compiled_gms
            # if "gm_data_list" in bd:
            #     gms = [
            #         self._replace_meta_constants(_deserialize_graphmodule(gm_data), self.device)
            #         for gm_data in bd["gm_data_list"]
            #     ]
            # else:
            #     gms = [self._replace_meta_constants(_deserialize_graphmodule(bd["gm_data"]), self.device)]
            
            if b_idx == 0:
                first_gm = gms[0]

            forward_args = list(bd["graphargs"])
            b_input_idxs = list(bd["input_idxs"])
            b_param_idxs = list(bd["param_idxs"])
            apply_zero = bool(bd.get("apply_zero", True))
            shared_placeholder_names = list(
                bd.get("shared_placeholder_names", bd.get("placeholder_names", []))
            )
            input_placeholder_names = [
                shared_placeholder_names[i] if 0 <= i < len(shared_placeholder_names) else f"<missing:{i}>"
                for i in b_input_idxs
            ]
            param_placeholder_names = [
                shared_placeholder_names[i] if 0 <= i < len(shared_placeholder_names) else f"<missing:{i}>"
                for i in b_param_idxs
            ]
            self.logger.log(VERBOSE,
                f"[load_stage_bucket] rank={self.global_rank} stage={stage_id} "
                f"ubid={ubid} shared_placeholders={shared_placeholder_names} "
                f"input_idxs={b_input_idxs} input_placeholders={input_placeholder_names} "
                f"param_idxs={b_param_idxs} param_placeholders={param_placeholder_names}"
            )

            # Extract FX placeholder names for each param index.
            self.bucket_param_names[ubid] = [
                shared_placeholder_names[i] if i < len(shared_placeholder_names) else f"ubid{ubid}_p{i}"
                for i in b_param_idxs
            ]

            self.logger.log(VERBOSE,
                f"Stage {stage_id} bucket {b_idx} (ubid={ubid}) input indices: {b_input_idxs}"
            )

            # Save input tensor metadata for pre-allocating FWD recv buffers.
            # Stored as a list of (shape, dtype, requires_grad) in input-slot order.
            recv_meta = []
            for i in b_input_idxs:
                meta = forward_args[i]
                if meta is not None:
                    recv_meta.append((tuple(meta.shape), meta.dtype,
                                      bool(getattr(meta, "requires_grad", False))))
                forward_args[i] = None  # clear slot; run_dag will fill it at execution time
            self.forward_input_meta[ubid] = recv_meta

            self.bucket_input_idxs[ubid] = b_input_idxs

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
                    # Non-trainable slot: try to fill from const attrs (freqs_cis, mask, …)
                    # before falling back to zeros.  Dynamo names direct model attrs as
                    # "l_self_<attr_name>", so strip that prefix to get the bare name.
                    ph_name = shared_placeholder_names[i] if i < len(shared_placeholder_names) else ""
                    attr_name = re.sub(r'^l_self_', '', ph_name)
                    const_val = self.model_const_attrs.get(attr_name)
                    if (
                        const_val is not None
                        and tuple(const_val.shape) == tuple(arg.shape)
                        and const_val.dtype == arg.dtype
                    ):
                        t.copy_(const_val)
                    else:
                        t.zero_()
                realized[i] = t

            shared_name_to_idx = {
                name: i for i, name in enumerate(shared_placeholder_names)
            }
            subgraph_specs = []
            for gm in gms:
                sub_placeholder_names = [
                    n.name for n in gm.graph.nodes if n.op == "placeholder"
                ]
                dynamic_names = [
                    name for name in sub_placeholder_names
                    if name not in shared_name_to_idx
                ]
                subgraph_specs.append((gm.forward, sub_placeholder_names, dynamic_names))

            def _bucket_forward_runner(
                shared_args,
                _specs=subgraph_specs,
                _name_to_idx=shared_name_to_idx,
                _use_ac=use_activation_checkpointing,
                _stage_id=stage_id,
                _ubid=ubid,
                _shared_placeholder_names=tuple(shared_placeholder_names),
            ):
                out = None
                for subgraph_idx, (forward_impl, placeholder_names, dynamic_names) in enumerate(_specs):
                    if len(dynamic_names) == 0:
                        dynamic_values = []
                    elif len(dynamic_names) == 1:
                        dyn_arg = out
                        if isinstance(dyn_arg, (tuple, list)) and len(dyn_arg) == 1:
                            dyn_arg = dyn_arg[0]
                        dynamic_values = [dyn_arg]
                    else:
                        if not isinstance(out, (tuple, list)):
                            raise RuntimeError(
                                f"Bucket forward expected {len(dynamic_names)} dynamic inputs "
                                f"but previous subgraph produced {type(out).__name__}"
                            )
                        dynamic_values = list(out)
                        if len(dynamic_values) != len(dynamic_names):
                            raise RuntimeError(
                                f"Bucket forward expected {len(dynamic_names)} dynamic inputs "
                                f"but previous subgraph produced {len(dynamic_values)} values"
                            )
                    dynamic_name_to_value = {
                        name: value for name, value in zip(dynamic_names, dynamic_values)
                    }
                    call_args = []
                    for name in placeholder_names:
                        if name in dynamic_name_to_value:
                            call_args.append(dynamic_name_to_value[name])
                        else:
                            call_args.append(shared_args[_name_to_idx[name]])
                    try:
                        self.logger.debug(
                            f"[bucket_fwd_call] rank={self.global_rank} stage={_stage_id} "
                            f"ubid={_ubid} subgraph={subgraph_idx} "
                            f"shared_placeholders={list(_shared_placeholder_names)} "
                            f"placeholders={placeholder_names} "
                            f"dynamic_names={dynamic_names}"
                        )
                        zero_storage = []
                        for name, arg in zip(placeholder_names, call_args):
                            if not isinstance(arg, torch.Tensor):
                                continue
                            storage = arg.untyped_storage()
                            storage_size = storage.size()
                            if storage_size == 0:
                                zero_storage.append(
                                    f"{name}: shape={tuple(arg.shape)} stride={tuple(arg.stride())} "
                                    f"requires_grad={arg.requires_grad} dtype={arg.dtype} "
                                    f"device={arg.device} storage_offset={arg.storage_offset()}"
                                )
                        if zero_storage:
                            self.logger.error(
                                f"[bucket_fwd_zero_storage] rank={self.global_rank} stage={_stage_id} "
                                f"ubid={_ubid} subgraph={subgraph_idx}: " + " | ".join(zero_storage)
                            )
                    except Exception:
                        self.logger.debug("bucket forward pre-call logging failed", exc_info=True)
                    if _use_ac:
                        out = torch.utils.checkpoint.checkpoint(
                            forward_impl,
                            *call_args,
                            use_reentrant=False,
                        )
                    else:
                        try:
                            out = forward_impl(*call_args)
                        except Exception as exc:
                            tensor_summaries = []
                            for name, arg in zip(placeholder_names, call_args):
                                if isinstance(arg, torch.Tensor):
                                    storage = arg.untyped_storage()
                                    tensor_summaries.append(
                                        f"{name}: shape={tuple(arg.shape)} stride={tuple(arg.stride())} "
                                        f"requires_grad={arg.requires_grad} dtype={arg.dtype} "
                                        f"device={arg.device} storage_size={storage.size()} "
                                        f"storage_offset={arg.storage_offset()}"
                                    )
                                else:
                                    tensor_summaries.append(f"{name}: type={type(arg).__name__}")
                            self.logger.error(
                                f"[bucket_fwd_exception] rank={self.global_rank} stage={_stage_id} "
                                f"ubid={_ubid} subgraph={subgraph_idx} "
                                f"shared_placeholders={list(_shared_placeholder_names)} "
                                f"placeholders={placeholder_names} "
                                f"dynamic_names={dynamic_names} "
                                f"exc_type={type(exc).__name__} exc={exc} "
                                f"args={' | '.join(tensor_summaries)}",
                                exc_info=True,
                            )
                            raise
                return out

            self.bucket_fwd_fns[ubid] = _bucket_forward_runner
            self.bucket_fwd_args[ubid] = realized
            self.bucket_param_idxs[ubid] = b_param_idxs
            self.bucket_ac_subgraph_counts[ubid] = ac_num_subgraphs
            trainable_idxs = [
                i for i in b_param_idxs
                if realized[i] is not None and realized[i].requires_grad
            ]
            self.bucket_trainable_param_idxs[ubid] = trainable_idxs

            if self.zero_stage > 0 and self.dp_degree > 1 and apply_zero and trainable_idxs:
                trainable = [realized[i] for i in trainable_idxs]
                flat_params = torch.cat([p.detach().view(-1) for p in trainable]).contiguous()
                flat_params.requires_grad_(False)
                orig_numel = flat_params.numel()
                dp = self.dp_degree
                shard_size = (orig_numel + dp - 1) // dp
                padded_numel = shard_size * dp
                if padded_numel > orig_numel:
                    padded = flat_params.new_zeros(padded_numel)
                    padded[:orig_numel].copy_(flat_params)
                    flat_params = padded

                offset = 0
                view_specs = []
                for idx, p in zip(trainable_idxs, trainable):
                    numel = p.numel()
                    realized[idx] = realized[idx].detach()
                    realized[idx].data = flat_params[offset:offset + numel].view(p.shape)
                    realized[idx].requires_grad_(True)
                    realized[idx].grad_dtype = self.zero_grad_dtype
                    view_specs.append((realized[idx], offset, numel, tuple(p.shape)))
                    offset += numel

                shard_start = self.dp_rank * shard_size
                if self.zero_stage == 3:
                    shard_param = flat_params[shard_start:shard_start + shard_size].detach().clone()
                else:
                    shard_param = flat_params[shard_start:shard_start + shard_size]
                shard_param.requires_grad_(True)
                # Keep ZeRO comm/storage buffers in fp32, but let the optimizer consume
                # grads in the shard-param dtype to match Adam's bf16 foreach path.
                shard_param.grad_dtype = None

                self.bucket_flat_params[ubid] = flat_params
                self.bucket_flat_grads[ubid] = (
                    None
                    if self.zero_stage in (2, 3)
                    else torch.zeros(padded_numel, dtype=self.zero_grad_dtype, device=self.device)
                )
                self.bucket_shard_params[ubid] = shard_param
                self.bucket_shard_optims[ubid] = self.optim_class([shard_param])
                self.bucket_rs_grads[ubid] = torch.zeros(
                    shard_size, dtype=self.zero_grad_dtype, device=self.device
                )
                self.param_shard_info[ubid] = (shard_start, shard_size, orig_numel)
                self.bucket_param_view_specs[ubid] = view_specs
                self.zero3_full_params_fresh[ubid] = False
                # self.bucket_param_locks[ubid] = threading.Lock()
                # self.bucket_grad_locks[ubid] = threading.Lock()

                if self.zero_stage == 3:
                    storage = flat_params.untyped_storage()
                    if storage.size() != 0:
                        storage.resize_(0)

                optim = None
            else:
                # Keep trainable params as separate tensors for the non-ZeRO path.
                self.bucket_param_view_specs[ubid] = []
                self.bucket_flat_params[ubid] = None
                self.bucket_flat_grads[ubid] = None
                self.bucket_param_locks.pop(ubid, None)
                self.bucket_grad_locks.pop(ubid, None)
                trainable_for_optim = [realized[i] for i in trainable_idxs]
                optim = self.optim_class(trainable_for_optim, fused=True) if trainable_for_optim else None
            self.bucket_optims[ubid] = optim

            self.logger.log(VERBOSE,
                f"[ac_bucket] rank={self.global_rank} stage={stage_id} bucket={b_idx} "
                f"ubid={ubid}: ac_enabled={use_activation_checkpointing} "
                f"requested_subgraphs={ac_requested_subgraphs} "
                f"actual_subgraphs={ac_num_subgraphs} inputs={len(b_input_idxs)} "
                f"params={len(b_param_idxs)} trainable_params={len(trainable_idxs)} "
                f"apply_zero={apply_zero}"
            )


        # Keep first GraphModule for compatibility with external inspection tools.
        self.graph_modules[stage_id] = first_gm

        # self._log_mem(f"post_load_stage_{stage_id}")

    # -----------------------------------------------------------------------
    # DAG-based execution
    # -----------------------------------------------------------------------

    def load_dag(self, dag: TaskDAG) -> None:
        """Store the per-rank TaskDAG for subsequent run_dag() calls."""
        self.dag = dag
        self.sorted_dag_nodes = sorted(dag.nodes, key=runtime_sort_key)

    def get_all_params(self) -> dict:
        """Return {unique_bucket_id: flat_cpu_tensor} for every trainable bucket."""
        result: dict = {}
        for ubid, trainable_idxs in self.bucket_trainable_param_idxs.items():
            params = []
            bucket_args = self.bucket_fwd_args.get(ubid, [])
            for idx in trainable_idxs:
                tensor = bucket_args[idx]
                if tensor is not None:
                    params.append(tensor.detach().cpu().view(-1))
            if params:
                result[ubid] = torch.cat(params).clone()
        return result

    def get_named_params(self) -> dict:
        """Return {fx_placeholder_name: cpu_float32_tensor} for all trainable params."""
        result: dict = {}
        for ubid, b_fwd_args in self.bucket_fwd_args.items():
            b_param_idxs = self.bucket_param_idxs[ubid]
            names = self.bucket_param_names.get(ubid, [])
            for i, idx in enumerate(b_param_idxs):
                p = b_fwd_args[idx]
                if p is not None and p.requires_grad:
                    name = names[i] if i < len(names) else f"ubid{ubid}_p{i}"
                    result[name] = p.detach().cpu().clone().float()
        return result

    def set_named_params(self, named_params: dict) -> None:
        """Load parameter values from ``{fx_placeholder_name: tensor}`` dict."""
        for ubid, b_fwd_args in self.bucket_fwd_args.items():
            b_param_idxs = self.bucket_param_idxs[ubid]
            names = self.bucket_param_names.get(ubid, [])
            for i, idx in enumerate(b_param_idxs):
                p = b_fwd_args[idx]
                if p is not None and p.requires_grad:
                    name = names[i] if i < len(names) else None
                    if name and name in named_params:
                        with torch.no_grad():
                            p.data.copy_(named_params[name].to(p.device, p.dtype))

    def get_bucket_fwd_counts(self) -> dict:
        """Return the set of unique_bucket_ids loaded on this actor."""
        return list(self.bucket_fwd_fns.keys())

    def _compute_stream_for_id(self, compute_stream_id: str) -> "torch.cuda.Stream":
        if compute_stream_id == "overlapped_comp_stream":
            return self.overlapped_comp_stream
        return self.comp_stream

    def _compute_stream_for_task(self, task: Task) -> "torch.cuda.Stream":
        return self._compute_stream_for_id(task.compute_stream_id)

    def _timing_stream(self, task) -> "torch.cuda.Stream":
        """Return the CUDA stream that drives work for the given task type."""
        task_type = task.task_type
        if task_type == TaskType.SEND:
            return self.p2p_send_stream
        if task_type == TaskType.RECV:
            return self.p2p_recv_stream
        if task_type in (TaskType.FWD_A2A, TaskType.BWD_A2A):
            return self.a2a_stream
        if task_type in (TaskType.ALL_REDUCE, TaskType.REDUCE_SCATTER, TaskType.ALL_GATHER):
            return self.ar_stream
        return self._compute_stream_for_task(task)

    def get_node_runtimes(self) -> dict:
        """Return ``{uid: profiling_measurements}`` for every profiled DAG node.

        Only nodes that have at least one measurement are included.  Call this
        after one or more ``run_dag(profiling=True)`` iterations.
        """
        return {
            node.uid: list(node.profiling_measurements)
            for node in self.dag.nodes
            if node.profiling_measurements
        }

    def _clear_param_grads(self) -> None:
        """Restore standard PyTorch semantics: inactive params keep grad=None."""
        for ubid, trainable_idxs in self.bucket_trainable_param_idxs.items():
            args = self.bucket_fwd_args[ubid]
            for idx in trainable_idxs:
                t = args[idx]
                if t is not None:
                    t.grad = None
    def _param_has_effective_grad(self, param: torch.Tensor | None) -> bool:
        """Return True when a param carries a materialized gradient."""
        return param is not None and param.grad is not None

    def _accumulate_zero_param_grads_to_flat(self, ubid: int | None) -> None:
        """Accumulate ZeRO-managed param grads into the fp32 flat grad buffer."""
        if ubid is None or self.zero_stage <= 0 or self.dp_degree <= 1:
            return
        specs = self.bucket_param_view_specs.get(ubid, [])
        if not specs:
            return
        flat_grads = self.bucket_flat_grads.get(ubid)
        if flat_grads is None:
            shard_info = self.param_shard_info.get(ubid)
            if shard_info is None:
                return
            _shard_start, shard_size, _orig_numel = shard_info
            flat_grads = torch.zeros(
                shard_size * self.dp_degree,
                dtype=self.zero_grad_dtype,
                device=self.device,
            )
            self.bucket_flat_grads[ubid] = flat_grads
        for param, offset, numel, _shape in specs:
            grad = param.grad
            if grad is None:
                continue
            flat_grads[offset:offset + numel].add_(grad.detach().reshape(-1).to(flat_grads.dtype))
            param.grad = None

    def _sync_payload_ubid(self, node: Task) -> int | None:
        if node.unique_bucket_id is not None:
            return node.unique_bucket_id
        if node.data_preds:
            return node.data_preds[0].unique_bucket_id
        return None

    def _sync_payload_ubids(self, node: Task) -> list[int]:
        sync_ubids = node.custom_metadata.get("sync_ubids")
        if sync_ubids:
            return list(sync_ubids)
        ubid = self._sync_payload_ubid(node)
        return [ubid] if ubid is not None else []

    def _collect_saved_tensors(
        self,
        obj: Any,
        seen_objs: set[int],
        seen_tensors: set[int],
        tensors: list[torch.Tensor],
    ) -> None:
        obj_id = id(obj)
        if obj_id in seen_objs:
            return
        seen_objs.add(obj_id)

        if isinstance(obj, torch.Tensor):
            tensor_id = id(obj)
            if tensor_id not in seen_tensors:
                seen_tensors.add(tensor_id)
                tensors.append(obj)
            return

        if isinstance(obj, dict):
            for v in obj.values():
                self._collect_saved_tensors(v, seen_objs, seen_tensors, tensors)
            return

        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                self._collect_saved_tensors(v, seen_objs, seen_tensors, tensors)
            return

        if isinstance(obj, Node):
            for name in dir(obj):
                if not name.startswith("_saved_"):
                    continue
                try:
                    value = getattr(obj, name)
                except Exception:
                    continue
                self._collect_saved_tensors(value, seen_objs, seen_tensors, tensors)

    def _summarize_param_groups(self, param_groups: list[dict[str, Any]]) -> dict[str, int]:
        grad_tensor_count = 0
        grad_numel = 0
        grad_bytes = 0
        intermediate_count = 0
        param_count = 0
        saved_tensors: list[torch.Tensor] = []
        seen_objs: set[int] = set()
        seen_tensors: set[int] = set()

        for param_group in param_groups:
            intermediates = param_group.get("intermediates", [])
            grads = param_group.get("grads", [])
            params = param_group.get("params", [])
            intermediate_count += len(intermediates)
            param_count += len(params)

            for grads_tuple in grads:
                if grads_tuple is None:
                    continue
                for grad in grads_tuple:
                    if grad is None:
                        continue
                    grad_tensor_count += 1
                    grad_numel += grad.numel()
                    grad_bytes += grad.numel() * grad.element_size()

            for intermediate in intermediates:
                self._collect_saved_tensors(intermediate, seen_objs, seen_tensors, saved_tensors)

        saved_numel = sum(t.numel() for t in saved_tensors)
        saved_bytes = sum(t.numel() * t.element_size() for t in saved_tensors)

        return {
            "groups": len(param_groups),
            "intermediates": intermediate_count,
            "params": param_count,
            "grad_tensors": grad_tensor_count,
            "grad_numel": grad_numel,
            "grad_bytes": grad_bytes,
            "saved_tensors": len(saved_tensors),
            "saved_numel": saved_numel,
            "saved_bytes": saved_bytes,
        }

    def _summarize_outstanding_bwdi(self) -> dict[str, int]:
        buffers = 0
        groups = 0
        params = 0
        intermediates = 0
        grad_tensors = 0
        grad_bytes = 0
        saved_tensors = 0
        saved_bytes = 0

        for value in self.task_buffer.values():
            if not isinstance(value, dict):
                continue
            param_groups = value.get("param_groups")
            if param_groups is None:
                continue
            buffers += 1
            summary = self._summarize_param_groups(param_groups)
            groups += summary["groups"]
            params += summary["params"]
            intermediates += summary["intermediates"]
            grad_tensors += summary["grad_tensors"]
            grad_bytes += summary["grad_bytes"]
            saved_tensors += summary["saved_tensors"]
            saved_bytes += summary["saved_bytes"]

        return {
            "buffers": buffers,
            "groups": groups,
            "params": params,
            "intermediates": intermediates,
            "grad_tensors": grad_tensors,
            "grad_bytes": grad_bytes,
            "saved_tensors": saved_tensors,
            "saved_bytes": saved_bytes,
        }

    def _accumulate_tensor_bytes(self, obj: Any, seen: set[int]) -> int:
        if isinstance(obj, torch.Tensor):
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            return obj.numel() * obj.element_size()

        if isinstance(obj, dict):
            return sum(self._accumulate_tensor_bytes(v, seen) for v in obj.values())

        if isinstance(obj, (list, tuple, set)):
            return sum(self._accumulate_tensor_bytes(v, seen) for v in obj)

        return 0

    def _summarize_task_buffer(self) -> dict[str, float]:
        categories = {
            "fwd_mb": 0,
            "shape_ref": 0,
            "recv": 0,
            "bwd_i": 0,
            "bwd": 0,
            "a2a": 0,
            "other": 0,
        }
        counts = {k: 0 for k in categories}

        for key, value in self.task_buffer.items():
            seen: set[int] = set()
            nbytes = self._accumulate_tensor_bytes(value, seen)

            category = "other"
            if isinstance(key, tuple) and len(key) == 2 and key[0] == "shape_ref":
                category = "shape_ref"
            elif isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], int):
                category = "fwd_mb"
            elif isinstance(value, list):
                category = "recv"
            elif isinstance(value, dict):
                if "param_groups" in value:
                    category = "bwd_i"
                elif "inp_grads" in value:
                    category = "bwd"
                elif "detached_outs" in value:
                    category = "a2a" if "pre_detach_outs" not in value else "other"

            categories[category] += nbytes
            counts[category] += 1

        summary = {}
        total_bytes = 0
        total_entries = 0
        for key, nbytes in categories.items():
            summary[f"{key}_bytes"] = nbytes
            summary[f"{key}_count"] = counts[key]
            total_bytes += nbytes
            total_entries += counts[key]
        summary["total_bytes"] = total_bytes
        summary["total_entries"] = total_entries
        return summary

    def _init_task_buffer_refcounts(self, dag: TaskDAG) -> None:
        self.task_buffer_refcounts = {
            node.uid: len(node.data_succs)
            for node in dag.nodes
            if node.data_succs
        }

    def _release_task_buffer_uid(self, uid: int) -> None:
        remaining = self.task_buffer_refcounts.get(uid)
        if remaining is None:
            self.task_buffer.pop(uid, None)
            return
        remaining -= 1
        if remaining <= 0:
            self.task_buffer_refcounts.pop(uid, None)
            self.task_buffer.pop(uid, None)
        else:
            self.task_buffer_refcounts[uid] = remaining

    def run_dag(self, loss_fn=None, profiling: bool = False):
        """Thin wrapper around ``_run_dag_body`` that dumps a CUDA memory
        snapshot if the body raises ``torch.cuda.OutOfMemoryError``.

        The dump is written to ``self._oom_snapshot_dir`` (set via
        ``set_oom_snapshot_dir``) or ``/tmp/piper_oom_snapshots`` as a fallback.
        The original exception is re-raised after the snapshot attempt so the
        caller (driver) still sees the OOM.
        """
        # Mark the entire iteration boundary for the memory-viz timeline.
        iter_idx = getattr(self, "_iter_counter", 0)
        self._iter_counter = iter_idx + 1
        self._nvtx_push(f"iter_{iter_idx}_rank_{self.global_rank}")
        try:
            # self._log_mem(f"iter_{iter_idx}_start")
            return self._run_dag_body(loss_fn=loss_fn, profiling=profiling)
        except torch.cuda.OutOfMemoryError as oom_exc:
            oom_dir = getattr(self, "_oom_snapshot_dir", None) or "/tmp/piper_oom_snapshots"
            try:
                os.makedirs(oom_dir, exist_ok=True)
                filepath = os.path.join(
                    oom_dir, f"oom_rank{self.global_rank}_memory.pickle"
                )
                torch.cuda.memory._dump_snapshot(filepath)
                self.logger.error(
                    f"[OOM] rank={self.global_rank} iter={iter_idx} "
                    f"dumped snapshot to {filepath}: {oom_exc}"
                )
            except Exception as dump_exc:
                self.logger.error(
                    f"[OOM] rank={self.global_rank} iter={iter_idx} "
                    f"snapshot dump FAILED: {dump_exc} (original OOM: {oom_exc})"
                )
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
            # self._log_mem(f"iter_{iter_idx}_oom")
            raise
        finally:
            self._nvtx_pop()

    def _run_dag_body(self, loss_fn=None, profiling: bool = False):
        """Execute the loaded TaskDAG in topological order.

        All inter-task data is routed through ``task_buffer``, keyed by the
        producing task's ``uid``.  Each task method receives its inputs as
        arguments and returns its outputs; ``run_dag`` is the only place that
        reads from and writes to ``task_buffer``.

        Synchronisation contract:
        - Before a SEND node: p2p_send_stream waits on the compute event so the
          activation / gradient tensor is fully computed before the send begins.
        - Before a compute node that has a RECV predecessor: comp_stream waits
          on the recv_event so the recv buffer is populated before compute reads it.
        - RECV records a recv_event after the dist.recv completes.
        """
        assert self.dag is not None, "load_dag() must be called before run_dag()"
        assert self.sorted_dag_nodes is not None, "load_dag() must initialize sorted node order"
        debug_enabled = self.logger.isEnabledFor(logging.DEBUG)
        verbose_enabled = self.logger.isEnabledFor(VERBOSE)
        # Reset per-iteration state
        self.task_buffer = {}
        self.task_buffer_refcounts = {}
        self.recv_events = {}
        self.a2a_events = {}
        self.ar_events = {}
        self.rs_events = {}
        if self.zero_stage == 3:
            self.ag_events = {}
        self.bwd_events = {}
        self._clear_param_grads()
        if self.zero_stage >= 1:
            with torch.cuda.stream(self.comp_stream):
                for buf in self.bucket_flat_grads.values():
                    if buf is not None:
                        buf.zero_()
        if self.zero_stage >= 2:
            with torch.cuda.stream(self.comp_stream):
                for buf in self.bucket_rs_grads.values():
                    if buf is not None:
                        buf.zero_()
        comp_events: dict = {}  # task_uid -> cuda.Event for compute tasks
        # profiling: list of (node, start_evt, end_evt, mem_before_bytes)
        # populated during the loop; elapsed_time() is only called after the
        # final synchronize() so we never block the CPU mid-dispatch.
        self._prof_records: list = []

        self._init_task_buffer_refcounts(self.dag)
        last_comp_event_by_stream: dict[str, torch.cuda.Event] = {}

        def _wait_for_all_gather(compute_node: Task) -> None:
            compute_stream = self._compute_stream_for_task(compute_node)
            for pred in compute_node.data_preds:
                if pred.task_type == TaskType.ALL_GATHER:
                    ag_evt = self.ag_events.get(pred.uid)
                    if ag_evt is not None:
                        compute_stream.wait_event(ag_evt)

        def _metadata_ubids(node: Task, key: str) -> list[int]:
            value = node.custom_metadata.get(key)
            if value is True:
                sync_ubids = self._sync_payload_ubids(node)
                if sync_ubids:
                    return sync_ubids
                return [node.unique_bucket_id] if node.unique_bucket_id is not None else []
            if isinstance(value, (list, tuple)):
                return [int(ubid) for ubid in value]
            return []

        def _run_free_full_grads(free_node: Task) -> None:
            self._nvtx_push(f"free_full_grads_ubid{free_node.unique_bucket_id}")
            try:
                for pred in free_node.data_preds:
                    if pred.task_type == TaskType.REDUCE_SCATTER:
                        rs_evt = self.rs_events.get(pred.uid)
                        if rs_evt is not None:
                            self.comp_stream.wait_event(rs_evt)
                            # self.logger.info("synchronizing free_full_grads on thread_id=%s", threading.get_ident())
                            rs_evt.synchronize()
                    elif pred.task_type == TaskType.ALL_REDUCE:
                        ar_evt = self.ar_events.get(pred.uid)
                        if ar_evt is not None:
                            self.comp_stream.wait_event(ar_evt)
                            # self.logger.info("synchronizing free_full_grads on thread_id=%s", threading.get_ident())
                            ar_evt.synchronize()
                self._free_full_grads(free_node.unique_bucket_id)
            finally:
                self._nvtx_pop()

        def _run_free_full_params(free_node: Task) -> None:
            self._nvtx_push(f"free_full_params_ubid{free_node.unique_bucket_id}")
            try:
                for pred in free_node.data_preds:
                    if pred.uid in comp_events:
                        # _free_full_params resizes CUDA storage from the host.
                        # Stream waits are not enough here: the CPU must not
                        # release the full ZeRO-3 buffer while a compute kernel
                        # launched on comp_stream may still be reading it.
                        comp_events[pred.uid].synchronize()
                self._free_full_params(free_node.unique_bucket_id)
            finally:
                self._nvtx_pop()

        for node in self.sorted_dag_nodes:

            task_type = node.task_type
            batch = node.batches[0]
            mb_idx = batch.mb_idx
            ubid = node.unique_bucket_id  # None for non-compute nodes
            node_comp_stream = self._compute_stream_for_task(node)

            if debug_enabled:
                self.logger.debug(
                    f"run_dag dispatch (time {node.time_step}): {task_type.value} "
                    f"mb{mb_idx} ubid={ubid}"
                )

            # Timeline marker visible in NVTX traces & PyTorch memory-viz.
            _task_label = f"{task_type.value}_mb{mb_idx}_ubid{ubid}_t{node.time_step}"
            self._nvtx_push(_task_label)
            _log_heavy_task = task_type in (
                TaskType.FWD,
                TaskType.BWD,
                TaskType.BWD_I,
                TaskType.BWD_W,
                TaskType.FWD_BWD,
                TaskType.UPD,
                TaskType.ALLOC_FULL_PARAMS,
                TaskType.FREE_FULL_PARAMS,
                TaskType.ALLOC_FULL_GRADS,
                TaskType.FREE_FULL_GRADS,
            )
            # if _log_heavy_task:
                # self._log_mem(f"before_{_task_label}")

            if profiling:
                _stream = self._timing_stream(node)
                _start_evt = torch.cuda.Event(enable_timing=True)
                _start_evt.record(_stream)
                _mem_before = torch.cuda.memory_allocated()

            match task_type:

                case TaskType.SEND:
                    # data_preds[0] is the compute task whose output we send.
                    compute_node = node.data_preds[0]
                    self.p2p_send_stream.wait_event(comp_events[compute_node.uid])
                    send_data = self.task_buffer[compute_node.uid]["send_output"]
                    self._start_timing(self.p2p_send_stream, "p2p_send")
                    self._exec_send(send_data, node.peer_pp_rank)
                    self._stop_timing(self.p2p_send_stream, "p2p_send")
                    # Free the send buffer.  For FWD/FWD_A2A the whole entry is stale
                    # after the send (task_buffer[(ubid,mb)] is the BWD consumer).
                    # For BWD/BWD_I/BWD_A2A keep the dict but drop the send tensor —
                    # BWD_W still needs param_groups (BWD_I), co-located BWD still
                    # needs inp_grads (BWD/BWD_A2A).
                    _send_buf = self.task_buffer.get(compute_node.uid)
                    if isinstance(_send_buf, dict):
                        _send_buf["send_output"] = None
                    self._release_task_buffer_uid(compute_node.uid)

                case TaskType.RECV:
                    # data_succs[0] is the downstream compute task.
                    compute_node = node.data_succs[0]
                    comp_evt = last_comp_event_by_stream.get(node.compute_stream_id)
                    if comp_evt is not None:
                        self.p2p_recv_stream.wait_event(comp_evt)
                    if compute_node.task_type == TaskType.FWD:
                        # FWD recv: pre-allocate based on target bucket's input metadata.
                        recv_ubid = compute_node.unique_bucket_id
                        self._start_timing(self.p2p_recv_stream, "p2p_recv")
                        recv_tensors = self._exec_recv_fwd(recv_ubid, node.peer_pp_rank)
                        self._stop_timing(self.p2p_recv_stream, "p2p_recv")
                    else:
                        # BWD recv: pre-allocate based on FWD output shapes.
                        fwd_ubid = node.custom_metadata["fwd_ubid"]
                        shape_meta = self.task_buffer[("shape_ref", fwd_ubid)]
                        self._start_timing(self.p2p_recv_stream, "p2p_recv")
                        recv_tensors = self._exec_recv_bwd(shape_meta, node.peer_pp_rank)
                        self._stop_timing(self.p2p_recv_stream, "p2p_recv")
                    self.task_buffer[node.uid] = recv_tensors
                    recv_evt = torch.cuda.Event()
                    recv_evt.record(self.p2p_recv_stream)
                    self.recv_events[node.uid] = recv_evt

                case TaskType.FWD_A2A:
                    fwd_pred = next(p for p in node.data_preds if p.task_type == TaskType.FWD)
                    self.a2a_stream.wait_event(comp_events[fwd_pred.uid])
                    tensor_idx = node.custom_metadata["a2a_tensor_idx"]
                    # Pass-through: copy upstream FWD task_buffer, apply A2A only at tensor_idx.
                    fwd_buf = dict(self.task_buffer[fwd_pred.uid])
                    # FWD_A2A now owns the copy; release predecessor after consuming it.
                    self._release_task_buffer_uid(fwd_pred.uid)
                    detached_outs = list(fwd_buf["detached_outs"])
                    self._nvtx_push(f"fwd_a2a_ubid{fwd_pred.unique_bucket_id}_mb{mb_idx}")
                    self._start_timing(self.a2a_stream, "fwd_a2a")
                    detached_outs[tensor_idx] = self._exec_a2a(detached_outs[tensor_idx]).requires_grad_(True)
                    self._stop_timing(self.a2a_stream, "fwd_a2a")
                    self._nvtx_pop()
                    fwd_buf["detached_outs"] = detached_outs
                    self.task_buffer[node.uid] = fwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(self.a2a_stream)
                    self.a2a_events[node.uid] = a2a_evt
                    node_comp_stream.wait_event(a2a_evt)

                case TaskType.BWD_A2A:
                    bwd_pred = next(
                        p for p in node.data_preds
                        if p.task_type in (TaskType.BWD, TaskType.BWD_I)
                    )
                    self.a2a_stream.wait_event(comp_events[bwd_pred.uid])
                    tensor_idx = node.custom_metadata["a2a_tensor_idx"]
                    # Pass-through: copy upstream BWD/BWD_I task_buffer, apply reversed A2A only at tensor_idx.
                    bwd_buf = dict(self.task_buffer[bwd_pred.uid])
                    self._release_task_buffer_uid(bwd_pred.uid)
                    inp_grads = list(bwd_buf["inp_grads"])
                    grad_a2a_out = inp_grads[tensor_idx]
                    assert grad_a2a_out is not None, (
                        f"BWD_A2A mb={mb_idx}: grad at a2a_tensor_idx={tensor_idx} is None"
                    )
                    if not grad_a2a_out.is_contiguous():
                        grad_a2a_out = grad_a2a_out.contiguous()
                    self._nvtx_push(f"bwd_a2a_ubid{bwd_pred.unique_bucket_id}_mb{mb_idx}")
                    self._start_timing(self.a2a_stream, "bwd_a2a")
                    inp_grads[tensor_idx] = self._exec_a2a(grad_a2a_out)
                    self._stop_timing(self.a2a_stream, "bwd_a2a")
                    self._nvtx_pop()
                    bwd_buf["inp_grads"] = inp_grads
                    self.task_buffer[node.uid] = bwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(self.a2a_stream)
                    self.a2a_events[node.uid] = a2a_evt
                    node_comp_stream.wait_event(a2a_evt)

                case TaskType.ALL_REDUCE:
                    bwd_node = node.data_preds[0]
                    ar_ubids = self._sync_payload_ubids(node)
                    ar_ubid = ar_ubids[0] if ar_ubids else bwd_node.unique_bucket_id
                    self._nvtx_push(f"all_reduce_ubid{ar_ubid}")
                    self.ar_stream.wait_event(comp_events[bwd_node.uid])
                    self._start_timing(self.ar_stream, "all_reduce")
                    runtime_bytes = sum(self._exec_all_reduce_grads(ubid) for ubid in ar_ubids)
                    self._stop_timing(self.ar_stream, "all_reduce")
                    if runtime_bytes > 0:
                        ar_evt = torch.cuda.Event()
                        ar_evt.record(self.ar_stream)
                        self.ar_events[node.uid] = ar_evt
                        for free_ubid in _metadata_ubids(node, "zero_free_full_grads_after_ubids"):
                            ar_evt.synchronize()
                            self._free_full_grads(free_ubid)
                    self._nvtx_pop()
                    self._release_task_buffer_uid(bwd_node.uid)

                case TaskType.REDUCE_SCATTER:
                    bwd_node = node.data_preds[0]
                    rs_ubid = node.unique_bucket_id if node.unique_bucket_id is not None else bwd_node.unique_bucket_id
                    self._nvtx_push(f"reduce_scatter_ubid{rs_ubid}")
                    self.ar_stream.wait_event(comp_events[bwd_node.uid])
                    self._start_timing(self.ar_stream, "reduce_scatter")
                    self._exec_reduce_scatter(rs_ubid)
                    self._stop_timing(self.ar_stream, "reduce_scatter")
                    rs_evt = torch.cuda.Event()
                    rs_evt.record(self.ar_stream)
                    self.rs_events[node.uid] = rs_evt
                    for free_ubid in _metadata_ubids(node, "zero_free_full_grads_after_ubids"):
                        rs_evt.synchronize()
                        self._free_full_grads(free_ubid)
                    self._nvtx_pop()
                    self._release_task_buffer_uid(bwd_node.uid)

                case TaskType.ALL_GATHER:
                    for alloc_ubid in _metadata_ubids(node, "zero_alloc_full_params_before"):
                        self._alloc_full_params(alloc_ubid)
                    self._nvtx_push(f"all_gather_s{batch.stage_id}_b{node.bucket_id}")
                    self._start_timing(self.ar_stream, "all_gather")
                    ag_launched = self._exec_all_gather(node.unique_bucket_id)
                    self._stop_timing(self.ar_stream, "all_gather")
                    if ag_launched:
                        ag_evt = torch.cuda.Event()
                        ag_evt.record(self.ar_stream)
                        self.ag_events[node.uid] = ag_evt
                    self._nvtx_pop()

                case TaskType.ALLOC_FULL_GRADS:
                    self._nvtx_push(f"alloc_full_grads_ubid{node.unique_bucket_id}")
                    self._alloc_full_grads(node.unique_bucket_id)
                    self._nvtx_pop()

                case TaskType.FREE_FULL_GRADS:
                    _run_free_full_grads(node)

                case TaskType.ALLOC_FULL_PARAMS:
                    self._nvtx_push(f"alloc_full_params_ubid{node.unique_bucket_id}")
                    self._alloc_full_params(node.unique_bucket_id)
                    self._nvtx_pop()

                case TaskType.FREE_FULL_PARAMS:
                    _run_free_full_params(node)

                case TaskType.FWD:
                    # Wait on any RECV predecessor.
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        node_comp_stream.wait_event(self.recv_events.pop(recv_pred.uid))

                    # Gather input tensors from task_buffer.
                    # FWD(bkt=0 of first stage): inputs come from self.inputs.
                    # FWD(bkt=0 of non-first stage): inputs come from RECV or prev FWD (co-located).
                    # FWD(bkt>0): inputs come from prev FWD or FWD_A2A output.
                    fwd_data_pred = next(
                        (p for p in node.data_preds
                         if p.task_type in (TaskType.FWD, TaskType.FWD_A2A)), None
                    )
                    if fwd_data_pred is not None:
                        # Both FWD and FWD_A2A task_buffer entries carry "detached_outs".
                        input_tensors = self.task_buffer[fwd_data_pred.uid]["detached_outs"]
                        self._release_task_buffer_uid(fwd_data_pred.uid)
                    elif recv_pred is not None:
                        input_tensors = self.task_buffer[recv_pred.uid]
                        self._release_task_buffer_uid(recv_pred.uid)
                    else:
                        input_tensors = self.inputs  # first stage, first bucket

                    _wait_for_all_gather(node)

                    self._nvtx_push(f"forward_ubid{ubid}_mb{mb_idx}")
                    self._start_timing(node_comp_stream, "forward")
                    fwd_out = self._forward_dag(ubid, mb_idx, input_tensors, node_comp_stream)
                    self._stop_timing(node_comp_stream, "forward")
                    self._nvtx_pop()
                    self.task_buffer[node.uid] = fwd_out
                    # Shape ref for BWD RECV: store lightweight metadata (shape, dtype) so
                    # _exec_recv_bwd can pre-allocate receive buffers without holding live tensors.
                    self.task_buffer[("shape_ref", ubid)] = [(t.shape, t.dtype) for t in fwd_out["out_with_grad"]]
                    self.task_buffer[(ubid, mb_idx)] = fwd_out  # per-mb FWD data for BWD
                    evt = torch.cuda.Event()
                    evt.record(node_comp_stream)
                    comp_events[node.uid] = evt
                    last_comp_event_by_stream[node.compute_stream_id] = evt
                    if node.custom_metadata.get("zero_free_full_params_after"):
                        evt.synchronize()
                        self._free_full_params(ubid)

                case TaskType.BWD:
                    # Wait on upstream grad RECV if present.
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        node_comp_stream.wait_event(self.recv_events.pop(recv_pred.uid))
                    _wait_for_all_gather(node)

                    # Resolve outputs and upstream grads via per-mb ubid lookup.
                    fwd_out = self.task_buffer[(ubid, mb_idx)]

                    if node.compute_loss:
                        assert loss_fn is not None
                        with torch.cuda.stream(node_comp_stream):
                            outputs_or_loss = [loss_fn(fwd_out["out_with_grad"][0], self.labels)]
                        upstream_grads = None
                    elif recv_pred is not None:
                        upstream_grads = self.task_buffer[recv_pred.uid]
                        self._release_task_buffer_uid(recv_pred.uid)
                        if not isinstance(upstream_grads, (list, tuple)):
                            upstream_grads = [upstream_grads]
                        outputs_or_loss = fwd_out["out_with_grad"]
                        upstream_grads = list(upstream_grads)
                    else:
                        outputs_or_loss = fwd_out["out_with_grad"]
                        upstream_grads = None  # last stage, no recv

                    bwd_a2a_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.BWD_A2A), None
                    )
                    if bwd_a2a_pred is not None:
                        a2a_buf = self.task_buffer[bwd_a2a_pred.uid]
                        pre_detach_outs = fwd_out["pre_detach_outs"]
                        detached_outs = fwd_out["detached_outs"]
                        # Set .grad for ALL requires_grad detached boundary tensors from
                        # BWD_A2A's inp_grads.  inp_grads[tensor_idx] is the reversed A2A
                        # gradient; other positions carry pass-through grads (residuals, etc.)
                        # computed by the preceding BWD bucket.  The assertion in _backward_dag
                        # requires every requires_grad entry to have a non-None .grad.
                        for i, (d, g) in enumerate(zip(detached_outs, a2a_buf["inp_grads"])):
                            if d is not None and isinstance(d, torch.Tensor) and d.requires_grad and g is not None:
                                d.grad = g
                        self._release_task_buffer_uid(bwd_a2a_pred.uid)
                    else:
                        bwd_pred = next(
                            (p for p in node.data_preds
                             if p.task_type in (TaskType.BWD, TaskType.BWD_I)), None
                        )
                        if bwd_pred is not None and recv_pred is None:
                            # Co-located boundary-bridge: the downstream BWD stored
                            # inp_grads for D' (its re-detached inputs). D' is parallel
                            # to this stage's detached_outs (D), so copy grad explicitly
                            # — same pattern as BWD_A2A.
                            # recv_pred is None ensures this is a true co-located pair,
                            # not a scheduling mb-chain edge from a cross-rank predecessor.
                            prev_buf = self.task_buffer[bwd_pred.uid]
                            pre_detach_outs = fwd_out.get("pre_detach_outs")
                            detached_outs = fwd_out.get("detached_outs")
                            for d, g in zip(detached_outs, prev_buf["inp_grads"]):
                                if (d is not None and isinstance(d, torch.Tensor)
                                        and d.requires_grad and g is not None):
                                    d.grad = g
                            self._release_task_buffer_uid(bwd_pred.uid)
                        else:
                            # Last (or only) stage in backward order, or has a RECV for
                            # upstream grads (cross-rank predecessor): standard backward.
                            pre_detach_outs = None
                            detached_outs = None

                    inp_with_grad = fwd_out.get("inp_with_grad")
                    if node.custom_metadata.get("zero_alloc_full_grads_before"):
                        self._alloc_full_grads(ubid)

                    self._nvtx_push(f"backward_ubid{ubid}_mb{mb_idx}")
                    self._start_timing(node_comp_stream, "backward")
                    bwd_out = self._backward_dag(
                        ubid, mb_idx, outputs_or_loss, upstream_grads,
                        pre_detach_outs, detached_outs, inp_with_grad,
                        fwd_out.get("out_with_grad"),
                        node_comp_stream,
                    )
                    self._stop_timing(node_comp_stream, "backward")
                    self._nvtx_pop()
                    buf = bwd_out if bwd_out is not None else {}
                    # Full-length inp_grads parallel to fwd_inputs / detached_outs so
                    # BWD_A2A can index with a2a_tensor_idx without filtering offset.
                    fwd_inputs_full = fwd_out.get("fwd_inputs")
                    buf["inp_grads"] = (
                        [t.grad if (t is not None and t.requires_grad) else None
                         for t in fwd_inputs_full]
                        if fwd_inputs_full is not None
                        else [t.grad for t in (inp_with_grad or [])]
                    )
                    self._accumulate_zero_param_grads_to_flat(ubid)
                    self.task_buffer[node.uid] = buf
                    # FWD stored the same dict under both node.uid and (ubid, mb_idx).
                    # Clear the dict in-place so any remaining alias stops keeping
                    # activations and autograd edges alive through the optimizer step.
                    fwd_out.clear()
                    del self.task_buffer[(ubid, mb_idx)]
                    evt = torch.cuda.Event()
                    evt.record(node_comp_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    last_comp_event_by_stream[node.compute_stream_id] = evt
                    if node.custom_metadata.get("zero_free_full_params_after"):
                        evt.synchronize()
                        self._free_full_params(ubid)

                case TaskType.BWD_I:
                    # Wait on upstream grad RECV if present.
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        node_comp_stream.wait_event(self.recv_events.pop(recv_pred.uid))
                    _wait_for_all_gather(node)

                    # Resolve stage_outputs_or_loss and output_grads.
                    fwd_out = self.task_buffer[(ubid, mb_idx)]

                    if node.compute_loss:
                        assert loss_fn is not None
                        with torch.cuda.stream(node_comp_stream):
                            stage_outputs_or_loss = [loss_fn(fwd_out["out_with_grad"][0], self.labels)]
                        output_grads = None
                    elif recv_pred is not None:
                        upstream_raw = self.task_buffer[recv_pred.uid]
                        self._release_task_buffer_uid(recv_pred.uid)
                        if not isinstance(upstream_raw, (list, tuple)):
                            upstream_raw = [upstream_raw]
                        stage_outputs_or_loss = fwd_out["out_with_grad"]
                        output_grads = list(upstream_raw)
                    else:
                        stage_outputs_or_loss = fwd_out["out_with_grad"]
                        output_grads = None

                    # If a BWD_A2A precedes this bucket, use the reversed grad as
                    # output_grads so backward flows correctly through the A2A boundary.
                    bwd_a2a_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.BWD_A2A), None
                    )
                    if bwd_a2a_pred is not None and not node.compute_loss and recv_pred is None:
                        a2a_buf = self.task_buffer[bwd_a2a_pred.uid]
                        # BWD_A2A passes through BWD_I(bkt+1)'s task_buffer with
                        # inp_grads[tensor_idx] replaced by the reversed A2A gradient.
                        # Build output_grads aligned with out_with_grad (requires_grad
                        # positions of detached_outs) then filter to non-None entries.
                        detached_outs = fwd_out["detached_outs"]
                        output_grads_full = [
                            a2a_buf["inp_grads"][i]
                            for i, t in enumerate(detached_outs)
                            if t is not None and getattr(t, "requires_grad", False)
                        ]
                        pairs = [
                            (o, g) for o, g in zip(stage_outputs_or_loss, output_grads_full)
                            if g is not None
                        ]
                        stage_outputs_or_loss = [p[0] for p in pairs]
                        output_grads = [p[1] for p in pairs]
                        self._release_task_buffer_uid(bwd_a2a_pred.uid)

                    input_values = fwd_out.get("inp_with_grad") or []

                    # Weights for this bucket.
                    b_fwd_args = self.bucket_fwd_args[ubid]
                    weights = [b_fwd_args[i] for i in self.bucket_param_idxs[ubid]
                               if b_fwd_args[i] is not None]

                    self._nvtx_push(f"backward_input_ubid{ubid}_mb{mb_idx}")
                    self._start_timing(node_comp_stream, "backward_input")
                    with torch.cuda.stream(node_comp_stream):
                        dinputs, param_groups = self._bucket_backward_input(
                            stage_outputs_or_loss, output_grads, input_values, iter(weights)
                        )
                    self._stop_timing(node_comp_stream, "backward_input")
                    self._nvtx_pop()
                    # Full-length inp_grads parallel to fwd_inputs / detached_outs.
                    fwd_inputs_full = fwd_out.get("fwd_inputs")
                    if fwd_inputs_full is not None:
                        inp_grads_full = [
                            t.grad if (t is not None and t.requires_grad) else None
                            for t in fwd_inputs_full
                        ]
                    else:
                        inp_grads_full = list(dinputs)
                    # Store param_groups for BWD_W (accessed via data_succs edge).
                    self.task_buffer[node.uid] = {
                        "inp_grads": inp_grads_full,
                        "param_groups": param_groups,
                    }
                    if verbose_enabled:
                        pg_summary = self._summarize_param_groups(param_groups)
                        outstanding = self._summarize_outstanding_bwdi()
                        self.logger.log(VERBOSE,
                            f"[bwd_i_retain] rank={self.global_rank} ubid={ubid} mb={mb_idx}: "
                            f"groups={pg_summary['groups']} "
                            f"params={pg_summary['params']} "
                            f"intermediates={pg_summary['intermediates']} "
                            f"grad_tensors={pg_summary['grad_tensors']} "
                            f"grad_bytes_gb={pg_summary['grad_bytes'] / (1024 ** 3):.2f} "
                            f"saved_tensors={pg_summary['saved_tensors']} "
                            f"saved_bytes_gb={pg_summary['saved_bytes'] / (1024 ** 3):.2f} "
                            f"outstanding_buffers={outstanding['buffers']} "
                            f"outstanding_grad_bytes_gb={outstanding['grad_bytes'] / (1024 ** 3):.2f} "
                            f"outstanding_saved_bytes_gb={outstanding['saved_bytes'] / (1024 ** 3):.2f} "
                            f"alloc_gb={torch.cuda.memory_allocated(self.device) / (1024 ** 3):.2f} "
                            f"reserved_gb={torch.cuda.memory_reserved(self.device) / (1024 ** 3):.2f}"
                        )
                    # Propagate input grad to SEND (if this is the first bucket and has predecessors).
                    if dinputs:
                        self.task_buffer[node.uid]["send_output"] = list(dinputs)
                    # FWD stored the same dict under both node.uid and (ubid, mb_idx).
                    # Clear the dict in-place so any remaining alias stops keeping
                    # activations and autograd edges alive through the optimizer step.
                    fwd_out.clear()
                    del stage_outputs_or_loss, fwd_out
                    del self.task_buffer[(ubid, mb_idx)]
                    evt = torch.cuda.Event()
                    evt.record(node_comp_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    last_comp_event_by_stream[node.compute_stream_id] = evt
                    if node.custom_metadata.get("zero_free_full_params_after"):
                        evt.synchronize()
                        self._free_full_params(ubid)

                case TaskType.BWD_W:
                    _wait_for_all_gather(node)
                    if node.custom_metadata.get("zero_alloc_full_grads_before"):
                        self._alloc_full_grads(ubid)
                    # data_preds[0] is the BWD_I task for bkt=K-1 (head of chain).
                    # For bkt<K-1 the chain-link edge (BWD_W predecessor) is prepended
                    # before Pass 2 appends the BWD_I edge, so data_preds[0] would be
                    # the wrong task.  Find BWD_I explicitly by type to be safe.
                    bwdi_node = next(p for p in node.data_preds if p.task_type == TaskType.BWD_I)
                    param_groups = self.task_buffer[bwdi_node.uid]["param_groups"]
                    if verbose_enabled:
                        pg_summary = self._summarize_param_groups(param_groups)
                        outstanding = self._summarize_outstanding_bwdi()
                        self.logger.log(VERBOSE,
                            f"[bwd_w_begin] rank={self.global_rank} ubid={ubid} mb={mb_idx}: "
                            f"groups={pg_summary['groups']} "
                            f"params={pg_summary['params']} "
                            f"intermediates={pg_summary['intermediates']} "
                            f"grad_tensors={pg_summary['grad_tensors']} "
                            f"grad_bytes_gb={pg_summary['grad_bytes'] / (1024 ** 3):.2f} "
                            f"saved_tensors={pg_summary['saved_tensors']} "
                            f"saved_bytes_gb={pg_summary['saved_bytes'] / (1024 ** 3):.2f} "
                            f"outstanding_buffers={outstanding['buffers']} "
                            f"outstanding_grad_bytes_gb={outstanding['grad_bytes'] / (1024 ** 3):.2f} "
                            f"outstanding_saved_bytes_gb={outstanding['saved_bytes'] / (1024 ** 3):.2f} "
                            f"alloc_gb={torch.cuda.memory_allocated(self.device) / (1024 ** 3):.2f} "
                            f"reserved_gb={torch.cuda.memory_reserved(self.device) / (1024 ** 3):.2f}"
                        )
                    b_fwd_args = self.bucket_fwd_args[ubid]
                    weights = [b_fwd_args[i] for i in self.bucket_param_idxs[ubid]
                               if b_fwd_args[i] is not None]
                    self._nvtx_push(f"backward_weight_ubid{ubid}_mb{mb_idx}")
                    self._start_timing(node_comp_stream, "backward_weight")
                    with torch.cuda.stream(node_comp_stream):
                        self._bucket_backward_weight(iter(weights), param_groups, ubid=ubid, mb_idx=mb_idx)
                    self._stop_timing(node_comp_stream, "backward_weight")
                    self._nvtx_pop()
                    self._accumulate_zero_param_grads_to_flat(ubid)
                    self.task_buffer[node.uid] = {}
                    # BWD_I buffer is no longer needed: _bucket_backward_weight already
                    # deleted param_groups["intermediates"] and param_groups["grads"] and
                    # the autograd graph was freed by retain_graph=False.  Drop the shell.
                    self._release_task_buffer_uid(bwdi_node.uid)
                    evt = torch.cuda.Event()
                    evt.record(node_comp_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    last_comp_event_by_stream[node.compute_stream_id] = evt
                    if node.custom_metadata.get("zero_free_full_params_after"):
                        evt.synchronize()
                        self._free_full_params(ubid)

                case TaskType.UPD:
                    self._nvtx_push("update")
                    self._start_timing(self.comp_stream, "optim_step")
                    self._update()
                    self._stop_timing(self.comp_stream, "optim_step")
                    self._nvtx_pop()

            if profiling:
                _end_evt = torch.cuda.Event(enable_timing=True)
                _end_evt.record(_stream)
                _mem_after = torch.cuda.memory_allocated()
                self._prof_records.append((node, mb_idx, ubid, _start_evt, _end_evt, _mem_before))
                self.logger.info(f"Rank {self.global_rank} task {node.task_type.value} (time {node.time_step}) mem before {_mem_before / 1024 ** 3:.2f} GB after {_mem_after / 1024 ** 3:.2f} GB")

            # if _log_heavy_task:
                # self._log_mem(f"after_{_task_label}")
            self._nvtx_pop()

            if verbose_enabled:
                tb_summary = self._summarize_task_buffer()
                self.logger.log(VERBOSE,
                    f"[task_mem] rank={self.global_rank} time={node.time_step} "
                    f"task={node.task_type.value} mb={mb_idx} ubid={ubid}: "
                    f"alloc_gb={torch.cuda.memory_allocated(self.device) / (1024 ** 3):.2f} "
                    f"reserved_gb={torch.cuda.memory_reserved(self.device) / (1024 ** 3):.2f} "
                    f"tb_entries={tb_summary['total_entries']} "
                    f"tb_total_gb={tb_summary['total_bytes'] / (1024 ** 3):.2f} "
                    f"fwd_mb_count={tb_summary['fwd_mb_count']} "
                    f"fwd_mb_gb={tb_summary['fwd_mb_bytes'] / (1024 ** 3):.2f} "
                    f"recv_count={tb_summary['recv_count']} "
                    f"recv_gb={tb_summary['recv_bytes'] / (1024 ** 3):.2f} "
                    f"bwd_i_count={tb_summary['bwd_i_count']} "
                    f"bwd_i_gb={tb_summary['bwd_i_bytes'] / (1024 ** 3):.2f} "
                    f"bwd_count={tb_summary['bwd_count']} "
                    f"bwd_gb={tb_summary['bwd_bytes'] / (1024 ** 3):.2f} "
                    f"shape_ref_count={tb_summary['shape_ref_count']} "
                    f"shape_ref_gb={tb_summary['shape_ref_bytes'] / (1024 ** 3):.2f} "
                    f"a2a_count={tb_summary['a2a_count']} "
                    f"a2a_gb={tb_summary['a2a_bytes'] / (1024 ** 3):.2f} "
                    f"other_count={tb_summary['other_count']} "
                    f"other_gb={tb_summary['other_bytes'] / (1024 ** 3):.2f}"
                )

    def _exec_send(self, send_data, peer_pp_rank: int) -> None:
        """Send tensors to peer_pp_rank on p2p_send_stream.

        The caller (run_dag) must have already made p2p_send_stream wait on the
        compute event before calling this method.
        """
        global_dst_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)
        if self.logger.isEnabledFor(VERBOSE):
            self.logger.log(VERBOSE, f"exec_send to global rank {global_dst_rank}")

        with torch.cuda.stream(self.p2p_send_stream):
            tensors = send_data if isinstance(send_data, (list, tuple)) else [send_data]
            pp_group = self.pp_lo_hi if global_dst_rank > self.global_rank else self.pp_hi_lo
            for tensor in tensors:
                dist.send(tensor, dst=global_dst_rank, group=pp_group)

    def _exec_recv_fwd(self, recv_ubid: int, peer_pp_rank: int) -> list:
        """Receive FWD activations from peer_pp_rank into pre-allocated buffers.

        Buffer shapes are derived from the target bucket's stored input metadata.
        Returns the received tensor list (stored in task_buffer by run_dag).
        """
        global_src_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)
        if self.logger.isEnabledFor(VERBOSE):
            self.logger.log(VERBOSE, f"exec_recv_fwd ubid={recv_ubid} from global rank {global_src_rank}")

        buf = [
            torch.empty(shape, dtype=dtype, requires_grad=requires_grad, device=self.device)
            for shape, dtype, requires_grad in self.forward_input_meta[recv_ubid]
        ]
        with torch.cuda.stream(self.p2p_recv_stream):
            pp_group = self.pp_hi_lo if global_src_rank > self.global_rank else self.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def _exec_recv_bwd(self, shape_meta: list, peer_pp_rank: int) -> list:
        """Receive BWD upstream gradients from peer_pp_rank.

        Buffer shapes are derived from shape_meta, a list of (shape, dtype) tuples
        stored in task_buffer[("shape_ref", ubid)] after the corresponding FWD.
        Returns the received gradient list (stored in task_buffer by run_dag).
        """
        global_src_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)
        if self.logger.isEnabledFor(VERBOSE):
            self.logger.log(VERBOSE, f"exec_recv_bwd from global rank {global_src_rank}")

        buf = [torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in shape_meta]
        with torch.cuda.stream(self.p2p_recv_stream):
            pp_group = self.pp_hi_lo if global_src_rank > self.global_rank else self.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def _exec_a2a(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply dist.all_to_all_single on a2a_stream and return the output tensor.

        Used for both FWD_A2A and BWD_A2A — the operation is symmetric.
        Stream waits and event recording are handled by run_dag.
        """
        output_buf = torch.empty_like(input_tensor, device=self.device)
        with torch.cuda.stream(self.a2a_stream):
            dist.all_to_all_single(output_buf, input_tensor, group=self.ep_group)
        return output_buf

    def _forward_dag(self, ubid: int, mb_idx: int, input_tensors, compute_stream) -> dict:
        """Run the forward function for bucket *ubid* and return a result dict.

        The result dict always contains:
        - ``"send_output"``: raw output tuple/tensor for SEND (last bucket only)
        - ``"out_with_grad"``: list of output tensors that require grad (last bucket only)
        - ``"pre_detach_outs"`` / ``"detached_outs"``: boundary tensors (non-last bucket only)
        - ``"inp_with_grad"``: input tensors that require grad (always; empty list if none)

        run_dag stores the result in task_buffer[node.uid] and passes fields to
        downstream tasks as appropriate.
        """
        fwd_fn = self.bucket_fwd_fns[ubid]
        fwd_args = self.bucket_fwd_args[ubid]
        input_idxs = self.bucket_input_idxs[ubid]

        # Place input tensors into the args array.
        # For co-located stages, input_tensors may be the raw output tuple from
        # the previous stage's FWD (a tuple or a single tensor).
        if not isinstance(input_tensors, (list, tuple)):
            input_tensors = [input_tensors]

        for i, tensor in zip(input_idxs, input_tensors):
            if isinstance(tensor, (list, tuple)):
                tensor = tensor[0]
            # Detach activations arriving from a co-located stage so backward
            # treats them as leaves (same device, no cross-stage grad flow).
            if tensor.requires_grad:
                tensor = tensor.detach().requires_grad_(True)
            fwd_args[i] = tensor

        # Full input list (after double-detach), parallel to detached_outs of the
        # predecessor FWD/FWD_A2A.  Stored so that BWD can build a full-length
        # inp_grads list (None for non-grad entries) that BWD_A2A can index with
        # a2a_tensor_idx without any requires_grad filtering offset.
        fwd_inputs = [fwd_args[i] for i in input_idxs]

        inp_with_grad = [t for t in fwd_inputs if t is not None and t.requires_grad]

        if ubid not in self._logged_bucket_forward_info:
            self._logged_bucket_forward_info.add(ubid)
            self.logger.log(VERBOSE,
                f"[ac_forward] rank={self.global_rank} ubid={ubid}: "
                f"ac_subgraphs={self.bucket_ac_subgraph_counts.get(ubid, 1)} "
                f"inputs={len(input_idxs)}"
            )

        # Run the forward function.
        with torch.cuda.stream(compute_stream):
            output = fwd_fn(fwd_args)

        # Clear input slots so we don't hold stale tensor references.
        for i in input_idxs:
            fwd_args[i] = None

        out_list = list(output) if isinstance(output, tuple) else [output]

        # Is this the last bucket of its stage?  Determined from DAG structure in
        # run_dag, but here we use a simpler proxy: if the bucket has an A2A
        # boundary entry it is NOT the last bucket.  More precisely, run_dag
        # calls us with correct inputs; we always produce both boundary and
        # last-bucket fields and let run_dag consume what's relevant.
        possibly_detached = [
            t.detach().requires_grad_(True) if isinstance(t, torch.Tensor) and t.requires_grad else t
            for t in out_list
        ]
        out_with_grad = [t for t in out_list if isinstance(t, torch.Tensor) and t.requires_grad]

        return {
            "pre_detach_outs": out_list,
            "detached_outs": possibly_detached,
            "out_with_grad": out_with_grad,
            "send_output": output,    # raw output for SEND node
            "inp_with_grad": inp_with_grad,
            "fwd_inputs": fwd_inputs,
        }

    def _backward_dag(
        self,
        ubid: int,
        mb_idx: int,
        outputs_or_loss: list,
        upstream_grads,
        pre_detach_outs,
        detached_outs,
        inp_with_grad,
        out_with_grad,
        compute_stream,
    ) -> dict | None:
        """Fused backward pass (BWD) for a single bucket.

        Handles both:
        - Last-bucket-of-stage case: backward from out_with_grad / loss.
        - Non-last-bucket case: backward through the (pre_detach, detached) boundary.

        Returns a dict with ``"send_output"`` (input grad list) when this is the
        first-bucket-of-stage and the stage has a predecessor, or None otherwise.
        """
        if pre_detach_outs is None:
            # Last bucket (or only bucket): backward from stage outputs or loss.
            if upstream_grads is not None:
                with torch.cuda.stream(compute_stream):
                    torch.autograd.backward(outputs_or_loss, upstream_grads)
            else:
                with torch.cuda.stream(compute_stream):
                    outputs_or_loss[0].backward()
        else:
            # Non-last bucket: propagate through the bucket boundary.
            bwd_pairs = [
                (p, d.grad)
                for p, d in zip(pre_detach_outs, detached_outs)
                if (isinstance(d, torch.Tensor) and d.requires_grad and d.grad is not None)
            ]
            assert bwd_pairs, (
                f"BWD ubid={ubid} mb={mb_idx}: detached boundary has no tensors "
                "with a materialized grad"
            )
            if bwd_pairs:
                outputs_bwd = [p for p, _g in bwd_pairs]
                grads_bwd = [g for _p, g in bwd_pairs]
                with torch.cuda.stream(compute_stream):
                    torch.autograd.backward(outputs_bwd, grads_bwd)

        # Collect stage-input grads for SEND — applies to both last-bucket and
        # non-last-bucket paths (e.g. when downstream of BWD_A2A).
        if inp_with_grad:
            output_grads = [t.grad for t in inp_with_grad if t.grad is not None]
            if output_grads:
                return {"send_output": output_grads}
        return None

    def _bucket_backward_input(
        self,
        stage_outputs_or_loss: list,
        output_grads,
        input_values: list,
        weights,
    ):
        """Compute input gradients for a single bucket (BWD_I).

        Matches ``torch.distributed.pipelining.stage_backward_input`` exactly,
        with "stage" replaced by "bucket".  Returns ``(dinputs, param_groups)``.
        """
        from .backward_utils import get_param_groups, construct_reverse_graph, _get_grad_fn_or_grad_acc

        stage_output_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, stage_outputs_or_loss)))
        stage_input_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, input_values)))
        weight_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, weights)))

        reverse_edges_dict = construct_reverse_graph(stage_output_grad_fns)
        param_groups = get_param_groups(stage_input_grad_fns, weight_grad_fns, reverse_edges_dict)

        handles = []
        for param_group in param_groups:
            for i, intermediate in enumerate(param_group["intermediates"]):
                def get_hook(pg, idx):
                    def hook(grad_inputs):
                        if pg.get("grads") is None:
                            pg["grads"] = [None] * len(pg["intermediates"])
                        pg["grads"][idx] = tuple(grad_inputs)
                    return hook
                handles.append(intermediate.register_prehook(get_hook(param_group, i)))

        if output_grads is None:
            output_grads = [torch.ones_like(o) for o in stage_outputs_or_loss]

        input_values = [inp for inp in input_values if inp.requires_grad]
        if input_values:
            dinputs = torch.autograd.grad(
                stage_outputs_or_loss,
                inputs=input_values,
                grad_outputs=output_grads,
                retain_graph=True,
                allow_unused=True,
            )
            for inp, dinput in zip(input_values, dinputs):
                if inp.grad is None:
                    inp.grad = dinput
                else:
                    inp.grad += dinput
        else:
            dinputs = ()

        # Match torch.distributed.pipelining.stage_backward_input(): once dinputs
        # are computed, these outputs are no longer needed by the later weight
        # pass, so detach them to let autograd release the dead part of the graph.
        for i, t in enumerate(stage_outputs_or_loss):
            if not isinstance(t, torch.Tensor):
                continue
            try:
                t.detach_()
            except RuntimeError:
                # Some tensors may be views; replace the list entry so the original
                # graph-carrying tensor can be released once local refs are dropped.
                stage_outputs_or_loss[i] = t.detach()

        for handle in handles:
            handle.remove()

        return dinputs, param_groups

    def _bucket_backward_weight(self, weights, param_groups: list, ubid: int | None = None, mb_idx: int | None = None) -> None:
        """Compute weight gradients for a single bucket (BWD_W).

        Matches ``torch.distributed.pipelining.stage_backward_weight`` exactly,
        with "stage" replaced by "bucket".  Accumulates gradients into param.grad.
        """
        grad_acc_to_weight: Dict[Node, Parameter] = {}
        for weight in weights:
            grad_acc = _get_grad_fn_or_grad_acc(weight)
            grad_acc_to_weight[grad_acc] = weight

        for param_group in param_groups:
            valid_edges: List[GradientEdge] = []
            valid_grad_outputs: List[torch.Tensor] = []

            for grads_tuple, intermediate in zip(
                param_group.get("grads", []), param_group["intermediates"]
            ):
                if grads_tuple is None:
                    continue
                for i, grad in enumerate(grads_tuple):
                    if grad is not None:
                        valid_edges.append(GradientEdge(intermediate, i))
                        valid_grad_outputs.append(grad)

            del param_group["intermediates"]

            if not valid_edges:
                continue

            weight_edges = tuple(GradientEdge(w, 0) for w in param_group["params"])
            if not weight_edges:
                continue

            try:
                dweights = torch.autograd.grad(
                    valid_edges,
                    weight_edges,
                    grad_outputs=valid_grad_outputs,
                    retain_graph=False,
                )
            except torch.OutOfMemoryError:
                self.logger.error(
                    f"[bwd_w_oom] rank={self.global_rank} ubid={ubid} mb={mb_idx}: "
                    f"group_params={len(param_group['params'])} "
                    f"group_valid_edges={len(valid_edges)} "
                    f"group_grad_outputs={len(valid_grad_outputs)} "
                    f"alloc_gb={torch.cuda.memory_allocated(self.device) / (1024 ** 3):.2f} "
                    f"reserved_gb={torch.cuda.memory_reserved(self.device) / (1024 ** 3):.2f}"
                )
                raise

            del param_group["grads"]

            for grad_acc, dw in zip(param_group["params"], dweights):
                if dw is None or grad_acc not in grad_acc_to_weight:
                    continue
                weight = grad_acc_to_weight[grad_acc]
                if weight.grad is None:
                    weight.grad = dw
                else:
                    weight.grad += dw

    def _exec_all_reduce_grads(self, ubid: int) -> int:
        """All-reduce each materialized parameter grad for *ubid* on ar_stream.

        Returns the total payload bytes actually dispatched at runtime.
        """
        if self.zero_stage == 1 and ubid in self.bucket_flat_grads:
            flat_grads = self.bucket_flat_grads.get(ubid)
            if flat_grads is None:
                self.logger.log(VERBOSE,
                    f"[all_reduce_bucket] rank={self.global_rank} ubid={ubid}: "
                    "skipped no ZeRO flat grad buffer"
                )
                return 0
            total_bytes = flat_grads.numel() * flat_grads.element_size()
            self.logger.log(VERBOSE,
                f"[collective_payload] op=all_reduce rank={self.global_rank} "
                f"ubid={ubid} mode=flat numel={flat_grads.numel()} "
                f"element_size={flat_grads.element_size()} dtype={flat_grads.dtype} "
                f"payload_bytes={total_bytes}"
            )
            with torch.cuda.stream(self.ar_stream):
                dist.all_reduce(flat_grads, group=self.dp_group)
            return total_bytes

        args = self.bucket_fwd_args.get(ubid, [])
        trainable_idxs = self.bucket_trainable_param_idxs.get(ubid, [])
        grad_tensors = []
        for idx in trainable_idxs:
            param = args[idx]
            if param is not None and param.grad is not None:
                grad_tensors.append((idx, param.grad))

        if not grad_tensors:
            self.logger.log(VERBOSE,
                f"[all_reduce_bucket] rank={self.global_rank} ubid={ubid}: "
                "skipped no materialized grads"
            )
            return 0

        total_numel = sum(grad.numel() for _, grad in grad_tensors)
        total_bytes = sum(grad.numel() * grad.element_size() for _, grad in grad_tensors)
        largest_idx, largest_grad = max(grad_tensors, key=lambda item: item[1].numel())
        largest_bytes = largest_grad.numel() * largest_grad.element_size()
        dtype_summary = ",".join(sorted({str(grad.dtype) for _, grad in grad_tensors}))
        self.logger.log(VERBOSE,
            f"[collective_payload] op=all_reduce rank={self.global_rank} "
            f"ubid={ubid} mode=per_param grad_tensors={len(grad_tensors)} "
            f"numel={total_numel} payload_bytes={total_bytes} dtypes={dtype_summary}"
        )
        self.logger.log(VERBOSE,
            f"[all_reduce_bucket] rank={self.global_rank} ubid={ubid}: "
            f"mode=per_param grad_tensors={len(grad_tensors)} "
            f"grad_numel={total_numel} "
            f"grad_gb={total_bytes / (1024 ** 3):.4f} "
            f"largest_idx={largest_idx} "
            f"largest_numel={largest_grad.numel()} "
            f"largest_gb={largest_bytes / (1024 ** 3):.4f} "
            f"grad_dtype={largest_grad.dtype} "
            f"dp_degree={self.dp_degree}"
        )
        for grad_order, (idx, grad) in enumerate(grad_tensors):
            grad_bytes = grad.numel() * grad.element_size()
            data_ptr_mod_256 = grad.data_ptr() % 256 if grad.is_cuda else -1
            self.logger.log(VERBOSE,
                f"[all_reduce_grad] rank={self.global_rank} ubid={ubid} "
                f"order={grad_order} idx={idx}: "
                f"shape={tuple(grad.shape)} stride={tuple(grad.stride())} "
                f"numel={grad.numel()} gb={grad_bytes / (1024 ** 3):.4f} "
                f"dtype={grad.dtype} device={grad.device} "
                f"contiguous={grad.is_contiguous()} "
                f"storage_offset={grad.storage_offset()} "
                f"data_ptr_mod_256={data_ptr_mod_256}"
            )
        with torch.cuda.stream(self.ar_stream):
            for _, grad in grad_tensors:
                dist.all_reduce(grad, group=self.dp_group)
        return total_bytes

    def _alloc_full_params(self, ubid: int | None) -> None:
        if self.zero_stage != 3:
            return
        if ubid is None:
            return
        shard_info = self.param_shard_info.get(ubid)
        if shard_info is None:
            return
        _shard_start, shard_size, _orig_numel = shard_info
        full = self.bucket_flat_params.get(ubid)
        if full is None:
            full = self.bucket_shard_params[ubid].new_empty(shard_size * self.dp_degree)
            self.bucket_flat_params[ubid] = full
        storage = full.untyped_storage()
        required_bytes = full.numel() * full.element_size()
        if storage.size() != required_bytes:
            storage.resize_(required_bytes)
        self.logger.debug(
            f"[zero3_alloc_full_params] rank={self.global_rank} ubid={ubid}: "
            f"numel={full.numel()} required_bytes={required_bytes} "
            f"storage_size={storage.size()} fresh={self.zero3_full_params_fresh.get(ubid, False)}"
        )
        for param, offset, numel, shape in self.bucket_param_view_specs.get(ubid, []):
            param.data = full[offset:offset + numel].view(shape)
            param.requires_grad_(True)
        param_names = self.bucket_param_names.get(ubid, [])
        zero_storage = []
        for i, (param, offset, numel, shape) in enumerate(self.bucket_param_view_specs.get(ubid, [])):
            p_storage = param.untyped_storage()
            if p_storage.size() == 0:
                name = param_names[i] if i < len(param_names) else f"param{i}"
                zero_storage.append(
                    f"{name}: offset={offset} numel={numel} shape={shape} stride={tuple(param.stride())}"
                )
        if zero_storage:
            self.logger.error(
                f"[zero3_alloc_full_params_zero_storage] rank={self.global_rank} ubid={ubid}: "
                + " | ".join(zero_storage)
            )

    def _free_full_params(self, ubid: int | None) -> None:
        if self.zero_stage != 3:
            return
        full = self.bucket_flat_params.get(ubid)
        if full is None:
            return
        storage = full.untyped_storage()
        self.logger.debug(
            f"[zero3_free_full_params] rank={self.global_rank} ubid={ubid}: "
            f"storage_size_before={storage.size()} fresh_before={self.zero3_full_params_fresh.get(ubid, False)}"
        )
        if storage.size() != 0:
            storage.resize_(0)
        self.zero3_full_params_fresh[ubid] = False

    def _alloc_full_grads(self, ubid: int | None) -> None:
        specs = self.bucket_param_view_specs.get(ubid, [])
        assert specs, f"Expected param view specs for ubid={ubid}"
        flat_grads = self.bucket_flat_grads.get(ubid)
        if flat_grads is None:
            shard_info = self.param_shard_info.get(ubid)
            if shard_info is None:
                return
            _shard_start, shard_size, _orig_numel = shard_info
            flat_grads = torch.zeros(
                shard_size * self.dp_degree,
                dtype=self.zero_grad_dtype,
                device=self.device,
            )
            self.bucket_flat_grads[ubid] = flat_grads
        else:
            flat_grads.zero_()
        if self.zero_stage in (2, 3) and self.bucket_rs_grads.get(ubid) is None:
            shard_info = self.param_shard_info.get(ubid)
            if shard_info is None:
                return
            _shard_start, shard_size, _orig_numel = shard_info
            self.bucket_rs_grads[ubid] = torch.zeros(
                shard_size, dtype=self.zero_grad_dtype, device=self.device
            )

    def _free_full_grads(self, ubid: int | None) -> None:
        for param, *_ in self.bucket_param_view_specs.get(ubid, []):
            param.grad = None
        if self.zero_stage in (2, 3):
            self.bucket_flat_grads[ubid] = None

    def _exec_reduce_scatter(self, ubid: int) -> int:
        with torch.cuda.stream(self.ar_stream):
            flat_grads = self.bucket_flat_grads.get(ubid)
            _shard_start, shard_size, _orig_numel = self.param_shard_info.get(ubid)
            # add logging here
            rs_out = self.bucket_rs_grads.get(ubid)
            if flat_grads is None or rs_out is None:
                if flat_grads is None:
                    self.logger.warning(f"[reduce_scatter_alloc] rank={self.global_rank} ubid={ubid}: Found no flat grad buffer")
                    flat_grads = torch.zeros(
                        shard_size * self.dp_degree,
                        dtype=self.zero_grad_dtype,
                        device=self.device,
                    )
                    self.bucket_flat_grads[ubid] = flat_grads
                if rs_out is None:
                    rs_out = torch.zeros(
                        shard_size, dtype=self.zero_grad_dtype, device=self.device
                    )
                    self.bucket_rs_grads[ubid] = rs_out
            assert flat_grads is not None and rs_out is not None, (
                f"REDUCE_SCATTER dispatched for ubid={ubid} without allocated gradient buffers"
            )
            total_bytes = flat_grads.numel() * flat_grads.element_size()
            self.logger.log(VERBOSE,
                f"[collective_payload] op=reduce_scatter rank={self.global_rank} "
                f"ubid={ubid} input_numel={flat_grads.numel()} "
                f"input_element_size={flat_grads.element_size()} input_dtype={flat_grads.dtype} "
                f"payload_bytes={total_bytes} output_numel={rs_out.numel()} "
                f"output_element_size={rs_out.element_size()} output_dtype={rs_out.dtype}"
            )
            tmp = torch.empty_like(rs_out)
            flat_grads[_shard_start:_shard_start + shard_size].add_(rs_out)
            dist.reduce_scatter_tensor(rs_out, flat_grads, group=self.dp_group)
            rs_out.add_(tmp)
            return total_bytes

    def _exec_all_gather(
        self,
        ubid: int | None,
    ):
        if ubid is None:
            return False
        with torch.cuda.stream(self.ar_stream):
            flat_params = self.bucket_flat_params.get(ubid)
            shard_in = self.bucket_shard_params.get(ubid)
            if flat_params is None or shard_in is None:
                return False
            if self.zero_stage == 3 and self.zero3_full_params_fresh.get(ubid, False):
                self.logger.debug(
                    f"[zero3_all_gather_skip] rank={self.global_rank} ubid={ubid}: "
                    f"storage_size={flat_params.untyped_storage().size()} fresh=True"
                )
                return False
            self.logger.debug(
                f"[zero3_all_gather_begin] rank={self.global_rank} ubid={ubid}: "
                f"flat_numel={flat_params.numel()} flat_storage={flat_params.untyped_storage().size()} "
                f"shard_numel={shard_in.numel()} shard_storage={shard_in.untyped_storage().size()} "
                f"fresh={self.zero3_full_params_fresh.get(ubid, False)}"
            )
            dist.all_gather_into_tensor(flat_params, shard_in, group=self.dp_group)
            if self.zero_stage == 3:
                self.zero3_full_params_fresh[ubid] = True
            self.logger.debug(
                f"[zero3_all_gather_end] rank={self.global_rank} ubid={ubid}: "
                f"flat_storage={flat_params.untyped_storage().size()} fresh={self.zero3_full_params_fresh.get(ubid, False)}"
            )
        return True

    def _update(self, *deps):
        if self.zero_stage > 0:
            sync_events = self.ar_events if self.zero_stage == 1 else self.rs_events
            for evt in sync_events.values():
                self.comp_stream.wait_event(evt)

            for ubid, shard_optim in self.bucket_shard_optims.items():
                shard_param = self.bucket_shard_params[ubid]
                shard_start, shard_size, _orig_numel = self.param_shard_info[ubid]
                if self.zero_stage == 1:
                    flat_grads = self.bucket_flat_grads.get(ubid)
                    if flat_grads is None:
                        continue
                    shard_param.grad = flat_grads[shard_start:shard_start + shard_size].to(shard_param.dtype)
                else:
                    rs_grad = self.bucket_rs_grads.get(ubid)
                    if rs_grad is None:
                        continue
                    shard_param.grad = rs_grad.to(shard_param.dtype)
                with torch.cuda.stream(self.comp_stream):
                    shard_optim.step()
                shard_param.grad = None

            if self.zero_stage == 3:
                for ubid, specs in self.bucket_param_view_specs.items():
                    for param, *_ in specs:
                        param.grad = None
                    full = self.bucket_flat_params.get(ubid)
                    if full is not None:
                        storage = full.untyped_storage()
                        if storage.size() != 0:
                            storage.resize_(0)
                    self.zero3_full_params_fresh[ubid] = False

            losses = self.loss
            self.loss.clear()
            torch.cuda.synchronize()
            return losses

        pre_step_mem: dict = {}
        grad_debug: dict = {}

        for ar_evt in self.ar_events.values():
            self.comp_stream.wait_event(ar_evt)

        for ubid, optim in self.bucket_optims.items():
            if optim is None:
                continue
            bwd_evt = self.bwd_events.get(ubid)
            if bwd_evt is not None:
                self.comp_stream.wait_event(bwd_evt)

            params = self.bucket_fwd_args.get(ubid, [])
            trainable_idxs = self.bucket_trainable_param_idxs.get(ubid, [])

            params_with_grad = 0
            total_trainable_params = 0
            total_grad_elems = 0
            for idx in trainable_idxs:
                p = params[idx]
                if p is None:
                    continue
                total_trainable_params += p.numel()
                if p.grad is not None:
                    params_with_grad += 1
                    total_grad_elems += p.grad.numel()

            # Exact zero-gradient detection requires scanning each grad tensor and
            # can allocate a large temporary near peak memory. Treat materialized
            # grads as active here; structural sparse expert metadata would be
            # needed to preserve skip-zero semantics without a dense value scan.
            nonzero_grad_params = params_with_grad
            nonzero_grad_elems = total_grad_elems
            zero_grad_params = 0

            state_entries_before = sum(len(optim.state[p]) > 0 for p in optim.param_groups[0]["params"])
            pre_step_mem[ubid] = (
                torch.cuda.memory_allocated(self.device),
                torch.cuda.memory_reserved(self.device),
            )
            grad_debug[ubid] = {
                "num_params": len(trainable_idxs),
                "params_with_grad": params_with_grad,
                "nonzero_grad_params": nonzero_grad_params,
                "total_trainable_params": total_trainable_params,
                "total_grad_elems": total_grad_elems,
                "nonzero_grad_elems": nonzero_grad_elems,
                "zero_grad_params": zero_grad_params,
                "state_entries_before": state_entries_before,
            }
            self.logger.log(VERBOSE,
                f"[optim_debug_pre] rank={self.global_rank} ubid={ubid}: "
                f"params={len(trainable_idxs)} "
                f"params_with_grad={params_with_grad} "
                f"nonzero_grad_params={nonzero_grad_params} "
                f"zero_grad_params={zero_grad_params} "
                f"trainable_elems={total_trainable_params} "
                f"grad_elems={total_grad_elems} "
                f"nonzero_grad_elems={nonzero_grad_elems} "
                f"state_entries_before={state_entries_before} "
                f"alloc_before_gb={pre_step_mem[ubid][0] / (1024 ** 3):.2f} "
                f"reserved_before_gb={pre_step_mem[ubid][1] / (1024 ** 3):.2f}"
            )
            try:
                with torch.cuda.stream(self.comp_stream):
                    optim.step()
            except torch.OutOfMemoryError:
                alloc_now = torch.cuda.memory_allocated(self.device)
                reserv_now = torch.cuda.memory_reserved(self.device)
                self.logger.error(
                    f"[optim_debug_oom] rank={self.global_rank} ubid={ubid}: "
                    f"state_entries_before={state_entries_before} "
                    f"alloc_now_gb={alloc_now / (1024 ** 3):.2f} "
                    f"reserved_now_gb={reserv_now / (1024 ** 3):.2f}"
                )
                raise

        losses = self.loss
        self.loss.clear()

        torch.cuda.synchronize()

        for ubid, optim in self.bucket_optims.items():
            if optim is None or ubid not in grad_debug:
                continue
            alloc_before, reserv_before = pre_step_mem[ubid]
            alloc_after = torch.cuda.memory_allocated(self.device)
            reserv_after = torch.cuda.memory_reserved(self.device)
            state_entries_after = sum(len(optim.state[p]) > 0 for p in optim.param_groups[0]["params"])
            dbg = grad_debug[ubid]
            self.logger.log(VERBOSE,
                f"[optim_debug] rank={self.global_rank} ubid={ubid}: "
                f"params={dbg['num_params']} "
                f"params_with_grad={dbg['params_with_grad']} "
                f"nonzero_grad_params={dbg['nonzero_grad_params']} "
                f"zero_grad_params={dbg['zero_grad_params']} "
                f"trainable_elems={dbg['total_trainable_params']} "
                f"grad_elems={dbg['total_grad_elems']} "
                f"nonzero_grad_elems={dbg['nonzero_grad_elems']} "
                f"state_entries_before={dbg['state_entries_before']} "
                f"state_entries_after={state_entries_after} "
                f"alloc_before_gb={alloc_before / (1024 ** 3):.2f} "
                f"alloc_after_gb={alloc_after / (1024 ** 3):.2f} "
                f"reserved_before_gb={reserv_before / (1024 ** 3):.2f} "
                f"reserved_after_gb={reserv_after / (1024 ** 3):.2f}"
            )

        # Process deferred profiling records now that all GPU work is complete.
        # elapsed_time() is safe to call after synchronize().
        if self._prof_records:
            for node, mb_idx, ubid, start_evt, end_evt, mem_before in self._prof_records:
                t_ms = start_evt.elapsed_time(end_evt)
                node.profiling_measurements.append(t_ms)
                self.logger.info(
                    f"[profiling] rank {self.global_rank} "
                    f"{node.task_type.value} mb{mb_idx} ubid{ubid} "
                    f"(time step {node.time_step}): time={t_ms:.3f}ms "
                    f"mem_before={mem_before/1e9:.3f}GB"
                )

        return {
            "losses": losses,
        }
