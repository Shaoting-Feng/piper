import ray
import torch
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import gc
import threading
from torch.nn import Parameter
from torch.autograd.graph import GradientEdge, Node, set_warn_on_accumulate_grad_stream_mismatch
import torch.distributed as dist
from collections import defaultdict

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .piper_utils import (
    _deserialize_graphmodule,
    create_logger,
    LOG_LEVEL,
    NcclOverlapDetector,
)
from .backward_utils import get_param_groups, construct_reverse_graph, _get_grad_fn_or_grad_acc
from .piper_exec import TaskType, TaskDAG, TaskNode

CLEANUP_MEMORY = True

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
            num_gpus=0.8,
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
        stage_to_device=None,
    ):
        self.logger = create_logger("piper_actor", LOG_LEVEL)

        # BWD_I uses torch.autograd.grad (not .backward), so AccumulateGrad nodes are
        # traversed for stream-sync bookkeeping but never accumulate to p.grad.
        # Suppress the spurious stream-mismatch warning.
        set_warn_on_accumulate_grad_stream_mismatch(False)

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
        self.comm_stream = torch.cuda.Stream()
        self.a2a_stream = torch.cuda.Stream()
        self.p2p_send_stream = torch.cuda.Stream()
        self.p2p_recv_stream = torch.cuda.Stream()

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
        # accumuate loss for each microbatch
        self.loss = []

        self.tracing = False
        self._pending_timing_events: list = []  # (label, start_event, stop_event)
        self.trace_data = defaultdict(list)
        self.memory_tracing_enabled = False

        # DAG execution state
        self.dag = None
        self.send_buffer: dict = {}  # ((stage_id, bucket_id_or_None), mb_idx) -> tensor(s); or (None, mb_idx) for BWD
        self.recv_buffer: dict = {}  # same key format as send_buffer -> tensor(s)
        self.recv_events: dict = {}  # same key format as send_buffer -> cuda.Event
        self.bucket_buffer: dict = {}  # (stage_id, mb_idx, bucket_id) -> (pre_detach_outs, detached_outs)
        # A2A boundaries per stage: stage_id -> {boundary_bucket_id -> tensor_idx}
        self.a2a_boundaries: dict = {}
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

        # Split backward 
        self.bw_param_groups = defaultdict(dict)  # stage_id -> mb_idx -> param group for split backward pass
        self.bw_grad_cache = defaultdict(dict)  # stage_id -> mb_idx -> cached activations between BWD_I and BWD_W passes
        self.upstream_grad_cache = defaultdict(dict)  # stage_id -> mb_idx -> cached upstream grads between BWD_I and BWD_W passes

        from .piper_utils import piper_metadata
        piper_metadata.actor_self = self

    def get_trace_data(self) -> dict:
        return self.global_rank, self.trace_data

    def clear_trace_data(self) -> None:
        self.trace_data.clear()
        self._pending_timing_events.clear()

    def set_tracing(self, enabled: bool) -> None:
        self.tracing = enabled
        self.logger.info(
            f"Actor {self.global_rank}: Tracing {'enabled' if enabled else 'disabled'}"
        )

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()

    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(
            f"/m-coriander/coriander/shubham/moe-scheduling/piper_profiling/actor{self.global_rank}_memory_snapshot_mb4_gpipe.pickle"
        )
        self.logger.info(
            f"Saved memory snapshot to actor{self.global_rank}_memory_snapshot_mb4_gpipe.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)
    
    def enable_memory_tracing(self, enabled: bool = True) -> None:
        """Enable/disable memory tracing."""
        self.memory_tracing_enabled = enabled
        if enabled:
            torch.cuda.memory._record_memory_history()
            self.logger.info(f"Actor {self.global_rank}: Memory tracing enabled")
        else:
            if hasattr(self, 'memory_tracing_enabled') and self.memory_tracing_enabled:
                # Dump snapshot before disabling
                snapshot_path = f"actor{self.global_rank}_iter_memory.pickle"
                os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                torch.cuda.memory._dump_snapshot(snapshot_path)
                self.logger.info(f"Saved memory snapshot to {snapshot_path}")
            torch.cuda.memory._record_memory_history(enabled=None)
            self.memory_tracing_enabled = False
            self.logger.info(f"Actor {self.global_rank}: Memory tracing disabled")

    def dump_memory_snapshot(self, filename: str = None) -> str:
        """Dump current memory snapshot. Returns snapshot path."""
        if not (hasattr(self, 'memory_tracing_enabled') and self.memory_tracing_enabled):
            self.logger.warning("Memory tracing not enabled, cannot dump snapshot")
            return None
        
        if filename is None:
            filename = f"actor{self.global_rank}_iter_memory.pickle"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.cuda.memory._dump_snapshot(filename)
        self.logger.info(f"Saved memory snapshot to {filename}")
        return filename

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
        init_method = f"tcp://{master_addr}:{master_port}"

        self.logger.info(f"Actor {self.global_rank} using GPU {os.environ['CUDA_VISIBLE_DEVICES']}, master addr {master_addr}:{master_port}")

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

        # Force cuBLAS context initialization on comp_stream so the first
        # backward pass does not encounter a "no current CUDA context" warning.
        with torch.cuda.stream(self.comp_stream):
            _w = torch.zeros(4, 4, device=self.device)
            torch.mm(_w, _w)
        torch.cuda.synchronize()

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

        for pp_group_id in range(num_pp_groups):
            group_ranks = [
                (pp_group_id * self.pp_degree + i) for i in range(self.pp_degree)
            ]
            lo_hi_group = dist.new_group(ranks=group_ranks, backend="nccl")
            hi_lo_group = dist.new_group(ranks=group_ranks, backend="nccl")
            
            if self.global_rank in group_ranks:
                self.pp_lo_hi = lo_hi_group
                self.pp_hi_lo = hi_lo_group
            
            # for i in range(len(group_ranks)):
            #     for j in range(i + 1, len(group_ranks)):
            #         rank_lo, rank_hi = group_ranks[i], group_ranks[j]
            #         pg_lo_to_hi = dist.new_group(ranks=[rank_lo, rank_hi], backend="nccl")
            #         pg_hi_to_lo = dist.new_group(ranks=[rank_lo, rank_hi], backend="nccl")
            #         if self.global_rank in (rank_lo, rank_hi):
            #             self.logger.debug(f"Global rank {self.global_rank} saving pp communicators for ranks {[rank_lo, rank_hi]}")
            #             self.pp_groups[(rank_lo, rank_hi)] = pg_lo_to_hi
            #             self.pp_groups[(rank_hi, rank_lo)] = pg_hi_to_lo

        self.logger.info(f"Global rank {self.global_rank} warmed up pp communicators")

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
                    logger.debug(f"Replacing parameter '{full_name}' from meta/fake to {device} (shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad})")
                    new_param = torch.empty_like(param, device=device)
                    new_param.requires_grad_(param.requires_grad)
                    setattr(module, name, torch.nn.Parameter(new_param, requires_grad=param.requires_grad))
                    params_count += 1
                    replaced_count += 1
            
            # replace all buffers that are on meta device or FakeTensor
            for name, buffer in module.named_buffers(recurse=False):
                if is_meta_or_fake(buffer):
                    full_name = f"{module_name}.{name}" if module_name else name
                    logger.debug(f"Replacing buffer '{full_name}' from meta/fake to {device} (shape={buffer.shape}, dtype={buffer.dtype})")
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
                                logger.debug(f"Replacing get_attr '{full_name}' from meta/fake to {device} (shape={attr_value.shape}, dtype={attr_value.dtype})")
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
                                logger.debug(f"Replacing node.meta['{key}'] for node '{node.name}' from meta/fake to {device} (shape={value.shape}, dtype={value.dtype})")
                                node.meta[key] = torch.empty_like(value, device=device)
                                node_meta_count += 1
                                replaced_count += 1
                
                # replace in constants dict if it exists
                if hasattr(module, '_constants'):
                    for key, value in module._constants.items():
                        if is_meta_or_fake(value):
                            full_name = f"{module_name}._constants['{key}']" if module_name else f"_constants['{key}']"
                            logger.debug(f"Replacing constant '{full_name}' from meta/fake to {device} (shape={value.shape}, dtype={value.dtype})")
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
                            logger.debug(f"Replacing meta/fake tensor in node '{full_name}' args (shape={arg.shape}, dtype={arg.dtype})")
                            new_args.append(torch.empty_like(arg, device=device))
                            node_args_count += 1
                            replaced_count += 1
                        elif isinstance(arg, torch.device) and arg.type == 'meta':
                            # replace meta device objects with the actual device
                            full_name = f"{module_name}.{node.name}" if module_name else node.name
                            logger.debug(f"Replacing meta device object in node '{full_name}' args with {device}")
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
                            logger.debug(f"Replacing meta/fake tensor in node '{full_name}' kwargs['{key}'] (shape={value.shape}, dtype={value.dtype})")
                            new_kwargs[key] = torch.empty_like(value, device=device)
                            node_args_count += 1
                            replaced_count += 1
                        elif isinstance(value, torch.device) and value.type == 'meta':
                            # replace meta device objects with the actual device
                            full_name = f"{module_name}.{node.name}" if module_name else node.name
                            logger.debug(f"Replacing meta device object in node '{full_name}' kwargs['{key}'] with {device}")
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
            logger.debug(
                f"_replace_meta_constants: Replaced {replaced_count} meta device tensors "
                f"(params: {params_count}, buffers: {buffers_count}, get_attr: {get_attr_count}, "
                f"constants: {constants_count}, attrs: {attrs_count}, node_args: {node_args_count}, "
                f"node_meta: {node_meta_count}, checked {submodule_count} submodules)"
            )
        else:
            logger.debug(f"_replace_meta_constants: No meta device tensors found to replace! (checked {submodule_count} submodules)")
        
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
            self.logger.debug(f"Actor {self.global_rank} imported and initialized Triton kernel modules")
        except ImportError as e:
            self.logger.warning(
                f"Actor {self.global_rank} failed to import Triton kernel modules: {e}. "
                f"This may cause issues with Triton kernels in the graph."
            )

        # HACK: manually register Triton kernel in the side table
        # the graph expects kernel_idx=0, so we need to register it at that index
        self.logger.debug(f"Manually registering Triton kernel at index 0 on actor {self.global_rank}...")
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
                    self.logger.debug(
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

        self._load_stage_kernels()

        # Restore triton kernel constant_args that were captured at compile time.
        # Each bucket may reference specific constant_args_idx values baked into the
        # compiled graph; we restore them at the exact indices so the graph runs correctly.
        try:
            from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
            for bd in modules_data:
                for idx, args in bd.get("triton_constant_args", {}).items():
                    kernel_side_table.constant_args[int(idx)] = args
        except Exception as e:
            self.logger.warning(f"Actor {self.global_rank} failed to restore triton constant_args: {e}")

        first_gm = None
        last_gm = None

        for b_idx, bd in enumerate(modules_data):
            gm = _deserialize_graphmodule(bd["gm_data"])
            gm = self._replace_meta_constants(gm, self.device)
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

    # -----------------------------------------------------------------------
    # DAG-based execution
    # -----------------------------------------------------------------------

    def load_dag(self, dag: TaskDAG) -> None:
        """Store the per-rank TaskDAG for subsequent run_dag() calls."""
        self.dag = dag

    def get_all_params(self) -> dict:
        """Return {(stage_id, bucket_id): flat_cpu_tensor} for every trainable bucket."""
        return {
            key: tensor.detach().cpu().clone()
            for key, tensor in self.bucket_flat_params.items()
            if tensor is not None
        }

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

        self.recv_buffer = {}
        self.recv_events = {}
        comp_events: dict = {}  # (stage_id, mb_idx, task_type, bucket_id) -> cuda.Event

        # Sort nodes by time_step.  Nodes that share a time_step run on different
        # CUDA streams (e.g. a RECV alongside the preceding compute task), so
        # having two nodes of the same type at the same time_step would indicate
        # a scheduling error — assert against it before dispatching.
        sorted_nodes = sorted(dag.nodes, key=lambda n: n.time_step)
        ts_types: dict[int, set] = {}
        for node in sorted_nodes:
            ts, ttype = node.time_step, node.task.type
            assert ttype not in ts_types.get(ts, set()), (
                f"run_dag: time_step={ts} has two nodes of type {ttype}"
            )
            ts_types.setdefault(ts, set()).add(ttype)

        for node in sorted_nodes:
            task = node.task
            # batches always has exactly one entry for single-stage tasks
            batch = task.batches[0]
            stage_id = batch.stage_id
            mb_idx = batch.mb_idx

            self.logger.debug(
                f"run_dag dispatch (time {node.time_step}): {task.type.value} s{stage_id} mb{mb_idx} bkt={node.bucket_id}"
            )

            match task.type:

                case TaskType.SEND:
                    compute_node = node.data_preds[0]
                    _cs = compute_node.task.batches[0].stage_id
                    _cm = compute_node.task.batches[0].mb_idx
                    _ct = compute_node.task.type
                    send_key = (_cs, compute_node.bucket_id) if _ct == TaskType.FWD else None
                    comp_key = (_cs, _cm, _ct, compute_node.bucket_id)
                    self.p2p_send_stream.wait_event(comp_events[comp_key])
                    self._exec_send(stage_id, mb_idx, send_key, node.peer_pp_rank)

                case TaskType.RECV:
                    compute_node = node.data_succs[0]
                    recv_key = (stage_id, compute_node.bucket_id) \
                        if compute_node.task.type == TaskType.FWD else None
                    self._exec_recv(stage_id, mb_idx, recv_key, node.peer_pp_rank)

                case TaskType.FWD_A2A:
                    bucket_id = node.bucket_id
                    torch.cuda.nvtx.range_push(f"fwd_a2a_s{stage_id}_b{bucket_id}_mb{mb_idx}")
                    self.a2a_stream.wait_event(
                        comp_events[(stage_id, mb_idx, TaskType.FWD, bucket_id)]
                    )
                    self._exec_fwd_a2a(stage_id, bucket_id, mb_idx)
                    torch.cuda.nvtx.range_pop()
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(self.a2a_stream)
                    self.a2a_events[(stage_id, mb_idx, TaskType.FWD_A2A, bucket_id)] = a2a_evt
                    self.comp_stream.wait_event(a2a_evt)

                case TaskType.BWD_A2A:
                    bucket_id = node.bucket_id
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
                    self.comp_stream.wait_event(a2a_evt)

                case TaskType.ALL_REDUCE:
                    bucket_id = node.bucket_id
                    bwd_node = node.data_preds[0]
                    bwd_key = (
                        bwd_node.task.batches[0].stage_id,
                        bwd_node.task.batches[0].mb_idx,
                        bwd_node.task.type,
                        bwd_node.bucket_id,
                    )
                    torch.cuda.nvtx.range_push(f"all_reduce_s{stage_id}_b{bucket_id}")
                    self._exec_all_reduce(stage_id, bucket_id, comp_events[bwd_key])
                    torch.cuda.nvtx.range_pop()
            
                case TaskType.FWD:
                    bucket_id = node.bucket_id
                    recv_key = ((stage_id, bucket_id), mb_idx)
                    if recv_key in self.recv_events:
                        self.comp_stream.wait_event(self.recv_events.pop(recv_key))
                    torch.cuda.nvtx.range_push(f"forward_s{stage_id}_b{bucket_id}_mb{mb_idx}")
                    self._start_timing(self.comp_stream, f"forward")
                    self._forward_dag(stage_id, bucket_id, mb_idx)
                    self._stop_timing(self.comp_stream, f"forward")
                    torch.cuda.nvtx.range_pop()
                    evt = torch.cuda.Event()
                    evt.record(self.comp_stream)
                    comp_events[(stage_id, mb_idx, TaskType.FWD, bucket_id)] = evt

                case TaskType.BWD:
                    bucket_id = node.bucket_id
                    recv_key = (None, mb_idx)
                    if recv_key in self.recv_events:
                        self.comp_stream.wait_event(self.recv_events.pop(recv_key))
                    torch.cuda.nvtx.range_push(f"backward_s{stage_id}_b{bucket_id}_mb{mb_idx}")
                    self._start_timing(self.comp_stream, f"backward")
                    self._backward_dag(stage_id, bucket_id, mb_idx, loss_fn=loss_fn)
                    self._stop_timing(self.comp_stream, f"backward")
                    torch.cuda.nvtx.range_pop()
                    evt = torch.cuda.Event()
                    evt.record(self.comp_stream)
                    comp_events[(stage_id, mb_idx, TaskType.BWD, bucket_id)] = evt

                case TaskType.BWD_I:
                    bucket_id = node.bucket_id
                    recv_key = (None, mb_idx)
                    if recv_key in self.recv_events:
                        self.comp_stream.wait_event(self.recv_events.pop(recv_key))
                    torch.cuda.nvtx.range_push(f"backward_input_stage_{stage_id}_b{bucket_id}_mb_{mb_idx}")
                    self._start_timing(self.comp_stream, f"backward_input")
                    self._backward_input_dag(stage_id, bucket_id, mb_idx, loss_fn=loss_fn)
                    self._stop_timing(self.comp_stream, f"backward_input")
                    torch.cuda.nvtx.range_pop()
                    evt = torch.cuda.Event()
                    evt.record(self.comp_stream)
                    comp_events[(stage_id, mb_idx, TaskType.BWD_I, bucket_id)] = evt

                case TaskType.BWD_W:
                    torch.cuda.nvtx.range_push(f"backward_weight_stage_{stage_id}_mb_{mb_idx}")
                    self._start_timing(self.comp_stream, f"backward_weight")
                    self._backward_weight_dag(stage_id, mb_idx, loss_fn=loss_fn)
                    self._stop_timing(self.comp_stream, f"backward_weight")
                    torch.cuda.nvtx.range_pop()

                case TaskType.UPD:
                    torch.cuda.nvtx.range_push("update")
                    self._update()
                    torch.cuda.nvtx.range_pop()

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
            self._start_timing(self.p2p_send_stream, "p2p_send")
            pp_group = self.pp_lo_hi if global_dst_rank > self.global_rank else self.pp_hi_lo
            for tensor in tensors:
                dist.send(
                    tensor,
                    dst=global_dst_rank,
                    # group=self.pp_groups[(self.global_rank, global_dst_rank)],
                    group=pp_group,
                )
            self._stop_timing(self.p2p_send_stream, "p2p_send")

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
            assert act_list is not None, (
                f"BWD recv for stage {stage_id} mb {mb_idx} before FWD has run"
            )
            buf = [torch.empty_like(a) for a in act_list]

        with torch.cuda.stream(self.p2p_recv_stream):
            tensors = buf if isinstance(buf, list) else [buf]
            self._start_timing(self.p2p_recv_stream, "p2p_recv")
            pp_group = self.pp_hi_lo if global_src_rank > self.global_rank else self.pp_lo_hi
            for tensor in tensors:
                dist.recv(
                    tensor,
                    src=global_src_rank,
                    # group=self.pp_groups[(global_src_rank, self.global_rank)],
                    group=pp_group,
                )
            self._stop_timing(self.p2p_recv_stream, "p2p_recv")

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
            self._start_timing(self.a2a_stream, "fwd_a2a")
            dist.all_to_all_single(output_buf, x_detached, group=self.ep_group)
            self._stop_timing(self.a2a_stream, "fwd_a2a")
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
            self._start_timing(self.a2a_stream, "bwd_a2a")
            dist.all_to_all_single(reversed_grad, grad_a2a_out, group=self.ep_group)
            self._stop_timing(self.a2a_stream, "bwd_a2a")
        x_detached.grad = reversed_grad

        # Restore x_detached in bucket_buffer so _backward_dag reads x_detached.grad
        _, detached_outs = self.bucket_buffer[(stage_id, mb_idx, boundary_bucket_id)]
        detached_outs[tensor_idx] = x_detached

    def _backward_input_dag(self, stage_id: int, bucket_id: int, mb_idx: int, *, loss_fn=None) -> None:
        """Backward-input pass for DAG execution (ZeroBubble split backward).

        Unified for both bucketed (A2A) and non-bucketed stages.  Non-bucketed
        stages use bucket_id=0 with n_buckets=1, so the first-bucket and
        last-bucket conditions both fire in the same call — identical to the
        single-pass behaviour of the old non-bucketed path.

        Must be called in reverse bucket order: n_buckets-1, …, 1, 0.
        BWD_A2A must run between consecutive calls at A2A boundaries.
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

        # Stage 0: no prev stage, so no input-grad send.  Cache upstream_grad for BWD_W
        # which runs the full backward for stage 0.
        # When the stage has A2A boundaries we still need to propagate through each
        # intermediate bucket to set x_a2a.grad for the BWD_A2A nodes.  Fall through
        # in that case; bucket 0 always returns early (nothing further to propagate).
        if stage_id == 0:
            if bucket_id == n_buckets - 1:
                ug = upstream_grads[0] if len(upstream_grads) == 1 else upstream_grads
                self.upstream_grad_cache[stage_id][mb_idx] = ug
            stage_a2a = self.a2a_boundaries.get(stage_id, {})
            if not stage_a2a or bucket_id == 0:
                return

        # ---- Step 2: activation inputs (what we compute grad w.r.t.) ----
        if bucket_id > 0:
            prev_bkt = bucket_id - 1
            _, detached_outs_prev = self.bucket_buffer[(stage_id, mb_idx, prev_bkt)]
            # Include all requires_grad outputs of prev_bkt, not just the A2A tensor.
            # _exec_fwd_a2a feeds all detached_outs to the next bucket, so other cross-segment
            # tensors (e.g. top_scores, token_indices) are also real inputs and need gradients.
            activation_inputs = [d for d in detached_outs_prev if d.requires_grad]
        else:
            # Bucket 0: grad w.r.t. stage input
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

        # Free output reference after processing the first-to-bwd bucket.
        # Stage 0 skips this: BWD_W reads out_activation directly for stage 0.
        if bucket_id == n_buckets - 1 and stage_id != 0:
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
        flat_grads = self.bucket_flat_grads.get(lookup_key)
        assert flat_grads is not None, (
            f"ALL_REDUCE dispatched for stage {stage_id} bucket {bucket_id} "
            f"but no flat_grads tensor was allocated — this bucket has no trainable parameters."
        )
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(bwd_event)
            self._start_timing(self.comm_stream, f"all_reduce")
            dist.all_reduce(flat_grads, group=self.dp_group)
            self._stop_timing(self.comm_stream, f"all_reduce")
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
            stage_a2a = self.a2a_boundaries.get(stage_id, {})
            if stage_a2a:
                # Bucketed stage 0 with A2A: out_activation (bkt=last) has no
                # differentiable path to pre-dispatch params because all
                # intermediate bucket inputs are detached leaves.
                # Two-step approach:
                #   Step 1 — ZeroBubble mechanism for intermediate-bucket params
                #             (e.g. expert params) captured by BWD_I hooks.
                #   Step 2 — full backward through bkt=0 using .grad values set
                #             on its detached_outs by BWD_I(bkt=1) and BWD_A2A(bkt=0).

                # Step 1: expert / intermediate-bucket params via GradientEdge
                grad_acc_to_weight_s0: Dict[Node, Parameter] = {}
                for param in stage_params:
                    node = _get_grad_fn_or_grad_acc(param)
                    grad_acc_to_weight_s0[node] = param

                param_groups_s0 = self.bw_param_groups[stage_id].get(mb_idx) or []
                for pg in param_groups_s0:
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

                    mapped_param_nodes = [p for p in pg["params"] if p in grad_acc_to_weight_s0]
                    if not mapped_param_nodes:
                        continue

                    weight_edges = tuple(GradientEdge(p, 0) for p in mapped_param_nodes)

                    with torch.cuda.stream(comp_stream):
                        gparams_s0 = torch.autograd.grad(
                            outputs=intermediate_edges,
                            inputs=weight_edges,
                            grad_outputs=intermediate_edge_grads,
                            retain_graph=False,
                        )

                    del pg["grads"]

                    assert len(gparams_s0) == len(mapped_param_nodes)
                    for param_node, dw in zip(mapped_param_nodes, gparams_s0):
                        if dw is None:
                            continue
                        weight = grad_acc_to_weight_s0[param_node]
                        if weight.grad is None:
                            weight.grad = dw
                        else:
                            weight.grad.add_(dw)
                        updated_params[id(weight)] = weight

                # Step 2: pre-dispatch params (embedding, router, attention) via
                #         .grad values set on bkt=0 detached_outs by BWD_A2A(bkt=0)
                #         and BWD_I(bkt=1).
                buf0 = self.bucket_buffer.pop((stage_id, mb_idx, 0), None)
                if buf0 is not None:
                    pre_detach_outs0, detached_outs0 = buf0
                    outputs_bwd = [
                        p for p, d in zip(pre_detach_outs0, detached_outs0)
                        if d.requires_grad and d.grad is not None
                    ]
                    grads_bwd = [
                        d.grad for p, d in zip(pre_detach_outs0, detached_outs0)
                        if d.requires_grad and d.grad is not None
                    ]
                    if outputs_bwd:
                        with torch.cuda.stream(comp_stream):
                            torch.autograd.backward(outputs_bwd, grads_bwd)
            else:
                # Non-bucketed stage 0: single-segment full backward from out_activation.
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

        self.bw_grad_cache[stage_id][mb_idx] = None
        self.upstream_grad_cache[stage_id][mb_idx] = None
        self.bw_param_groups[stage_id][mb_idx] = None
        self.out_activation[stage_id][mb_idx] = None

    def _update(self, *deps):
        if self.ar_events:
            # DAG execution path: wait for in-flight all-reduces then step optimizers.
            self._start_timing(self.comp_stream, "backward_sync")
            for ar_evt in self.ar_events.values():
                self.comp_stream.wait_event(ar_evt)
            self._stop_timing(self.comp_stream, "backward_sync")
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

        return losses
    