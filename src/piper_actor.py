import ray
import torch
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set
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
from .piper_exec import TaskType, TaskDAG, Task

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
        nccl_env = {
            "env_vars": {
                "NCCL_SOCKET_IFNAME": "^lo,docker",
                "NCCL_DEBUG": "WARN",
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
        # accumuate loss for each microbatch
        self.loss = []

        self.tracing = False
        self._pending_timing_events: list = []  # (label, start_event, stop_event)
        self.trace_data = defaultdict(list)
        self.memory_tracing_enabled = False

        # DAG execution state
        self.dag = None

        # Unified inter-task buffer.  Keyed by task.uid (int), storing whatever
        # data the task produced for downstream consumers.
        # Cleared at the start of each run_dag() call.
        self.task_buffer: dict = {}

        # CUDA events for synchronisation between streams
        self.recv_events: dict = {}   # task_uid -> cuda.Event (for RECV tasks)
        self.a2a_events: dict = {}    # task_uid -> cuda.Event (for FWD_A2A/BWD_A2A tasks)
        self.ar_events: dict = {}     # unique_bucket_id -> cuda.Event (for ALL_REDUCE tasks)
        self.bwd_events: dict = {}    # unique_bucket_id -> cuda.Event (for BWD/BWD_I/BWD_W tasks)

        # Per-bucket data keyed by unique_bucket_id (populated by _load_stage).
        # "Bucket" here is a globally unique unit of computation; each stage may
        # contribute one or more buckets.  The actor is agnostic to which stage
        # owns a bucket.
        self.bucket_fwd_fns: dict = {}          # ubid -> fwd function
        self.bucket_fwd_args: dict = {}         # ubid -> args list (with None holes for inputs)
        self.bucket_input_idxs: dict = {}       # ubid -> list of input slot indices
        self.bucket_param_idxs: dict = {}       # ubid -> list of param slot indices
        self.bucket_param_names: dict = {}      # ubid -> [FX placeholder name per param]
        self.bucket_optims: dict = {}           # ubid -> optimizer (or None)
        self.bucket_flat_params: dict = {}      # ubid -> flat param tensor (or None)
        self.bucket_flat_grads: dict = {}       # ubid -> flat grad tensor (or None)
        self.bucket_trainable_param_idxs: dict = {}  # ubid -> list of trainable param slot indices

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
        self.logger.debug(
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
        init_method = f"tcp://{master_addr}:{master_port}"

        self.logger.info(f"Actor {self.global_rank} using GPU {os.environ['CUDA_VISIBLE_DEVICES']}, master addr {master_addr}:{master_port}")

        self.device = f"cuda:{self.global_rank % torch.cuda.device_count()}"
        torch.cuda.set_device(self.device)

        if self.pp_degree > 1 or self.dp_degree > 1:
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

        # Warm up both communicators to force eager NCCL initialization.
        # Without this, NCCL defers init to first use and the entire first
        # training iteration pays the (very expensive) init cost.
        dummy = torch.zeros(1, device=self.device)
        dist.all_reduce(dummy, group=self.pp_lo_hi)
        dist.all_reduce(dummy, group=self.pp_hi_lo)
        torch.cuda.synchronize()
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
                for idx, args in bd.get("triton_constant_args", {}).items():
                    kernel_side_table.constant_args[int(idx)] = args
        except Exception as e:
            self.logger.warning(f"Actor {self.global_rank} failed to restore triton constant_args: {e}")

        first_gm = None

        for b_idx, bd in enumerate(modules_data):
            ubid: int = bd["unique_bucket_id"]
            gm = _deserialize_graphmodule(bd["gm_data"])
            gm = self._replace_meta_constants(gm, self.device)
            if b_idx == 0:
                first_gm = gm

            forward_args = list(bd["graphargs"])
            b_input_idxs = list(bd["input_idxs"])
            b_param_idxs = list(bd["param_idxs"])

            # Extract FX placeholder names for each param index.
            gm_placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
            self.bucket_param_names[ubid] = [
                gm_placeholders[i].name if i < len(gm_placeholders) else f"ubid{ubid}_p{i}"
                for i in b_param_idxs
            ]

            self.logger.debug(
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
                    ph_name = gm_placeholders[i].name if i < len(gm_placeholders) else ""
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

            # Forward function (with optional activation checkpointing).
            if use_activation_checkpointing:
                fwd = gm.forward
                forward_fn = lambda *args, _fn=fwd: torch.utils.checkpoint.checkpoint(
                    _fn, *args, use_reentrant=False
                )
            else:
                forward_fn = gm.forward

            self.bucket_fwd_fns[ubid] = forward_fn
            self.bucket_fwd_args[ubid] = realized
            self.bucket_param_idxs[ubid] = b_param_idxs

            # Collect trainable parameters and build a contiguous flat tensor so
            # a single all-reduce call can sync the entire module's gradients.
            trainable_idxs = [
                i for i in b_param_idxs
                if realized[i] is not None and realized[i].requires_grad
            ]
            trainable = [realized[i] for i in trainable_idxs]
            self.bucket_trainable_param_idxs[ubid] = trainable_idxs

            if trainable:
                flat_params = torch.cat([t.detach().view(-1) for t in trainable]).contiguous()
                flat_params.requires_grad_(False)
                flat_grads = torch.zeros_like(flat_params)
                offset = 0
                for tidx, t in zip(trainable_idxs, trainable):
                    numel = t.numel()
                    realized[tidx] = realized[tidx].detach()
                    realized[tidx].data = flat_params[offset:offset + numel].view(t.shape)
                    realized[tidx].requires_grad_(True)
                    offset += numel
                self.bucket_flat_params[ubid] = flat_params
                self.bucket_flat_grads[ubid] = flat_grads
            else:
                self.bucket_flat_params[ubid] = None
                self.bucket_flat_grads[ubid] = None

            # Optimizer for this module's trainable parameters.
            trainable_for_optim = [realized[i] for i in trainable_idxs]
            optim = self.optim_class(trainable_for_optim, fused=True) if trainable_for_optim else None
            self.bucket_optims[ubid] = optim


        # Keep first GraphModule for compatibility with external inspection tools.
        self.graph_modules[stage_id] = first_gm

    # -----------------------------------------------------------------------
    # DAG-based execution
    # -----------------------------------------------------------------------

    def load_dag(self, dag: TaskDAG) -> None:
        """Store the per-rank TaskDAG for subsequent run_dag() calls."""
        self.dag = dag

    def get_all_params(self) -> dict:
        """Return {unique_bucket_id: flat_cpu_tensor} for every trainable bucket."""
        return {
            ubid: tensor.detach().cpu().clone()
            for ubid, tensor in self.bucket_flat_params.items()
            if tensor is not None
        }

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

    def _timing_stream(self, task_type) -> "torch.cuda.Stream":
        """Return the CUDA stream that drives work for the given task type."""
        if task_type == TaskType.SEND:
            return self.p2p_send_stream
        if task_type == TaskType.RECV:
            return self.p2p_recv_stream
        if task_type in (TaskType.FWD_A2A, TaskType.BWD_A2A):
            return self.a2a_stream
        if task_type == TaskType.ALL_REDUCE:
            return self.comm_stream
        return self.comp_stream

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

    def run_dag(self, loss_fn=None, profiling: bool = False):
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
        dag = self.dag

        # Reset per-iteration state
        self.task_buffer = {}
        self.recv_events = {}
        self.a2a_events = {}
        self.ar_events = {}
        self.bwd_events = {}

        # Point each trainable param's .grad at the appropriate slice of its
        # bucket's flat_grads tensor so backward accumulates contiguously.
        for ubid, trainable_idxs in self.bucket_trainable_param_idxs.items():
            flat_grads = self.bucket_flat_grads.get(ubid)
            if flat_grads is None:
                continue
            flat_grads.zero_()
            args = self.bucket_fwd_args[ubid]
            offset = 0
            for i in trainable_idxs:
                t = args[i]
                if t is not None:
                    numel = t.numel()
                    t.grad = flat_grads[offset:offset + numel].view(t.shape)
                    offset += numel

        comp_events: dict = {}  # task_uid -> cuda.Event for compute tasks

        # profiling: list of (node, start_evt, end_evt, mem_before_bytes)
        # populated during the loop; elapsed_time() is only called after the
        # final synchronize() so we never block the CPU mid-dispatch.
        self._prof_records: list = []

        # Sort nodes by time_step with tie-breaking: SEND/AR/A2A first (0),
        # compute second (1), RECV last (2).  RECV is deferred so that when a
        # RECV overlaps with compute at the same time_step the recv kernel only
        # launches after compute finishes (p2p_recv_stream waits on the compute
        # event), avoiding SM contention from an idle-spinning recv kernel.
        _SEND_PRIORITY_TYPES = {TaskType.SEND, TaskType.ALL_REDUCE, TaskType.FWD_A2A, TaskType.BWD_A2A}

        def _ts_key(n):
            t = n.chunk.type
            if t in _SEND_PRIORITY_TYPES:
                sub = 0
            elif t == TaskType.RECV:
                sub = 2
            else:
                sub = 1
            return (n.time_step, sub)

        sorted_nodes = sorted(dag.nodes, key=_ts_key)
        ts_to_comp_event: dict[int, torch.cuda.Event] = {}

        for node in sorted_nodes:

            chunk = node.chunk
            batch = chunk.batches[0]
            mb_idx = batch.mb_idx
            ubid = node.unique_bucket_id  # None for non-compute nodes

            self.logger.info(
                f"run_dag dispatch (time {node.time_step}): {chunk.type.value} "
                f"mb{mb_idx} ubid={ubid}"
            )

            if profiling:
                _stream = self._timing_stream(chunk.type)
                _start_evt = torch.cuda.Event(enable_timing=True)
                _start_evt.record(_stream)
                _mem_before = torch.cuda.memory_allocated()

            match chunk.type:

                case TaskType.SEND:
                    # data_preds[0] is the compute task whose output we send.
                    compute_node = node.data_preds[0]
                    self.p2p_send_stream.wait_event(comp_events[compute_node.uid])
                    send_data = self.task_buffer[compute_node.uid]["send_output"]
                    self._exec_send(send_data, node.peer_pp_rank)
                    # Free the send buffer.  For FWD/FWD_A2A the whole entry is stale
                    # after the send (task_buffer[(ubid,mb)] is the BWD consumer).
                    # For BWD/BWD_I/BWD_A2A keep the dict but drop the send tensor —
                    # BWD_W still needs param_groups (BWD_I), co-located BWD still
                    # needs inp_grads (BWD/BWD_A2A).
                    _send_buf = self.task_buffer.get(compute_node.uid)
                    if compute_node.chunk.type in (TaskType.FWD, TaskType.FWD_A2A):
                        del self.task_buffer[compute_node.uid]
                    elif isinstance(_send_buf, dict):
                        _send_buf["send_output"] = None

                case TaskType.RECV:
                    # data_succs[0] is the downstream compute task.
                    compute_node = node.data_succs[0]
                    comp_evt = ts_to_comp_event.get(node.time_step)
                    if comp_evt is not None:
                        self.p2p_recv_stream.wait_event(comp_evt)
                    if compute_node.chunk.type == TaskType.FWD:
                        # FWD recv: pre-allocate based on target bucket's input metadata.
                        recv_ubid = compute_node.unique_bucket_id
                        recv_tensors = self._exec_recv_fwd(recv_ubid, node.peer_pp_rank)
                    else:
                        # BWD recv: pre-allocate based on FWD output shapes.
                        fwd_ubid = node.custom_metadata["fwd_ubid"]
                        out_with_grad = self.task_buffer[("shape_ref", fwd_ubid)]
                        recv_tensors = self._exec_recv_bwd(out_with_grad, node.peer_pp_rank)
                    self.task_buffer[node.uid] = recv_tensors
                    recv_evt = torch.cuda.Event()
                    recv_evt.record(self.p2p_recv_stream)
                    self.recv_events[node.uid] = recv_evt

                case TaskType.FWD_A2A:
                    fwd_pred = next(p for p in node.data_preds if p.chunk.type == TaskType.FWD)
                    self.a2a_stream.wait_event(comp_events[fwd_pred.uid])
                    tensor_idx = node.custom_metadata["a2a_tensor_idx"]
                    # Pass-through: copy upstream FWD task_buffer, apply A2A only at tensor_idx.
                    fwd_buf = dict(self.task_buffer[fwd_pred.uid])
                    # FWD_A2A now owns the copy; original FWD entry is no longer needed.
                    del self.task_buffer[fwd_pred.uid]
                    detached_outs = list(fwd_buf["detached_outs"])
                    torch.cuda.nvtx.range_push(f"fwd_a2a_ubid{fwd_pred.unique_bucket_id}_mb{mb_idx}")
                    self._start_timing(self.a2a_stream, "fwd_a2a")
                    detached_outs[tensor_idx] = self._exec_a2a(detached_outs[tensor_idx]).requires_grad_(True)
                    self._stop_timing(self.a2a_stream, "fwd_a2a")
                    torch.cuda.nvtx.range_pop()
                    fwd_buf["detached_outs"] = detached_outs
                    self.task_buffer[node.uid] = fwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(self.a2a_stream)
                    self.a2a_events[node.uid] = a2a_evt
                    self.comp_stream.wait_event(a2a_evt)

                case TaskType.BWD_A2A:
                    bwd_pred = next(
                        p for p in node.data_preds
                        if p.chunk.type in (TaskType.BWD, TaskType.BWD_I)
                    )
                    self.a2a_stream.wait_event(comp_events[bwd_pred.uid])
                    tensor_idx = node.custom_metadata["a2a_tensor_idx"]
                    # Pass-through: copy upstream BWD/BWD_I task_buffer, apply reversed A2A only at tensor_idx.
                    bwd_buf = dict(self.task_buffer[bwd_pred.uid])
                    # Analog of FWD_A2A change 3: free predecessor buffer after copying.
                    # Skip for BWD_I — its param_groups must survive until BWD_W.
                    if bwd_pred.chunk.type == TaskType.BWD:
                        del self.task_buffer[bwd_pred.uid]
                    inp_grads = list(bwd_buf["inp_grads"])
                    grad_a2a_out = inp_grads[tensor_idx]
                    assert grad_a2a_out is not None, (
                        f"BWD_A2A mb={mb_idx}: grad at a2a_tensor_idx={tensor_idx} is None"
                    )
                    if not grad_a2a_out.is_contiguous():
                        grad_a2a_out = grad_a2a_out.contiguous()
                    torch.cuda.nvtx.range_push(f"bwd_a2a_ubid{bwd_pred.unique_bucket_id}_mb{mb_idx}")
                    self._start_timing(self.a2a_stream, "bwd_a2a")
                    inp_grads[tensor_idx] = self._exec_a2a(grad_a2a_out)
                    self._stop_timing(self.a2a_stream, "bwd_a2a")
                    torch.cuda.nvtx.range_pop()
                    bwd_buf["inp_grads"] = inp_grads
                    self.task_buffer[node.uid] = bwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(self.a2a_stream)
                    self.a2a_events[node.uid] = a2a_evt
                    self.comp_stream.wait_event(a2a_evt)

                case TaskType.ALL_REDUCE:
                    bwd_node = node.data_preds[0]
                    flat_grads = self.task_buffer[bwd_node.uid].get("flat_grads")
                    if flat_grads is not None:
                        torch.cuda.nvtx.range_push(f"all_reduce_ubid{bwd_node.unique_bucket_id}")
                        self._exec_all_reduce(flat_grads, comp_events[bwd_node.uid], bwd_node.unique_bucket_id)
                        torch.cuda.nvtx.range_pop()
                    else:
                        self.logger.debug(
                            f"ALL_REDUCE skipped: no trainable params at time_step={node.time_step}"
                        )

                case TaskType.FWD:
                    # Wait on any RECV predecessor.
                    recv_pred = next(
                        (p for p in node.data_preds if p.chunk.type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        self.comp_stream.wait_event(self.recv_events.pop(recv_pred.uid))

                    # Gather input tensors from task_buffer.
                    # FWD(bkt=0 of first stage): inputs come from self.inputs.
                    # FWD(bkt=0 of non-first stage): inputs come from RECV or prev FWD (co-located).
                    # FWD(bkt>0): inputs come from prev FWD or FWD_A2A output.
                    fwd_data_pred = next(
                        (p for p in node.data_preds
                         if p.chunk.type in (TaskType.FWD, TaskType.FWD_A2A)), None
                    )
                    if fwd_data_pred is not None:
                        # Both FWD and FWD_A2A task_buffer entries carry "detached_outs".
                        input_tensors = self.task_buffer[fwd_data_pred.uid]["detached_outs"]
                        # Upstream bucket consumed; free its buffer (no further reader).
                        del self.task_buffer[fwd_data_pred.uid]
                    elif recv_pred is not None:
                        input_tensors = self.task_buffer[recv_pred.uid]
                    else:
                        input_tensors = self.inputs  # first stage, first bucket

                    torch.cuda.nvtx.range_push(f"forward_ubid{ubid}_mb{mb_idx}")
                    self._start_timing(self.comp_stream, "forward")
                    fwd_out = self._forward_dag(ubid, mb_idx, input_tensors)
                    self._stop_timing(self.comp_stream, "forward")
                    torch.cuda.nvtx.range_pop()
                    self.task_buffer[node.uid] = fwd_out
                    # Shape ref for BWD RECV: store empty placeholder tensors under a
                    # tuple key so it never aliases task_buffer[node.uid] (both are ints).
                    # Allows fwd_out to be freed after BWD_I consumes (ubid, mb_idx).
                    self.task_buffer[("shape_ref", ubid)] = [torch.empty_like(t) for t in fwd_out["out_with_grad"]]
                    self.task_buffer[(ubid, mb_idx)] = fwd_out  # per-mb FWD data for BWD
                    evt = torch.cuda.Event()
                    evt.record(self.comp_stream)
                    comp_events[node.uid] = evt
                    ts_to_comp_event[node.time_step] = evt

                case TaskType.BWD:
                    # Wait on upstream grad RECV if present.
                    recv_pred = next(
                        (p for p in node.data_preds if p.chunk.type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        self.comp_stream.wait_event(self.recv_events.pop(recv_pred.uid))

                    # Resolve outputs and upstream grads via per-mb ubid lookup.
                    fwd_out = self.task_buffer[(ubid, mb_idx)]

                    if node.compute_loss:
                        assert loss_fn is not None
                        outputs_or_loss = [loss_fn(fwd_out["out_with_grad"][0], self.labels)]
                        upstream_grads = None
                    elif recv_pred is not None:
                        upstream_grads = self.task_buffer[recv_pred.uid]
                        if not isinstance(upstream_grads, (list, tuple)):
                            upstream_grads = [upstream_grads]
                        outputs_or_loss = fwd_out["out_with_grad"]
                        upstream_grads = list(upstream_grads)
                    else:
                        outputs_or_loss = fwd_out["out_with_grad"]
                        upstream_grads = None  # last stage, no recv

                    bwd_a2a_pred = next(
                        (p for p in node.data_preds if p.chunk.type == TaskType.BWD_A2A), None
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
                    else:
                        bwd_pred = next(
                            (p for p in node.data_preds
                             if p.chunk.type in (TaskType.BWD, TaskType.BWD_I)), None
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
                            # Analog of FWD change 2: free predecessor buffer after consuming
                            # inp_grads.  Only for plain BWD — BWD_I must survive until BWD_W.
                            if bwd_pred.chunk.type == TaskType.BWD:
                                del self.task_buffer[bwd_pred.uid]
                        else:
                            # Last (or only) stage in backward order, or has a RECV for
                            # upstream grads (cross-rank predecessor): standard backward.
                            pre_detach_outs = None
                            detached_outs = None

                    inp_with_grad = fwd_out.get("inp_with_grad")

                    torch.cuda.nvtx.range_push(f"backward_ubid{ubid}_mb{mb_idx}")
                    self._start_timing(self.comp_stream, "backward")
                    bwd_out = self._backward_dag(
                        ubid, mb_idx, outputs_or_loss, upstream_grads,
                        pre_detach_outs, detached_outs, inp_with_grad,
                        fwd_out.get("out_with_grad"),
                    )
                    self._stop_timing(self.comp_stream, "backward")
                    torch.cuda.nvtx.range_pop()
                    buf = bwd_out if bwd_out is not None else {}
                    buf["flat_grads"] = self.bucket_flat_grads.get(ubid)
                    # Full-length inp_grads parallel to fwd_inputs / detached_outs so
                    # BWD_A2A can index with a2a_tensor_idx without filtering offset.
                    fwd_inputs_full = fwd_out.get("fwd_inputs")
                    buf["inp_grads"] = (
                        [t.grad if (t is not None and t.requires_grad) else None
                         for t in fwd_inputs_full]
                        if fwd_inputs_full is not None
                        else [t.grad for t in (inp_with_grad or [])]
                    )
                    self.task_buffer[node.uid] = buf
                    evt = torch.cuda.Event()
                    evt.record(self.comp_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    ts_to_comp_event[node.time_step] = evt

                case TaskType.BWD_I:
                    # Wait on upstream grad RECV if present.
                    recv_pred = next(
                        (p for p in node.data_preds if p.chunk.type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        self.comp_stream.wait_event(self.recv_events.pop(recv_pred.uid))

                    # Resolve stage_outputs_or_loss and output_grads.
                    fwd_out = self.task_buffer[(ubid, mb_idx)]

                    if node.compute_loss:
                        assert loss_fn is not None
                        stage_outputs_or_loss = [loss_fn(fwd_out["out_with_grad"][0], self.labels)]
                        output_grads = None
                    elif recv_pred is not None:
                        upstream_raw = self.task_buffer[recv_pred.uid]
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
                        (p for p in node.data_preds if p.chunk.type == TaskType.BWD_A2A), None
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

                    input_values = fwd_out.get("inp_with_grad") or []

                    # Weights for this bucket.
                    b_fwd_args = self.bucket_fwd_args[ubid]
                    weights = [b_fwd_args[i] for i in self.bucket_param_idxs[ubid]
                               if b_fwd_args[i] is not None]

                    torch.cuda.nvtx.range_push(f"backward_input_ubid{ubid}_mb{mb_idx}")
                    self._start_timing(self.comp_stream, "backward_input")
                    with torch.cuda.stream(self.comp_stream):
                        dinputs, param_groups = self._bucket_backward_input(
                            stage_outputs_or_loss, output_grads, input_values, iter(weights)
                        )
                    self._stop_timing(self.comp_stream, "backward_input")
                    torch.cuda.nvtx.range_pop()
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
                        "dinputs": dinputs,
                        "inp_grads": inp_grads_full,
                        "param_groups": param_groups,
                        "flat_grads": self.bucket_flat_grads.get(ubid),
                    }
                    # Propagate input grad to SEND (if this is the first bucket and has predecessors).
                    if dinputs:
                        self.task_buffer[node.uid]["send_output"] = list(dinputs)
                    # Free per-microbatch FWD data: all tensors from it have been consumed
                    # by _bucket_backward_input.  This removes the last Python reference to
                    # fwd_out (task_buffer[ubid] now holds only shape placeholders after
                    # the FWD task change; task_buffer[node_fwd.uid] was freed by SEND),
                    # allowing the stage-boundary activation tensors to be GC'd.
                    del self.task_buffer[(ubid, mb_idx)]
                    evt = torch.cuda.Event()
                    evt.record(self.comp_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    ts_to_comp_event[node.time_step] = evt

                case TaskType.BWD_W:
                    # data_preds[0] is the BWD_I task for bkt=K-1 (head of chain).
                    # For bkt<K-1 the chain-link edge (BWD_W predecessor) is prepended
                    # before Pass 2 appends the BWD_I edge, so data_preds[0] would be
                    # the wrong task.  Find BWD_I explicitly by type to be safe.
                    bwdi_node = next(p for p in node.data_preds if p.chunk.type == TaskType.BWD_I)
                    param_groups = self.task_buffer[bwdi_node.uid]["param_groups"]
                    b_fwd_args = self.bucket_fwd_args[ubid]
                    weights = [b_fwd_args[i] for i in self.bucket_param_idxs[ubid]
                               if b_fwd_args[i] is not None]
                    torch.cuda.nvtx.range_push(f"backward_weight_ubid{ubid}_mb{mb_idx}")
                    self._start_timing(self.comp_stream, "backward_weight")
                    with torch.cuda.stream(self.comp_stream):
                        self._bucket_backward_weight(iter(weights), param_groups)
                    self._stop_timing(self.comp_stream, "backward_weight")
                    torch.cuda.nvtx.range_pop()
                    # Store flat_grads so ALL_REDUCE can find them when BWD_W is the trigger.
                    # bucket_flat_grads.get(ubid) is None for non-trainable buckets; the
                    # ALL_REDUCE handler's "if flat_grads is not None" guard skips those.
                    self.task_buffer[node.uid] = {"flat_grads": self.bucket_flat_grads.get(ubid)}
                    # BWD_I buffer is no longer needed: _bucket_backward_weight already
                    # deleted param_groups["intermediates"] and param_groups["grads"] and
                    # the autograd graph was freed by retain_graph=False.  Drop the shell.
                    del self.task_buffer[bwdi_node.uid]
                    evt = torch.cuda.Event()
                    evt.record(self.comp_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    ts_to_comp_event[node.time_step] = evt

                case TaskType.UPD:
                    torch.cuda.nvtx.range_push("update")
                    self._update()
                    torch.cuda.nvtx.range_pop()

            if profiling:
                _end_evt = torch.cuda.Event(enable_timing=True)
                _end_evt.record(_stream)
                _mem_after = torch.cuda.memory_allocated()
                self._prof_records.append((node, mb_idx, ubid, _start_evt, _end_evt, _mem_before))
                self.logger.info(f"Rank {self.global_rank} task {chunk.type.value} (time {node.time_step}) mem before {_mem_before / 1024 ** 3:.2f} GB after {_mem_after / 1024 ** 3:.2f} GB")

    def _exec_send(self, send_data, peer_pp_rank: int) -> None:
        """Send tensors to peer_pp_rank on p2p_send_stream.

        The caller (run_dag) must have already made p2p_send_stream wait on the
        compute event before calling this method.
        """
        global_dst_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)
        self.logger.debug(f"exec_send to global rank {global_dst_rank}")

        with torch.cuda.stream(self.p2p_send_stream):
            tensors = send_data if isinstance(send_data, (list, tuple)) else [send_data]
            self._start_timing(self.p2p_send_stream, "p2p_send")
            pp_group = self.pp_lo_hi if global_dst_rank > self.global_rank else self.pp_hi_lo
            for tensor in tensors:
                dist.send(tensor, dst=global_dst_rank, group=pp_group)
            self._stop_timing(self.p2p_send_stream, "p2p_send")

    def _exec_recv_fwd(self, recv_ubid: int, peer_pp_rank: int) -> list:
        """Receive FWD activations from peer_pp_rank into pre-allocated buffers.

        Buffer shapes are derived from the target bucket's stored input metadata.
        Returns the received tensor list (stored in task_buffer by run_dag).
        """
        global_src_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)
        self.logger.debug(f"exec_recv_fwd ubid={recv_ubid} from global rank {global_src_rank}")

        buf = [
            torch.empty(shape, dtype=dtype, requires_grad=requires_grad, device=self.device)
            for shape, dtype, requires_grad in self.forward_input_meta[recv_ubid]
        ]
        with torch.cuda.stream(self.p2p_recv_stream):
            self._start_timing(self.p2p_recv_stream, "p2p_recv")
            pp_group = self.pp_hi_lo if global_src_rank > self.global_rank else self.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
            self._stop_timing(self.p2p_recv_stream, "p2p_recv")
        return buf

    def _exec_recv_bwd(self, out_with_grad: list, peer_pp_rank: int) -> list:
        """Receive BWD upstream gradients from peer_pp_rank.

        Buffer shapes are derived from out_with_grad (FWD outputs saved in task_buffer).
        Returns the received gradient list (stored in task_buffer by run_dag).
        """
        global_src_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)
        self.logger.debug(f"exec_recv_bwd from global rank {global_src_rank}")

        buf = [torch.empty_like(a) for a in out_with_grad]
        with torch.cuda.stream(self.p2p_recv_stream):
            self._start_timing(self.p2p_recv_stream, "p2p_recv")
            pp_group = self.pp_hi_lo if global_src_rank > self.global_rank else self.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
            self._stop_timing(self.p2p_recv_stream, "p2p_recv")
        return buf

    def _exec_a2a(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply dist.all_to_all_single on a2a_stream and return the output tensor.

        Used for both FWD_A2A and BWD_A2A — the operation is symmetric.
        Stream waits and event recording are handled by run_dag.
        """
        output_buf = torch.empty_like(input_tensor)
        with torch.cuda.stream(self.a2a_stream):
            dist.all_to_all_single(output_buf, input_tensor, group=self.ep_group)
        return output_buf

    def _forward_dag(self, ubid: int, mb_idx: int, input_tensors) -> dict:
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

        # Run the forward function.
        with torch.cuda.stream(self.comp_stream):
            output = fwd_fn(*fwd_args)

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
    ) -> dict | None:
        """Fused backward pass (BWD) for a single bucket.

        Handles both:
        - Last-bucket-of-stage case: backward from out_with_grad / loss.
        - Non-last-bucket case: backward through the (pre_detach, detached) boundary.

        Returns a dict with ``"send_output"`` (input grad list) when this is the
        first-bucket-of-stage and the stage has a predecessor, or None otherwise.
        """
        comp_stream = self.comp_stream

        if pre_detach_outs is None:
            # Last bucket (or only bucket): backward from stage outputs or loss.
            if upstream_grads is not None:
                with torch.cuda.stream(comp_stream):
                    torch.autograd.backward(outputs_or_loss, upstream_grads)
            else:
                with torch.cuda.stream(comp_stream):
                    outputs_or_loss[0].backward()
        else:
            # Non-last bucket: propagate through the bucket boundary.
            outputs_bwd = [p for p, d in zip(pre_detach_outs, detached_outs) if d.requires_grad]
            grads_bwd = [d.grad for p, d in zip(pre_detach_outs, detached_outs) if d.requires_grad]
            assert all(g is not None for g in grads_bwd), (
                f"BWD ubid={ubid} mb={mb_idx}: detached boundary has no .grad"
            )
            with torch.cuda.stream(comp_stream):
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

        for t in stage_outputs_or_loss:
            t.detach_()
        # stage_outputs_or_loss = [t.detach() for t in stage_outputs_or_loss]

        for handle in handles:
            handle.remove()

        return dinputs, param_groups

    def _bucket_backward_weight(self, weights, param_groups: list) -> None:
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

            dweights = torch.autograd.grad(
                valid_edges,
                weight_edges,
                grad_outputs=valid_grad_outputs,
                retain_graph=False,
            )

            del param_group["grads"]

            for grad_acc, dw in zip(param_group["params"], dweights):
                if dw is None or grad_acc not in grad_acc_to_weight:
                    continue
                weight = grad_acc_to_weight[grad_acc]
                if weight.grad is None:
                    weight.grad = dw
                else:
                    weight.grad += dw

    def _exec_all_reduce(self, flat_grads: torch.Tensor, bwd_event: torch.cuda.Event, ubid: int) -> None:
        """Launch an all-reduce for *flat_grads* on comm_stream.

        *flat_grads* comes directly from the predecessor BWD/BWD_I task's task_buffer
        entry.  comm_stream waits on *bwd_event* (ensuring all gradient accumulation is
        complete) then records an event so _update can wait before stepping.
        """
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(bwd_event)
            self._start_timing(self.comm_stream, "all_reduce")
            dist.all_reduce(flat_grads, group=self.dp_group)
            self._stop_timing(self.comm_stream, "all_reduce")
            evt = torch.cuda.Event()
            evt.record(self.comm_stream)
        self.ar_events[ubid] = evt

    def _update(self, *deps):
        self._start_timing(self.comp_stream, "optim_step")

        # Snapshot flat_params and mean grad before stepping, to log the change.
        # We clone on comp_stream (after all bwd events) then log after synchronize().
        pre_snapshots: dict = {}  # ubid -> (params_before, mean_grad)

        for ubid, optim in self.bucket_optims.items():
            if optim is None:
                continue
            bwd_evt = self.bwd_events.get(ubid)
            if bwd_evt is not None:
                self.comp_stream.wait_event(bwd_evt)
            ar_evt = self.ar_events.get(ubid)
            if ar_evt is not None:
                self.comp_stream.wait_event(ar_evt)

            flat_params = self.bucket_flat_params.get(ubid)
            flat_grads = self.bucket_flat_grads.get(ubid)
            if flat_params is not None:
                with torch.cuda.stream(self.comp_stream):
                    params_before = flat_params.clone()
                    mean_grad = flat_grads.abs().mean() if flat_grads is not None else None
                pre_snapshots[ubid] = (params_before, mean_grad)

            with torch.cuda.stream(self.comp_stream):
                optim.step()
        self._stop_timing(self.comp_stream, "optim_step")

        losses = self.loss
        self.loss.clear()

        torch.cuda.synchronize()

        # Log mean grad and mean param change per bucket to confirm optimizer is updating.
        # for ubid, (params_before, mean_grad_t) in pre_snapshots.items():
        #     flat_params = self.bucket_flat_params[ubid]
        #     diff = (flat_params - params_before).abs().mean().item()
        #     mean_grad_val = mean_grad_t.item() if mean_grad_t is not None else float("nan")
        #     self.logger.info(
        #         f"[update] rank={self.global_rank} ubid={ubid}: "
        #         f"mean_grad={mean_grad_val:.4e}  mean_param_diff={diff:.4e}"
        #     )

        # Process deferred profiling records now that all GPU work is complete.
        # elapsed_time() is safe to call after synchronize().
        if self._prof_records:
            for node, mb_idx, ubid, start_evt, end_evt, mem_before in self._prof_records:
                t_ms = start_evt.elapsed_time(end_evt)
                node.profiling_measurements.append(t_ms)
                self.logger.info(
                    f"[profiling] rank {self.global_rank} "
                    f"{node.chunk.type.value} mb{mb_idx} ubid{ubid} "
                    f"(time step {node.time_step}): time={t_ms:.3f}ms "
                    f"mem_before={mem_before/1e9:.3f}GB"
                )

        return losses
    