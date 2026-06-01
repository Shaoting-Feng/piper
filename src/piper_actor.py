import ray
import torch
import logging
import os
import re
from typing import Any, Dict, List
import gc
from concurrent.futures import Future, ThreadPoolExecutor
from torch.nn import Parameter
from torch.autograd.graph import GradientEdge, Node
from torch.autograd.graph import set_warn_on_accumulate_grad_stream_mismatch
import torch.distributed as dist
from collections import defaultdict

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .piper_utils import (
    _serialize_graphmodule,
    _deserialize_graphmodule,
    create_logger,
    LOG_LEVEL,
)
from .backward_utils import get_param_groups, construct_reverse_graph, _get_grad_fn_or_grad_acc
from .piper_exec import TaskType
from .piper import _serial_topological_order, _training_dag_task_type

CLEANUP_MEMORY = False

logger = create_logger("piper_actor", LOG_LEVEL)

def _get_rank(pp_rank, dp_rank, pp_degree):
    return pp_rank + dp_rank * pp_degree


def _create_actors(
    num_actors,
    optim_class,
    profile=False,
    no_nvtx: bool = False,
    pg=None,
    temp_dir: str = None,
    use_inductor: bool = False,
    pp_outer: bool = False,
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
                # "NCCL_SOCKET_IFNAME": "ens32",
                # "GLOO_SOCKET_IFNAME": "ens32",
                # "NCCL_P2P_DISABLE": "1",
                # "NCCL_DEBUG": "INFO",
                **({"TMPDIR": temp_dir} if (profile and temp_dir) else {}),
            }
        }
        # When pp_outer=True, one bundle corresponds to one pipeline stage and
        # holds all DP replicas for that stage (placement group shape is
        # [{"GPU": dp}] * pp). Otherwise one bundle is one DP replica holding
        # all PP ranks (shape is [{"GPU": pp}] * dp).
        bundle_index = pp_rank if pp_outer else dp_rank
        actor = PiperActor.options(
            num_gpus=0.6,
            runtime_env={**nsight_env, **nccl_env},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
            ),
        ).remote(
            pp_rank,
            optim_class,
            world_size,
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            pp_degree=pp_degree,
            no_nvtx=no_nvtx,
            use_inductor=use_inductor,
        )
        piper_metadata.actors[pp_rank] = actor


@ray.remote
class PiperActor:
    def __init__(
        self,
        pp_rank,
        optim_class,
        world_size,
        dp_rank=0,
        dp_degree=1,
        pp_degree=1,
        no_nvtx: bool = False,
        use_inductor: bool = False,
    ):
        self.logger = create_logger("piper_actor", LOG_LEVEL)

        # BWD_I uses torch.autograd.grad (not .backward), so AccumulateGrad nodes are
        # traversed for stream-sync bookkeeping but never accumulate to p.grad.
        # Suppress the spurious stream-mismatch warning.
        set_warn_on_accumulate_grad_stream_mismatch(False)

        self.pp_rank = pp_rank
        self.optim_class = optim_class
        self.no_nvtx = no_nvtx
        self.use_inductor = bool(use_inductor)

        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.pp_degree = pp_degree
        self.world_size = world_size

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

        self.streams: dict[str, torch.cuda.Stream] = {}

        # map stage id -> original GraphModule (for hook registration)
        self.graph_modules = dict()
        # map stage id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.forward_args = dict()
        # map stage id -> input idx -> input tensor metadata
        self.forward_input_meta = defaultdict(dict)
        # accumuate loss for each microbatch
        self.loss = []

        # DAG execution state
        self.dag = None
        self.sorted_dag_nodes = None

        # PyTorch profiler state (enabled on demand via start_pytorch_profiler).
        self._pytorch_profiler_enabled = False
        self._torch_profiler = None

        # Unified inter-task buffer.  Keyed by task.uid (int), storing whatever
        # data the task produced for downstream consumers.
        # Cleared at the start of each run_dag() call.
        self.task_buffer: dict = {}
        self.task_buffer_refcounts: dict[int, int] = {}

        # CUDA events for synchronisation between streams
        self.recv_events: dict = {}   # task_uid -> cuda.Event (for RECV tasks)
        self.a2a_events: dict = {}    # task_uid -> cuda.Event (for FWD_A2A/BWD_A2A tasks)
        self.ar_events: dict = {}     # task_uid -> cuda.Event (for ALL_REDUCE tasks)
        self.rs_events: dict = {}     # task_uid -> cuda.Event (for REDUCE_SCATTER tasks)
        self.ag_events: dict = {}     # ALL_GATHER task_uid -> cuda.Event
        self.bwd_events: dict = {}    # bucket_key -> cuda.Event (for BWD/BWD_I/BWD_W tasks)

        # Per-bucket data keyed by bucket_key (populated by _load_stage).
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
        self.bucket_flat_params: dict = {}       # ubid -> full flat param tensor for sharded params
        self.bucket_flat_grads: dict = {}        # ubid -> flat grad tensor for sharded grads
        self.bucket_shard_params: dict = {}      # ubid -> local optimizer shard param tensor
        self.bucket_shard_optims: dict = {}      # ubid -> local shard optimizer
        self.bucket_rs_grads: dict = {}          # ubid -> reduced local shard grad tensor
        self.grad_buffer_dtype = torch.float32   # sharded gradient buffers stay fp32
        self.param_shard_info: dict = {}         # ubid -> (shard_start, shard_size, orig_numel)
        self.bucket_param_view_specs: dict = {}  # ubid -> [(param, offset, numel, shape)]
        self.full_params_fresh: dict = {}        # ubid -> bool
        self.param_sharded_ubids: set = set()
        self.grad_sharded_ubids: set = set()
        self.zero_managed_ubids: set = set()
        self.cleanup_executor = ThreadPoolExecutor(max_workers=1)
        self.pending_param_frees: dict[Any, Future] = {}
        self.pending_grad_frees: dict[Any, Future] = {}

        # Non-trainable constant tensor attributes (e.g. freqs_cis, mask) pushed
        # from the coordinator before compilation so _load_stage can fill them in
        # instead of zero-initializing.  Keyed by bare attribute name (e.g. "freqs_cis").
        self.model_const_attrs: dict = {}

    def get_and_reset_peak_memory_stats(self) -> tuple:
        """Return (global_rank, max_memory_allocated_bytes) and reset peak stats."""
        max_alloc = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return self.global_rank, max_alloc

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()

    def _nvtx_push(self, label: str) -> None:
        if not self.no_nvtx:
            torch.cuda.nvtx.range_push(label)

    def _nvtx_pop(self) -> None:
        if not self.no_nvtx:
            torch.cuda.nvtx.range_pop()

    def start_pytorch_profiler(self) -> None:
        """Begin a torch.profiler session spanning the next run_dag iterations.

        Each node's execution in _run_dag_body is wrapped in a record_function
        labelled the same as its NVTX range, so the resulting trace identifies
        per-node work.
        """
        self._torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self._torch_profiler.__enter__()
        self._pytorch_profiler_enabled = True
        self.logger.info(f"Actor {self.global_rank}: PyTorch profiler started")

    def stop_pytorch_profiler(self, profile_dir: str) -> str:
        """End the profiler session and export a chrome trace.

        The file is named ``dp{dp_rank}_pp{pp_rank}.json`` inside *profile_dir*
        so the harness can group same-dp-rank actors. Returns the written path.
        """
        self._pytorch_profiler_enabled = False
        self._torch_profiler.__exit__(None, None, None)
        os.makedirs(profile_dir, exist_ok=True)
        filepath = os.path.join(
            profile_dir, f"dp{self.dp_rank}_pp{self.pp_rank}.json"
        )
        self._torch_profiler.export_chrome_trace(filepath)
        self._torch_profiler = None
        self.logger.info(
            f"Actor {self.global_rank}: PyTorch profiler trace written to {filepath}"
        )
        return filepath

    def _rf_enter(self, label: str):
        """Enter a profiler record_function for *label* (no-op unless profiling).

        Returns the entered context (or None) to pass to _rf_exit, mirroring the
        _nvtx_push/_nvtx_pop pattern so the large match block need not be
        re-indented under a ``with``.

        While profiling, we also disable multithreaded autograd for the duration
        of the task. record_function pushes its range onto the *calling* thread's
        thread-local stack, but the autograd engine normally runs backward on a
        separate per-device worker thread, so kernels it launches fall outside the
        label window and appear un-labelled on the GPU streams. Running backward on
        the calling thread makes the label span every backward kernel. This perturbs
        timing slightly, which is acceptable for a diagnostic profiling run.
        """
        if not self._pytorch_profiler_enabled:
            return None
        mt = torch.autograd.set_multithreading_enabled(False)
        mt.__enter__()
        rf = torch.profiler.record_function(label)
        rf.__enter__()
        return (rf, mt)

    def _rf_exit(self, rf) -> None:
        if rf is not None:
            rf_ctx, mt = rf
            rf_ctx.__exit__(None, None, None)
            mt.__exit__(None, None, None)

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

            if self.dp_degree > 1:
                self._join_dp_process_group()
            if self.pp_degree > 1:
                self._join_pp_process_group()

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

    def _derive_dag_bucket_modes(self, training_dag: Any) -> None:
        self.param_sharded_ubids = set()
        self.grad_sharded_ubids = set()
        for node in training_dag.nodes.values():
            meta = getattr(node, "node_meta", {}) or {}
            ubid = meta.get("bucket_key")
            if ubid is None:
                continue
            if node.node_kind == "ALL_GATHER_COMM":
                self.param_sharded_ubids.add(ubid)
            elif node.node_kind == "REDUCE_SCATTER_COMM":
                self.grad_sharded_ubids.add(ubid)
        self.zero_managed_ubids = self.param_sharded_ubids | self.grad_sharded_ubids

    @staticmethod
    def _node_meta(node: Any) -> dict:
        meta = getattr(node, "node_meta", None)
        return meta if isinstance(meta, dict) else {}

    def _node_bucket_key(self, node: Any) -> Any | None:
        return self._node_meta(node).get("bucket_key")

    def _relocate_meta_devices(self, gm) -> None:
        """Rewrite baked ``torch.device('meta')`` literals to the actor device.

        Meta-device compilation captures whatever device was current at trace
        time (inside ``type_as`` / ``.to(device=x.device)`` / device-aware
        factory ops like ``torch.zeros(..., device=...)``). That literal
        round-trips through serialization as ``meta`` and would otherwise make
        those ops emit meta tensors at runtime, mismatching the CUDA inputs.
        Lifted params/buffers/inputs are unaffected (placed on device directly),
        so this only touches device literals embedded in node args/kwargs.
        """
        import torch.fx as fx

        device = self.device
        replaced = 0

        def _fix(value):
            nonlocal replaced
            if isinstance(value, torch.device):
                if value.type == "meta":
                    replaced += 1
                    return torch.device(device)
                return value
            if isinstance(value, tuple):
                return tuple(_fix(v) for v in value)
            if isinstance(value, list):
                return [_fix(v) for v in value]
            if isinstance(value, dict):
                return {k: _fix(v) for k, v in value.items()}
            return value

        for module in gm.modules():
            if not isinstance(module, fx.GraphModule):
                continue
            before = replaced
            for node in module.graph.nodes:
                node.args = _fix(node.args)
                node.kwargs = _fix(node.kwargs)
            if replaced != before:
                module.recompile()

    def _load_stage(
        self,
        stage_id: int,
        modules_data: list,
        a2a_boundaries: dict = None,
        use_activation_checkpointing: bool = False,
    ) -> None:
        """Load a (possibly bucketed) stage.

        *modules_data* is a list of dicts, one per module/bucket, each with keys:
        ``gm_data``, ``graphargs``, ``input_idxs``, ``param_idxs``, ``bucket_key``.

        A non-bucketed stage is represented as a single-element list.
        All per-bucket data structures are keyed by ``bucket_key`` so run_dag
        can look up bucket data without knowing which stage owns a bucket.
        """
        self.logger.debug(
            f"Loading stage {stage_id} ({len(modules_data)} module(s)) on actor {self.global_rank}"
        )

        g = torch.Generator(device=self.device)
        g.manual_seed(1000 * self.global_rank + stage_id)

        first_gm = None

        for b_idx, bd in enumerate(modules_data):
            ubid: Any = bd["bucket_key"]
            ac_num_subgraphs = int(bd.get("ac_num_subgraphs", 1))
            ac_requested_subgraphs = int(bd.get("ac_requested_subgraphs", ac_num_subgraphs))
            gms = [_deserialize_graphmodule(gm_data) for gm_data in bd["gm_data_list"]] if "gm_data_list" in bd else [_deserialize_graphmodule(bd["gm_data"])]
            for gm in gms:
                self._relocate_meta_devices(gm)
            if self.use_inductor:
                compiled_gms = []
                for subgraph_idx, gm in enumerate(gms):
                    compiled_gm = torch.compile(gm)
                    compiled_gms.append(compiled_gm)
                    self.logger.debug(
                        f"[load_stage_compile] rank={self.global_rank} stage={stage_id} "
                        f"ubid={ubid} subgraph={subgraph_idx} compiled=True"
                    )
                gms = compiled_gms

            if b_idx == 0:
                first_gm = gms[0]

            forward_args = list(bd["graphargs"])
            b_input_idxs = list(bd["input_idxs"])
            b_param_idxs = list(bd["param_idxs"])
            apply_zero = bool(bd.get("apply_zero", True))
            shared_placeholder_names = list(
                bd.get("shared_placeholder_names", bd.get("placeholder_names", []))
            )
            # Extract FX placeholder names for each param index.
            self.bucket_param_names[ubid] = [
                shared_placeholder_names[i] if i < len(shared_placeholder_names) else f"ubid{ubid}_p{i}"
                for i in b_param_idxs
            ]

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
                    if _use_ac:
                        out = torch.utils.checkpoint.checkpoint(
                            forward_impl,
                            *call_args,
                            use_reentrant=False,
                        )
                    else:
                        out = forward_impl(*call_args)
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

            zero_managed = (
                self.dp_degree > 1
                and apply_zero
                and bool(trainable_idxs)
                and ubid in self.zero_managed_ubids
            )
            params_sharded = ubid in self.param_sharded_ubids
            grads_sharded = ubid in self.grad_sharded_ubids

            if zero_managed:
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
                    realized[idx].grad_dtype = self.grad_buffer_dtype
                    view_specs.append((realized[idx], offset, numel, tuple(p.shape)))
                    offset += numel

                shard_start = self.dp_rank * shard_size
                if params_sharded:
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
                    if grads_sharded
                    else torch.zeros(padded_numel, dtype=self.grad_buffer_dtype, device=self.device)
                )
                self.bucket_shard_params[ubid] = shard_param
                self.bucket_shard_optims[ubid] = self.optim_class([shard_param])
                self.bucket_rs_grads[ubid] = (
                    torch.zeros(shard_size, dtype=self.grad_buffer_dtype, device=self.device)
                    if grads_sharded
                    else None
                )
                self.param_shard_info[ubid] = (shard_start, shard_size, orig_numel)
                self.bucket_param_view_specs[ubid] = view_specs
                self.full_params_fresh[ubid] = False

                if params_sharded:
                    storage = flat_params.untyped_storage()
                    if storage.size() != 0:
                        storage.resize_(0)

                optim = None
            else:
                # Keep trainable params as separate tensors for the non-ZeRO path.
                self.bucket_param_view_specs[ubid] = []
                self.bucket_flat_params[ubid] = None
                self.bucket_flat_grads[ubid] = None
                trainable_for_optim = [realized[i] for i in trainable_idxs]
                optim = self.optim_class(trainable_for_optim, fused=True) if trainable_for_optim else None
            self.bucket_optims[ubid] = optim

        # Keep first GraphModule for compatibility with external inspection tools.
        self.graph_modules[stage_id] = first_gm

    # -----------------------------------------------------------------------
    # DAG-based execution
    # -----------------------------------------------------------------------

    def _stream_id(self, node_or_stream: Any) -> str:
        if isinstance(node_or_stream, str):
            return node_or_stream
        return str(getattr(node_or_stream, "stream", "default_stream"))

    def _initialize_streams_for_training_dag(self, training_dag: Any) -> None:
        stream_ids = {
            self._stream_id(n)
            for n in training_dag.nodes.values()
            if getattr(n, "stream", None) is not None
        }
        stream_ids.add("default_stream")
        self.streams = {
            stream_id: torch.cuda.Stream(device=self.device)
            for stream_id in sorted(stream_ids)
        }

        # Force cuBLAS context initialization on every logical stream used by
        # this DAG so the first backward pass does not hit lazy CUDA warnings.
        for stream in self.streams.values():
            with torch.cuda.stream(stream):
                _w = torch.zeros(4, 4, device=self.device)
                torch.mm(_w, _w)

    def _stream_for_id(self, stream_id: str) -> "torch.cuda.Stream":
        assert stream_id in self.streams, (
            f"TrainingDAG referenced stream={stream_id!r}, but load_training_dag "
            f"initialized only {sorted(self.streams)}"
        )
        return self.streams[stream_id]

    def _stream_for_task(self, task: Any) -> "torch.cuda.Stream":
        return self._stream_for_id(self._stream_id(task))

    def _default_stream(self) -> "torch.cuda.Stream":
        return self._stream_for_id("default_stream")

    def load_training_dag(self, training_dag: Any) -> None:
        """Load per-PP-rank TrainingDAG compute nodes using existing _load_stage logic.

        This is an adapter for the new TrainingDAG representation where each
        COMPUTE/FWD node with ``node_meta['gm']`` is equivalent to one bucket/module.
        The runtime executes TrainingDAG nodes via ``run_dag``; this method
        only populates actor-side bucket/module state.
        """
        if training_dag is None:
            raise ValueError("load_training_dag requires non-None training_dag")
        if not hasattr(training_dag, "nodes"):
            raise TypeError("load_training_dag expected object with 'nodes' field")

        self._derive_dag_bucket_modes(training_dag)
        self._initialize_streams_for_training_dag(training_dag)

        # Clear any prior module state before loading a new training DAG.
        self.graph_modules.clear()
        self.forward_args.clear()
        self.forward_input_meta.clear()

        stage_to_modules: dict[int, list[dict]] = defaultdict(list)

        # Deterministic order by stage/segment then uid.
        compute_nodes = [
            n for n in training_dag.nodes.values()
            if getattr(n, "node_kind", None) == "COMPUTE"
            and getattr(n, "compute_subkind", None) == "FWD"
        ]
        compute_nodes.sort(
            key=lambda n: (
                int(getattr(n, "node_meta", {}).get("stage_id", 10**9)),
                int(getattr(n, "node_meta", {}).get("segment_id", 10**9)),
                str(getattr(n, "uid", "")),
            )
        )

        seen_bucket_keys: set[Any] = set()
        skipped_dupe_fwd_nodes = 0
        for node in compute_nodes:
            meta = getattr(node, "node_meta", {}) or {}
            bucket_key = meta.get("bucket_key", getattr(node, "uid", None))
            if bucket_key in seen_bucket_keys:
                skipped_dupe_fwd_nodes += 1
                continue
            seen_bucket_keys.add(bucket_key)
            gm = meta.get("gm")
            gm_data = meta.get("gm_data")
            stage_id = meta.get("stage_id")
            input_idxs = meta.get("input_idxs")
            param_idxs = meta.get("param_idxs")
            graphargs = meta.get("graphargs")
            input_names = meta.get("input_names", [])
            output_names = meta.get("output_names", [])

            if gm_data is None and gm is None:
                raise ValueError(
                    f"load_training_dag: compute node {getattr(node, 'uid', '<unknown>')} "
                    f"is missing required metadata field 'gm_data' (or fallback 'gm')"
                )
            if stage_id is None or input_idxs is None or param_idxs is None or graphargs is None:
                raise ValueError(
                    f"load_training_dag: compute node {getattr(node, 'uid', '<unknown>')} "
                    f"is missing required metadata fields"
                )

            module_data = {
                "gm_data": gm_data if gm_data is not None else _serialize_graphmodule(gm),
                "graphargs": list(graphargs),
                "input_idxs": list(input_idxs),
                "param_idxs": list(param_idxs),
                "placeholder_names": list(input_names),
                "output_names": list(output_names),
                "shared_placeholder_names": list(input_names),
                "bucket_key": bucket_key,
                "ac_num_subgraphs": 1,
                "ac_requested_subgraphs": 1,
                "apply_zero": bool(meta.get("apply_zero", True)),
                "training_dag_uid": getattr(node, "uid", None),
                "triton_constant_args": dict(meta.get("triton_constant_args", {})),
            }
            stage_to_modules[int(stage_id)].append(module_data)

        assert stage_to_modules, (
            "load_training_dag: no FWD compute nodes with gm metadata found"
        )

        for stage_id in sorted(stage_to_modules.keys()):
            self._load_stage(
                stage_id=stage_id,
                modules_data=stage_to_modules[stage_id],
                a2a_boundaries={},
                use_activation_checkpointing=False,
            )

        # Build runtime adjacency on TrainingDAG nodes so run_dag can execute directly.
        for n in training_dag.nodes.values():
            n.data_preds = []
            n.data_succs = []
            n.temporal_preds = []
            n.temporal_succs = []
            if n.node_kind == "SEND_COMM":
                n.peer_pp_rank = n.node_meta.get("peer_pp_rank")
            elif n.node_kind == "RECV_COMM":
                n.peer_pp_rank = n.node_meta.get("peer_pp_rank")
            else:
                n.peer_pp_rank = None
        for e in training_dag.edges:
            if e.src_uid not in training_dag.nodes or e.dst_uid not in training_dag.nodes:
                continue
            src = training_dag.nodes[e.src_uid]
            dst = training_dag.nodes[e.dst_uid]
            if e.dep_kind == "temporal":
                src.temporal_succs.append(dst)
                dst.temporal_preds.append(src)
            else:
                src.data_succs.append(dst)
                dst.data_preds.append(src)

        for n in training_dag.nodes.values():
            n.task_type = _training_dag_task_type(n)
            mb = n.tag.get("MB", 0)
            st = n.tag.get("PP", 0)
            n.batches = [type("RuntimeBatch", (), {"stage_id": st, "mb_idx": mb})()]

        self.dag = training_dag
        self.sorted_dag_nodes = [
            training_dag.nodes[uid] for uid in _serial_topological_order(training_dag)
        ]

    def _clear_param_grads(self) -> None:
        """Restore standard PyTorch semantics: inactive params keep grad=None."""
        for ubid, trainable_idxs in self.bucket_trainable_param_idxs.items():
            args = self.bucket_fwd_args[ubid]
            for idx in trainable_idxs:
                t = args[idx]
                if t is not None:
                    t.grad = None

    def _accumulate_zero_param_grads_to_flat(
        self, ubid: Any | None, stream: "torch.cuda.Stream"
    ) -> None:
        """Accumulate ZeRO-managed param grads into the fp32 flat grad buffer."""
        if ubid is None or ubid not in self.zero_managed_ubids or self.dp_degree <= 1:
            return
        if ubid in self.grad_sharded_ubids:
            self._wait_pending_free(self.pending_grad_frees, ubid)
        specs = self.bucket_param_view_specs.get(ubid, [])
        if not specs:
            return
        with torch.cuda.stream(stream):
            flat_grads = self.bucket_flat_grads.get(ubid)
            if flat_grads is None:
                shard_info = self.param_shard_info.get(ubid)
                if shard_info is None:
                    return
                _shard_start, shard_size, _orig_numel = shard_info
                flat_grads = torch.zeros(
                    shard_size * self.dp_degree,
                    dtype=self.grad_buffer_dtype,
                    device=self.device,
                )
                self.bucket_flat_grads[ubid] = flat_grads
            for param, offset, numel, _shape in specs:
                grad = param.grad
                if grad is None:
                    continue
                flat_grads[offset:offset + numel].add_(grad.detach().reshape(-1).to(flat_grads.dtype))
                param.grad = None

    def _sync_payload_ubid(self, node: Any) -> Any | None:
        bk = self._node_bucket_key(node)
        if bk is not None:
            return bk
        if node.data_preds:
            pred_bk = self._node_bucket_key(node.data_preds[0])
            if pred_bk is not None:
                return pred_bk
        return None

    def _sync_payload_ubids(self, node: Any) -> list[Any]:
        sync_ubids = self._node_meta(node).get("sync_ubids")
        if sync_ubids:
            return list(sync_ubids)
        ubid = self._sync_payload_ubid(node)
        return [ubid] if ubid is not None else []

    def _wait_pending_free(self, pending: dict[Any, Future], ubid: Any | None) -> None:
        if ubid is None:
            return
        fut = pending.pop(ubid, None)
        if fut is not None:
            fut.result()

    def _drain_pending_frees(self) -> None:
        for pending in (self.pending_param_frees, self.pending_grad_frees):
            futures = list(pending.values())
            pending.clear()
            for fut in futures:
                fut.result()

    def _defer_free_full_params(self, ubid: Any | None, evt: torch.cuda.Event) -> None:
        if ubid is None or ubid not in self.param_sharded_ubids:
            return
        self._wait_pending_free(self.pending_param_frees, ubid)
        self.pending_param_frees[ubid] = self.cleanup_executor.submit(
            self._wait_then_free_full_params,
            ubid,
            evt,
        )

    def _defer_free_full_grads(self, ubid: Any | None, evt: torch.cuda.Event) -> None:
        if ubid is None or ubid not in self.grad_sharded_ubids:
            return
        self._wait_pending_free(self.pending_grad_frees, ubid)
        self.pending_grad_frees[ubid] = self.cleanup_executor.submit(
            self._wait_then_free_full_grads,
            ubid,
            evt,
        )

    def _wait_then_free_full_params(self, ubid: Any, evt: torch.cuda.Event) -> None:
        evt.synchronize()
        self._free_full_params(ubid)

    def _wait_then_free_full_grads(self, ubid: Any, evt: torch.cuda.Event) -> None:
        evt.synchronize()
        self._free_full_grads(ubid)

    def _init_task_buffer_refcounts(self, dag: Any) -> None:
        nodes_iter = dag.nodes.values() if isinstance(dag.nodes, dict) else dag.nodes
        self.task_buffer_refcounts = {
            node.uid: len(node.data_succs)
            for node in nodes_iter
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

    def run_dag(self, loss_fn=None):
        # Mark the entire iteration boundary for the NVTX timeline.
        iter_idx = getattr(self, "_iter_counter", 0)
        self._iter_counter = iter_idx + 1
        self._nvtx_push(f"iter_{iter_idx}_rank_{self.global_rank}")
        self._run_dag_body(loss_fn=loss_fn)
        self._nvtx_pop()

    def _run_dag_body(self, loss_fn=None):
        """Execute the loaded TrainingDAG in topological order.

        All inter-task data is routed through ``task_buffer``, keyed by the
        producing task's ``uid``.  Each task method receives its inputs as
        arguments and returns its outputs; ``run_dag`` is the only place that
        reads from and writes to ``task_buffer``.

        Synchronisation contract:
        - Before a SEND node: the node's stream waits on the compute event so the
          activation / gradient tensor is fully computed before the send begins.
        - Before a compute node that has a RECV predecessor: the compute stream
          waits on the recv_event so the recv buffer is populated before compute reads it.
        - RECV records a recv_event after the dist.recv completes.
        """
        assert self.dag is not None, "load_training_dag() must be called before run_dag()"
        assert self.sorted_dag_nodes is not None, "load_training_dag() must initialize sorted node order"
        self._drain_pending_frees()
        debug_enabled = self.logger.isEnabledFor(logging.DEBUG)
        # Reset per-iteration state
        self.task_buffer = {}
        self.task_buffer_refcounts = {}
        self.recv_events = {}
        self.a2a_events = {}
        self.ar_events = {}
        self.rs_events = {}
        self.ag_events = {}
        self.bwd_events = {}
        self._clear_param_grads()
        default_stream = self._default_stream()
        with torch.cuda.stream(default_stream):
            for buf in self.bucket_flat_grads.values():
                if buf is not None:
                    buf.zero_()
            for buf in self.bucket_rs_grads.values():
                if buf is not None:
                    buf.zero_()
        zero_evt = torch.cuda.Event()
        zero_evt.record(default_stream)
        for stream in self.streams.values():
            if stream is not default_stream:
                stream.wait_event(zero_evt)
        comp_events: dict = {}  # task_uid -> cuda.Event for compute tasks

        self._init_task_buffer_refcounts(self.dag)
        last_comp_event_by_stream: dict[str, torch.cuda.Event] = {}
        def _node_tag_str(node: Any) -> str:
            tag = getattr(node, "tag", None)
            if not isinstance(tag, dict) or not tag:
                return "{}"
            items = sorted(tag.items(), key=lambda kv: kv[0])
            return "{" + ",".join(f"{k}={v}" for k, v in items) + "}"

        def _wait_for_all_gather(compute_node: Any) -> None:
            compute_stream = self._stream_for_task(compute_node)
            for pred in compute_node.data_preds:
                if pred.task_type == TaskType.ALL_GATHER:
                    ag_evt = self.ag_events.get(pred.uid)
                    if ag_evt is not None:
                        compute_stream.wait_event(ag_evt)

        for node in self.sorted_dag_nodes:

            task_type = node.task_type
            batch = node.batches[0]
            mb_idx = batch.mb_idx
            ubid = self._node_bucket_key(node)  # None for non-compute nodes
            node_stream = self._stream_for_task(node)
            node_stream_id = self._stream_id(node)
            node_tag = _node_tag_str(node)

            if debug_enabled:
                self.logger.debug(
                    f"run_dag dispatch: {task_type.value} "
                    f"tag={node_tag}"
                )

            # Timeline marker visible in NVTX traces / PyTorch profiler.
            _task_label = f"{task_type.value}:{node_tag}:uid{node.uid}"
            self._nvtx_push(_task_label)
            _rf = self._rf_enter(_task_label)

            match task_type:

                case TaskType.SEND:
                    # data_preds[0] is the compute task whose output we send.
                    compute_node = node.data_preds[0]
                    node_stream.wait_event(comp_events[compute_node.uid])
                    send_data = self.task_buffer[compute_node.uid]["send_output"]
                    self._exec_send(send_data, node.peer_pp_rank, stream=node_stream)
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
                    comp_evt = last_comp_event_by_stream.get(self._stream_id(compute_node))
                    if comp_evt is not None:
                        node_stream.wait_event(comp_evt)
                    if compute_node.task_type == TaskType.FWD:
                        # FWD recv: pre-allocate based on target bucket's input metadata.
                        recv_ubid = self._node_bucket_key(compute_node)
                        recv_tensors = self._exec_recv_fwd(
                            recv_ubid, node.peer_pp_rank, stream=node_stream
                        )
                    else:
                        # BWD recv: pre-allocate based on FWD output shapes.
                        fwd_uid = compute_node.node_meta.get("fwd_uid")
                        fwd_key = (compute_node.node_meta.get("bucket_key"), fwd_uid)
                        shape_meta = self.task_buffer[("shape_ref",) + fwd_key]
                        recv_tensors = self._exec_recv_bwd(
                            shape_meta, node.peer_pp_rank, stream=node_stream
                        )
                    self.task_buffer[node.uid] = recv_tensors
                    recv_evt = torch.cuda.Event()
                    recv_evt.record(node_stream)
                    self.recv_events[node.uid] = recv_evt

                case TaskType.FWD_A2A:
                    fwd_pred = next(p for p in node.data_preds if p.task_type == TaskType.FWD)
                    node_stream.wait_event(comp_events[fwd_pred.uid])
                    tensor_idx = self._node_meta(node)["a2a_tensor_idx"]
                    # Pass-through: copy upstream FWD task_buffer, apply A2A only at tensor_idx.
                    fwd_buf = dict(self.task_buffer[fwd_pred.uid])
                    # FWD_A2A now owns the copy; release predecessor after consuming it.
                    self._release_task_buffer_uid(fwd_pred.uid)
                    detached_outs = list(fwd_buf["detached_outs"])
                    detached_outs[tensor_idx] = self._exec_a2a(
                        detached_outs[tensor_idx], stream=node_stream
                    ).requires_grad_(True)
                    fwd_buf["detached_outs"] = detached_outs
                    self.task_buffer[node.uid] = fwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(node_stream)
                    self.a2a_events[node.uid] = a2a_evt
                    # node_stream.wait_event(a2a_evt)

                case TaskType.BWD_A2A:
                    bwd_pred = next(
                        p for p in node.data_preds
                        if p.task_type in (TaskType.BWD, TaskType.BWD_I)
                    )
                    node_stream.wait_event(comp_events[bwd_pred.uid])
                    tensor_idx = self._node_meta(node)["a2a_tensor_idx"]
                    # Pass-through: copy upstream BWD/BWD_I task_buffer, apply reversed A2A only at tensor_idx.
                    bwd_buf = dict(self.task_buffer[bwd_pred.uid])
                    self._release_task_buffer_uid(bwd_pred.uid)
                    inp_grads = list(bwd_buf["inp_grads"])
                    grad_a2a_out = inp_grads[tensor_idx]
                    assert grad_a2a_out is not None, (
                        f"BWD_A2A tag={node_tag}: grad at a2a_tensor_idx={tensor_idx} is None"
                    )
                    if not grad_a2a_out.is_contiguous():
                        grad_a2a_out = grad_a2a_out.contiguous()
                    inp_grads[tensor_idx] = self._exec_a2a(grad_a2a_out, stream=node_stream)
                    bwd_buf["inp_grads"] = inp_grads
                    self.task_buffer[node.uid] = bwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(node_stream)
                    self.a2a_events[node.uid] = a2a_evt
                    # node_stream.wait_event(a2a_evt)

                case TaskType.ALL_REDUCE:
                    bwd_node = node.data_preds[0]
                    ar_ubids = self._sync_payload_ubids(node)
                    assert ar_ubids, (
                        f"ALL_REDUCE node uid={node.uid} has no sync_payload_ubids"
                    )
                    node_stream.wait_event(comp_events[bwd_node.uid])
                    for ubid in ar_ubids:
                        self._exec_all_reduce_grads(ubid, stream=node_stream)
                    ar_evt = torch.cuda.Event()
                    ar_evt.record(node_stream)
                    self.ar_events[node.uid] = ar_evt
                    self._release_task_buffer_uid(bwd_node.uid)

                case TaskType.REDUCE_SCATTER:
                    bwd_node = node.data_preds[0]
                    rs_ubid = self._node_bucket_key(node)
                    assert rs_ubid is not None, (
                        f"REDUCE_SCATTER node uid={node.uid} has no bucket_key"
                    )
                    node_stream.wait_event(comp_events[bwd_node.uid])
                    rs_bytes = self._exec_reduce_scatter(rs_ubid, stream=node_stream)
                    rs_evt = torch.cuda.Event()
                    rs_evt.record(node_stream)
                    self.rs_events[node.uid] = rs_evt
                    if rs_bytes:
                        self._defer_free_full_grads(rs_ubid, rs_evt)
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        assert False and "Param free should happen after a compute node"
                    self._release_task_buffer_uid(bwd_node.uid)

                case TaskType.ALL_GATHER:
                    ag_ubid = self._node_bucket_key(node)
                    assert ag_ubid is not None, (
                        f"ALL_GATHER node uid={node.uid} has no bucket_key"
                    )
                    self._exec_all_gather(ag_ubid, stream=node_stream)
                    ag_evt = torch.cuda.Event()
                    ag_evt.record(node_stream)
                    self.ag_events[node.uid] = ag_evt

                case TaskType.FWD:
                    # Wait on any RECV predecessor.
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        node_stream.wait_event(self.recv_events.pop(recv_pred.uid))

                    # Wait on any FWD_A2A predecessor.
                    a2a_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.FWD_A2A), None
                    )
                    if a2a_pred is not None and a2a_pred.uid in self.a2a_events:
                        node_stream.wait_event(self.a2a_events.pop(a2a_pred.uid))

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

                    fwd_out = self._forward_dag(ubid, mb_idx, input_tensors, node_stream)
                    self.task_buffer[node.uid] = fwd_out
                    # Shape ref for BWD RECV: store lightweight metadata (shape, dtype) so
                    # _exec_recv_bwd can pre-allocate receive buffers without holding live tensors.
                    bkey = node.node_meta.get("bucket_key")
                    fwd_key = (bkey, node.uid)
                    self.task_buffer[("shape_ref",) + fwd_key] = [(t.shape, t.dtype) for t in fwd_out["out_with_grad"]]
                    self.task_buffer[fwd_key] = fwd_out  # per-node FWD data for BWD
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self._defer_free_full_params(ubid, evt)

                case TaskType.BWD:
                    # Wait on upstream grad RECV if present.
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        node_stream.wait_event(self.recv_events.pop(recv_pred.uid))
                    _wait_for_all_gather(node)

                    # Wait on any BWD_A2A predecessor.
                    a2a_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.BWD_A2A), None
                    )
                    if a2a_pred is not None and a2a_pred.uid in self.a2a_events:
                        node_stream.wait_event(self.a2a_events.pop(a2a_pred.uid))

                    # Resolve outputs and upstream grads via per-mb ubid lookup.
                    fwd_uid = node.node_meta.get("fwd_uid")
                    fwd_key = (node.node_meta.get("bucket_key"), fwd_uid)
                    fwd_out = self.task_buffer[fwd_key]

                    if self._node_meta(node).get("compute_loss", False):
                        assert loss_fn is not None
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self._log_compute_loss_inputs(node, fwd_key, fwd_out)
                        with torch.cuda.stream(node_stream):
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

                    if a2a_pred is not None:
                        a2a_buf = self.task_buffer[a2a_pred.uid]
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
                        self._release_task_buffer_uid(a2a_pred.uid)
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
                    if self._node_meta(node).get("zero_alloc_full_grads_before"):
                        self._alloc_full_grads(ubid, node_stream)

                    bwd_out = self._backward_dag(
                        ubid, mb_idx, outputs_or_loss, upstream_grads,
                        pre_detach_outs, detached_outs, inp_with_grad,
                        fwd_out.get("out_with_grad"),
                        node_stream,
                    )
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
                    self._accumulate_zero_param_grads_to_flat(ubid, node_stream)
                    self.task_buffer[node.uid] = buf
                    # FWD stored the same dict under both node.uid and (ubid, mb_idx).
                    # Clear the dict in-place so any remaining alias stops keeping
                    # activations and autograd edges alive through the optimizer step.
                    fwd_out.clear()
                    del self.task_buffer[fwd_key]
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self._defer_free_full_params(ubid, evt)

                case TaskType.BWD_I:
                    # Wait on upstream grad RECV if present.
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.recv_events:
                        node_stream.wait_event(self.recv_events.pop(recv_pred.uid))
                    _wait_for_all_gather(node)

                    # Resolve stage_outputs_or_loss and output_grads.
                    fwd_uid = node.node_meta.get("fwd_uid")
                    fwd_key = (node.node_meta.get("bucket_key"), fwd_uid)
                    fwd_out = self.task_buffer[fwd_key]

                    if self._node_meta(node).get("compute_loss", False):
                        assert loss_fn is not None
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self._log_compute_loss_inputs(node, fwd_key, fwd_out)
                        with torch.cuda.stream(node_stream):
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
                    if (
                        bwd_a2a_pred is not None
                        and not self._node_meta(node).get("compute_loss", False)
                        and recv_pred is None
                    ):
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

                    with torch.cuda.stream(node_stream):
                        dinputs, param_groups = self._bucket_backward_input(
                            stage_outputs_or_loss, output_grads, input_values, iter(weights)
                        )
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
                    # Propagate input grad to SEND (if this is the first bucket and has predecessors).
                    if dinputs:
                        self.task_buffer[node.uid]["send_output"] = list(dinputs)
                    # FWD stored the same dict under both node.uid and (ubid, mb_idx).
                    # Clear the dict in-place so any remaining alias stops keeping
                    # activations and autograd edges alive through the optimizer step.
                    fwd_out.clear()
                    del stage_outputs_or_loss, fwd_out
                    del self.task_buffer[fwd_key]
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self._defer_free_full_params(ubid, evt)

                case TaskType.BWD_W:
                    _wait_for_all_gather(node)
                    if self._node_meta(node).get("zero_alloc_full_grads_before"):
                        self._alloc_full_grads(ubid, node_stream)
                    # data_preds[0] is the BWD_I task for bkt=K-1 (head of chain).
                    # For bkt<K-1 the chain-link edge (BWD_W predecessor) is prepended
                    # before Pass 2 appends the BWD_I edge, so data_preds[0] would be
                    # the wrong task.  Find BWD_I explicitly by type to be safe.
                    bwdi_node = next(p for p in node.data_preds if p.task_type == TaskType.BWD_I)
                    param_groups = self.task_buffer[bwdi_node.uid]["param_groups"]
                    b_fwd_args = self.bucket_fwd_args[ubid]
                    weights = [b_fwd_args[i] for i in self.bucket_param_idxs[ubid]
                               if b_fwd_args[i] is not None]
                    with torch.cuda.stream(node_stream):
                        self._bucket_backward_weight(iter(weights), param_groups, ubid=ubid, mb_idx=mb_idx)
                    self._accumulate_zero_param_grads_to_flat(ubid, node_stream)
                    self.task_buffer[node.uid] = {}
                    # BWD_I buffer is no longer needed: _bucket_backward_weight already
                    # deleted param_groups["intermediates"] and param_groups["grads"] and
                    # the autograd graph was freed by retain_graph=False.  Drop the shell.
                    self._release_task_buffer_uid(bwdi_node.uid)
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    self.bwd_events[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self._defer_free_full_params(ubid, evt)

                case TaskType.UPD:
                    self._update(node_stream)

                case TaskType.ORDER_DUMMY:
                    pass

            self._rf_exit(_rf)
            self._nvtx_pop()

    def _log_compute_loss_inputs(self, node: Any, fwd_key: Any, fwd_out: dict) -> None:
        def _summarize_value(value: Any) -> str:
            if isinstance(value, torch.Tensor):
                return (
                    f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                    f"requires_grad={value.requires_grad}, device={value.device})"
                )
            if isinstance(value, (list, tuple)):
                return "[" + ", ".join(_summarize_value(v) for v in value) + "]"
            if value is None:
                return "None"
            return type(value).__name__

        self.logger.debug(
            "compute_loss inputs: rank=%s node_uid=%s node_type=%s tag=%s "
            "fwd_key=%s labels=%s out_with_grad=%s pre_detach_outs=%s "
            "detached_outs=%s send_output=%s",
            self.global_rank,
            getattr(node, "uid", None),
            getattr(getattr(node, "task_type", None), "value", None),
            getattr(node, "tag", None),
            fwd_key,
            _summarize_value(self.labels),
            _summarize_value(fwd_out.get("out_with_grad")),
            _summarize_value(fwd_out.get("pre_detach_outs")),
            _summarize_value(fwd_out.get("detached_outs")),
            _summarize_value(fwd_out.get("send_output")),
        )

    def _exec_send(self, send_data, peer_pp_rank: int, stream) -> None:
        """Send tensors to peer_pp_rank on the stream assigned to the SEND node.

        The caller (run_dag) must have already made the node stream wait on the
        compute event before calling this method.
        """
        global_dst_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)

        with torch.cuda.stream(stream):
            tensors = send_data if isinstance(send_data, (list, tuple)) else [send_data]
            use_lo_hi = global_dst_rank > self.global_rank
            pp_group = self.pp_lo_hi if use_lo_hi else self.pp_hi_lo
            for tensor in tensors:
                dist.send(tensor, dst=global_dst_rank, group=pp_group)

    def _exec_recv_fwd(self, recv_ubid: Any, peer_pp_rank: int, stream) -> list:
        """Receive FWD activations from peer_pp_rank into pre-allocated buffers.

        Buffer shapes are derived from the target bucket's stored input metadata.
        Returns the received tensor list (stored in task_buffer by run_dag).
        """
        global_src_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)

        buf = [
            torch.empty(shape, dtype=dtype, requires_grad=requires_grad, device=self.device)
            for shape, dtype, requires_grad in self.forward_input_meta[recv_ubid]
        ]
        with torch.cuda.stream(stream):
            use_hi_lo = global_src_rank > self.global_rank
            pp_group = self.pp_hi_lo if use_hi_lo else self.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def _exec_recv_bwd(self, shape_meta: list, peer_pp_rank: int, stream) -> list:
        """Receive BWD upstream gradients from peer_pp_rank.

        Buffer shapes are derived from shape_meta, a list of (shape, dtype) tuples
        stored in task_buffer[("shape_ref", ubid)] after the corresponding FWD.
        Returns the received gradient list (stored in task_buffer by run_dag).
        """
        global_src_rank = _get_rank(peer_pp_rank, self.dp_rank, self.pp_degree)

        buf = [torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in shape_meta]
        with torch.cuda.stream(stream):
            use_hi_lo = global_src_rank > self.global_rank
            pp_group = self.pp_hi_lo if use_hi_lo else self.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def _exec_a2a(self, input_tensor: torch.Tensor, stream) -> torch.Tensor:
        """Apply dist.all_to_all_single on the node stream and return the output tensor.

        Used for both FWD_A2A and BWD_A2A — the operation is symmetric.
        Stream waits and event recording are handled by run_dag.
        """
        output_buf = torch.empty_like(input_tensor, device=self.device)
        with torch.cuda.stream(stream):
            dist.all_to_all_single(output_buf, input_tensor, group=self.ep_group)
        return output_buf

    def _forward_dag(self, ubid: Any, mb_idx: int, input_tensors, compute_stream) -> dict:
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
        ubid: Any,
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

    @staticmethod
    def _grad_with_param_layout(weight: Parameter, grad: torch.Tensor) -> torch.Tensor:
        if (
            grad.dtype == weight.dtype
            and grad.device == weight.device
            and grad.layout == weight.layout
            and tuple(grad.stride()) == tuple(weight.stride())
        ):
            return grad
        if weight.layout == torch.strided and grad.layout != torch.strided:
            grad = grad.to_dense()
        out = torch.empty_strided(
            tuple(weight.shape),
            tuple(weight.stride()),
            dtype=weight.dtype,
            device=weight.device,
        )
        out.copy_(grad)
        return out

    def _fused_backward(self, stage_outputs_or_loss: list, output_grads, weights: list) -> None:
        """Fused backward writing weight grads directly into ``weight.grad``.

        Used for the first pipeline stage, whose inputs (token ids) do not
        require grad. The split B/W path relies on input-grad prehooks firing to
        capture intermediate grads, but with no input grads to compute those
        prehooks never run and stage 0 would never accumulate weight gradients.
        Computing dW in one pass here (and no-opping the paired BWD_W) keeps the
        first stage training without redundant work.
        """
        wts = [w for w in weights if w.requires_grad]
        if not wts:
            return
        dweights = torch.autograd.grad(
            stage_outputs_or_loss,
            inputs=wts,
            grad_outputs=output_grads,
            retain_graph=False,
            allow_unused=True,
        )
        for w, dw in zip(wts, dweights):
            if dw is None:
                continue
            dw = self._grad_with_param_layout(w, dw)
            if w.grad is None:
                w.grad = dw
            else:
                if (
                    w.grad.dtype != w.dtype
                    or w.grad.device != w.device
                    or w.grad.layout != w.layout
                    or tuple(w.grad.stride()) != tuple(w.stride())
                ):
                    w.grad = self._grad_with_param_layout(w, w.grad)
                w.grad += dw

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

        weights = list(weights)

        if output_grads is None:
            output_grads = [torch.ones_like(o) for o in stage_outputs_or_loss]

        input_values = [inp for inp in input_values if inp.requires_grad]
        if not input_values:
            # First pipeline stage: no input gradient to compute or send. The
            # split B/W mechanism would never fire its prehooks here, so compute
            # weight grads in a single fused backward and return empty
            # param_groups so the paired BWD_W is a no-op.
            self._fused_backward(stage_outputs_or_loss, output_grads, weights)
            for i, t in enumerate(stage_outputs_or_loss):
                if not isinstance(t, torch.Tensor):
                    continue
                stage_outputs_or_loss[i] = t.detach()
            return (), []

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

        # Match torch.distributed.pipelining.stage_backward_input(): once dinputs
        # are computed, these outputs are no longer needed by the later weight
        # pass, so detach them to let autograd release the dead part of the graph.
        for i, t in enumerate(stage_outputs_or_loss):
            if not isinstance(t, torch.Tensor):
                continue
            stage_outputs_or_loss[i] = t.detach()

        for handle in handles:
            handle.remove()

        return dinputs, param_groups

    def _bucket_backward_weight(self, weights, param_groups: list, ubid: Any | None = None, mb_idx: int | None = None) -> None:
        """Compute weight gradients for a single bucket (BWD_W).

        Matches ``torch.distributed.pipelining.stage_backward_weight`` exactly,
        with "stage" replaced by "bucket".  Accumulates gradients into param.grad.
        """
        if not param_groups:
            # First-stage BWD_W is a no-op: BWD_I already ran the fused backward and
            # wrote weight.grad directly, so there are no param_groups to consume.
            # Skip the grad-accumulator preamble (one throwaway view per weight).
            return

        grad_acc_to_weight: Dict[Node, Parameter] = {}
        _grad_with_param_layout = self._grad_with_param_layout

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
                dw = _grad_with_param_layout(weight, dw)
                if weight.grad is None:
                    weight.grad = dw
                else:
                    if (
                        weight.grad.dtype != weight.dtype
                        or weight.grad.device != weight.device
                        or weight.grad.layout != weight.layout
                        or tuple(weight.grad.stride()) != tuple(weight.stride())
                    ):
                        weight.grad = _grad_with_param_layout(weight, weight.grad)
                    weight.grad += dw

    def _exec_all_reduce_grads(self, ubid: Any, stream) -> int:
        """All-reduce each materialized parameter grad for *ubid* on *stream*.

        Returns the total payload bytes actually dispatched at runtime.
        """
        assert ubid is not None, "_exec_all_reduce_grads requires a non-None ubid"
        if not self._has_trainable_params_for_collective(ubid, "_exec_all_reduce_grads"):
            return 0
        args = self.bucket_fwd_args.get(ubid, [])
        trainable_idxs = self.bucket_trainable_param_idxs.get(ubid, [])
        grad_tensors = []
        for idx in trainable_idxs:
            param = args[idx]
            assert param is not None, (
                f"_exec_all_reduce_grads: ubid={ubid} idx={idx} param is None"
            )
            assert param.grad is not None, (
                f"_exec_all_reduce_grads: ubid={ubid} idx={idx} param.grad is None"
            )
            grad_tensors.append((idx, param.grad))

        total_bytes = sum(grad.numel() * grad.element_size() for _, grad in grad_tensors)
        with torch.cuda.stream(stream):
            for _, grad in grad_tensors:
                dist.all_reduce(grad, group=self.dp_group)
        return total_bytes

    def _has_trainable_params_for_collective(self, ubid: Any, op_name: str) -> bool:
        trainable_idxs = self.bucket_trainable_param_idxs.get(ubid, [])
        if trainable_idxs:
            return True
        self.logger.warning(
            "%s: skipping collective for ubid=%s because it has no trainable param indices",
            op_name,
            ubid,
        )
        return False

    def _alloc_full_params(self, ubid: Any) -> None:
        assert ubid is not None, "_alloc_full_params requires a non-None ubid"
        assert ubid in self.param_sharded_ubids, (
            f"_alloc_full_params: ubid={ubid} is not in param_sharded_ubids={self.param_sharded_ubids}"
        )
        self._wait_pending_free(self.pending_param_frees, ubid)
        assert self.param_shard_info.get(ubid) is not None, (
            f"_alloc_full_params: missing param_shard_info for ubid={ubid}"
        )
        full = self.bucket_flat_params.get(ubid)
        assert full is not None, (
            f"_alloc_full_params: missing bucket_flat_params buffer for ubid={ubid}"
        )
        specs = self.bucket_param_view_specs.get(ubid, [])
        assert specs, f"_alloc_full_params: missing bucket_param_view_specs for ubid={ubid}"
        storage = full.untyped_storage()
        required_bytes = full.numel() * full.element_size()
        storage.resize_(required_bytes)
        self.logger.debug(
            f"[alloc_full_params] rank={self.global_rank} ubid={ubid}: "
            f"numel={full.numel()} required_bytes={required_bytes} "
            f"storage_size={storage.size()} fresh={self.full_params_fresh.get(ubid, False)}"
        )
        for param, offset, numel, shape in specs:
            param.data = full[offset:offset + numel].view(shape)
            param.requires_grad_(True)
        param_names = self.bucket_param_names.get(ubid, [])
        zero_storage = []
        for i, (param, offset, numel, shape) in enumerate(specs):
            p_storage = param.untyped_storage()
            if p_storage.size() == 0:
                name = param_names[i] if i < len(param_names) else f"param{i}"
                zero_storage.append(
                    f"{name}: offset={offset} numel={numel} shape={shape} stride={tuple(param.stride())}"
                )
        assert not zero_storage, (
            f"[alloc_full_params_zero_storage] rank={self.global_rank} ubid={ubid}: "
            + " | ".join(zero_storage)
        )

    def _free_full_params(self, ubid: Any) -> None:
        assert ubid is not None, "_free_full_params requires a non-None ubid"
        assert ubid in self.param_sharded_ubids, (
            f"_free_full_params: ubid={ubid} is not in param_sharded_ubids={self.param_sharded_ubids}"
        )
        full = self.bucket_flat_params.get(ubid)
        assert full is not None, (
            f"_free_full_params: missing bucket_flat_params buffer for ubid={ubid}"
        )
        storage = full.untyped_storage()
        self.logger.debug(
            f"[free_full_params] rank={self.global_rank} ubid={ubid}: "
            f"storage_size_before={storage.size()} fresh_before={self.full_params_fresh.get(ubid, False)}"
        )
        storage.resize_(0)
        self.full_params_fresh[ubid] = False

    def _alloc_full_grads(self, ubid: Any, stream: "torch.cuda.Stream") -> None:
        assert ubid is not None, "_alloc_full_grads requires a non-None ubid"
        assert ubid in self.grad_sharded_ubids, (
            f"_alloc_full_grads: ubid={ubid} is not in grad_sharded_ubids={self.grad_sharded_ubids}"
        )
        self._wait_pending_free(self.pending_grad_frees, ubid)
        specs = self.bucket_param_view_specs.get(ubid, [])
        assert specs, f"_alloc_full_grads: missing bucket_param_view_specs for ubid={ubid}"
        shard_info = self.param_shard_info.get(ubid)
        assert shard_info is not None, (
            f"_alloc_full_grads: missing param_shard_info for ubid={ubid}"
        )
        shard_size = shard_info[1]
        with torch.cuda.stream(stream):
            if self.bucket_flat_grads.get(ubid) is None:
                self.bucket_flat_grads[ubid] = torch.zeros(
                    shard_size * self.dp_degree,
                    dtype=self.grad_buffer_dtype,
                    device=self.device,
                )
            if self.bucket_rs_grads.get(ubid) is None:
                self.bucket_rs_grads[ubid] = torch.zeros(
                    shard_size, dtype=self.grad_buffer_dtype, device=self.device
                )

    def _free_full_grads(self, ubid: Any) -> None:
        assert ubid is not None, "_free_full_grads requires a non-None ubid"
        assert ubid in self.grad_sharded_ubids, (
            f"_free_full_grads: ubid={ubid} is not in grad_sharded_ubids={self.grad_sharded_ubids}"
        )
        specs = self.bucket_param_view_specs.get(ubid, [])
        assert specs, f"_free_full_grads: missing bucket_param_view_specs for ubid={ubid}"
        for param, *_ in specs:
            param.grad = None
        self.bucket_flat_grads[ubid] = None

    def _exec_reduce_scatter(self, ubid: Any, stream) -> int:
        assert ubid is not None, "_exec_reduce_scatter requires a non-None ubid"
        if not self._has_trainable_params_for_collective(ubid, "_exec_reduce_scatter"):
            return 0
        assert ubid in self.grad_sharded_ubids, (
            f"_exec_reduce_scatter: ubid={ubid} is not in grad_sharded_ubids={self.grad_sharded_ubids}"
        )
        assert self.param_shard_info.get(ubid) is not None, (
            f"_exec_reduce_scatter: missing param_shard_info for ubid={ubid}"
        )
        flat_grads = self.bucket_flat_grads.get(ubid)
        rs_out = self.bucket_rs_grads.get(ubid)
        assert flat_grads is not None, (
            f"_exec_reduce_scatter: missing bucket_flat_grads buffer for ubid={ubid}"
        )
        assert rs_out is not None, (
            f"_exec_reduce_scatter: missing bucket_rs_grads buffer for ubid={ubid}"
        )
        with torch.cuda.stream(stream):
            total_bytes = flat_grads.numel() * flat_grads.element_size()
            tmp = torch.empty_like(rs_out)
            dist.reduce_scatter_tensor(tmp, flat_grads, group=self.dp_group)
            rs_out.add_(tmp)
            return total_bytes

    def _exec_all_gather(
        self,
        ubid: Any,
        stream,
    ) -> int:
        assert ubid is not None, "_exec_all_gather requires a non-None ubid"
        if not self._has_trainable_params_for_collective(ubid, "_exec_all_gather"):
            return 0
        assert ubid in self.param_sharded_ubids, (
            f"_exec_all_gather: ubid={ubid} is not in param_sharded_ubids={self.param_sharded_ubids}"
        )
        self._alloc_full_params(ubid)
        flat_params = self.bucket_flat_params.get(ubid)
        shard_in = self.bucket_shard_params.get(ubid)
        assert flat_params is not None, (
            f"_exec_all_gather: missing bucket_flat_params buffer for ubid={ubid}"
        )
        assert shard_in is not None, (
            f"_exec_all_gather: missing bucket_shard_params buffer for ubid={ubid}"
        )
        assert not self.full_params_fresh.get(ubid, False), (
            f"_exec_all_gather: ubid={ubid} dispatched but full params are already fresh; "
            "DAG is constructing a redundant ALL_GATHER"
        )
        with torch.cuda.stream(stream):
            self.logger.debug(
                f"[all_gather_begin] rank={self.global_rank} ubid={ubid}: "
                f"flat_numel={flat_params.numel()} flat_storage={flat_params.untyped_storage().size()} "
                f"shard_numel={shard_in.numel()} shard_storage={shard_in.untyped_storage().size()}"
            )
            dist.all_gather_into_tensor(flat_params, shard_in, group=self.dp_group)
            self.full_params_fresh[ubid] = True
            self.logger.debug(
                f"[all_gather_end] rank={self.global_rank} ubid={ubid}: "
                f"flat_storage={flat_params.untyped_storage().size()}"
            )
            return flat_params.numel() * flat_params.element_size()

    def _update(self, stream, *deps):
        self._drain_pending_frees()
        if self.bucket_shard_optims:
            for evt in self.rs_events.values():
                stream.wait_event(evt)

            for ubid, shard_optim in self.bucket_shard_optims.items():
                shard_param = self.bucket_shard_params[ubid]
                rs_grad = self.bucket_rs_grads.get(ubid)
                if rs_grad is None:
                    continue
                with torch.cuda.stream(stream):
                    shard_param.grad = rs_grad.to(shard_param.dtype)
                    shard_optim.step()
                shard_param.grad = None

            for ubid in self.param_sharded_ubids:
                for param, *_ in self.bucket_param_view_specs.get(ubid, []):
                    param.grad = None
                full = self.bucket_flat_params.get(ubid)
                if full is not None:
                    storage = full.untyped_storage()
                    if storage.size() != 0:
                        storage.resize_(0)
                self.full_params_fresh[ubid] = False

            losses = self.loss
            self.loss.clear()
            torch.cuda.synchronize()
            return losses

        for ar_evt in self.ar_events.values():
            stream.wait_event(ar_evt)

        for ubid, optim in self.bucket_optims.items():
            if optim is None:
                continue
            bwd_evt = self.bwd_events.get(ubid)
            if bwd_evt is not None:
                stream.wait_event(bwd_evt)

            with torch.cuda.stream(stream):
                optim.step()

        losses = self.loss
        self.loss.clear()

        torch.cuda.synchronize()

        return {
            "losses": losses,
        }
