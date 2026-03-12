"""
Bucketed ZeRO optimizer stages for Piper.

Each trainable parameter is divided into dp_degree contiguous shards; each DP rank
owns and optimises one shard. Parameters are grouped into 25 MB gradient-sync buckets
ordered by reverse forward-use (approximating backward arrival order), enabling
overlap between communication and backward computation.

Per-step communication pipeline (per bucket, pipelined across buckets):
  1. stage-1: all_reduce flat gradients (async, on comm_stream), or
     stage-2: reduce_scatter gradient flat buffer → local grad shard (async)
  2. Optimizer step on local param shard                      (comp_stream)
  3. all_gather local param shard → flat param buffer         (async, on comm_stream)
  4. Scatter flat param buffer back to p.data

See docs/zero1_design.md for the full design rationale.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Type

import torch
import torch.distributed as dist
import torch.fx
from torch.fx.passes.split_module import split_module
from torch.nn import Parameter
from torch.optim import Optimizer

from .piper_utils import create_logger, LOG_LEVEL

logger = create_logger("piper_zero", LOG_LEVEL)

# Default bucket size in bytes — matches PyTorch DDP's default of 25 MB.
BUCKET_SIZE_BYTES_DEFAULT: int = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# FX graph analysis
# ---------------------------------------------------------------------------

def _get_param_forward_order(
    gm: torch.fx.GraphModule,
    param_idxs: list[int],
    params: list[Parameter],
) -> list[Parameter]:
    """
    Return ``params`` sorted by the first position in ``gm``'s FX graph where
    each parameter placeholder is used as an input to a computation node.

    Parameters used earlier in the forward graph (lower node index) come first.
    Parameters never referenced as inputs are appended last in original order.

    Args:
        gm: The FX GraphModule for this pipeline stage.
        param_idxs: Indices of parameter placeholders within the graph's full
            placeholder list (matches the positional order of ``forward_args``).
        params: Trainable parameter tensors, one per entry in ``param_idxs``.

    Returns:
        ``params`` in forward-use order.
    """
    placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    param_idx_set = set(param_idxs)

    # Build a set of placeholder nodes that correspond to parameters.
    param_ph_nodes: set[torch.fx.Node] = {
        placeholder_nodes[i]
        for i in param_idxs
        if i < len(placeholder_nodes)
    }

    # Walk graph nodes to find the first computation node that uses each param.
    first_use_pos: dict[torch.fx.Node, int] = {}
    for node_pos, node in enumerate(gm.graph.nodes):
        if node.op == "placeholder":
            continue
        for inp in node.all_input_nodes:
            if inp in param_ph_nodes and inp not in first_use_pos:
                first_use_pos[inp] = node_pos

    # Sort params by first-use position.
    param_ph_by_idx = {
        i: placeholder_nodes[i]
        for i in param_idxs
        if i < len(placeholder_nodes)
    }
    pairs: list[tuple[Parameter, float]] = []
    for idx, p in zip(param_idxs, params):
        ph = param_ph_by_idx.get(idx)
        pos = first_use_pos.get(ph, math.inf) if ph is not None else math.inf
        pairs.append((p, pos))

    pairs.sort(key=lambda x: x[1])
    return [p for p, _ in pairs]


def _build_bucket_segments(
    gm: torch.fx.GraphModule,
    param_idxs: list[int],
    params: list[Parameter],
    param_to_bucket: dict[int, "ZeROOneBucket"],
) -> list[tuple[int, "ZeROOneBucket"]]:
    """
    For each bucket, find the last graph position at which any of its parameters
    are used as an input to a computation node.

    Returns a list of ``(last_use_pos, bucket)`` tuples sorted in ascending
    order of ``last_use_pos`` — i.e. in *forward* order.

    Args:
        gm: The FX GraphModule for this pipeline stage.
        param_idxs: Indices of parameter placeholder nodes in the graph.
        params: Trainable parameter tensors (one per entry in ``param_idxs``).
        param_to_bucket: Mapping from ``id(param)`` to owning bucket.
    """
    placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]

    # Map placeholder node → bucket.
    ph_to_bucket: dict[torch.fx.Node, "ZeROOneBucket"] = {}
    for i, p in zip(param_idxs, params):
        if i < len(placeholder_nodes):
            bucket = param_to_bucket.get(id(p))
            if bucket is not None:
                ph_to_bucket[placeholder_nodes[i]] = bucket

    # Walk graph to find last-use position per bucket.
    last_use_pos: dict[int, int] = {}  # bucket_id → node position
    for node_pos, node in enumerate(gm.graph.nodes):
        if node.op == "placeholder":
            continue
        for inp in node.all_input_nodes:
            if inp in ph_to_bucket:
                bid = ph_to_bucket[inp].bucket_id
                if bid not in last_use_pos or node_pos > last_use_pos[bid]:
                    last_use_pos[bid] = node_pos

    # Collect unique buckets and sort by last-use position.
    seen: dict[int, "ZeROOneBucket"] = {b.bucket_id: b for b in ph_to_bucket.values()}
    segments = [(last_use_pos.get(bid, 0), bucket) for bid, bucket in seen.items()]
    segments.sort(key=lambda x: x[0])
    return segments


def _make_zero3_split_gm(
    gm: torch.fx.GraphModule,
    fwd_segments: list[tuple[int, "ZeROOneBucket"]],
) -> torch.fx.GraphModule:
    """
    Split *gm* into ``len(fwd_segments)`` sub-modules using
    ``torch.fx.passes.split_module``.

    Partition k contains all computation nodes whose topological position falls
    at or before ``fwd_segments[k][0]`` (the last-use position of bucket k).
    Placeholder nodes are all assigned to partition 0; ``split_module``
    automatically threads them through as inputs to later partitions when needed.

    Returns the new wrapper ``GraphModule`` whose sub-modules are named
    ``submod_0``, ``submod_1``, …
    """
    topo_idx: dict[torch.fx.Node, int] = {
        node: i for i, node in enumerate(gm.graph.nodes)
    }
    num_partitions = len(fwd_segments)

    def split_callback(node: torch.fx.Node) -> int:
        if node.op == "placeholder":
            return 0
        pos = topo_idx[node]
        for k, (last_pos, _) in enumerate(fwd_segments):
            if pos <= last_pos:
                return k
        return num_partitions - 1

    return split_module(gm, gm, split_callback)


# ---------------------------------------------------------------------------
# Bucket
# ---------------------------------------------------------------------------

class ZeROOneBucket:
    """
    One gradient-synchronisation bucket for ZeRO-1.

    Holds a flat parameter buffer and a flat gradient buffer covering all params
    in the bucket. The local shard (owned by this DP rank) is a contiguous slice
    of those flat buffers. One optimizer instance manages the local param shard.

    Lifecycle per training step:
      ``on_grad_accumulated`` → (when bucket full) ``launch_reduce_scatter``
      → (in finalize_step) ``optimizer_step`` → ``launch_all_gather``
      → ``copy_flat_to_params`` → ``reset_step_state``
    """

    def __init__(
        self,
        bucket_id: int,
        params: list[Parameter],
        dp_rank: int,
        dp_degree: int,
        dp_group: dist.ProcessGroup,
        device: str,
        dtype: torch.dtype,
        optim_class: Callable[[list[Parameter]], Optimizer],
        num_mbs: int,
        zero_stage: int,
    ) -> None:
        self.bucket_id = bucket_id
        self.params = params
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.dp_group = dp_group
        self.num_mbs = num_mbs
        self.zero_stage = zero_stage

        # ------------------------------------------------------------------
        # Flat buffer layout: [p0 elems | p1 elems | ... | padding]
        # ------------------------------------------------------------------
        # Capture original shapes and numels before any p.data reassignment.
        self.param_original_shapes: list[torch.Size] = [p.shape for p in params]
        self.param_offsets: list[int] = []
        self.param_numels: list[int] = []
        total_numel = 0
        for p in params:
            self.param_offsets.append(total_numel)
            self.param_numels.append(p.numel())
            total_numel += p.numel()

        # Pad so that padded_numel is divisible by dp_degree (for equal shards).
        self.shard_numel: int = math.ceil(total_numel / dp_degree)
        self.padded_numel: int = self.shard_numel * dp_degree
        self.shard_start: int = dp_rank * self.shard_numel
        self.shard_end: int = self.shard_start + self.shard_numel

        # Persistent flat buffers (allocated once, reused each step).
        self.flat_param_buf: torch.Tensor = torch.empty(
            self.padded_numel, device=device, dtype=dtype
        )
        # ZeRO-1: persistent full gradient buffer (all_reduce writes into it).
        # ZeRO-2: None — a transient buffer is allocated per-step inside
        #         launch_grad_sync and released once reduce_scatter returns.
        self.flat_grad_buf: Optional[torch.Tensor] = (
            torch.zeros(self.padded_numel, device=device, dtype=dtype)
            if zero_stage == 1
            else None
        )

        # Initialise flat_param_buf from current parameter values.
        for p, off, numel in zip(params, self.param_offsets, self.param_numels):
            self.flat_param_buf[off : off + numel].copy_(p.data.view(-1))

        # Local shard: a separate tensor (not a view of flat_param_buf) to
        # avoid aliasing when all_gather writes into flat_param_buf.
        self.local_param_shard: Parameter = Parameter(
            self.flat_param_buf[self.shard_start : self.shard_end].clone(),
            requires_grad=True,
        )
        # Receives the reduce-scattered gradient for the local shard.
        self.local_grad_shard: torch.Tensor = torch.zeros(
            self.shard_numel, device=device, dtype=dtype
        )

        # One optimizer per bucket, managing only the local param shard.
        self.optimizer: Optimizer = optim_class([self.local_param_shard])

        # ------------------------------------------------------------------
        # Per-step state (reset by reset_step_state)
        # ------------------------------------------------------------------
        # Counts how many microbatch backward passes have touched each param.
        self._mb_accumulated: dict[int, int] = {id(p): 0 for p in params}
        # Number of params that have fully accumulated all num_mbs microbatches.
        self._ready_param_count: int = 0

        # Async collective handles (None until launched).
        self.grad_sync_handle: Optional[dist.Work] = None
        self.all_gather_handle: Optional[dist.Work] = None

        # ------------------------------------------------------------------
        # ZeRO-3: CUDA events for all-gather pipelining.
        # ``_ag_fwd_event`` fires when the forward all-gather into
        # ``flat_param_buf`` completes; ``_ag_bwd_event`` for backward.
        # ------------------------------------------------------------------
        if zero_stage == 3:
            self._ag_fwd_event: Optional[torch.cuda.Event] = torch.cuda.Event()
            self._ag_bwd_event: Optional[torch.cuda.Event] = torch.cuda.Event()
            # Redirect each param's .data to only its owned shard slice inside
            # local_param_shard.  Full-shape views are installed temporarily by
            # swap_to_full_params() in the per-bucket pre-hooks and torn down by
            # restore_to_shard_params() in the corresponding post-hooks.
            for p, off, numel in zip(self.params, self.param_offsets, self.param_numels):
                flat_lo = max(off, self.shard_start)
                flat_hi = min(off + numel, self.shard_end)
                if flat_lo < flat_hi:
                    p.data = self.local_param_shard[
                        flat_lo - self.shard_start : flat_hi - self.shard_start
                    ]
                    logger.debug(f"Bucket {bucket_id} param shard: flat[{flat_lo}:{flat_hi}] → {p.shape}")
                else:
                    # This param has no owned elements on this rank.
                    p.data = self.local_param_shard.new_empty(0)
        else:
            self._ag_fwd_event = None
            self._ag_bwd_event = None

        logger.debug(
            f"ZeROOneBucket {bucket_id}: {len(params)} params, "
            f"{total_numel} elems, shard [{self.shard_start}:{self.shard_end}]"
        )

    # ------------------------------------------------------------------
    # Per-step trigger (called from grad hook)
    # ------------------------------------------------------------------

    def on_grad_accumulated(self, param: Parameter) -> bool:
        """
        Record one microbatch gradient accumulation for *param*.

        Returns ``True`` when every parameter in this bucket has accumulated
        gradients from all ``num_mbs`` microbatches, signalling that the bucket
        is ready for reduce_scatter.
        """
        pid = id(param)
        self._mb_accumulated[pid] += 1
        if self._mb_accumulated[pid] == self.num_mbs:
            self._ready_param_count += 1
        return self._ready_param_count == len(self.params)

    # ------------------------------------------------------------------
    # Communication and computation
    # ------------------------------------------------------------------

    def launch_grad_sync(
        self,
        comm_stream: torch.cuda.Stream,
        # producer_stream: torch.cuda.Stream,
    ) -> dist.Work:
        """
        Copy accumulated per-param gradients into the flat gradient buffer, then
        launch stage-specific async gradient synchronisation on *comm_stream*.

        - stage 1: ``all_reduce`` on ``flat_grad_buf`` (full grads preserved)
        - stage 2: ``reduce_scatter_tensor`` into ``local_grad_shard``

        Returns the ``dist.Work`` handle for the caller to wait on.
        """
        # comm_stream.wait_stream(producer_stream)
        with torch.cuda.stream(comm_stream):
            if self.zero_stage == 1:
                for p, off in zip(self.params, self.param_offsets):
                    if p.grad is not None:
                        self.flat_grad_buf[off : off + p.numel()].copy_(p.grad.view(-1))
                    else:
                        self.flat_grad_buf[off : off + p.numel()].zero_()
                handle = dist.all_reduce(
                    self.flat_grad_buf,
                    op=dist.ReduceOp.AVG,
                    group=self.dp_group,
                    # async_op=True,
                )
            else:
                # Transient buffer: exists only for the duration of this call.
                # Because the collective is currently synchronous (async_op
                # commented out), the buffer is safe to release on return.
                # NOTE: if async_op=True is re-enabled, store this tensor as
                # self._tmp_flat_grad_buf and clear it in prepare_local_grad_shard
                # after grad_sync_handle.wait() so it stays alive long enough.
                tmp_flat = torch.empty(
                    self.padded_numel,
                    device=self.local_grad_shard.device,
                    dtype=self.local_grad_shard.dtype,
                )
                for p, off in zip(self.params, self.param_offsets):
                    if p.grad is not None:
                        tmp_flat[off : off + p.numel()].copy_(p.grad.view(-1))
                    else:
                        tmp_flat[off : off + p.numel()].zero_()
                handle = dist.reduce_scatter_tensor(
                    self.local_grad_shard,
                    tmp_flat,
                    op=dist.ReduceOp.AVG,
                    group=self.dp_group,
                    # async_op=True,
                )

        # Free per-param gradients; the flat buffer (persistent for ZeRO-1,
        # transient for ZeRO-2) has already captured or scattered them.
        for p in self.params:
            p.grad = None

        return handle

    def prepare_local_grad_shard(self) -> None:
        """Prepare local shard gradient after grad sync."""
        if self.zero_stage == 1:
            self.local_grad_shard.copy_(
                self.flat_grad_buf[self.shard_start : self.shard_end]
            )
            self.flat_grad_buf.zero_()
        # In stage-2, local_grad_shard is already the reduce_scatter output.
        # The transient flat buffer was released when launch_grad_sync returned.

    def optimizer_step(self) -> None:
        """
        Attach the reduce-scattered gradient to ``local_param_shard`` and run
        the optimizer.

        Must be called after ``reduce_scatter_handle.wait()``.
        """
        if self.local_param_shard.grad is None:
            self.local_param_shard.grad = self.local_grad_shard.clone()
        else:
            self.local_param_shard.grad.copy_(self.local_grad_shard)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def launch_all_gather(
        self,
        comm_stream: torch.cuda.Stream,
        comp_stream: torch.cuda.Stream,
    ) -> dist.Work:
        """
        Launch an async ``all_gather_into_tensor`` that reconstructs
        ``flat_param_buf`` from all ranks' updated ``local_param_shard``.

        Must be called after ``optimizer_step()``.
        Returns the ``dist.Work`` handle.
        """
        comm_stream.wait_stream(comp_stream)
        with torch.cuda.stream(comm_stream):
            handle = dist.all_gather_into_tensor(
                self.flat_param_buf,
                self.local_param_shard.data,
                group=self.dp_group,
                # async_op=True,
            )
        return handle

    def copy_flat_to_params(self) -> None:
        """
        Scatter ``flat_param_buf`` back into each parameter's ``.data``.

        Must be called after ``all_gather_handle.wait()``.
        Used by ZeRO-1/2 finalize_step; ZeRO-3 uses ``swap_to_full_params``
        instead (a zero-copy view rather than a copy).
        """
        for p, off, numel, shape in zip(
            self.params, self.param_offsets, self.param_numels, self.param_original_shapes
        ):
            p.data.copy_(self.flat_param_buf[off : off + numel].view(shape))

    def reset_step_state(self) -> None:
        """Reset per-step counters and handles for the next training step."""
        for pid in self._mb_accumulated:
            self._mb_accumulated[pid] = 0
        self._ready_param_count = 0
        self.grad_sync_handle = None
        self.all_gather_handle = None

    # ------------------------------------------------------------------
    # ZeRO-3 helpers
    # ------------------------------------------------------------------

    def swap_to_full_params(self) -> None:
        """
        Redirect each parameter's ``.data`` to a view of ``flat_param_buf``
        with the parameter's original shape.

        ``flat_param_buf`` must already be populated by ``launch_all_gather_params``
        before this is called.  This is a zero-copy operation — ``flat_param_buf``
        IS the temporary buffer; no extra allocation is performed.

        Called in the forward/backward pre-hook for ZeRO-3 to give the FX
        sub-module access to fully-gathered parameters.
        """
        logger.debug(f"Bucket {self.bucket_id} swapping to full param views")
        for p, off, numel, shape in zip(
            self.params, self.param_offsets, self.param_numels, self.param_original_shapes
        ):
            p.data = self.flat_param_buf[off : off + numel].view(shape)

    def restore_to_shard_params(self) -> None:
        """
        Restore each parameter's ``.data`` to its owned shard slice inside
        ``local_param_shard``, releasing the reference to the full-shape view
        of ``flat_param_buf`` set by ``swap_to_full_params``.

        Called in the forward/backward post-hook for ZeRO-3 after sub-module
        execution.  The released view has no remaining strong references so the
        GPU memory occupied by the non-owned regions becomes reclaimable once the
        autograd graph releases its own references (after backward completes).
        """
        for p, off, numel in zip(self.params, self.param_offsets, self.param_numels):
            flat_lo = max(off, self.shard_start)
            flat_hi = min(off + numel, self.shard_end)
            if flat_lo < flat_hi:
                p.data = self.local_param_shard[
                    flat_lo - self.shard_start : flat_hi - self.shard_start
                ]
            else:
                p.data = self.local_param_shard.new_empty(0)

    def launch_all_gather_params(
        self,
        comm_stream: torch.cuda.Stream,
        is_bwd: bool = False,
    ) -> None:
        """
        Launch an all-gather of ``local_param_shard`` → ``flat_param_buf`` on
        *comm_stream*, then record the corresponding CUDA event.

        ``is_bwd=True`` records ``_ag_bwd_event`` (used by backward pre-hooks);
        ``is_bwd=False`` records ``_ag_fwd_event`` (used by forward pre-hooks).

        Call ``swap_to_full_params()`` inside ``comp_stream.wait_event(ag_event)``
        to install gathered parameters into each ``p.data`` as a zero-copy view.
        """
        logger.debug(f"Bucket {self.bucket_id} launching all-gather for {'backward' if is_bwd else 'forward'}")
        ag_event = self._ag_bwd_event if is_bwd else self._ag_fwd_event
        with torch.cuda.stream(comm_stream):
            dist.all_gather_into_tensor(
                self.flat_param_buf,
                self.local_param_shard.data,
                group=self.dp_group,
            )
            ag_event.record()  # records on comm_stream (current stream)

    @property
    def size_bytes(self) -> int:
        """Total byte size of parameters in this bucket (based on original shapes)."""
        return sum(n * p.element_size() for n, p in zip(self.param_numels, self.params))


# ---------------------------------------------------------------------------
# State manager
# ---------------------------------------------------------------------------

class ZeROOneState:
    """
    Manages ZeRO-1 optimizer state partitioning for one pipeline stage.

    At construction time, parameters are grouped into gradient-sync buckets in
    reverse forward-use order. Post-accumulate-grad hooks are registered on every
    parameter; each hook decrements a per-bucket counter and launches an async
    reduce_scatter when the bucket's counter reaches zero.

    At the end of each training step, call ``finalize_step`` to complete the
    pipelined reduce_scatter → optimizer step → all_gather sequence.

    Args:
        all_params: All trainable parameters for this stage.
        params_in_forward_order: Same parameters sorted by first use in the
            FX graph forward pass (from ``_get_param_forward_order``).
        dp_rank: Data-parallel rank of this actor.
        dp_degree: Total number of data-parallel replicas.
        dp_group: ``torch.distributed`` process group for this DP replica set.
        device: CUDA device string for the actor.
        num_mbs: Number of microbatches per training step.
        optim_class: Callable ``(params) -> Optimizer`` used per bucket.
        comm_stream: CUDA stream for collective communication ops.
        comp_stream: CUDA stream for computation (backward / optimizer).
        bucket_size_bytes: Maximum byte size per bucket (default 25 MB).
    """

    def __init__(
        self,
        all_params: list[Parameter],
        params_in_forward_order: list[Parameter],
        dp_rank: int,
        dp_degree: int,
        dp_group: dist.ProcessGroup,
        device: str,
        num_mbs: int,
        optim_class: Callable[[list[Parameter]], Optimizer],
        comm_stream: torch.cuda.Stream,
        comp_stream: torch.cuda.Stream,
        bucket_size_bytes: int = BUCKET_SIZE_BYTES_DEFAULT,
        zero_stage: int = 1,
        # ZeRO-3 only: FX graph and parameter index list for graph splitting.
        gm: Optional[torch.fx.GraphModule] = None,
        param_idxs: Optional[list[int]] = None,
    ) -> None:
        if zero_stage not in (1, 2, 3):
            raise ValueError(f"ZeRO state must be 1, 2, or 3, got {zero_stage}")

        self.all_params = all_params
        self.zero_stage = zero_stage
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.dp_group = dp_group
        self._comm_stream = comm_stream
        self._comp_stream = comp_stream
        self._num_mbs = num_mbs

        # Build buckets in reverse forward order ≈ backward arrival order.
        self._buckets: list[ZeROOneBucket] = _build_buckets(
            params_in_bwd_order=list(reversed(params_in_forward_order)),
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            dp_group=dp_group,
            device=device,
            num_mbs=num_mbs,
            optim_class=optim_class,
            bucket_size_bytes=bucket_size_bytes,
            zero_stage=zero_stage,
        )

        # Fast lookup: parameter id → owning bucket.
        self._param_to_bucket: dict[int, ZeROOneBucket] = {
            id(p): bucket
            for bucket in self._buckets
            for p in bucket.params
        }

        if zero_stage == 3:
            if gm is None or param_idxs is None:
                raise ValueError("gm and param_idxs are required for ZeRO-3")
            # Build per-bucket FX sub-modules for pipelined all-gather.
            fwd_segments = _build_bucket_segments(
                gm, param_idxs, all_params, self._param_to_bucket
            )
            # Buckets in forward order (sub-module k ↔ fwd_ordered_buckets[k]).
            self._fwd_ordered_buckets: list[ZeROOneBucket] = [
                b for _, b in fwd_segments
            ]
            self.split_gm: Optional[torch.fx.GraphModule] = _make_zero3_split_gm(
                gm, fwd_segments
            )
            self._setup_zero3_module_hooks()
        else:
            self._fwd_ordered_buckets = []
            self.split_gm = None
            # Grad hooks drive reduce-scatter for ZeRO-1/2 (same mechanism).
            self._setup_grad_hooks()

        logger.debug(
            f"ZeROOneState: {len(self._buckets)} buckets, "
            f"{len(all_params)} params, dp_rank={dp_rank}/{dp_degree}, "
            f"zero_stage={zero_stage}"
        )

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _setup_grad_hooks(self) -> None:
        """Register a post-accumulate-grad hook on every tracked parameter."""
        for p in self.all_params:
            if id(p) in self._param_to_bucket:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param: Parameter) -> Callable[[Parameter], None]:
        """Return a closure that fires when *param*'s gradient is accumulated."""
        bucket = self._param_to_bucket[id(param)]

        def _hook(_param: Parameter) -> None:
            if bucket.on_grad_accumulated(_param):
                # Wait for backward on comp stream to complete
                dep_event = torch.cuda.Event()
                dep_event.record(self._comp_stream)
                self._comm_stream.wait_event(dep_event)
                # All params in bucket have full gradients: launch reduce_scatter.
                bucket.grad_sync_handle = bucket.launch_grad_sync(
                    self._comm_stream,
                    # torch.cuda.current_stream(),
                )
                logger.debug(
                    f"Launched grad sync for bucket {bucket.bucket_id}"
                )

        return _hook

    def _setup_zero3_module_hooks(self) -> None:
        """
        Register forward hooks on each sub-module of ``split_gm`` that also
        install tensor-level backward hooks to pipeline all-gather with
        computation during the backward pass.

        Forward hooks (per sub-module k):
          - ``forward_pre_hook``: wait for fwd AG event, swap params to full
            shape; also register a post-backward tensor hook on the first
            requires-grad input (fires after backward through this sub-module)
            to restore params to shard and seed backward AG lookahead.
          - ``forward_hook``: restore params to shard after sub-module's forward;
            seed fwd AG for bucket k+2; register a pre-backward tensor hook on
            the output (fires when grad arrives at this output, i.e., before
            backward through this sub-module) to wait for bwd AG and swap params
            to full shape.

        ``register_full_backward_pre_hook`` is intentionally avoided because it
        does not fire reliably on ``GraphModule`` sub-modules created by
        ``split_module`` where parameters arrive as positional placeholders
        rather than registered ``nn.Parameter`` children.
        """
        fwd_buckets = self._fwd_ordered_buckets
        N = len(fwd_buckets)
        comm_stream = self._comm_stream
        comp_stream = self._comp_stream

        for k in range(N):
            bucket = fwd_buckets[k]
            submod = getattr(self.split_gm, f"submod_{k}")

            # ---- forward pre-hook ----------------------------------------
            def _make_fwd_pre(bkt: ZeROOneBucket, k_: int):
                def _fwd_pre(module, args):
                    # GPU-side wait: computation doesn't start until the
                    # all-gather into flat_param_buf has completed.
                    logger.debug(f"Bucket {bkt.bucket_id} pre-forward hook waiting for fwd AG event")
                    comp_stream.wait_event(bkt._ag_fwd_event)
                    bkt.swap_to_full_params()

                    # Register a post-backward hook on the first requires-grad
                    # input tensor.  Its gradient is computed AFTER backward has
                    # finished processing all ops inside this sub-module, making
                    # it the right signal to restore params to shard storage and
                    # seed the lookahead all-gather for backward.
                    lookahead_k = k_ - 2  # 2 positions back = 2 ahead in bwd
                    for inp in (args if isinstance(args, (list, tuple)) else [args]):
                        if isinstance(inp, torch.Tensor) and inp.requires_grad:
                            def _post_bwd(grad, _bkt=bkt, _lk=lookahead_k):
                                logger.debug(f"Bucket {bkt.bucket_id} post-backward hook releasing params")
                                _bkt.restore_to_shard_params()
                                if _lk >= 0:
                                    fwd_buckets[_lk].launch_all_gather_params(
                                        comm_stream, is_bwd=True
                                    )
                            inp.register_hook(_post_bwd)
                            break

                return _fwd_pre

            # ---- forward post-hook ----------------------------------------
            def _make_fwd_post(bkt: ZeROOneBucket, k_: int):
                def _fwd_post(module, args, output):
                    # Restore p.data to the owned shard slice after forward.
                    logger.debug(f"Bucket {bkt.bucket_id} post-forward hook releasing params")
                    bkt.restore_to_shard_params()
                    # Seed forward all-gather lookahead for bucket k+2.
                    if k_ + 2 < N:
                        fwd_buckets[k_ + 2].launch_all_gather_params(
                            comm_stream, is_bwd=False
                        )

                    # Register a pre-backward hook on the output tensor.
                    # Its gradient is computed AFTER backward finishes all ops
                    # that consume this output but BEFORE backward processes
                    # this sub-module's own ops — i.e., exactly when we need
                    # to swap params back to full shape for this sub-module's
                    # backward pass.
                    out: Optional[torch.Tensor] = None
                    if isinstance(output, torch.Tensor):
                        out = output
                    elif isinstance(output, (list, tuple)):
                        for t in output:
                            if isinstance(t, torch.Tensor) and t.requires_grad:
                                out = t
                                break
                    if out is not None and out.requires_grad:
                        def _pre_bwd(grad, _bkt=bkt):
                            logger.debug(f"Bucket {bkt.bucket_id} pre-backward hook waiting for bwd AG event")
                            comp_stream.wait_event(_bkt._ag_bwd_event)
                            _bkt.swap_to_full_params()
                            return grad
                        out.register_hook(_pre_bwd)

                return _fwd_post

            submod.register_forward_pre_hook(_make_fwd_pre(bucket, k))
            submod.register_forward_hook(_make_fwd_post(bucket, k))

    # ------------------------------------------------------------------
    # ZeRO-3 per-microbatch seeding
    # ------------------------------------------------------------------

    def before_microbatch_forward(self) -> None:
        """
        Seed the forward all-gather pipeline for one microbatch.

        Launches AGs for the first (and second) buckets in forward order so
        that at least two buckets are in-flight before the first sub-module's
        forward pre-hook fires.  Must be called immediately before the
        microbatch's forward pass (outside ``torch.cuda.stream(comp_stream)``).
        """
        if self.zero_stage != 3:
            return
        fwd_buckets = self._fwd_ordered_buckets
        N = len(fwd_buckets)
        if N > 0:
            fwd_buckets[0].launch_all_gather_params(self._comm_stream, is_bwd=False)
        if N > 1:
            fwd_buckets[1].launch_all_gather_params(self._comm_stream, is_bwd=False)

    def before_microbatch_backward(self) -> None:
        """
        Seed the backward all-gather pipeline for one microbatch.

        Launches AGs for the last two buckets in forward order (which are the
        first two encountered during backward, since backward runs in reverse).
        The tensor-level hooks installed during forward take over from there,
        pipelining subsequent all-gathers two buckets ahead as backward proceeds.

        Must be called immediately before ``backward()``/``loss.backward()``.
        """
        if self.zero_stage != 3:
            return
        fwd_buckets = self._fwd_ordered_buckets
        N = len(fwd_buckets)
        if N > 0:
            fwd_buckets[N - 1].launch_all_gather_params(self._comm_stream, is_bwd=True)
        if N > 1:
            fwd_buckets[N - 2].launch_all_gather_params(self._comm_stream, is_bwd=True)

    def after_microbatch_backward(self) -> None:
        """
        Restore all parameters to their shard-only storage after the microbatch
        backward pass completes.

        Must be called after ``backward()``/``loss.backward()`` returns.
        """
        if self.zero_stage != 3:
            return
        for bucket in self._buckets:
            bucket.restore_to_shard_params()

    # ------------------------------------------------------------------
    # End-of-step pipeline
    # ------------------------------------------------------------------

    def finalize_step(
        self,
        comm_stream: torch.cuda.Stream,
        comp_stream: torch.cuda.Stream,
    ) -> None:
        """
        Complete the pipelined reduce_scatter → optimizer → all_gather sequence
        for all buckets, then reset state for the next step.

        Should be called in ``_update`` after all backward passes are complete.

        Pipeline structure:
          For bucket k: wait rs_k → optim_k → launch ag_k
          The all_gather of bucket k-1 overlaps with optim_k and launch_ag_k.
        """
        # Ensure every bucket has gradient sync in flight (handles buckets
        # whose hooks were never triggered, e.g. params with no gradient).
        # for bucket in self._buckets:
        #     if bucket.grad_sync_handle is None:
        #         bucket.grad_sync_handle = bucket.launch_grad_sync(
        #             comm_stream,
        #             torch.cuda.current_stream(),
        #         )

        if self.zero_stage == 3:
            # ZeRO-3: optimizer step on local shard only.  No all-gather and no
            # copy back needed: each p.data is a live view of its owned slice of
            # local_param_shard (set in __init__ and restored by
            # restore_to_shard_params after each sub-module), so optimizer
            # updates to local_param_shard are instantly visible through p.data.
            for bucket in self._buckets:
                bucket.prepare_local_grad_shard()
                with torch.cuda.stream(comp_stream):
                    bucket.optimizer_step()

            # Wait for all comp_stream work to finish.
            event = torch.cuda.Event()
            with torch.cuda.stream(comp_stream):
                event.record()
            event.synchronize()

            for p in self.all_params:
                p.grad = None
            for bucket in self._buckets:
                bucket.reset_step_state()
            return

        # ZeRO-1 / ZeRO-2: pipeline reduce_scatter → optimizer → all_gather.
        for bucket in self._buckets:
            # bucket.grad_sync_handle.wait()
            bucket.prepare_local_grad_shard()
            with torch.cuda.stream(comp_stream):
                bucket.optimizer_step()
            bucket.all_gather_handle = bucket.launch_all_gather(
                comm_stream, comp_stream
            )

        event = torch.cuda.Event()
        with torch.cuda.stream(comm_stream):
            event.record()
        event.synchronize() # Ensure all_gathers complete before copying params back.

        # Wait for all all_gathers and write results back into p.data.
        for bucket in self._buckets:
            # bucket.all_gather_handle.wait()
            bucket.copy_flat_to_params()

        # Ensure no stale parameter gradients remain.
        for p in self.all_params:
            p.grad = None
        for bucket in self._buckets:
            bucket.reset_step_state()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def num_buckets(self) -> int:
        return len(self._buckets)

    def bucket_sizes(self) -> list[int]:
        return [b.size_bytes for b in self._buckets]

    def __repr__(self) -> str:
        return (
            f"ZeROOneState(dp_rank={self.dp_rank}/{self.dp_degree}, "
            f"zero_stage={self.zero_stage}, "
            f"buckets={self.num_buckets()}, "
            f"params={len(self.all_params)})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_buckets(
    params_in_bwd_order: list[Parameter],
    dp_rank: int,
    dp_degree: int,
    dp_group: dist.ProcessGroup,
    device: str,
    num_mbs: int,
    optim_class: Callable[[list[Parameter]], Optimizer],
    bucket_size_bytes: int,
    zero_stage: int,
) -> list[ZeROOneBucket]:
    """
    Fill buckets greedily: add parameters in *params_in_bwd_order* until a
    bucket would exceed *bucket_size_bytes*, then start a new bucket.

    The first parameter always starts a new bucket regardless of size (no
    parameter is skipped).
    """
    buckets: list[ZeROOneBucket] = []
    current: list[Parameter] = []
    current_bytes = 0

    for p in params_in_bwd_order:
        p_bytes = p.numel() * p.element_size()
        if current and current_bytes + p_bytes > bucket_size_bytes:
            dtype = _bucket_dtype(current)
            buckets.append(
                ZeROOneBucket(
                    len(buckets), current,
                    dp_rank, dp_degree, dp_group,
                    device, dtype, optim_class, num_mbs, zero_stage,
                )
            )
            current = []
            current_bytes = 0
        current.append(p)
        current_bytes += p_bytes

    if current:
        dtype = _bucket_dtype(current)
        buckets.append(
            ZeROOneBucket(
                len(buckets), current,
                dp_rank, dp_degree, dp_group,
                device, dtype, optim_class, num_mbs, zero_stage,
            )
        )

    return buckets


def _bucket_dtype(params: list[Parameter]) -> torch.dtype:
    """Return the dtype of the first parameter (all params in a bucket share dtype)."""
    return params[0].dtype
