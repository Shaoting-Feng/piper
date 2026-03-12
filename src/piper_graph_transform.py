import hashlib
import json
import ray
import torch
import torch.fx as fx
import torch.distributed as dist
from collections import defaultdict
from pathlib import Path
from typing import Optional
import operator
import os
import time
from torch.autograd import Function

from .piper_utils import create_logger, LOG_LEVEL, piper_metadata

logger = create_logger("piper_graph_transform", LOG_LEVEL)


class CommOp:
    def __init__(self, tensor_id: int, name: str, dep: str, pass_type: str, op: str, group: str):
        self.tensor = None # this gets set on the actor
        self.tensor_id = tensor_id
        self.name = name
        self.dep = dep # "pre" or "post"
        self.pass_type = pass_type # "forward" or "backward"
        self.op = op # "allreduce", "allgather", "scatter", "alltoall"
        self.group = group # "dp" or "pp"

def _get_dp_comm_ops(inputs, placeholders):
    dp_comm_ops = []
    ids = []
    for p, t in zip(placeholders, inputs):
        ids.append(id(t))
        if isinstance(t, torch.nn.Parameter) and t.requires_grad:
            dp_comm_ops.append(CommOp(id(t), p.name, "post", "backward", "allreduce", "dp"))
    return dp_comm_ops, ids

class AllToAllSingleFunction(torch.autograd.Function):
    """
    Custom autograd function for all_to_all_single that ensures gradients flow correctly.
    
    This wraps torch.distributed.all_to_all_single to ensure that gradients
    from the input tensor properly flow to the output buffer.
    The backward pass performs another all_to_all_single to reverse the communication.
    """
    @staticmethod
    def forward(ctx, output, input_tensor):
        """
        Forward pass: performs all_to_all_single communication.

        Args:
            ctx: Context for storing information for backward pass
            output: Output buffer (will be modified in-place)
            input_tensor: Input tensor to communicate
            group: Process group (optional)
        """
        # Store group for backward pass
        from .piper_utils import piper_metadata
        ctx.actor_self = piper_metadata.actor_self
        ctx.group = ctx.actor_self.ep_group
        ctx.global_rank = ctx.actor_self.global_rank
        ctx.stream = ctx.actor_self.a2a_stream

        comp_stream = torch.cuda.current_stream()
        comm_finished_event = torch.cuda.Event()

        ctx.actor_self._start_timing(ctx.stream, "fwd_a2a")

        ctx.stream.wait_stream(comp_stream)
        _ov_token = ctx.actor_self.overlap_detector.before_kernel(ctx.stream, "fwd_a2a", "a2a_stream")

        input_tensor = input_tensor.contiguous()
        with torch.cuda.stream(ctx.stream):
            dist.all_to_all_single(output, input_tensor, group=ctx.group)
            comm_finished_event.record()
        ctx.actor_self.overlap_detector.after_kernel(ctx.stream, _ov_token)

        ctx.actor_self._stop_timing(ctx.stream, "fwd_a2a")

        # POST-HOOK: wait for bwd to reach its corresponding A2A
        if ctx.actor_self.overlap_a2a_ops:
            idx = ctx.actor_self.fwd_a2a_counter
            ctx.actor_self.fwd_a2a_submitted[idx].set()
            ctx.actor_self.bwd_a2a_submitted[idx].wait()
            ctx.actor_self.fwd_a2a_counter += 1
        comp_stream.wait_event(comm_finished_event)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: performs reverse all_to_all_single to propagate gradients.

        The backward of all_to_all_single is another all_to_all_single operation
        that reverses the communication pattern.
        """
        logger.debug(f"Dispatch AllToAllSingleFunction backward rank={ctx.global_rank}, shape={grad_output.shape}")

        if grad_output is None:
            return None, None, None

        comp_stream = torch.cuda.current_stream()
        comm_finished_event = torch.cuda.Event()

        ctx.actor_self._start_timing(ctx.stream, "bwd_a2a")

        ctx.stream.wait_stream(comp_stream)
        _ov_token = ctx.actor_self.overlap_detector.before_kernel(ctx.stream, "bwd_a2a", "a2a_stream")

        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        with torch.cuda.stream(ctx.stream):
            dist.all_to_all_single(grad_input, grad_output, group=ctx.group)
            comm_finished_event.record()
        ctx.actor_self.overlap_detector.after_kernel(ctx.stream, _ov_token)

        ctx.actor_self._stop_timing(ctx.stream, "bwd_a2a")

        # POST-HOOK: wait for fwd to reach the next A2A
        if ctx.actor_self.overlap_a2a_ops:
            idx = ctx.actor_self.bwd_a2a_counter
            ctx.actor_self.bwd_a2a_submitted[idx].set()
            if idx < len(ctx.actor_self.fwd_a2a_submitted) - 1:
                ctx.actor_self.fwd_a2a_submitted[idx + 1].wait()
            ctx.actor_self.bwd_a2a_counter += 1
        comp_stream.wait_event(comm_finished_event)

        # Return gradients: grad_output flows to grad_input, None for group
        return grad_input, grad_input, None

def _dispatch_a2a_single(output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Dispatch a Ray remote call to run an all_to_all_single operation
    using a custom autograd Function. This function is called
    from the FX graph, so it needs to be allowed in the graph.
    """
    return AllToAllSingleFunction.apply(output, input_tensor)

# Allow the dispatch function in the graph
torch.compiler.allow_in_graph(_dispatch_a2a_single)

def _profile_and_split_gm(gm, num_stages) -> tuple[fx.GraphModule, list[tuple[int, fx.GraphModule, list[int], list]]]:
    """
    Profile each node in the graph module, then split into num_stages stages
    of roughly equal execution time. This replaces annotation-based splitting
    when users don't provide stage annotations.

    Returns the same format as _split_gm_by_stages:
        (top_level_gm, [(stage_id, stage_gm, input_idxs, param_idxs, graphargs, placeholders), ...])
    """
    # Step 1: Collect compute nodes (everything that isn't placeholder, get_attr, or output)
    compute_nodes = []
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        compute_nodes.append(node)

    if not compute_nodes:
        return gm, []

    # Check for cached split from a previous profiling run
    cache_dir = Path.home() / ".cache" / "piper" / "splits"
    graph_signature = hashlib.sha256(
        json.dumps([(n.name, n.op, str(n.target)) for n in compute_nodes] + [num_stages]).encode()
    ).hexdigest()[:16]
    cache_file = cache_dir / f"{graph_signature}.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            split_node_names = cached["split_node_names"]
            # Rebuild split_indices from cached node names
            node_name_to_idx = {n.name: i for i, n in enumerate(compute_nodes)}
            split_indices = [node_name_to_idx[name] for name in split_node_names]
            logger.info(f"Loaded cached split from {cache_file} (splits: {split_node_names})")

            # Assign stages and inject annotations
            stage_assignments = {}
            current_stage = 0
            split_iter = iter(split_indices)
            next_split = next(split_iter, len(compute_nodes))
            for idx, node in enumerate(compute_nodes):
                if idx == next_split:
                    current_stage += 1
                    next_split = next(split_iter, len(compute_nodes))
                stage_assignments[node] = current_stage

            for node, stage_id in stage_assignments.items():
                if 'custom' not in node.meta:
                    node.meta['custom'] = {}
                node.meta['custom']['stage'] = stage_id

            return _split_gm_by_stages(gm)
        except Exception as e:
            logger.warning(f"Failed to load cached split from {cache_file}: {e}, re-profiling")

    # Step 2: Build environment mapping for executing individual nodes
    # We need real tensors on device to profile. To handle models too large for a single
    # GPU, we eagerly free intermediate tensors once all their consumers have been profiled.
    env = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build reference counts: for each node, count how many compute nodes (+ the output node)
    # consume it. When refcount hits 0 after profiling a consumer, we free the tensor.
    compute_node_set = set(compute_nodes)
    ref_counts = {}  # node.name -> int
    for node in compute_nodes:
        for inp in node.all_input_nodes:
            ref_counts[inp.name] = ref_counts.get(inp.name, 0) + 1

    def _materialize_tensor(ex, device):
        """Create a real tensor on device from an example (possibly meta) tensor."""
        if isinstance(ex, torch.Tensor):
            shape = tuple(ex.shape)
            if ex.device.type != "meta" and ex.device.type == device:
                return ex
            if ex.is_floating_point():
                t = torch.randn(shape, dtype=ex.dtype, device=device)
            else:
                t = torch.ones(shape, dtype=ex.dtype, device=device)
            if isinstance(ex, torch.nn.Parameter):
                t = torch.nn.Parameter(t, requires_grad=ex.requires_grad)
            return t
        return ex

    def _ensure_in_env(node):
        """Lazily materialize a placeholder or get_attr node into env on first access."""
        if node.name in env:
            return
        if node.op == "placeholder":
            ex = node.meta.get("example_value")
            if ex is not None:
                env[node.name] = _materialize_tensor(ex, device)
            else:
                env[node.name] = None
        elif node.op == "get_attr":
            target_atoms = node.target.split(".")
            attr = gm
            for atom in target_atoms:
                attr = getattr(attr, atom)
            if isinstance(attr, torch.Tensor):
                env[node.name] = _materialize_tensor(attr, device)
            else:
                env[node.name] = attr

    def _fetch_arg(arg):
        if isinstance(arg, fx.Node):
            _ensure_in_env(arg)
            return env.get(arg.name)
        elif isinstance(arg, (list, tuple)):
            vals = [_fetch_arg(a) for a in arg]
            return type(arg)(vals)
        elif isinstance(arg, dict):
            return {k: _fetch_arg(v) for k, v in arg.items()}
        return arg

    def _dec_ref(node):
        """Decrement refcount for a node and free its env entry when it reaches 0."""
        name = node.name
        if name not in ref_counts:
            return
        ref_counts[name] -= 1
        if ref_counts[name] <= 0 and name in env and isinstance(env[name], torch.Tensor):
            del env[name]

    def _exec_node(node, args, kwargs, fn):
        """Execute a single graph node and return the result."""
        if node.op == "call_method":
            return getattr(args[0], node.target)(*args[1:], **kwargs)
        elif node.op == "call_module":
            return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    def _log_env_memory():
        """Log all active tensors in env and their total memory usage."""
        active = []
        total_bytes = 0
        for name, val in env.items():
            if isinstance(val, torch.Tensor) and val.device.type != "meta":
                nbytes = val.nelement() * val.element_size()
                total_bytes += nbytes
                active.append((name, tuple(val.shape), val.dtype, nbytes))
        total_mb = total_bytes / (1024 * 1024)
        cuda_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024) if device == "cuda" else 0
        logger.debug(
            f"Active env tensors: {len(active)}, env total: {total_mb:.1f} MB, "
            f"CUDA allocated: {cuda_allocated_mb:.1f} MB"
        )
        # Log the top 10 largest tensors
        active.sort(key=lambda x: x[3], reverse=True)
        for name, shape, dtype, nbytes in active[:10]:
            logger.debug(f"  {name}: {shape} {dtype} ({nbytes / (1024*1024):.1f} MB)")

    # Step 3: Profile each compute node with eager memory freeing
    # Use no_grad to prevent autograd graph accumulation — we only need forward timing.
    node_times = {}  # node -> time in ms

    num_warmup = 3
    num_profile = 5
    log_interval = max(1, len(compute_nodes) // 20)  # Log ~20 times during profiling
    for node_idx, node in enumerate(compute_nodes):
        if node_idx % log_interval == 0:
            _log_env_memory()

        args = _fetch_arg(node.args)
        kwargs = _fetch_arg(node.kwargs)

        # Check for None inputs from failed predecessor nodes
        has_none_input = False
        for inp in node.all_input_nodes:
            if inp.op not in ("placeholder", "get_attr") and env.get(inp.name) is None:
                has_none_input = True
                break

        if has_none_input:
            logger.debug(f"Skipping node {node.name}: predecessor produced None")
            node_times[node.name] = 0.0
            env[node.name] = None
            for inp in node.all_input_nodes:
                _dec_ref(inp)
            continue

        # Resolve the callable
        if node.op == "call_function":
            fn = node.target
        elif node.op == "call_method":
            fn = getattr(type(args[0]), node.target) if args else None
        elif node.op == "call_module":
            fn = gm.get_submodule(node.target)
            fn.to(device)
        else:
            node_times[node.name] = 0.0
            env[node.name] = None
            for inp in node.all_input_nodes:
                _dec_ref(inp)
            continue

        # Execute and profile under no_grad to prevent autograd graph accumulation
        try:
            with torch.no_grad():
                for _ in range(num_warmup):
                    result = _exec_node(node, args, kwargs, fn)

                if device == "cuda":
                    torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(num_profile):
                    result = _exec_node(node, args, kwargs, fn)
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / num_profile
            node_times[node.name] = elapsed
            env[node.name] = result
        except Exception as e:
            logger.warning(f"Failed to profile node {node.name} ({node.op}): {e}")
            node_times[node.name] = 0.0
            try:
                with torch.no_grad():
                    env[node.name] = _exec_node(node, args, kwargs, fn)
            except Exception:
                env[node.name] = None

        # Move call_module submodules back to meta to free GPU memory
        if node.op == "call_module":
            fn.to("meta")

        # Decrement refcounts for all inputs of this node
        for inp in node.all_input_nodes:
            _dec_ref(inp)

    # Step 4: Partition compute nodes into num_stages groups of roughly equal time
    total_time = sum(node_times.get(n.name, 0.0) for n in compute_nodes)
    target_time = total_time / num_stages

    # Precompute prefix sums for fast time range queries
    prefix_times = [0.0]
    for n in compute_nodes:
        prefix_times.append(prefix_times[-1] + node_times.get(n.name, 0.0))

    def _count_cross_boundary_deps(split_idx):
        """Count how many compute nodes before split_idx are used by nodes at or after split_idx."""
        before_set = set(compute_nodes[:split_idx])
        cross = set()
        for node in compute_nodes[split_idx:]:
            for arg in node.all_input_nodes:
                if arg in before_set and arg.op not in ("placeholder", "get_attr"):
                    cross.add(arg)
        return len(cross)

    def _find_best_split(ideal_idx, stage_start_idx):
        """Search all valid split points and return the one closest to ideal_idx in time."""
        ideal_time = prefix_times[ideal_idx]
        best_split = None
        best_time_dist = float('inf')
        for candidate in range(stage_start_idx + 1, len(compute_nodes)):
            n_deps = _count_cross_boundary_deps(candidate)
            if n_deps <= 1:
                time_dist = abs(prefix_times[candidate] - ideal_time)
                if time_dist < best_time_dist:
                    best_time_dist = time_dist
                    best_split = candidate
        return best_split

    # Find split points: we need (num_stages - 1) splits
    split_indices = []
    stage_start_idx = 0
    for s in range(num_stages - 1):
        # Target cumulative time for end of this stage
        remaining_time = prefix_times[-1] - prefix_times[stage_start_idx]
        remaining_stages = num_stages - s
        stage_target = remaining_time / remaining_stages

        target_cum_time = prefix_times[stage_start_idx] + stage_target
        # Find the ideal split index (first node where cumulative time exceeds target)
        ideal_idx = stage_start_idx + 1
        for i in range(stage_start_idx + 1, len(compute_nodes)):
            if prefix_times[i] >= target_cum_time:
                ideal_idx = i
                break
        else:
            ideal_idx = len(compute_nodes) - 1

        best_split = _find_best_split(ideal_idx, stage_start_idx)
        if best_split is not None:
            split_indices.append(best_split)
            time_offset = prefix_times[best_split] - prefix_times[ideal_idx]
            logger.info(
                f"Stage {s}/{s+1} boundary: split before node {compute_nodes[best_split].name} "
                f"(index {best_split}, time offset from ideal: {time_offset:+.3f} ms)"
            )
            stage_start_idx = best_split
        else:
            logger.warning(f"Could not find valid split point near node {ideal_idx}, skipping")

    # Save split to cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        split_node_names = [compute_nodes[i].name for i in split_indices]
        cache_file.write_text(json.dumps({"split_node_names": split_node_names}, indent=2))
        logger.info(f"Saved split cache to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save split cache: {e}")

    # Assign stages based on split indices
    stage_assignments = {}
    current_stage = 0
    split_iter = iter(split_indices)
    next_split = next(split_iter, len(compute_nodes))
    for idx, node in enumerate(compute_nodes):
        if idx == next_split:
            current_stage += 1
            next_split = next(split_iter, len(compute_nodes))
        stage_assignments[node] = current_stage

    # Ensure example_value metadata matches the profiled result's requires_grad.
    # This prevents assertions in _forward_impl from failing when non-grad tensors
    # (e.g., masks, positional encodings) are passed between stages as outputs.
    for node in compute_nodes:
        result = env.get(node.name)
        if result is not None and isinstance(result, torch.Tensor) and not result.requires_grad:
            ex = node.meta.get("example_value")
            if ex is not None and isinstance(ex, torch.Tensor) and ex.requires_grad:
                node.meta["example_value"] = ex.detach()

    # Inject stage annotations into node metadata so _split_gm_by_stages can handle it
    for node, stage_id in stage_assignments.items():
        if 'custom' not in node.meta:
            node.meta['custom'] = {}
        node.meta['custom']['stage'] = stage_id

    # Log stage times
    stage_times = defaultdict(float)
    stage_node_counts = defaultdict(int)
    for node, stage_id in stage_assignments.items():
        stage_times[stage_id] += node_times.get(node.name, 0.0)
        stage_node_counts[stage_id] += 1
    for stage_id in sorted(stage_times.keys()):
        logger.info(
            f"Stage {stage_id}: {stage_times[stage_id]:.3f} ms "
            f"({stage_node_counts[stage_id]} nodes)"
        )

    # Free all profiling tensors before training starts
    import gc
    env.clear()
    del env
    gc.collect()
    torch.cuda.empty_cache()

    # Delegate to _split_gm_by_stages which handles all the submodule creation
    return _split_gm_by_stages(gm)


def _meta_tensor_like(example: torch.Tensor, *, requires_grad: bool, as_parameter: bool):
    t = torch.empty(example.shape, dtype=example.dtype, device="meta")
    t.requires_grad_(requires_grad)
    if as_parameter:
        return torch.nn.Parameter(t, requires_grad=requires_grad)
    return t

def _split_gm_by_stages(gm) -> tuple[fx.GraphModule, list[tuple[int, fx.GraphModule, list[int], list]]]:
    """
    Transform a graph module with stage annotations by extracting
    stage computations into submodules and distributing the stage
    submodules to Ray actors.
    
    Returns:
        A tuple of (new_graph_module, list_of_stage_info) where:
        - new_graph_module: The transformed top-level graph module
        - list_of_stage_info: List of (stage_id, submodule, input_idxs, all_inputs) tuples for each extracted stage
    """
    # Collect all nodes with custom metadata
    nodes_with_metadata = []
    for node in gm.graph.nodes:
        if 'custom' in node.meta:
            custom_meta = node.meta['custom']
            # Extract stage from metadata
            if isinstance(custom_meta, dict):
                stage_annotation_id = custom_meta.get('stage')
                if stage_annotation_id is not None:
                    nodes_with_metadata.append((node, stage_annotation_id))
    
    if not nodes_with_metadata:
        return gm, []
    
    # Group nodes by stage ID for creating stage modules
    stage_code = defaultdict(list)  # stage_id -> list of all nodes for this stage
    
    for node, stage_annotation_id in nodes_with_metadata:
        stage_code[stage_annotation_id].append(node)
    
    stage_modules = {}
    
    # Process stages in sorted order to track outputs from previous stages
    for stage_annotation_id, stage_nodes in sorted(stage_code.items(), key=lambda x: x[0]):
        stage_graph = fx.Graph()
        stage_node_mapping = {}
        
        stage_node_set = set(stage_nodes)
        
        # Collect all nodes from other stages to identify stage boundaries
        other_stage_nodes = set()
        for other_stage_id, other_stage_nodes_list in stage_code.items():
            if other_stage_id != stage_annotation_id:
                other_stage_nodes.update(other_stage_nodes_list)
        
        # Check if any nodes are outputs from previous stages
        prev_stage_outputs = set()
        for prev_stage_id in range(stage_annotation_id):
            if prev_stage_id in stage_modules:
                _, _, _, prev_outputs, _, _, _, _, _, _ = stage_modules[prev_stage_id]
                prev_stage_outputs.update(prev_outputs)
        
        # Find all inputs needed by this stage (nodes that are not in the stage set)
        # Check both args and kwargs to find all dependencies
        stage_inputs = set()
        get_attr_nodes = set()  # Track get_attr nodes separately
        
        def extract_nodes_from_arg(arg):
            """Recursively extract all fx.Node objects from an argument"""
            nodes = []
            if isinstance(arg, fx.Node):
                nodes.append(arg)
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    nodes.extend(extract_nodes_from_arg(item))
            elif isinstance(arg, dict):
                for value in arg.values():
                    nodes.extend(extract_nodes_from_arg(value))
            return nodes
        
        for node in stage_nodes:
            # Check args - use both map_arg and explicit extraction for robustness
            for arg in fx.graph.map_arg(node.args, lambda n: n):
                if isinstance(arg, fx.Node) and arg not in stage_node_set:
                    if arg.op == "get_attr":
                        get_attr_nodes.add(arg)
                    elif arg.op == "placeholder":
                        # Placeholders are always inputs
                        stage_inputs.add(arg)
                    elif arg in other_stage_nodes or arg in prev_stage_outputs:
                        # Nodes from other stages or outputs from previous stages are inputs
                        stage_inputs.add(arg)
                    else:
                        # Could be computed in this stage or an input - will determine later
                        stage_inputs.add(arg)
            
            # Check kwargs - explicitly extract nodes from dict values
            # fx.graph.map_arg should work, but let's also manually check kwargs dict
            if node.kwargs:
                # Use map_arg first (should handle most cases)
                for arg in fx.graph.map_arg(node.kwargs, lambda n: n):
                    if isinstance(arg, fx.Node) and arg not in stage_node_set:
                        if arg.op == "get_attr":
                            get_attr_nodes.add(arg)
                        elif arg.op == "placeholder":
                            # Placeholders are always inputs
                            stage_inputs.add(arg)
                        elif arg in other_stage_nodes or arg in prev_stage_outputs:
                            # Nodes from other stages or outputs from previous stages are inputs
                            stage_inputs.add(arg)
                        else:
                            # Could be computed in this stage or an input - will determine later
                            stage_inputs.add(arg)
                
                # Also explicitly extract nodes from kwargs dict values as a fallback
                for kwarg_value in extract_nodes_from_arg(node.kwargs):
                    if kwarg_value not in stage_node_set:
                        if kwarg_value.op == "get_attr":
                            get_attr_nodes.add(kwarg_value)
                        elif kwarg_value.op == "placeholder":
                            stage_inputs.add(kwarg_value)
                        elif kwarg_value in other_stage_nodes or kwarg_value in prev_stage_outputs:
                            stage_inputs.add(kwarg_value)
                        else:
                            stage_inputs.add(kwarg_value)
        
        # Copy get_attr nodes (parameters) to the stage graph first
        # These should be available directly in the stage, not passed as inputs
        for node in gm.graph.nodes:
            if node in get_attr_nodes:
                new_get_attr = stage_graph.node_copy(node, lambda n: n)
                stage_node_mapping[node] = new_get_attr
        
        # Create placeholders for inputs with proper ordering
        # For stages > 0, inputs from previous stage outputs should be ordered
        # according to their order in the previous stage's outputs
        input_placeholders = {}
        input_order = []  # Track order of inputs
        
        # Get outputs from previous stage (if it exists) to determine ordering
        prev_stage_outputs_ordered = []
        if stage_annotation_id > 0:
            prev_stage_id = stage_annotation_id - 1
            if prev_stage_id in stage_modules:
                _, _, _, prev_stage_outputs_list, _, _, _, _, _, _ = stage_modules[prev_stage_id]
                prev_stage_outputs_ordered = prev_stage_outputs_list
        
        # For stages > 0, first add inputs that come from previous stage outputs
        # in the order they appear in the previous stage's outputs
        if stage_annotation_id > 0 and prev_stage_outputs_ordered:
            for prev_output_node in prev_stage_outputs_ordered:
                if prev_output_node in stage_inputs and prev_output_node not in input_placeholders:
                    # If the input is already a placeholder, use node_copy to preserve all metadata
                    if prev_output_node.op == "placeholder":
                        placeholder = stage_graph.node_copy(prev_output_node, lambda n: n)
                    else:
                        # For computed nodes, create a placeholder but preserve metadata
                        placeholder = stage_graph.placeholder(prev_output_node.name)
                        # Copy all metadata from the original node
                        placeholder.meta.update(prev_output_node.meta)
                    stage_node_mapping[prev_output_node] = placeholder
                    input_placeholders[prev_output_node] = placeholder
                    input_order.append(prev_output_node)
        
        # Then add any remaining inputs in original graph order
        # (these are typically placeholders or other inputs not from previous stage)
        for node in gm.graph.nodes:
            if node in stage_inputs and node not in input_placeholders:
                # If the input is already a placeholder, use node_copy to preserve all metadata
                if node.op == "placeholder":
                    placeholder = stage_graph.node_copy(node, lambda n: n)
                else:
                    # For computed nodes, create a placeholder but preserve metadata
                    placeholder = stage_graph.placeholder(node.name)
                    # Copy all metadata from the original node
                    placeholder.meta.update(node.meta)
                stage_node_mapping[node] = placeholder
                input_placeholders[node] = placeholder
                input_order.append(node)
        
        # Find outputs of this stage (nodes used outside the stage set)
        # First, collect all nodes from other stages to identify stage boundaries
        other_stage_nodes = set()
        for other_stage_id, other_stage_nodes_list in stage_code.items():
            if other_stage_id != stage_annotation_id:
                other_stage_nodes.update(other_stage_nodes_list)
        
        # Check if any nodes are outputs from previous stages
        prev_stage_outputs = set()
        for prev_stage_id in range(stage_annotation_id):
            if prev_stage_id in stage_modules:
                _, _, _, prev_outputs, _, _, _, _, _, _ = stage_modules[prev_stage_id]
                prev_stage_outputs.update(prev_outputs)
        
        # Find all nodes computed in this stage (dependencies of stage_nodes)
        # These are nodes that are arguments to stage_nodes but not inputs from other stages
        # We need to recursively collect all dependencies
        stage_computed_nodes = set(stage_nodes)
        visited = set()
        
        def collect_dependencies(node):
            """Recursively collect all node dependencies that are computed in this stage"""
            if node in visited:
                return
            visited.add(node)
            
            # Skip if already in computed nodes
            if node in stage_computed_nodes:
                return
            
            # Skip placeholders, get_attr, nodes from other stages, and outputs from previous stages
            if (node.op == "placeholder" or node.op == "get_attr" or 
                node in other_stage_nodes or node in prev_stage_outputs):
                return
            
            # This node is computed in this stage (it's a dependency of a stage node)
            if node in stage_inputs:
                # Remove from inputs if it's computed in this stage
                stage_inputs.discard(node)
            stage_computed_nodes.add(node)
            
            # Recursively collect dependencies of this node from args
            for arg in fx.graph.map_arg(node.args, lambda n: n):
                if isinstance(arg, fx.Node):
                    collect_dependencies(arg)
            
            # Recursively collect dependencies from kwargs - use both map_arg and explicit extraction
            if node.kwargs:
                # Use map_arg first
                for arg in fx.graph.map_arg(node.kwargs, lambda n: n):
                    if isinstance(arg, fx.Node):
                        collect_dependencies(arg)
                
                # Also explicitly extract nodes from kwargs dict values as a fallback
                for kwarg_node in extract_nodes_from_arg(node.kwargs):
                    collect_dependencies(kwarg_node)
        
        # Collect all dependencies of stage nodes
        for stage_node in stage_nodes:
            # Collect from args
            for arg in fx.graph.map_arg(stage_node.args, lambda n: n):
                if isinstance(arg, fx.Node):
                    collect_dependencies(arg)
            
            # Collect from kwargs - use both map_arg and explicit extraction
            if stage_node.kwargs:
                for arg in fx.graph.map_arg(stage_node.kwargs, lambda n: n):
                    if isinstance(arg, fx.Node):
                        collect_dependencies(arg)
                
                # Also explicitly extract nodes from kwargs dict values
                for kwarg_node in extract_nodes_from_arg(stage_node.kwargs):
                    collect_dependencies(kwarg_node)
        
        # Find outputs: nodes computed in this stage that are used by other stages
        # Check if users have stage annotations from other stages, or if they're in already-processed stages
        # Only consider non-placeholder nodes as outputs — placeholders (e.g. model attributes
        # like freqs_cis, mask) are copied into each stage via get_attr or re-created as
        # placeholders, so they don't need to be passed between stages.
        stage_outputs = []
        for node in stage_computed_nodes:
            if node.op == "placeholder":
                continue
            for user in node.users:
                # Check if user has a stage annotation from a different stage
                user_in_other_stage = False
                if 'custom' in user.meta:
                    custom_meta = user.meta['custom']
                    if isinstance(custom_meta, dict):
                        user_stage = custom_meta.get('stage')
                        if user_stage is not None and user_stage != stage_annotation_id:
                            user_in_other_stage = True
                
                # Also check if user is in an already-processed stage's computed nodes
                if not user_in_other_stage:
                    for other_stage_id in range(stage_annotation_id):
                        if other_stage_id in stage_modules:
                            _, _, _, _, _, other_ordered_nodes, _, _, _, _ = stage_modules[other_stage_id]
                            if user in other_ordered_nodes:
                                user_in_other_stage = True
                                break
                
                # Check if user is in another stage (annotated nodes)
                if not user_in_other_stage and user in other_stage_nodes:
                    user_in_other_stage = True
                
                if user_in_other_stage:
                    if node not in stage_outputs:
                        stage_outputs.append(node)
                # Also check if user is the output node (which uses stage outputs)
                elif user.op == "output":
                    if node not in stage_outputs:
                        stage_outputs.append(node)
        
        # If no external users, use the last node(s) as output
        if not stage_outputs:
            # Use the last computed node in topological order
            ordered_computed = [n for n in gm.graph.nodes if n in stage_computed_nodes]
            stage_outputs = [ordered_computed[-1]] if ordered_computed else []
        
        # Preserve original graph order by iterating through nodes in original order
        # This ensures correctness of computation order
        # Include ALL computed nodes, not just annotated ones
        ordered_stage_nodes = []
        for node in gm.graph.nodes:
            if node in stage_computed_nodes:
                ordered_stage_nodes.append(node)
        
        # Copy stage nodes to the stage graph in original graph order
        for node in ordered_stage_nodes:
            new_stage_node = stage_graph.node_copy(
                node,
                lambda n: stage_node_mapping.get(n, input_placeholders.get(n))
            )
            stage_node_mapping[node] = new_stage_node

        # Create output node
        if stage_outputs:
            output_values = [stage_node_mapping[node] for node in stage_outputs]
            if len(output_values) == 1:
                stage_graph.output(output_values[0])
            else:
                stage_graph.output(tuple(output_values))
        else:
            # No outputs, create a dummy output
            stage_graph.output(stage_graph.placeholder('dummy'))
        
        stage_gm = fx.GraphModule(torch.nn.Module(), stage_graph)
        
        # Store the stage module and related info
        for node in ordered_stage_nodes:
            if node.op == "placeholder":
                input_placeholders[node] = node

        # Load the module on the corresponding actor
        # Track which inputs come from previous stage outputs
        input_idxs = []
        param_idxs = []
        graphargs = []
        placeholders = stage_gm.graph.find_nodes(op="placeholder")
        
        # Create reverse mapping from placeholder to original node
        placeholder_to_original = {placeholder: orig_node for orig_node, placeholder in input_placeholders.items()}
        
        # Get outputs from previous stage (if it exists)
        prev_stage_outputs = set()
        prev_stage_id = stage_annotation_id - 1
        if prev_stage_id in stage_modules:
            _, _, _, prev_stage_outputs_list, _, _, _, _, _, _ = stage_modules[prev_stage_id]
            prev_stage_outputs = set(prev_stage_outputs_list)
        
        for i, placeholder in enumerate(placeholders):
            ex = placeholder.meta["example_value"]

            # Parameter-like placeholders
            is_param_like = ("grapharg" in placeholder.meta) or ("self" in placeholder.name)

            # we may not actually need to call _metathis if we're already placing example inputs on meta, 
            # but it's guaranteed to be safe with this function call
            graphargs.append(
                _meta_tensor_like(
                    ex,
                    requires_grad=bool(getattr(ex, "requires_grad", False)),
                    as_parameter=is_param_like,
                )
            )

            if isinstance(ex, torch.nn.Parameter):
                param_idxs.append(i)

            # For the first stage, the input indices are everything that's not an attribute
            if stage_annotation_id == 0:
                if 'self' not in placeholder.name:
                    input_idxs.append(i)
            # For subsequent stages, check if this placeholder corresponds to an output from the previous stage
            else:
                orig_node = placeholder_to_original.get(placeholder)
                is_from_prev_stage = orig_node is not None and orig_node in prev_stage_outputs
                if is_from_prev_stage:
                    input_idxs.append(i)

        assert set(input_idxs) & set(param_idxs) == set(), "Input and parameter indices should be disjoint"

        stage_modules[stage_annotation_id] = (
            stage_gm,
            input_placeholders,
            input_order,
            stage_outputs,
            stage_nodes,
            ordered_stage_nodes,
            stage_annotation_id,
            input_idxs,
            param_idxs,
            graphargs,
        )

    # Create a new top-level graph and replace stage nodes with call_module
    new_graph = fx.Graph()
    node_mapping = {}
    
    # Copy placeholders
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
            node_mapping[node] = new_node
    
    # Track which stage calls have been replaced
    stage_call_replaced = {}
    
    # Copy all get_attr nodes (parameters) so they're available for stage calls
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            if node not in node_mapping:
                new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
                node_mapping[node] = new_node
    
    # Process nodes in the original graph order to preserve structure
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Already handled
            continue
        
        if node.op == "output":
            # Handle output separately
            continue
        
        if node.op == "get_attr":
            # Already handled above
            continue
        
        if node in node_mapping:
            # Already processed
            continue
        
        # Check if this node belongs to a stage and find which stage call it's part of
        in_stage = False
        for stage_annotation_id, stage_nodes in stage_code.items():
            if node in stage_nodes:
                in_stage = True
                
                # Get the stage module info
                stage_gm, input_placeholders, input_order, stage_outputs, stage_nodes, ordered_stage_nodes, _, _, _, _= stage_modules[stage_annotation_id]
                
                # Check if we've already created the call_module for this stage call
                if stage_annotation_id not in stage_call_replaced:
                    # Find first node in this stage's nodes (in original graph order)
                    first_node = None
                    for n in gm.graph.nodes:
                        if n in stage_nodes:
                            first_node = n
                            break
                    
                    if first_node is None:
                        first_node = stage_nodes[0]
                    
                    # Map inputs in the same order as placeholders (original graph order)
                    mapped_args = []
                    for input_node in input_order:
                        if input_node in node_mapping:
                            mapped_args.append(node_mapping[input_node])
                        else:
                            # Input node not yet mapped, need to copy it first
                            new_input_node = new_graph.node_copy(input_node, lambda n: node_mapping.get(n, n))
                            node_mapping[input_node] = new_input_node
                            mapped_args.append(new_input_node)
                    
                    # Create call_module node for the stage
                    module_name = f"stage_{stage_annotation_id}"
                    call_node = new_graph.call_module(module_name, tuple(mapped_args), {})
                    stage_call_replaced[stage_annotation_id] = (call_node, stage_outputs)
                
                # Map stage nodes to the call_module output
                call_node, stage_outputs = stage_call_replaced[stage_annotation_id]
                
                # Only create output mapping for output nodes (nodes used outside stage)
                if node in stage_outputs:
                    if len(stage_outputs) == 1:
                        # Single output, use call_node directly
                        node_mapping[node] = call_node
                    else:
                        # Multiple outputs, need to getitem
                        output_idx = stage_outputs.index(node)
                        getitem_node = new_graph.call_function(
                            operator.getitem,
                            (call_node, output_idx)
                        )
                        node_mapping[node] = getitem_node
                else:
                    node_mapping[node] = call_node
                break
        
        if not in_stage:
            # Regular node, copy it
            new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
            node_mapping[node] = new_node
    
    
    # Handle output node
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
            break
    
    if output_node:
        output_args = fx.graph.map_arg(output_node.args, lambda n: node_mapping.get(n, n))
        if isinstance(output_args, tuple) and len(output_args) == 1:
            new_graph.output(output_args[0])
        else:
            new_graph.output(output_args)
    
    # Create root module and add stage modules as submodules
    root_module = torch.nn.Module()
    submodule_list = []
    for stage_annotation_id, (stage_gm, placeholders, _, _, _, _, _, input_idxs, param_idxs, params) in stage_modules.items():
        module_name = f"stage_{stage_annotation_id}"
        root_module.add_module(module_name, stage_gm)
        submodule_list.append((stage_annotation_id, stage_gm, input_idxs, param_idxs, params, placeholders))
    
    new_gm = fx.GraphModule(root_module, new_graph)
    
    # Copy parameters, buffers, and modules from original module
    for name, param in gm.named_parameters(recurse=False):
        if name not in new_gm._parameters:
            new_gm.register_parameter(name, param)
    
    for name, buffer in gm.named_buffers(recurse=False):
        if name not in new_gm._buffers:
            new_gm.register_buffer(name, buffer)
    
    for name, module in gm.named_children():
        if name not in [f"stage_{stage_annotation_id}" for stage_annotation_id in stage_modules.keys()]:
            new_gm.add_module(name, module)
    
    new_gm.recompile()
    
    return new_gm, submodule_list



def _insert_a2a_ops(gm: fx.GraphModule) -> tuple[fx.GraphModule, int]:
    """
    Transform a graph module by inserting communication operations on annotated nodes.
    
    This function finds nodes annotated with torch.fx.traceback.annotate containing
    "collective": "all_to_all_single" and inserts the corresponding communication
    operations into the graph.
    
    For each annotated node, it inserts:
    1. (Optional) Reshape input if "reshape": ("input", shape) is specified
    2. buf = torch.empty_like(annotated_tensor)
    3. torch.distributed.all_to_all_single(buf, annotated_tensor)  # Uses default process group
    4. (Optional) Reshape output if "reshape": ("output", shape) is specified
    5. Replaces all uses of annotated_tensor with the result
    
    The communication uses the default distributed process group (no group parameter specified).
    
    The reshape annotation format is: ("input"|"output", shape_tuple)
    where shape_tuple is a tuple of dimensions (can include -1 for inferred dimension).
    
    Args:
        gm: The graph module to transform
        
    Returns:
        A new graph module with communication operations inserted
    """
    # Find all nodes with collective communication annotations
    # We need to identify contiguous blocks of annotations, not just group by metadata
    # Each contiguous block should get its own communication operation
    
    # First, collect all annotated nodes with their metadata
    annotated_nodes = []
    node_list = list(gm.graph.nodes)
    for idx, node in enumerate(node_list):
        if 'custom' in node.meta:
            custom_meta = node.meta['custom']
            if isinstance(custom_meta, dict):
                collective = custom_meta.get('collective')
                if collective == 'all_to_all_single':
                    reshape = custom_meta.get('reshape')
                    annotation_key = tuple(sorted(custom_meta.items()))
                    annotated_nodes.append((idx, node, annotation_key, reshape))
    
    if not annotated_nodes:
        return gm, 0
    
    # Group annotated nodes into contiguous blocks
    # A contiguous block is a sequence of nodes with the same annotation_key
    # that appear sequentially in the graph. Each block gets its own communication operation.
    # We split blocks when:
    # 1. The annotation_key changes (different annotation pattern)
    # 2. The same annotation_key appears again after being interrupted (new instance of same pattern)
    #    This handles the case where the same annotation appears in different layers
    # Group *all* nodes with the same annotation_key into a single block,
    # regardless of how far apart they are in the FX graph.
    # This ensures that every logically-related annotated region (e.g., the
    # "input" or "output" side of a given MoE layer) is treated as one block.
    blocks_by_key = {}
    for idx, node, annotation_key, reshape in annotated_nodes:
        if annotation_key not in blocks_by_key:
            blocks_by_key[annotation_key] = {
                'annotation_key': annotation_key,
                'nodes': [],
                'last_idx': idx,
            }
        block = blocks_by_key[annotation_key]
        block['nodes'].append((node, reshape))
        block['last_idx'] = idx

    annotation_blocks = list(blocks_by_key.values())
    
    n_a2a_ops = len(annotation_blocks)
    
    # For each contiguous block, find the output node (the one used outside the annotation)
    # This is the node that should have communication applied to it
    annotated_output_nodes = []
    for block_idx, block in enumerate(annotation_blocks):
        nodes_in_block = block['nodes']
        nodes_set = {node for node, _ in nodes_in_block}
        
        if len(nodes_in_block) == 1:
            # Single node, use it directly
            annotated_output_nodes.append(nodes_in_block[0])
        else:
            # Multiple nodes in the same block
            # Find the one that is used by nodes outside the annotation block
            # This is the output of the annotated block
            output_candidates = []
            for node, reshape in nodes_in_block:
                # Check if this node is used by any node not in the annotation block
                for user in node.users:
                    if user not in nodes_set:
                        output_candidates.append((node, reshape))
                        break
            
            if output_candidates:
                # Use the first output candidate (should be the output of the annotated block)
                annotated_output_nodes.append(output_candidates[0])
            else:
                # No external users found, use the last node in topological order within the block
                # This happens if the annotated block's output isn't used yet
                last_node = max(nodes_in_block, key=lambda x: node_list.index(x[0]))
                annotated_output_nodes.append(last_node)
    
    # Create a new graph to build the transformed version
    new_graph = fx.Graph()
    node_mapping = {}
    
    # Copy placeholders first
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
            node_mapping[node] = new_node
    
    # Copy all get_attr nodes (parameters) so they're available for use
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            if node not in node_mapping:
                new_node = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
                node_mapping[node] = new_node
    
    # Process nodes in topological order
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Already handled
            continue
        
        if node.op == "output":
            # Handle output separately at the end
            continue
        
        if node.op == "get_attr":
            # Already handled above
            continue
        
        # Check if this node needs communication inserted
        # This should be the output node of an annotated block
        needs_comm = False
        reshape_info = None
        for annotated_node, reshape in annotated_output_nodes:
            if node == annotated_node:
                needs_comm = True
                reshape_info = reshape
                break
        
        if needs_comm:
            # First, copy the node to get its output
            new_node = new_graph.node_copy(
                node,
                lambda n: node_mapping.get(n, n)
            )
            
            # Handle input reshape if specified
            # reshape_info format: ("input"|"output", shape_tuple)
            if reshape_info is not None:
                reshape_type, reshape_shape = reshape_info
                if reshape_type == "input":
                    # Reshape before communication
                    new_node = new_graph.call_function(
                        torch.reshape,
                        (new_node, reshape_shape)
                    )
            
            # Insert communication operations after this node
            # Following the exact pattern from the comments in mixtral.py:
            # buf = torch.empty_like(expert_inputs)
            # torch.distributed.all_to_all_single(buf, expert_inputs, group=device_mesh.get_group())
            # buf = buf.reshape(...)  # if output reshape
            # expert_inputs = buf
            
            # 1. Create buffer: buf = torch.empty_like(new_node)
            buf_node = new_graph.call_function(
                torch.empty_like,
                (new_node,)
            )
            
            # 2. Call all_to_all_single using custom autograd function
            # Use AllToAllSingleFunction to ensure gradients flow correctly
            # This wraps all_to_all_single with proper backward pass
            buf_node = new_graph.call_function(
                _dispatch_a2a_single,
                (buf_node, new_node)
            )
            # After all_to_all_single, buf_node contains the result
            # and maintains gradient connection from new_node through the custom Function
            
            # Handle output reshape if specified
            if reshape_info is not None:
                reshape_type, reshape_shape = reshape_info
                if reshape_type == "output":
                    # Reshape after communication
                    buf_node = new_graph.call_function(
                        torch.reshape,
                        (buf_node, reshape_shape)
                    )
            
            # 3. Replace all uses of the original node with the buffer
            # The buffer now contains the result after communication (and optional reshape)
            node_mapping[node] = buf_node
        else:
            # Regular node, just copy it
            new_node = new_graph.node_copy(
                node,
                lambda n: node_mapping.get(n, n)
            )
            node_mapping[node] = new_node
    
    # Handle output node
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
            break
    
    if output_node:
        output_args = fx.graph.map_arg(output_node.args, lambda n: node_mapping.get(n, n))
        if isinstance(output_args, tuple) and len(output_args) == 1:
            new_graph.output(output_args[0])
        else:
            new_graph.output(output_args)
    
    # Create new graph module with a fresh root module
    root_module = torch.nn.Module()
    
    # Copy parameters, buffers, and modules from original module
    for name, param in gm.named_parameters(recurse=False):
        root_module.register_parameter(name, param)
    
    for name, buffer in gm.named_buffers(recurse=False):
        root_module.register_buffer(name, buffer)
    
    for name, module in gm.named_children():
        root_module.add_module(name, module)
    
    new_gm = fx.GraphModule(root_module, new_graph)
    
    new_gm.recompile()
    
    return new_gm, n_a2a_ops

def _insert_p2p_ops(
    gm: fx.GraphModule,
    input_idxs: list[int],
    prev_stage_id: int,
    next_stage_id: int,
) -> fx.GraphModule:
    """
    Insert point-to-point recv/send ops into a stage graph.

    For each placeholder index in input_idxs, this will:
      1. Create an empty_like buffer from the corresponding placeholder.
      2. Replace all uses of that placeholder with a tensor received from
         prev_stage_id via torch.distributed.recv.

    For each output tensor, this will:
      1. Wrap the tensor in a call to torch.distributed.send with dst set
         to next_stage_id.

    Args:
        gm: GraphModule to transform.
        input_idxs: Indices of placeholders that should receive tensors
            from the previous stage.
        prev_stage_id: Rank used as src for recv operations.
        next_stage_id: Rank used as dst for send operations.

    Returns:
        A new GraphModule with inserted P2P recv/send operations.
    """
    new_graph = fx.Graph()
    node_mapping: dict[fx.Node, fx.Node] = {}

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    # Copy placeholders first so we preserve the original signature.
    for idx, node in enumerate(placeholders):
        new_ph = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
        node_mapping[node] = new_ph

        if idx in input_idxs:
            # Directly recv into the placeholder tensor and use that value
            # in place of the original placeholder.
            recv_node = new_graph.call_function(
                _dispatch_p2p_recv,
                (new_ph, prev_stage_id),
            )
            node_mapping[node] = recv_node

    # Copy get_attr nodes (parameters/buffers).
    for node in gm.graph.nodes:
        if node.op == "get_attr" and node not in node_mapping:
            new_attr = new_graph.node_copy(node, lambda n: node_mapping.get(n, n))
            node_mapping[node] = new_attr

    # Copy all other nodes except output, rewiring to use P2P-recv inputs.
    output_node: fx.Node | None = None
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "output":
            output_node = node
            continue

        new_node = new_graph.node_copy(
            node,
            lambda n: node_mapping.get(n, n),
        )
        node_mapping[node] = new_node

    if output_node is None:
        raise RuntimeError("GraphModule has no output node")

    # Map original outputs into the new graph.
    raw_outputs = fx.graph.map_arg(
        output_node.args,
        lambda n: node_mapping.get(n, n),
    )

    # For each tensor-valued node in the outputs, add a send op and
    # return the result of the send so it is kept in the dataflow.
    def _wrap_with_send(arg):
        if isinstance(arg, fx.Node):
            return new_graph.call_function(
                _dispatch_p2p_send,
                (arg, next_stage_id),
            )
        return arg

    wrapped_outputs = fx.graph.map_arg(raw_outputs, _wrap_with_send)

    if isinstance(wrapped_outputs, tuple) and len(wrapped_outputs) == 1:
        new_graph.output(wrapped_outputs[0])
    else:
        new_graph.output(wrapped_outputs)

    root_module = torch.nn.Module()

    # Copy parameters, buffers, and submodules from the original module.
    for name, param in gm.named_parameters(recurse=False):
        root_module.register_parameter(name, param)

    for name, buffer in gm.named_buffers(recurse=False):
        root_module.register_buffer(name, buffer)

    for name, module in gm.named_children():
        root_module.add_module(name, module)

    new_gm = fx.GraphModule(root_module, new_graph)
    new_gm.recompile()
    return new_gm