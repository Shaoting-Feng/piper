import hashlib
import json
import ray
import torch
import torch.fx as fx
import torch.distributed as dist
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional
import operator
import os
import time
from torch.autograd import Function

from .piper_utils import create_logger, LOG_LEVEL, VERBOSE, piper_metadata
from .piper_exec import TaskType, PipelineSchedule, Chunk, Task, TaskDAG, runtime_sort_key

logger = create_logger("piper_graph_transform", LOG_LEVEL)


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

    def _stable_target(target):
        """Return a stable string for a node target, avoiding memory addresses."""
        if isinstance(target, str):
            return target
        if callable(target) and hasattr(target, '__module__') and hasattr(target, '__qualname__'):
            return f"{target.__module__}.{target.__qualname__}"
        return str(target)

    graph_signature = hashlib.sha256(
        json.dumps([(n.name, n.op, _stable_target(n.target)) for n in compute_nodes] + [num_stages]).encode()
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

    def _ensure_real(val):
        """Recursively replace any meta tensors with real tensors on device."""
        if isinstance(val, torch.Tensor) and val.device.type == "meta":
            return _materialize_tensor(val, device)
        elif isinstance(val, (list, tuple)):
            return type(val)(_ensure_real(v) for v in val)
        elif isinstance(val, dict):
            return {k: _ensure_real(v) for k, v in val.items()}
        return val

    def _fetch_arg(arg):
        if isinstance(arg, fx.Node):
            _ensure_in_env(arg)
            val = env.get(arg.name)
            real_val = _ensure_real(val)
            if real_val is not val:
                env[arg.name] = real_val  # cache so future accesses also get real tensor
            return real_val
        elif isinstance(arg, (list, tuple)):
            vals = [_fetch_arg(a) for a in arg]
            return type(arg)(vals)
        elif isinstance(arg, dict):
            return {k: _fetch_arg(v) for k, v in arg.items()}
        elif isinstance(arg, torch.Tensor) and arg.device.type == "meta":
            # Constants baked into node.args at trace time (e.g. RoPE frequency
            # matrices) are meta tensors when the model was traced on meta device.
            # Materialize them on the profiling device so call_function nodes don't fail.
            return _materialize_tensor(arg, device)
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

        # Log the top 10 largest tensors
        active.sort(key=lambda x: x[3], reverse=True)
        for name, shape, dtype, nbytes in active[:10]:
            logger.log(VERBOSE, f"  {name}: {shape} {dtype} ({nbytes / (1024*1024):.1f} MB)")

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
            logger.log(VERBOSE, f"Skipping node {node.name}: predecessor produced None")
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
            logger.debug(
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
        logger.debug(f"Saved split cache to {cache_file}")
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
        logger.debug(
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
        logger.log(VERBOSE, "No stage nodes found in graph")
        return gm, []
    else:
        logger.log(VERBOSE, f"Found stage nodes in graph")
    
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
    logger.log(VERBOSE, f"Found {len(annotation_blocks)} contiguous annotation blocks")
    
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
    
    logger.log(VERBOSE, f"Found {len(annotated_output_nodes)} annotation blocks requiring communication operations")
    
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
                    # DEBUG: Log reshape information
                    logger.log(VERBOSE, f"Applying input reshape {reshape_shape} to node {node.name}")
                    if isinstance(reshape_shape, (list, tuple)) and -1 in reshape_shape:
                        logger.warning(f"Input reshape shape contains -1: {reshape_shape}, this may cause shape inference issues")
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
                    # DEBUG: Log reshape information
                    logger.log(VERBOSE, f"Applying output reshape {reshape_shape} to node {node.name}, buf_node shape inference needed")
                    # Log the reshape shape for debugging
                    if isinstance(reshape_shape, (list, tuple)) and -1 in reshape_shape:
                        logger.warning(f"Reshape shape contains -1: {reshape_shape}, this may cause shape inference issues")
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


# ---------------------------------------------------------------------------
# Per-chunk task-chain builder (internal helper)
# ---------------------------------------------------------------------------

def _splice_a2a_nodes(
    chain: list,
    boundaries: dict,
    task_type,
    pp_rank: int,
    source_chunk,
) -> list:
    """Splice A2A nodes into a bucket chain at all-to-all boundaries.

    For FWD: inserts FWD_A2A after FWD(bkt_id) when bkt_id is a boundary.
    For BWD/BWD_I: inserts BWD_A2A before BWD(bkt_id) in execution order
    (i.e. between BWD(bkt_id+1) and BWD(bkt_id)).

    Updates data and temporal edges in-place; returns the new chain list
    with A2A nodes interleaved.
    """
    a2a_type = TaskType.FWD_A2A if task_type == TaskType.FWD else TaskType.BWD_A2A
    new_chain: list = []

    for i, task in enumerate(chain):
        bkt_id = task.bucket_id
        is_last = (i == len(chain) - 1)

        if task_type == TaskType.FWD:
            new_chain.append(task)
            if bkt_id in boundaries and not is_last:
                next_task = chain[i + 1]
                a2a_node = Task(
                    task_type=a2a_type,
                    batches=list(task.batches),
                    task_pp_rank=pp_rank,
                    pp_rank=pp_rank,
                    time_step=0,
                    bucket_id=bkt_id,
                    resource="ep_stream",
                    custom_metadata={"a2a_tensor_idx": boundaries[bkt_id]},
                    source_chunk=source_chunk,
                    associated_chunk=source_chunk,
                )
                # Rewire data: task → next_task  ⟹  task → a2a → next_task
                task.data_succs.remove(next_task)
                next_task.data_preds.remove(task)
                task.data_succs.append(a2a_node)
                a2a_node.data_preds.append(task)
                a2a_node.data_succs.append(next_task)
                next_task.data_preds.append(a2a_node)
                # Rewire temporal
                task.temporal_succs.remove(next_task)
                next_task.temporal_preds.remove(task)
                task.temporal_succs.append(a2a_node)
                a2a_node.temporal_preds.append(task)
                a2a_node.temporal_succs.append(next_task)
                next_task.temporal_preds.append(a2a_node)
                new_chain.append(a2a_node)
        else:
            # BWD / BWD_I: insert A2A before task (i.e. between chain[i-1] and chain[i])
            if bkt_id in boundaries and i > 0:
                prev_task = chain[i - 1]
                a2a_node = Task(
                    task_type=a2a_type,
                    batches=list(task.batches),
                    task_pp_rank=pp_rank,
                    pp_rank=pp_rank,
                    time_step=0,
                    bucket_id=bkt_id,
                    resource="ep_stream",
                    custom_metadata={"a2a_tensor_idx": boundaries[bkt_id]},
                    source_chunk=source_chunk,
                    associated_chunk=source_chunk,
                )
                # Rewire data: prev_task → task  ⟹  prev_task → a2a → task
                prev_task.data_succs.remove(task)
                task.data_preds.remove(prev_task)
                prev_task.data_succs.append(a2a_node)
                a2a_node.data_preds.append(prev_task)
                a2a_node.data_succs.append(task)
                task.data_preds.append(a2a_node)
                # Rewire temporal
                prev_task.temporal_succs.remove(task)
                task.temporal_preds.remove(prev_task)
                prev_task.temporal_succs.append(a2a_node)
                a2a_node.temporal_preds.append(prev_task)
                a2a_node.temporal_succs.append(task)
                task.temporal_preds.append(a2a_node)
                new_chain.append(a2a_node)
            new_chain.append(task)

    return new_chain


def _build_chunk_task_chain(
    chunk,
    pp_rank: int,
    col_idx: int,
    bucket_counts: dict,
    a2a_info: dict,
) -> tuple:
    """Build the expanded task chain for a single schedule chunk.

    For FWD / BWD / BWD_I chunks with bucketing: returns a chain of K bucket
    tasks with internal data + temporal edges.  A2A nodes are spliced in at
    boundaries.  For all other chunk types a single task is returned.

    All created tasks have ``source_chunk`` set to *chunk*.

    Returns:
        ``(all_tasks, head, tail)`` where *head* is the first task in
        execution order and *tail* is the last.
    """
    TIME_SCALE = 1000

    task_type = chunk.type
    stage_id = chunk.batches[0].stage_id if chunk.batches else None
    K = bucket_counts.get(stage_id, 1) if stage_id is not None else 1

    # FWD, BWD, BWD_I, BWD_W all support per-bucket tasks
    if task_type not in (TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W):
        K = 1

    if K == 1:
        task = Task(
            task_type=chunk.type,
            batches=list(chunk.batches),
            task_pp_rank=chunk.pp_rank,
            pp_rank=pp_rank,
            time_step=col_idx,
            source_chunk=chunk,
            associated_chunk=chunk,
        )
        return [task], task, task

    # --- Bucketed chain ---
    if task_type == TaskType.FWD:
        # Execution order: bucket 0, 1, …, K-1
        chain = [
            Task(
                task_type=TaskType.FWD,
                batches=list(chunk.batches),
                task_pp_rank=pp_rank,
                pp_rank=pp_rank,
                time_step=col_idx * TIME_SCALE + i,
                bucket_id=b,
                source_chunk=chunk,
                associated_chunk=chunk,
            )
            for i, b in enumerate(range(K))
        ]
    else:
        # BWD / BWD_I: execution order K-1, K-2, …, 0
        chain = [
            Task(
                task_type=task_type,
                batches=list(chunk.batches),
                task_pp_rank=pp_rank,
                pp_rank=pp_rank,
                time_step=col_idx * TIME_SCALE + i,
                bucket_id=K - 1 - i,
                source_chunk=chunk,
                associated_chunk=chunk,
            )
            for i in range(K)
        ]

    # Connect consecutive bucket tasks with data + temporal edges
    for i in range(K - 1):
        chain[i].data_succs.append(chain[i + 1])
        chain[i + 1].data_preds.append(chain[i])
        chain[i].temporal_succs.append(chain[i + 1])
        chain[i + 1].temporal_preds.append(chain[i])

    # Insert A2A nodes at boundaries if applicable.
    # BWD_W (weight gradients) are computed locally and don't need A2A separation.
    if task_type != TaskType.BWD_W:
        boundaries = a2a_info.get(stage_id, {}) if stage_id is not None else {}
        if boundaries:
            chain = _splice_a2a_nodes(chain, boundaries, task_type, pp_rank, chunk)

    return chain, chain[0], chain[-1]


# ---------------------------------------------------------------------------
# expand_chunks_to_dags: build the full task DAG chunk-by-chunk
# ---------------------------------------------------------------------------

def expand_chunks_to_dags(
    schedule,
    bucket_counts: dict,
    a2a_info: dict,
) -> "TaskDAG":
    """Build a :class:`TaskDAG` by expanding every schedule chunk in-place.

    Replaces the former pipeline of ``schedule_to_dag`` →
    ``expand_bucket_tasks`` → ``expand_a2a_tasks`` →
    ``insert_p2p_ops`` → ``insert_ar_ops``.

    For each non-``None`` cell of *schedule*, :func:`_build_chunk_task_chain`
    produces the expanded task chain (with bucketing and A2A nodes already
    wired).  Cross-chunk data edges, SEND/RECV nodes, and ALL_REDUCE nodes
    are then added in subsequent passes.

    Every task created here has its ``source_chunk`` field set to the
    originating :class:`Chunk` from the pipeline schedule.

    The returned :class:`TaskDAG` has data edges and within-chain temporal
    edges only.  Between-chunk temporal edges are inserted by the separate
    :func:`add_temporal_dependencies` pass.

    Args:
        schedule: The 2-D pipeline pipeline schedule.
        bucket_counts: Mapping ``stage_id -> K`` for stages with ``K > 1``
            buckets.  Stages absent from this dict use a single bucket.
        a2a_info: Mapping ``stage_id -> {boundary_bucket_id -> tensor_idx}``
            for expert-parallel stages.  May be empty.
        dp_degree: Number of data-parallel replicas.  When ``> 1``,
            ALL_REDUCE nodes are inserted for every BWD/BWD_W bucket.
            Whether a bucket actually has trainable parameters is checked
            at runtime in run_dag, keeping the schedule model-agnostic.
        a2a_ar_no_overlap: When true, adds same-rank temporal dependencies
            from the last backward compute task to ALL_REDUCE nodes so
            gradient all-reduces run after backward compute on that actor.

    Returns:
        A new :class:`TaskDAG` containing all compute, A2A, SEND/RECV, and
        (if *dp_degree* > 1) ALL_REDUCE task nodes.
    """
    # mb_to_entries[mb_idx] = list of (col_idx, sub_idx, head, tail, pp_rank)
    # Used to build cross-chunk data edges in time order.
    mb_to_entries: dict = defaultdict(list)

    # (pp_rank, stage_id, mb_idx, bucket_id) -> BWD_I task (for per-bucket BWD_I → BWD_W edges)
    bwdi_tasks: dict = {}

    all_tasks: list = []

    # ------------------------------------------------------------------
    # Pass 1: build per-chunk task chains
    # ------------------------------------------------------------------
    # Precompute logical start times for cross-rank ordering keys.
    start_times = schedule._compute_start_times()

    for pp_rank, row in enumerate(schedule.grid):
        for col_idx, chunk in enumerate(row):
            if chunk.type == TaskType.FWD_BWD:
                fwd_batch, bwd_batch = chunk.batches[0], chunk.batches[1]

                fwd_sub = Chunk(pp_rank=pp_rank, batches=[fwd_batch], type=TaskType.FWD)
                fwd_chain, fwd_head, fwd_tail = _build_chunk_task_chain(
                    fwd_sub, pp_rank, col_idx, bucket_counts, a2a_info
                )
                for t in fwd_chain:
                    t.source_chunk = chunk  # override to point at original FWD_BWD cell

                bwd_sub = Chunk(pp_rank=pp_rank, batches=[bwd_batch], type=TaskType.BWD)
                bwd_chain, bwd_head, bwd_tail = _build_chunk_task_chain(
                    bwd_sub, pp_rank, col_idx, bucket_counts, a2a_info
                )
                for t in bwd_chain:
                    t.source_chunk = chunk

                # Within-cell ordering: FWD executes before BWD
                fwd_tail.temporal_succs.append(bwd_head)
                bwd_head.temporal_preds.append(fwd_tail)

                all_tasks.extend(fwd_chain + bwd_chain)
                logical_t = start_times[pp_rank][col_idx]
                mb_to_entries[fwd_batch.mb_idx].append((logical_t, 0, fwd_head, fwd_tail, pp_rank))
                mb_to_entries[bwd_batch.mb_idx].append((logical_t, 1, bwd_head, bwd_tail, pp_rank))

            else:
                tasks, head, tail = _build_chunk_task_chain(
                    chunk, pp_rank, col_idx, bucket_counts, a2a_info
                )
                all_tasks.extend(tasks)

                # UPD and BWD_W are excluded from the cross-stage mb chain
                if chunk.type not in (TaskType.UPD, TaskType.BWD_W):
                    logical_t = start_times[pp_rank][col_idx]
                    for batch in chunk.batches:
                        mb_to_entries[batch.mb_idx].append(
                            (logical_t, 0, head, tail, pp_rank)
                        )

                # Track per-bucket BWD_I tasks for BWD_I → BWD_W data edges
                if chunk.type == TaskType.BWD_I and chunk.batches:
                    b = chunk.batches[0]
                    for t in tasks:
                        if t.task_type == TaskType.BWD_I:
                            bwdi_tasks[(pp_rank, b.stage_id, b.mb_idx, t.bucket_id)] = t

    # ------------------------------------------------------------------
    # Pass 1.5: assign unique_bucket_id and compute_loss
    # ------------------------------------------------------------------
    # Build a sorted (stage_id, stage_bucket_id) → unique_bucket_id mapping.
    # Sorting by stage_id ensures the mapping is deterministic and consistent
    # with the offset computation in piper.py that passes unique_bucket_ids
    # to _load_stage.
    _compute_types = (TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W)
    _stage_bucket_pairs = sorted(
        {(t.batches[0].stage_id, t.bucket_id)
         for t in all_tasks
         if t.task_type in _compute_types and t.batches},
        key=lambda x: (x[0], x[1]),
    )
    _sb_to_ubid: dict = {sb: i for i, sb in enumerate(_stage_bucket_pairs)}

    _last_stage_id = max(
        (t.batches[0].stage_id
         for t in all_tasks
         if t.task_type in _compute_types and t.batches),
        default=None,
    )

    for task in all_tasks:
        if task.task_type in _compute_types and task.batches:
            s = task.batches[0].stage_id
            task.unique_bucket_id = _sb_to_ubid.get((s, task.bucket_id))

    # Set compute_loss=True on the first-to-run BWD/BWD_I task of the last stage.
    # "First to run" = highest bucket_id (BWD executes in reverse bucket order).
    if _last_stage_id is not None:
        _last_stage_K = bucket_counts.get(_last_stage_id, 1)
        _last_bkt = _last_stage_K - 1
        for task in all_tasks:
            if (task.task_type in (TaskType.BWD, TaskType.BWD_I)
                    and task.batches
                    and task.batches[0].stage_id == _last_stage_id
                    and task.bucket_id == _last_bkt):
                task.compute_loss = True

    # ------------------------------------------------------------------
    # Pass 2: per-bucket BWD_I → BWD_W data edges
    # ------------------------------------------------------------------
    bwd_w_tasks: dict = {}  # (pp_rank, stage_id, mb_idx, bucket_id) -> Task
    for task in all_tasks:
        if task.task_type == TaskType.BWD_W and task.batches:
            b = task.batches[0]
            bwd_w_tasks[(task.pp_rank, b.stage_id, b.mb_idx, task.bucket_id)] = task

    for key, bwdi_task in bwdi_tasks.items():
        bwdw_task = bwd_w_tasks.get(key)
        if bwdw_task is not None:
            bwdi_task.data_succs.append(bwdw_task)
            bwdw_task.data_preds.append(bwdi_task)

    # ------------------------------------------------------------------
    # Pass 3: cross-chunk data edges
    # ------------------------------------------------------------------
    for mb_idx, entries in mb_to_entries.items():
        entries.sort(key=lambda e: (e[0], e[1], e[4]))  # col_idx, sub_idx, pp_rank
        for i in range(len(entries) - 1):
            _ci, _si, _head_i, tail_i, _ri = entries[i]
            _cn, _sn, head_n, _tn, _rn = entries[i + 1]
            tail_i.data_succs.append(head_n)
            head_n.data_preds.append(tail_i)


    # ------------------------------------------------------------------
    # Pass 4: SEND / RECV nodes for cross-rank data edges
    # ------------------------------------------------------------------
    cross_edges = [
        (src, dst)
        for src in list(all_tasks)
        for dst in list(src.data_succs)
        if src.pp_rank != dst.pp_rank
    ]
    for src, dst in cross_edges:
        src.data_succs.remove(dst)
        dst.data_preds.remove(src)

        send_node = Task(
            task_type=TaskType.SEND,
            batches=list(src.batches),
            task_pp_rank=src.pp_rank,
            pp_rank=src.pp_rank,
            time_step=0,  # assigned later by assign_time_steps
            peer_pp_rank=dst.pp_rank,
            resource="pp_stream",
            source_chunk=src.source_chunk,
            associated_chunk=src.associated_chunk,
        )
        src.data_succs.append(send_node)
        send_node.data_preds.append(src)

        recv_metadata = {}
        if dst.task_type in (TaskType.BWD, TaskType.BWD_I):
            recv_metadata["fwd_ubid"] = dst.unique_bucket_id
        recv_node = Task(
            task_type=TaskType.RECV,
            batches=list(dst.batches),
            task_pp_rank=dst.pp_rank,
            pp_rank=dst.pp_rank,
            time_step=0,  # assigned later by assign_time_steps
            peer_pp_rank=src.pp_rank,
            resource="pp_stream",
            source_chunk=dst.source_chunk,
            associated_chunk=dst.associated_chunk,
            custom_metadata=recv_metadata,
        )
        recv_node.data_succs.append(dst)
        dst.data_preds.append(recv_node)

        all_tasks.append(send_node)
        all_tasks.append(recv_node)

    return TaskDAG(nodes=all_tasks)


# ---------------------------------------------------------------------------
# add_temporal_dependencies: insert temporal edges between adjacent chunks
# ---------------------------------------------------------------------------

_CRITICAL_PATH_TYPES = frozenset({
    TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W,
    TaskType.FWD_BWD, TaskType.FWD_A2A, TaskType.BWD_A2A,
})


def add_temporal_dependencies(dag: "TaskDAG", schedule) -> None:
    """Insert temporal edges that enforce execution order between adjacent chunks.

    For each row of *schedule*, consecutive non-``None`` chunk pairs ``(c1,
    c2)`` receive a temporal edge from the last critical-path task of *c1* to
    the first critical-path task of *c2*.

    The **critical-path chain** for a chunk is the sub-DAG of tasks associated
    with that chunk whose type is in :data:`_CRITICAL_PATH_TYPES` (i.e.
    compute and A2A tasks; SEND, RECV, and ALL_REDUCE are excluded).  Head and
    tail are found by following both data-dependency *and* within-chunk
    temporal edges.

    A :data:`~TaskType.UPD` node is also synthesised for each pipeline rank
    and appended to ``dag.nodes`` as the terminal temporal task for that rank.

    Args:
        dag: :class:`TaskDAG` produced by :func:`expand_chunks_to_dags`.
            Must contain only data edges and within-chain temporal edges —
            no between-chunk temporal edges should exist yet.
        schedule: The :class:`PipelineSchedule` whose grid determines
            adjacency and row membership.
    """
    # Group all tasks by id(source_chunk) for fast lookup.
    sc_to_tasks: dict = defaultdict(list)
    for task in dag.nodes:
        if task.source_chunk is not None:
            sc_to_tasks[id(task.source_chunk)].append(task)

    def _cp_endpoints(chunk_id: int):
        """Return (head, tail) of the critical-path sub-DAG for *chunk_id*.

        Uses both data and temporal edges within the filtered task set so
        that FWD_BWD cells (where FWD and BWD sub-chains are joined only by
        a temporal edge) are handled correctly.
        """
        tasks = [t for t in sc_to_tasks.get(chunk_id, [])
                 if t.task_type in _CRITICAL_PATH_TYPES]
        if not tasks:
            return None, None
        task_ids = {id(t) for t in tasks}
        head = next(
            t for t in tasks
            if not any(id(p) in task_ids
                       for p in list(t.data_preds) + list(t.temporal_preds))
        )
        tail = next(
            t for t in tasks
            if not any(id(s) in task_ids
                       for s in list(t.data_succs) + list(t.temporal_succs))
        )
        return head, tail

    new_nodes: list = []

    for pp_rank, row in enumerate(schedule.grid):
        non_none = list(enumerate(row))
        if not non_none:
            continue

        # Between-chunk temporal edges for consecutive pairs
        for i in range(len(non_none) - 1):
            _, c1 = non_none[i]
            _, c2 = non_none[i + 1]
            _, tail_c1 = _cp_endpoints(id(c1))
            head_c2, _ = _cp_endpoints(id(c2))
            if tail_c1 is not None and head_c2 is not None:
                tail_c1.temporal_succs.append(head_c2)
                head_c2.temporal_preds.append(tail_c1)

        # UPD node: final optimizer step for this rank.
        # In the no-DP case, every BWD-type chunk's critical-path tail feeds
        # into UPD so that UPD correctly waits for ALL backward passes, not
        # just the last one in the temporal chain.  In the DP/ZeRO case,
        # _wire_sync_to_upd (called inside insert_zero_ops) additionally
        # connects each AR/RS → UPD temporal edge.
        _BWD_CHUNK_TYPES = frozenset({
            TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W, TaskType.FWD_BWD,
        })

        # Use the last BWD-type chunk's tail for UPD's pp_rank / batches.
        ref_tail = None
        for _, chunk in reversed(non_none):
            if chunk.type in _BWD_CHUNK_TYPES:
                _, ref_tail = _cp_endpoints(id(chunk))
                if ref_tail is not None:
                    break
        # Fall back to the very last chunk if no BWD chunk exists.
        if ref_tail is None:
            _, ref_tail = _cp_endpoints(id(non_none[-1][1]))
        if ref_tail is None:
            continue

        upd_node = Task(
            task_type=TaskType.UPD,
            batches=list(ref_tail.batches),
            task_pp_rank=ref_tail.pp_rank,
            pp_rank=ref_tail.pp_rank,
            time_step=0,
            source_chunk=non_none[-1][1],
        )

        # Wire every BWD chunk tail → UPD.
        seen_tails: set[int] = set()
        for _, chunk in non_none:
            if chunk.type not in _BWD_CHUNK_TYPES:
                continue
            _, tail = _cp_endpoints(id(chunk))
            if tail is not None and id(tail) not in seen_tails:
                seen_tails.add(id(tail))
                tail.temporal_succs.append(upd_node)
                upd_node.temporal_preds.append(tail)

        # If no BWD chunks (FWD-only schedule), fall back to last chunk.
        if not upd_node.temporal_preds:
            ref_tail.temporal_succs.append(upd_node)
            upd_node.temporal_preds.append(ref_tail)

        new_nodes.append(upd_node)

    dag.nodes.extend(new_nodes)


# ---------------------------------------------------------------------------
# split_dag_by_rank: partition unified DAG into per-rank DAGs
# ---------------------------------------------------------------------------

def split_dag_by_rank(dag: "TaskDAG") -> list:
    """Split a unified :class:`TaskDAG` into one :class:`TaskDAG` per PP rank.

    Asserts that no cross-rank data or temporal edges remain (they should
    have been replaced by SEND/RECV nodes in :func:`expand_chunks_to_dags`).

    Returns:
        List of :class:`TaskDAG` objects ordered by ascending PP rank.
    """
    for node in dag.nodes:
        for succ in node.data_succs:
            assert node.pp_rank == succ.pp_rank, (
                f"Cross-rank data edge still present: "
                f"{node.node_id()} (rank {node.pp_rank}) → "
                f"{succ.node_id()} (rank {succ.pp_rank})"
            )
        for ts in node.temporal_succs:
            assert node.pp_rank == ts.pp_rank, (
                f"Cross-rank temporal edge present: "
                f"{node.node_id()} (rank {node.pp_rank}) → "
                f"{ts.node_id()} (rank {ts.pp_rank})"
            )

    rank_nodes: dict = defaultdict(list)
    for node in dag.nodes:
        rank_nodes[node.pp_rank].append(node)

    return [TaskDAG(nodes=rank_nodes[r]) for r in sorted(rank_nodes)]


def _remove_ar_nodes(rank_dag: TaskDAG) -> None:
    ar_set = {id(n) for n in rank_dag.nodes if n.task_type == TaskType.ALL_REDUCE}
    if not ar_set:
        return
    for n in rank_dag.nodes:
        if id(n) not in ar_set:
            n.data_preds = [p for p in n.data_preds if id(p) not in ar_set]
            n.data_succs = [s for s in n.data_succs if id(s) not in ar_set]
            n.temporal_preds = [p for p in n.temporal_preds if id(p) not in ar_set]
            n.temporal_succs = [s for s in n.temporal_succs if id(s) not in ar_set]
    rank_dag.nodes = [n for n in rank_dag.nodes if id(n) not in ar_set]


def _zero_task(
    ref: Task,
    bucket_id: int,
    task_type: TaskType,
    time_step: int = 0,
    *,
    ubid: int | None = None,
    source_chunk: Chunk | None = None,
) -> Task:
    resource = "compute_stream"
    if task_type in {
        TaskType.ALL_REDUCE,
        TaskType.REDUCE_SCATTER,
        TaskType.ALL_GATHER,
        TaskType.ALLOC_FULL_GRADS,
        TaskType.FREE_FULL_GRADS,
        TaskType.ALLOC_FULL_PARAMS,
        TaskType.FREE_FULL_PARAMS,
    }:
        resource = "ep_stream" if piper_metadata.ar_a2a_same_stream else "dp_stream"
    return Task(
        task_type=task_type,
        batches=list(ref.batches),
        task_pp_rank=ref.pp_rank,
        pp_rank=ref.pp_rank,
        time_step=time_step,
        bucket_id=bucket_id,
        resource=resource,
        unique_bucket_id=ubid,
        source_chunk=source_chunk or ref.source_chunk,
        associated_chunk=ref.associated_chunk,
    )


def _bucket_compute_tasks(
    rank_dag: TaskDAG,
    bucket_filter_keys: set | None = None,
) -> dict[int, list[Task]]:
    compute_types = {TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W}
    by_ubid: dict[int, list[Task]] = defaultdict(list)
    for node in rank_dag.nodes:
        if node.task_type not in compute_types or node.unique_bucket_id is None:
            continue
        if bucket_filter_keys is not None:
            if not node.batches:
                continue
            stage_id = node.batches[0].stage_id
            if (stage_id, node.bucket_id) not in bucket_filter_keys:
                continue
        by_ubid[node.unique_bucket_id].append(node)
    for nodes in by_ubid.values():
        nodes.sort(key=lambda n: n.time_step)
    return by_ubid


def _add_data_edge(src: Task, dst: Task) -> None:
    if dst not in src.data_succs:
        src.data_succs.append(dst)
    if src not in dst.data_preds:
        dst.data_preds.append(src)


def _add_temporal_edge(src: Task, dst: Task) -> None:
    if dst not in src.temporal_succs:
        src.temporal_succs.append(dst)
    if src not in dst.temporal_preds:
        dst.temporal_preds.append(src)


def _remove_data_edge(src: Task, dst: Task) -> None:
    if dst in src.data_succs:
        src.data_succs.remove(dst)
    if src in dst.data_preds:
        dst.data_preds.remove(src)


def _remove_temporal_edge(src: Task, dst: Task) -> None:
    if dst in src.temporal_succs:
        src.temporal_succs.remove(dst)
    if src in dst.temporal_preds:
        dst.temporal_preds.remove(src)


def _append_metadata_ubid(node: Task, key: str, ubid: int | None) -> None:
    if ubid is None:
        return
    values = node.custom_metadata.setdefault(key, [])
    if ubid not in values:
        values.append(ubid)


def _insert_zero_grad_ops(
    rank_dag: TaskDAG,
    *,
    sync_type: TaskType,
    use_grad_lifetime: bool = False,
    zero_bucket_keys: set | None = None,
    gradient_accumulation: bool = True,
) -> None:
    """Insert gradient-sync nodes for ZeRO-0/1/2.

    With ``gradient_accumulation=True``, a bucket's sync is attached only to the
    last BWD/BWD_W occurrence of that bucket. For ZeRO-2, the full-grad
    lifetime widens to first occurrence → last occurrence.
    """
    _remove_ar_nodes(rank_dag)

    grad_sync_types = {TaskType.BWD, TaskType.BWD_W}
    new_nodes: list[Task] = []

    for ubid, bucket_nodes in _bucket_compute_tasks(rank_dag, zero_bucket_keys).items():
        grad_nodes = [node for node in bucket_nodes if node.task_type in grad_sync_types]
        if not grad_nodes:
            continue
        first_compute = min(grad_nodes, key=lambda node: (node.time_step, node.uid))
        last_compute = max(grad_nodes, key=lambda node: (node.time_step, node.uid))

        if gradient_accumulation:
            sync_sources = [last_compute]
        else:
            sync_sources = grad_nodes

        for compute_node in sync_sources:
            sync = _zero_task(compute_node, compute_node.bucket_id, sync_type, ubid=ubid)
            _add_data_edge(compute_node, sync)
            new_nodes.append(sync)

        if use_grad_lifetime:
            if gradient_accumulation:
                first_compute.custom_metadata["zero_alloc_full_grads_before"] = True
                _append_metadata_ubid(new_nodes[-1], "zero_free_full_grads_after_ubids", ubid)
            else:
                for compute_node in grad_nodes:
                    sync = next(
                        node for node in new_nodes
                        if node.task_type == sync_type and node.unique_bucket_id == ubid and compute_node in node.data_preds
                    )
                    compute_node.custom_metadata["zero_alloc_full_grads_before"] = True
                    _append_metadata_ubid(sync, "zero_free_full_grads_after_ubids", ubid)

    rank_dag.nodes.extend(new_nodes)


def _insert_zero1_ops(
    rank_dag: TaskDAG,
    zero_bucket_keys: set | None = None,
    gradient_accumulation: bool = True,
) -> None:
    """ZeRO-1: all-reduce after each bucket or after its last occurrence."""
    _insert_zero_grad_ops(
        rank_dag,
        sync_type=TaskType.ALL_REDUCE,
        zero_bucket_keys=zero_bucket_keys,
        gradient_accumulation=gradient_accumulation,
    )


def _insert_zero2_ops(
    rank_dag: TaskDAG,
    zero_bucket_keys: set | None = None,
    gradient_accumulation: bool = True,
) -> None:
    """ZeRO-2: reduce-scatter with widened grad lifetime under accumulation."""
    _insert_zero_grad_ops(
        rank_dag,
        sync_type=TaskType.REDUCE_SCATTER,
        use_grad_lifetime=True,
        zero_bucket_keys=zero_bucket_keys,
        gradient_accumulation=gradient_accumulation,
    )


def _insert_zero3_ops(
    rank_dag: TaskDAG,
    zero_bucket_keys: set | None = None,
    gradient_accumulation: bool = True,
) -> None:
    """ZeRO-3: param gather + per-grad-node reduce-scatter lifetimes.

    * F / B_I  →  P+ → AG → compute → P-
    * B / B_W  →  (G+ →) and (P+ → AG →) compute (→ RS → G-) and (→ P-)
    """
    _remove_ar_nodes(rank_dag)

    # Types that only consume params (no weight-grad accumulation).
    param_only_types = frozenset({TaskType.FWD, TaskType.BWD_I})
    # Types that consume params AND accumulate weight gradients.
    grad_types = frozenset({TaskType.BWD, TaskType.BWD_W})

    new_nodes: list[Task] = []

    for ubid, bucket_nodes in _bucket_compute_tasks(rank_dag, zero_bucket_keys).items():
        for compute_node in bucket_nodes:
            t = compute_node.task_type
            bid = compute_node.bucket_id

            if t in param_only_types:
                ag      = _zero_task(compute_node, bid, TaskType.ALL_GATHER,        ubid=ubid)
                _add_data_edge(ag, compute_node)
                ag.custom_metadata["zero_alloc_full_params_before"] = True
                compute_node.custom_metadata["zero_free_full_params_after"] = True
                new_nodes.append(ag)

            elif t in grad_types:
                ag      = _zero_task(compute_node, bid, TaskType.ALL_GATHER,        ubid=ubid)
                _add_data_edge(ag, compute_node)
                ag.custom_metadata["zero_alloc_full_params_before"] = True
                compute_node.custom_metadata["zero_free_full_params_after"] = True
                new_nodes.append(ag)

    # Insert grad alloc/reduce/free for grad-producing ZeRO-3 nodes.
    for ubid, bucket_nodes in _bucket_compute_tasks(rank_dag, zero_bucket_keys).items():
        grad_nodes = [node for node in bucket_nodes if node.task_type in grad_types]
        if not grad_nodes:
            continue
        first_compute = min(grad_nodes, key=lambda node: (node.time_step, node.uid))
        last_compute = max(grad_nodes, key=lambda node: (node.time_step, node.uid))

        if gradient_accumulation:
            rs = _zero_task(last_compute, last_compute.bucket_id, TaskType.REDUCE_SCATTER, ubid=ubid)
            _add_data_edge(last_compute, rs)
            first_compute.custom_metadata["zero_alloc_full_grads_before"] = True
            _append_metadata_ubid(rs, "zero_free_full_grads_after_ubids", ubid)
            new_nodes.append(rs)
        else:
            for compute_node in grad_nodes:
                rs = _zero_task(compute_node, compute_node.bucket_id, TaskType.REDUCE_SCATTER, ubid=ubid)
                _add_data_edge(compute_node, rs)
                compute_node.custom_metadata["zero_alloc_full_grads_before"] = True
                _append_metadata_ubid(rs, "zero_free_full_grads_after_ubids", ubid)
                new_nodes.append(rs)

    rank_dag.nodes.extend(new_nodes)


def _fuse_consecutive_zero3_param_sequences(
    rank_dag: TaskDAG,
    *,
    zero_bucket_keys: set | None = None,
) -> None:
    """Fuse ZeRO runtime metadata across consecutive same-ubid compute runs."""

    def _ordered_compute_nodes() -> list[Task]:
        compute_types = {TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W}
        selected: list[Task] = []
        for node in rank_dag.nodes:
            if node.task_type not in compute_types or node.unique_bucket_id is None:
                continue
            if zero_bucket_keys is not None:
                if not node.batches:
                    continue
                stage_id = node.batches[0].stage_id
                if (stage_id, node.bucket_id) not in zero_bucket_keys:
                    continue
            selected.append(node)

        if not selected:
            return []

        selected_ids = {id(node) for node in selected}
        in_degree: dict[int, int] = {id(node): 0 for node in selected}
        succs: dict[int, list[Task]] = {id(node): [] for node in selected}
        for node in selected:
            for succ in node.temporal_succs:
                if id(succ) not in selected_ids:
                    continue
                succs[id(node)].append(succ)
                in_degree[id(succ)] += 1

        ready = sorted(
            [node for node in selected if in_degree[id(node)] == 0],
            key=lambda node: (node.time_step, node.uid),
        )
        ordered: list[Task] = []
        while ready:
            node = ready.pop(0)
            ordered.append(node)
            newly_ready: list[Task] = []
            for succ in sorted(succs[id(node)], key=lambda cur: (cur.time_step, cur.uid)):
                succ_id = id(succ)
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    newly_ready.append(succ)
            if newly_ready:
                ready.extend(newly_ready)
                ready.sort(key=lambda cur: (cur.time_step, cur.uid))

        if len(ordered) != len(selected):
            return sorted(selected, key=lambda node: (node.time_step, node.uid))
        return ordered

    ordered = _ordered_compute_nodes()
    runs: list[list[Task]] = []
    cur_run: list[Task] = []
    for node in ordered:
        if cur_run and node.unique_bucket_id == cur_run[-1].unique_bucket_id:
            cur_run.append(node)
            continue
        if len(cur_run) > 1:
            runs.append(cur_run)
        cur_run = [node]
    if len(cur_run) > 1:
        runs.append(cur_run)

    for run in runs:
        ubid = run[0].unique_bucket_id
        ag_nodes: list[Task] = []
        for compute_node in run:
            ag = _find_zero_neighbor(compute_node, TaskType.ALL_GATHER, preds=True, ubid=ubid)
            if ag is None:
                ag_nodes = []
                break
            ag_nodes.append(ag)
        if ag_nodes:
            for ag in ag_nodes[1:]:
                ag.custom_metadata.pop("zero_alloc_full_params_before", None)
            for compute_node in run[:-1]:
                compute_node.custom_metadata.pop("zero_free_full_params_after", None)

        grad_run = [node for node in run if node.task_type in {TaskType.BWD, TaskType.BWD_W}]
        if not grad_run:
            continue
        rs_nodes: list[Task] = []
        for compute_node in grad_run:
            rs = _find_zero_neighbor(compute_node, TaskType.REDUCE_SCATTER, preds=False, ubid=ubid)
            if rs is None:
                rs_nodes = []
                break
            rs_nodes.append(rs)
        if not rs_nodes:
            continue
        for compute_node in grad_run[1:]:
            compute_node.custom_metadata.pop("zero_alloc_full_grads_before", None)
        for rs in rs_nodes[:-1]:
            ubids = rs.custom_metadata.get("zero_free_full_grads_after_ubids", [])
            rs.custom_metadata["zero_free_full_grads_after_ubids"] = [
                cur for cur in ubids if cur != ubid
            ]
            if not rs.custom_metadata["zero_free_full_grads_after_ubids"]:
                rs.custom_metadata.pop("zero_free_full_grads_after_ubids", None)


def _zero_sequence_compute_nodes(rank_dag: TaskDAG, zero_stage: int) -> list[Task]:
    if zero_stage == 2:
        compute_types = {TaskType.BWD, TaskType.BWD_W}
    elif zero_stage == 3:
        compute_types = {TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W}
    else:
        return []
    return sorted(
        [node for node in rank_dag.nodes if node.task_type in compute_types],
        key=lambda node: (node.time_step, node.uid),
    )


def _find_zero_neighbor(node: Task, task_type: TaskType, *, preds: bool, ubid: int | None) -> Task | None:
    neighbors = node.data_preds if preds else node.data_succs
    for neighbor in neighbors:
        if neighbor.task_type != task_type:
            continue
        if ubid is not None and neighbor.unique_bucket_id != ubid:
            continue
        return neighbor
    return None


def _collect_zero_sequence_nodes(compute_node: Task, zero_stage: int) -> dict[TaskType, Task]:
    ubid = compute_node.unique_bucket_id
    seq: dict[TaskType, Task] = {compute_node.task_type: compute_node}
    if zero_stage == 2:
        alloc_g = _find_zero_neighbor(compute_node, TaskType.ALLOC_FULL_GRADS, preds=True, ubid=ubid)
        rs = _find_zero_neighbor(compute_node, TaskType.REDUCE_SCATTER, preds=False, ubid=ubid)
        free_g = _find_zero_neighbor(rs, TaskType.FREE_FULL_GRADS, preds=False, ubid=ubid) if rs else None
        if alloc_g is not None:
            seq[TaskType.ALLOC_FULL_GRADS] = alloc_g
        if rs is not None:
            seq[TaskType.REDUCE_SCATTER] = rs
        if free_g is not None:
            seq[TaskType.FREE_FULL_GRADS] = free_g
        return seq

    alloc_p = _find_zero_neighbor(compute_node, TaskType.ALL_GATHER, preds=True, ubid=ubid)
    if alloc_p is not None:
        ag = alloc_p
        alloc_p = _find_zero_neighbor(ag, TaskType.ALLOC_FULL_PARAMS, preds=True, ubid=ubid)
        if alloc_p is not None:
            seq[TaskType.ALLOC_FULL_PARAMS] = alloc_p
        seq[TaskType.ALL_GATHER] = ag
    free_p = _find_zero_neighbor(compute_node, TaskType.FREE_FULL_PARAMS, preds=False, ubid=ubid)
    if free_p is not None:
        seq[TaskType.FREE_FULL_PARAMS] = free_p
    if compute_node.task_type in {TaskType.BWD, TaskType.BWD_W}:
        alloc_g = _find_zero_neighbor(compute_node, TaskType.ALLOC_FULL_GRADS, preds=True, ubid=ubid)
        rs = _find_zero_neighbor(compute_node, TaskType.REDUCE_SCATTER, preds=False, ubid=ubid)
        free_g = _find_zero_neighbor(rs, TaskType.FREE_FULL_GRADS, preds=False, ubid=ubid) if rs else None
        if alloc_g is not None:
            seq[TaskType.ALLOC_FULL_GRADS] = alloc_g
        if rs is not None:
            seq[TaskType.REDUCE_SCATTER] = rs
        if free_g is not None:
            seq[TaskType.FREE_FULL_GRADS] = free_g
    return seq


def _reachable_downstream(node: Task) -> set[int]:
    seen: set[int] = set()
    stack = list(node.data_succs) + list(node.temporal_succs)
    while stack:
        cur = stack.pop()
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)
        stack.extend(cur.data_succs)
        stack.extend(cur.temporal_succs)
    return seen


def _wire_sync_to_upd(rank_dag: TaskDAG) -> None:
    """Add temporal edges from every AR/RS node → UPD on this rank.

    In the DP/ZeRO case the optimizer update must not start until all
    gradient-synchronisation collectives have completed.  This pass wires
    that constraint so that ``assign_time_steps`` Pass 3 can push UPD's
    time_step past the last AR/RS.
    """
    upd_nodes  = [n for n in rank_dag.nodes if n.task_type == TaskType.UPD]
    sync_nodes = [n for n in rank_dag.nodes
                  if n.task_type in (TaskType.ALL_REDUCE, TaskType.REDUCE_SCATTER)]
    for upd in upd_nodes:
        for sync in sync_nodes:
            if upd not in sync.temporal_succs:
                sync.temporal_succs.append(upd)
            if sync not in upd.temporal_preds:
                upd.temporal_preds.append(sync)


def insert_zero_ops(
    per_rank_dags: list,
    zero_stage: int,
    zero_bucket_keys: set | None = None,
    gradient_accumulation: bool = True,
) -> list:
    for rank_dag in per_rank_dags:
        if zero_stage == 0:
            _insert_zero1_ops(
                rank_dag,
                zero_bucket_keys=zero_bucket_keys,
                gradient_accumulation=gradient_accumulation,
            )
        elif zero_stage == 1:
            _insert_zero1_ops(
                rank_dag,
                zero_bucket_keys=zero_bucket_keys,
                gradient_accumulation=gradient_accumulation,
            )
        elif zero_stage == 2:
            _insert_zero2_ops(
                rank_dag,
                zero_bucket_keys=zero_bucket_keys,
                gradient_accumulation=gradient_accumulation,
            )
        elif zero_stage == 3:
            _insert_zero3_ops(
                rank_dag,
                zero_bucket_keys=zero_bucket_keys,
                gradient_accumulation=gradient_accumulation,
            )
        else:
            raise ValueError(f"Unsupported ZeRO stage: {zero_stage}")
        _wire_sync_to_upd(rank_dag)
    return per_rank_dags




# ---------------------------------------------------------------------------
# Graphviz visualisation
# ---------------------------------------------------------------------------

def _task_node_label(node: Task) -> str:
    """Short human-readable label for a Task."""
    task_type = node.task_type
    batches = node.batches

    if task_type in (TaskType.SEND, TaskType.RECV):
        op = "SEND" if task_type == TaskType.SEND else "RECV"
        return op

    type_abbrev = {
        TaskType.FWD:     "F",
        TaskType.BWD:     "B",
        TaskType.UPD:     "U",
        TaskType.BWD_I:   "BI",
        TaskType.BWD_W:   "Bw",
        TaskType.FWD_BWD: "FB",
        TaskType.ALL_REDUCE: "AR",
        TaskType.REDUCE_SCATTER: "RS",
        TaskType.ALL_GATHER: "AG",
        TaskType.ALLOC_FULL_GRADS: "G+",
        TaskType.FREE_FULL_GRADS: "G-",
        TaskType.ALLOC_FULL_PARAMS: "P+",
        TaskType.FREE_FULL_PARAMS: "P-",
        TaskType.FWD_A2A: "F_A2A",
        TaskType.BWD_A2A: "B_A2A",
    }.get(task_type, "?")

    if task_type == TaskType.UPD:
        return type_abbrev

    if task_type in (
        TaskType.ALL_REDUCE,
        TaskType.REDUCE_SCATTER,
        TaskType.ALL_GATHER,
        TaskType.ALLOC_FULL_GRADS,
        TaskType.FREE_FULL_GRADS,
        TaskType.ALLOC_FULL_PARAMS,
        TaskType.FREE_FULL_PARAMS,
    ):
        return type_abbrev

    if task_type in (TaskType.FWD_A2A, TaskType.BWD_A2A):
        return type_abbrev

    parts = " + ".join(f"S{b.stage_id} M{b.mb_idx}" for b in batches)
    return f"{type_abbrev} {parts}"


def compute_critical_path(dag: TaskDAG) -> set:
    """Return the set of UIDs of nodes on the critical path of *dag*.

    Uses both data-dependency and temporal edges.  Nodes whose ``runtime``
    field is ``None`` are treated as having zero duration (so they appear on
    the critical path only when they sit between two critical nodes).

    Returns:
        A ``set[int]`` of ``node.uid`` values for every critical node.
    """
    from collections import deque as _deque2

    # Topological sort (Kahn's algorithm over both edge kinds).
    in_deg = {n.uid: len(n.data_preds) + len(n.temporal_preds) for n in dag.nodes}
    queue = _deque2(n for n in dag.nodes if in_deg[n.uid] == 0)
    topo: list = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for s in list(node.data_succs) + list(node.temporal_succs):
            in_deg[s.uid] -= 1
            if in_deg[s.uid] == 0:
                queue.append(s)

    # Forward pass: EST[uid] = earliest start time.
    est: dict = {}
    for node in topo:
        preds = list(node.data_preds) + list(node.temporal_preds)
        est[node.uid] = max(
            (est[p.uid] + (p.runtime or 0.0) for p in preds),
            default=0.0,
        )

    makespan = max(est[n.uid] + (n.runtime or 0.0) for n in dag.nodes)

    # Backward pass: LST[uid] = latest start time that doesn't extend makespan.
    lst: dict = {}
    for node in reversed(topo):
        w = node.runtime or 0.0
        succs = list(node.data_succs) + list(node.temporal_succs)
        lst[node.uid] = (
            min(lst[s.uid] for s in succs) - w if succs else makespan - w
        )

    _EPS = 1e-6
    return {n.uid for n in dag.nodes if abs(lst[n.uid] - est[n.uid]) < _EPS}


def visualize_dag(
    dag: TaskDAG,
    output_path: str = "dag",
    fmt: str = "png",
    critical_path_nodes: "set | None" = None,
) -> None:
    """Render a :class:`TaskDAG` as a labelled image using *graphviz*.

    Nodes are coloured by pipeline rank.  Two edge styles:

    * **Dashed grey** – temporal edges (serialisation within one actor).
    * **Solid black** – data-dependency edges.

    Args:
        dag: The task DAG to render.
        output_path: Path prefix for the output file (no extension).
        fmt: Output format passed to graphviz (``"png"``, ``"svg"``, …).
        critical_path_nodes: Optional set of ``node.uid`` integers.  Nodes
            whose uid is in this set are outlined in red with a thick border.
            If ``None`` no critical-path highlighting is applied.

    The image is saved to ``{output_path}.{fmt}``.
    If *graphviz* is not installed the function logs a warning and returns.
    """
    try:
        import graphviz
    except ImportError:
        logger.warning("graphviz Python package not installed; skipping DAG visualisation")
        return

    if not dag.nodes:
        logger.warning("Empty task DAG; nothing to visualise")
        return

    # Build a mapping from pp_rank -> sorted list of stage_ids on that rank.
    pp_rank_stages: dict[int, list[int]] = {}
    for n in dag.nodes:
        if n.batches:
            rank = n.task_pp_rank
            sid = n.batches[0].stage_id
            if rank not in pp_rank_stages:
                pp_rank_stages[rank] = []
            if sid not in pp_rank_stages[rank]:
                pp_rank_stages[rank].append(sid)
    for rank in pp_rank_stages:
        pp_rank_stages[rank].sort()

    _COMPUTE_TYPES = (
        TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W, TaskType.FWD_BWD,
    )

    def _node_fill(node: Task) -> str:
        t = node.task_type
        if t in (
            TaskType.SEND, TaskType.RECV, TaskType.FWD_A2A, TaskType.BWD_A2A,
            TaskType.ALL_REDUCE, TaskType.REDUCE_SCATTER, TaskType.ALL_GATHER,
            TaskType.ALLOC_FULL_GRADS, TaskType.FREE_FULL_GRADS,
            TaskType.ALLOC_FULL_PARAMS, TaskType.FREE_FULL_PARAMS,
        ):
            return "#FFFFFF"
        if t == TaskType.FWD:
            return "#FFA500"  # orange
        if t in (TaskType.BWD, TaskType.BWD_I):
            return "#27AE60"  # green
        if t == TaskType.BWD_W:
            return "#2E86C1"  # blue
        return "#D5D8DC"

    def _node_fontcolor(node: Task) -> str:
        t = node.task_type
        if t not in _COMPUTE_TYPES or not node.batches:
            return "black"
        rank = node.task_pp_rank
        stages = pp_rank_stages.get(rank, [])
        if len(stages) < 2:
            return "black"
        sid = node.batches[0].stage_id
        # First stage -> black text, second (and beyond) stage -> white text
        return "black" if sid == stages[0] else "white"

    # Compute topological order using Kahn's algorithm (data + temporal edges).
    from collections import deque as _deque
    _in_degree: dict[str, int] = {
        n.node_id(): len(n.data_preds) + len(n.temporal_preds)
        for n in dag.nodes
    }
    _topo_order: dict[str, int] = {}
    _queue: _deque = _deque(n for n in dag.nodes if _in_degree[n.node_id()] == 0)
    _step = 0
    while _queue:
        _n = _queue.popleft()
        _topo_order[_n.node_id()] = _step
        _step += 1
        for _s in list(_n.data_succs) + list(_n.temporal_succs):
            _in_degree[_s.node_id()] -= 1
            if _in_degree[_s.node_id()] == 0:
                _queue.append(_s)

    dot = graphviz.Digraph("PiperDAG", comment="Piper Chunk DAG")
    dot.attr(rankdir="LR", splines="ortho", nodesep="0.4", ranksep="0.6", fontname="Helvetica")
    dot.attr("node", shape="box", style="filled", fontsize="9", fontname="Helvetica")
    dot.attr("edge", fontsize="8", fontname="Helvetica")

    for node in dag.nodes:
        label = f"{node.time_step}\n{_task_node_label(node)}"
        if node.runtime is not None:
            label += f"\n{node.runtime:.1f}ms"
        on_critical = critical_path_nodes is not None and node.uid in critical_path_nodes
        dot.node(
            node.node_id(),
            label=label,
            fillcolor=_node_fill(node),
            fontcolor=_node_fontcolor(node),
            tooltip=repr(node.source_chunk) if node.source_chunk is not None else node.task_type.value,
            color="red" if on_critical else "black",
            penwidth="3.0" if on_critical else "1.0",
        )

    # Group nodes by time_step so that tasks scheduled at the same step share
    # a column (graphviz rank="same").
    depth_to_nodes: dict[int, list[Task]] = defaultdict(list)
    for node in dag.nodes:
        depth_to_nodes[node.time_step].append(node)

    for col_nodes in depth_to_nodes.values():
        with dot.subgraph() as sub:
            sub.attr(rank="same")
            for node in col_nodes:
                sub.node(node.node_id())

    for node in dag.nodes:
        for ts in node.temporal_succs:
            key = (node.node_id(), ts.node_id())
            dot.edge(
                key[0], key[1],
                style="dashed", color="grey60",
                penwidth="1.0", arrowsize="0.6", constraint="true",
            )

    seen_edges: set[tuple[str, str]] = set()
    for node in dag.nodes:
        for succ in node.data_succs:
            key = (node.node_id(), succ.node_id())
            if key in seen_edges:
                continue
            seen_edges.add(key)
            cross_rank = node.pp_rank != succ.pp_rank
            dot.edge(
                key[0], key[1],
                color="black",
                penwidth="2.0", arrowsize="0.8",
                constraint="false" if cross_rank else "true",
            )

    out = dot.render(output_path, format=fmt, cleanup=False)
    logger.info(f"DAG visualisation saved to {out}")



def print_dag_order(
    dag: TaskDAG,
    label: str = "",
    rank: int = 0,
    out_dir: str = "out",
) -> None:
    """Write the execution order of nodes in a :class:`TaskDAG` to a file.

    Nodes are sorted by :func:`runtime_sort_key`, exactly as in ``run_dag``,
    so the output reflects exactly what the actor will execute.
    Useful for debugging schedule issues.

    Output is written to ``{out_dir}/dag_order_rank{rank}``.
    """
    import os

    sorted_nodes = sorted(dag.nodes, key=runtime_sort_key)

    header = f"--- DAG execution order{': ' + label if label else ''} ---"
    lines = [header]
    for step, node in enumerate(sorted_nodes):
        ttype = node.task_type.value if node.task_type is not None else "?"
        batches_str = ", ".join(
            f"s{b.stage_id} mb{b.mb_idx}" for b in node.batches
        ) if node.batches else ""
        bkt = f" bkt={node.bucket_id}" if node.bucket_id else ""
        ubid = f" ubid={node.unique_bucket_id}" if node.unique_bucket_id is not None else ""
        line = f"  {step:3d}  ts={node.time_step:3d}  rank={node.pp_rank}  {ttype:<14s}  {batches_str}{bkt}{ubid}"
        lines.append(line)
        logger.info(line)
    lines.append("-" * len(header))

    for line in lines:
        logger.debug(line)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dag_order_rank{rank}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# bucket_parameters – FX graph transformation
# ---------------------------------------------------------------------------

def _iter_node_args(node: fx.Node):
    """Yield every :class:`fx.Node` appearing in *node*'s args and kwargs."""
    def _collect(obj):
        if isinstance(obj, fx.Node):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from _collect(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _collect(v)
    yield from _collect(node.args)
    yield from _collect(node.kwargs)


def bucket_parameters(
    gm: fx.GraphModule,
    bucket_size_bytes: int = 25 * 1024 * 1024,
) -> list[fx.GraphModule]:
    """Split an FX GraphModule into sequential sub-modules based on parameter buckets.

    Adjacent parameters (in graph order) are greedily grouped into buckets of
    approximately *bucket_size_bytes* bytes.  If a parameter is used by compute
    nodes that fall into more than one bucket's node-range, it is promoted to
    its own singleton bucket so the graph can be cut cleanly.

    The graph is then cut at bucket boundaries and each segment is wrapped in a
    new :class:`fx.GraphModule`.  All original model inputs (``placeholder``
    nodes) are re-created in every sub-module.  Intermediate activation tensors
    that flow between segments become additional placeholder inputs on the
    receiving module.

    The modules are ordered so that the *i*-th module's outputs are the
    additional inputs of the *(i+1)*-th module (appended after the original
    model inputs), producing a pipeline that replicates the original graph.

    Args:
        gm: Source ``fx.GraphModule``.  ``node.meta["example_value"]`` is used
            for parameter size if available; falls back to direct attribute
            inspection.
        bucket_size_bytes: Target bucket size in bytes (default 25 MB).

    Returns:
        A list of ``fx.GraphModule`` objects in execution order.  If there is
        only one bucket (or no parameters), the list contains *gm* unchanged.
    """
    nodes = list(gm.graph.nodes)
    node_idx: dict[fx.Node, int] = {nd: i for i, nd in enumerate(nodes)}

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _is_param(nd: fx.Node) -> bool:
        if nd.op != "get_attr":
            return False
        obj = gm
        for part in nd.target.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return False
        return isinstance(obj, torch.nn.Parameter)

    def _attr_size(nd: fx.Node) -> int:
        ev = nd.meta.get("example_value")
        if ev is not None and hasattr(ev, "numel"):
            return int(ev.numel() * ev.element_size())
        obj = gm
        for part in nd.target.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return 0
        return int(obj.numel() * obj.element_size()) if isinstance(obj, torch.Tensor) else 0

    # ------------------------------------------------------------------ #
    # Step 1 – Collect parameter nodes                                   #
    # ------------------------------------------------------------------ #
    param_nodes: list[fx.Node] = [nd for nd in nodes if _is_param(nd)]
    if not param_nodes:
        return [gm]

    # ------------------------------------------------------------------ #
    # Step 2 – Per-parameter compute-node use range                      #
    # ------------------------------------------------------------------ #
    compute_set: frozenset[int] = frozenset(
        i for i, nd in enumerate(nodes)
        if nd.op not in ("placeholder", "get_attr", "output")
    )

    def _use_range(pnd: fx.Node) -> tuple[int, int]:
        idxs = [node_idx[u] for u in pnd.users if node_idx[u] in compute_set]
        if not idxs:
            return (node_idx[pnd], node_idx[pnd])
        return (min(idxs), max(idxs))

    param_ranges: dict[fx.Node, tuple[int, int]] = {pn: _use_range(pn) for pn in param_nodes}

    # ------------------------------------------------------------------ #
    # Step 3 – Greedy bucket assignment                                  #
    # ------------------------------------------------------------------ #
    bucket_id: dict[fx.Node, int] = {}
    cur_bucket = 0
    cur_size = 0
    for pn in param_nodes:
        sz = _attr_size(pn)
        if cur_size + sz > bucket_size_bytes and cur_size > 0:
            cur_bucket += 1
            cur_size = 0
        bucket_id[pn] = cur_bucket
        cur_size += sz
    n_initial = cur_bucket + 1

    # ------------------------------------------------------------------ #
    # Step 4 – Singleton promotion for cross-bucket parameters           #
    # ------------------------------------------------------------------ #
    def _initial_bucket_range(bid: int) -> tuple[int, int]:
        members = [pn for pn in param_nodes if bucket_id[pn] == bid]
        if not members:
            return (0, 0)
        return (
            min(param_ranges[pn][0] for pn in members),
            max(param_ranges[pn][1] for pn in members),
        )

    init_ranges = [_initial_bucket_range(b) for b in range(n_initial)]

    def _initial_bucket_at(idx: int) -> int:
        for bid, (lo, hi) in enumerate(init_ranges):
            if lo <= idx <= hi:
                return bid
        return 0

    singleton_next = n_initial
    for pn in param_nodes:
        first, last = param_ranges[pn]
        if _initial_bucket_at(first) != _initial_bucket_at(last):
            bucket_id[pn] = singleton_next
            singleton_next += 1

    # ------------------------------------------------------------------ #
    # Step 5 – Build final bucket list, sort, merge overlapping ranges  #
    # ------------------------------------------------------------------ #
    bucket_members: dict[int, list[fx.Node]] = defaultdict(list)
    for pn in param_nodes:
        bucket_members[bucket_id[pn]].append(pn)

    def _bucket_use_range(members: list[fx.Node]) -> tuple[int, int]:
        return (
            min(param_ranges[pn][0] for pn in members),
            max(param_ranges[pn][1] for pn in members),
        )

    bucket_list: list[tuple[int, int, list[fx.Node]]] = []
    for bid, members in bucket_members.items():
        f, l = _bucket_use_range(members)
        bucket_list.append((f, l, members))
    bucket_list.sort(key=lambda t: t[0])

    merged: list[tuple[int, int, list[fx.Node]]] = []
    for f, l, members in bucket_list:
        if merged and f <= merged[-1][1]:
            pf, pl, pm = merged[-1]
            merged[-1] = (pf, max(pl, l), pm + members)
        else:
            merged.append((f, l, members))

    n_segs = len(merged)
    if n_segs == 1:
        return [gm]

    # ------------------------------------------------------------------ #
    # Step 6 – Cut points (max last-use per bucket except the final one) #
    # ------------------------------------------------------------------ #
    # cut_after[i] is the node index after which we cut between seg i and i+1.
    cut_after: list[int] = [merged[i][1] for i in range(n_segs - 1)]

    # ------------------------------------------------------------------ #
    # Step 7 – Assign each node to a segment                            #
    # ------------------------------------------------------------------ #
    def _seg_of(idx: int) -> int:
        return sum(1 for c in cut_after if c < idx)

    node_seg: dict[fx.Node, int] = {}
    for nd in nodes:
        if nd.op == "placeholder":
            node_seg[nd] = 0
        elif nd.op == "output":
            node_seg[nd] = n_segs - 1
        elif nd.op == "get_attr":
            user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
            node_seg[nd] = _seg_of(min(user_idxs)) if user_idxs else 0
        else:
            node_seg[nd] = _seg_of(node_idx[nd])

    # ------------------------------------------------------------------ #
    # Step 8 – Compute cross-segment activation sets                    #
    # ------------------------------------------------------------------ #
    # seg_cross_in[seg] = ordered list of compute nodes produced in segment
    # < seg that are consumed by any segment >= seg.  Placeholders and
    # get_attr nodes are excluded: placeholder (model input) tensors are
    # assumed to be available in every segment directly; parameters never
    # cross segments after singleton promotion and can be re-issued.
    node_max_user_seg: dict[fx.Node, int] = {}
    for nd in nodes:
        if nd.op == "output":
            continue
        node_max_user_seg[nd] = (
            max(node_seg[u] for u in nd.users) if nd.users else node_seg[nd]
        )

    seg_cross_in: list[list[fx.Node]] = [[] for _ in range(n_segs)]
    for nd in nodes:
        if nd.op in ("placeholder", "get_attr", "output"):
            continue
        s = node_seg[nd]
        mu = node_max_user_seg[nd]
        for seg in range(s + 1, mu + 1):
            seg_cross_in[seg].append(nd)
    # Lists are already in graph order since we iterate nodes in graph order.

    # For each segment, determine which original placeholder nodes it uses
    # directly (so we only add the placeholders each segment actually needs).
    seg_placeholders: list[list[fx.Node]] = [[] for _ in range(n_segs)]
    for nd in nodes:
        if nd.op != "placeholder":
            continue
        for seg in range(n_segs):
            if any(node_seg[u] == seg for u in nd.users):
                seg_placeholders[seg].append(nd)

    # ------------------------------------------------------------------ #
    # Step 9 – Build sub-graphs                                          #
    # ------------------------------------------------------------------ #
    sub_graphs: list[fx.GraphModule] = []

    for seg in range(n_segs):
        sub_g = fx.Graph()
        remap: dict[fx.Node, fx.Node] = {}

        # Original model inputs used directly by this segment.
        for nd in seg_placeholders[seg]:
            new_ph = sub_g.placeholder(nd.name)
            new_ph.type = nd.type
            remap[nd] = new_ph

        # Cross-segment activation inputs from earlier segments.
        for orig in seg_cross_in[seg]:
            new_ph = sub_g.placeholder(f"_xseg_{orig.name}")
            new_ph.type = orig.type
            remap[orig] = new_ph

        # get_attr nodes whose attributes belong to this segment.
        for nd in nodes:
            if nd.op == "get_attr" and node_seg[nd] == seg:
                new_ga = sub_g.get_attr(nd.target)
                new_ga.type = nd.type
                remap[nd] = new_ga

        # Compute nodes for this segment (graph order preserved).
        for nd in nodes:
            if nd.op in ("placeholder", "get_attr", "output"):
                continue
            if node_seg[nd] != seg:
                continue
            new_nd = sub_g.node_copy(nd, arg_transform=lambda x, r=remap: r[x])
            remap[nd] = new_nd

        # Output node.
        if seg == n_segs - 1:
            orig_out = next(nd for nd in nodes if nd.op == "output")
            # orig_out.args[0] is the actual return value (node, tuple of nodes, etc.)
            sub_g.output(fx.map_arg(orig_out.args[0], lambda x: remap[x]))
        else:
            out_nodes = [remap[orig] for orig in seg_cross_in[seg + 1]]
            sub_g.output(tuple(out_nodes) if len(out_nodes) != 1 else out_nodes[0])

        sub_g.lint()
        sub_graphs.append(fx.GraphModule(gm, sub_g))

    return sub_graphs


# ---------------------------------------------------------------------------
# bucket_stage – stage-level parameter bucketing (placeholder params)
# ---------------------------------------------------------------------------

def bucket_stage(
    stage_gm: fx.GraphModule,
    graphargs: list,
    input_idxs: list[int],
    param_idxs: list[int],
    bucket_size_bytes: int = 25 * 1024 * 1024,
) -> list[tuple[fx.GraphModule, list[int], list[int], list]]:
    """Split a stage GraphModule into per-parameter-bucket sub-modules.

    Unlike :func:`bucket_parameters`, which handles ``get_attr`` parameter
    nodes, this function handles stages produced by :func:`_split_gm_by_stages`
    where trainable parameters are passed as **placeholder inputs** identified
    by *param_idxs*.

    Args:
        stage_gm: The stage ``fx.GraphModule`` to split.
        graphargs: Flat arg list for ``stage_gm.forward``; entries at
            *param_idxs* are meta tensors, entries at *input_idxs* are None.
        input_idxs: Positions of activation-input placeholders.
        param_idxs: Positions of parameter placeholders.
        bucket_size_bytes: Target bucket size (default 25 MB).

    Returns:
        A list of ``(bucket_gm, bucket_input_idxs, bucket_param_idxs,
        bucket_graphargs)`` tuples in execution order.  Returns a
        single-element list containing the original stage when no split is
        needed.
    """
    nodes = list(stage_gm.graph.nodes)
    node_idx: dict[fx.Node, int] = {nd: i for i, nd in enumerate(nodes)}
    ph_nodes = [nd for nd in nodes if nd.op == "placeholder"]

    param_ph_set = {ph_nodes[i] for i in param_idxs if i < len(ph_nodes)}
    param_ph_list = [ph_nodes[i] for i in param_idxs if i < len(ph_nodes)]
    input_ph_set  = {ph_nodes[i] for i in input_idxs  if i < len(ph_nodes)}

    if not param_ph_list:
        return [(stage_gm, list(input_idxs), list(param_idxs), list(graphargs))]

    compute_set: frozenset[int] = frozenset(
        i for i, nd in enumerate(nodes)
        if nd.op not in ("placeholder", "get_attr", "output")
    )

    def _use_range(pnd: fx.Node) -> tuple[int, int]:
        idxs = [node_idx[u] for u in pnd.users if node_idx[u] in compute_set]
        return (min(idxs), max(idxs)) if idxs else (node_idx[pnd], node_idx[pnd])

    def _size(pnd: fx.Node) -> int:
        i = ph_nodes.index(pnd)
        if i < len(graphargs) and graphargs[i] is not None and hasattr(graphargs[i], "numel"):
            return int(graphargs[i].numel() * graphargs[i].element_size())
        ev = pnd.meta.get("example_value")
        if ev is not None and hasattr(ev, "numel"):
            return int(ev.numel() * ev.element_size())
        return 0

    param_ranges = {pn: _use_range(pn) for pn in param_ph_list}

    # Greedy bucket assignment
    bucket_id: dict[fx.Node, int] = {}
    cur_b, cur_sz = 0, 0
    for pn in param_ph_list:
        sz = _size(pn)
        if cur_sz + sz > bucket_size_bytes and cur_sz > 0:
            cur_b += 1
            cur_sz = 0
        bucket_id[pn] = cur_b
        cur_sz += sz
    n_init = cur_b + 1

    # Singleton promotion
    def _init_range(b: int) -> tuple[int, int]:
        ms = [pn for pn in param_ph_list if bucket_id[pn] == b]
        if not ms:
            return (0, 0)
        return (min(param_ranges[pn][0] for pn in ms), max(param_ranges[pn][1] for pn in ms))

    init_ranges = [_init_range(b) for b in range(n_init)]

    def _bucket_at(idx: int) -> int:
        for b, (lo, hi) in enumerate(init_ranges):
            if lo <= idx <= hi:
                return b
        return 0

    sn = n_init
    for pn in param_ph_list:
        f, l = param_ranges[pn]
        if _bucket_at(f) != _bucket_at(l):
            bucket_id[pn] = sn
            sn += 1

    # Build bucket list, sort, merge overlapping ranges
    bm: dict[int, list[fx.Node]] = defaultdict(list)
    for pn in param_ph_list:
        bm[bucket_id[pn]].append(pn)

    blist: list[tuple[int, int, list[fx.Node]]] = []
    for members in bm.values():
        f = min(param_ranges[pn][0] for pn in members)
        l = max(param_ranges[pn][1] for pn in members)
        blist.append((f, l, members))
    blist.sort(key=lambda t: t[0])

    merged: list[tuple[int, int, list[fx.Node]]] = []
    for f, l, ms in blist:
        if merged and f <= merged[-1][1]:
            pf, pl, pm = merged[-1]
            merged[-1] = (pf, max(pl, l), pm + ms)
        else:
            merged.append((f, l, ms))

    alias_methods = {
        "view", "_unsafe_view", "reshape", "transpose", "permute", "t",
        "movedim", "moveaxis", "swapdims", "swapaxes",
        "select", "narrow", "slice", "split", "chunk", "unbind",
        "unsqueeze", "squeeze", "flatten", "expand", "diagonal",
        "detach", "alias", "as_strided",
    }
    multi_output_alias_methods = {"split", "chunk", "unbind"}

    alias_function_names = {
        "view", "_unsafe_view", "reshape", "transpose", "permute", "t",
        "movedim", "moveaxis", "swapdims", "swapaxes",
        "select", "narrow", "slice", "split", "chunk", "unbind",
        "unsqueeze", "squeeze", "flatten", "expand", "diagonal",
        "detach", "alias", "as_strided",
    }
    multi_output_alias_function_names = {"split", "chunk", "unbind"}
    hard_forbid_cross_bucket_prefixes = ("bfloat16_",)

    def _target_name(target: object) -> str | None:
        if isinstance(target, str):
            return target
        return getattr(target, "__name__", None)

    def _alias_passthrough_sources(nd: fx.Node) -> set[fx.Node]:
        if nd.op == "call_method":
            if nd.target in alias_methods:
                base = next(iter(nd.all_input_nodes), None)
                return set(alias_sources.get(base, set())) if base is not None else set()
            return set()

        if nd.op != "call_function":
            return set()

        if nd.target == operator.getitem:
            base = nd.args[0] if nd.args else None
            if isinstance(base, fx.Node):
                return set(alias_sources.get(base, set()))
            return set()

        name = _target_name(nd.target)
        if name in alias_function_names:
            base = next(iter(nd.all_input_nodes), None)
            return set(alias_sources.get(base, set())) if base is not None else set()
        return set()

    alias_sources: dict[fx.Node, set[fx.Node]] = {}
    for nd in nodes:
        if nd.op == "placeholder":
            alias_sources[nd] = {nd} if nd in param_ph_set else set()
            continue
        if nd.op in ("get_attr", "output"):
            alias_sources[nd] = set()
            continue
        alias_sources[nd] = _alias_passthrough_sources(nd)

    def _compute_seg_metadata(
        seg_ranges: list[tuple[int, int, list[fx.Node]]]
    ) -> tuple[dict[fx.Node, int], dict[fx.Node, int], list[list[fx.Node]]]:
        """Return (node_seg, node_max_user_seg, seg_cross_in) for segment ranges."""
        cut_after = [seg_ranges[i][1] for i in range(len(seg_ranges) - 1)]

        def _seg_of(idx: int) -> int:
            return sum(1 for c in cut_after if c < idx)

        node_seg: dict[fx.Node, int] = {}
        for nd in nodes:
            if nd.op == "output":
                node_seg[nd] = len(seg_ranges) - 1
            elif nd.op == "placeholder":
                if nd in param_ph_set:
                    user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
                    node_seg[nd] = _seg_of(min(user_idxs)) if user_idxs else 0
                else:
                    node_seg[nd] = 0
            elif nd.op == "get_attr":
                user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
                node_seg[nd] = _seg_of(min(user_idxs)) if user_idxs else 0
            else:
                node_seg[nd] = _seg_of(node_idx[nd])

        node_max_user_seg: dict[fx.Node, int] = {}
        for nd in nodes:
            if nd.op == "output":
                continue
            node_max_user_seg[nd] = (
                max(node_seg[u] for u in nd.users) if nd.users else node_seg[nd]
            )

        seg_cross_in: list[list[fx.Node]] = [[] for _ in range(len(seg_ranges))]
        for nd in nodes:
            if nd.op in ("get_attr", "output"):
                continue
            if nd.op == "placeholder" and nd in param_ph_set:
                continue
            s = node_seg[nd]
            mu = node_max_user_seg[nd]
            for seg in range(s + 1, mu + 1):
                seg_cross_in[seg].append(nd)

        return node_seg, node_max_user_seg, seg_cross_in

    n_segs = len(merged)
    if n_segs == 1:
        return [(stage_gm, list(input_idxs), list(param_idxs), list(graphargs))]

    while True:
        node_seg, node_max_user_seg, seg_cross_in = _compute_seg_metadata(merged)
        merged_boundary = False
        for seg in range(1, len(merged)):
            alias_crossers = [
                nd for nd in seg_cross_in[seg]
                if alias_sources.get(nd) or any(
                    nd.name.startswith(prefix) for prefix in hard_forbid_cross_bucket_prefixes
                )
            ]
            if not alias_crossers:
                continue
            pf, _pl, pms = merged[seg - 1]
            _cf, cl, cms = merged[seg]
            merged[seg - 1] = (pf, cl, pms + cms)
            del merged[seg]
            merged_boundary = True
            break
        if not merged_boundary:
            break

    n_segs = len(merged)
    if n_segs == 1:
        return [(stage_gm, list(input_idxs), list(param_idxs), list(graphargs))]

    # Build sub-graphs
    results: list[tuple[fx.GraphModule, list[int], list[int], list]] = []

    for seg in range(n_segs):
        sub_g = fx.Graph()
        remap: dict[fx.Node, fx.Node] = {}
        new_input_idxs: list[int] = []
        new_param_idxs: list[int] = []
        new_graphargs: list = []
        pos = 0

        def _meta_tensor_for(nd: fx.Node):
            """Return a meta tensor matching nd's shape/dtype, or None if unavailable."""
            ev = nd.meta.get("example_value")
            if ev is None:
                ev = nd.meta.get("val")
            if ev is not None and hasattr(ev, "shape"):
                return torch.empty(ev.shape, dtype=ev.dtype, device="meta", requires_grad=ev.requires_grad)
            return None

        def _add_ph(nd: fx.Node, name: str, is_input: bool) -> None:
            nonlocal pos
            new_ph = sub_g.placeholder(name)
            new_ph.type = nd.type
            remap[nd] = new_ph
            if is_input:
                new_input_idxs.append(pos)
                meta = _meta_tensor_for(nd)
                if meta is None:
                    # Fallback: use the corresponding entry from the incoming graphargs.
                    # This handles sub-graphs produced by split_by_a2a whose placeholder
                    # nodes are freshly created and lack example_value metadata.
                    i_orig = ph_nodes.index(nd) if nd in ph_nodes else -1
                    if 0 <= i_orig < len(graphargs):
                        meta = graphargs[i_orig]
                new_graphargs.append(meta)
            else:
                new_param_idxs.append(pos)
                i_orig = ph_nodes.index(nd) if nd in ph_nodes else -1
                new_graphargs.append(graphargs[i_orig] if 0 <= i_orig < len(graphargs) else None)
            pos += 1

        if seg == 0:
            # Non-param placeholders (activation inputs + other non-param inputs).
            for nd in nodes:
                if nd.op == "placeholder" and nd not in param_ph_set:
                    _add_ph(nd, nd.name, is_input=(nd in input_ph_set))
        else:
            # Cross-segment activation inputs from the previous segment.
            for orig in seg_cross_in[seg]:
                _add_ph(orig, f"_xseg_{orig.name}", is_input=True)

        # Parameter placeholders belonging to this segment.
        for nd in nodes:
            if nd.op == "placeholder" and nd in param_ph_set and node_seg[nd] == seg:
                _add_ph(nd, nd.name, is_input=False)

        # get_attr nodes for this segment.
        for nd in nodes:
            if nd.op == "get_attr" and node_seg[nd] == seg:
                new_ga = sub_g.get_attr(nd.target)
                new_ga.type = nd.type
                remap[nd] = new_ga

        # Compute nodes for this segment (graph order preserved).
        for nd in nodes:
            if nd.op in ("placeholder", "get_attr", "output"):
                continue
            if node_seg[nd] != seg:
                continue
            remap[nd] = sub_g.node_copy(nd, arg_transform=lambda x, r=remap: r[x])

        # Output node.
        if seg == n_segs - 1:
            orig_out = next(nd for nd in nodes if nd.op == "output")
            sub_g.output(fx.map_arg(orig_out.args[0], lambda x: remap[x]))
        else:
            out_nodes = [remap[orig] for orig in seg_cross_in[seg + 1]]
            sub_g.output(tuple(out_nodes) if len(out_nodes) != 1 else out_nodes[0])

        sub_g.lint()
        results.append((fx.GraphModule(stage_gm, sub_g), new_input_idxs, new_param_idxs, new_graphargs))

    return results


def apply_activation_checkpointing(
    stage_gm: fx.GraphModule,
    graphargs: list,
    input_idxs: list[int],
    param_idxs: list[int],
    num_checkpoints: int,
) -> list[fx.GraphModule]:
    """Split a bucket into sequential activation-checkpoint regions.

    The regions are chosen to be roughly balanced by parameter bytes. Live
    intermediate tensors can flow between adjacent regions as needed, so a
    later region may receive and forward multiple carried tensors.
    """
    num_checkpoints = max(1, int(num_checkpoints))
    if num_checkpoints <= 1:
        return [stage_gm]
    gm_name = getattr(stage_gm, "_get_name", lambda: type(stage_gm).__name__)()

    nodes = list(stage_gm.graph.nodes)
    node_idx: dict[fx.Node, int] = {nd: i for i, nd in enumerate(nodes)}
    ph_nodes = [nd for nd in nodes if nd.op == "placeholder"]
    param_ph_set = {ph_nodes[i] for i in param_idxs if i < len(ph_nodes)}
    param_ph_list = [ph_nodes[i] for i in param_idxs if i < len(ph_nodes)]
    input_ph_set = {ph_nodes[i] for i in input_idxs if i < len(ph_nodes)}
    compute_nodes = [nd for nd in nodes if nd.op not in ("placeholder", "get_attr", "output")]

    if not param_ph_list or len(compute_nodes) < 2:
        logger.warning(
            f"[ac_split_skip] gm={gm_name} requested_subgraphs={num_checkpoints} "
            f"params={len(param_ph_list)} compute_nodes={len(compute_nodes)} "
            "reason=insufficient_params_or_compute"
        )
        return [stage_gm]

    compute_set: frozenset[int] = frozenset(node_idx[nd] for nd in compute_nodes)

    def _tensor_meta(nd: fx.Node):
        ev = nd.meta.get("example_value")
        if ev is None:
            ev = nd.meta.get("val")
        if isinstance(ev, torch.Tensor):
            return ev
        return None

    def _param_size(pnd: fx.Node) -> int:
        i = ph_nodes.index(pnd)
        if i < len(graphargs) and graphargs[i] is not None and hasattr(graphargs[i], "numel"):
            return int(graphargs[i].numel() * graphargs[i].element_size())
        ev = _tensor_meta(pnd)
        if ev is not None:
            return int(ev.numel() * ev.element_size())
        return 0

    def _use_indices(pnd: fx.Node) -> list[int]:
        return [node_idx[u] for u in pnd.users if node_idx[u] in compute_set]

    param_first_use: dict[fx.Node, int] = {}
    for pn in param_ph_list:
        uses = _use_indices(pn)
        param_first_use[pn] = min(uses) if uses else node_idx[pn]

    param_sizes = {pn: _param_size(pn) for pn in param_ph_list}
    total_param_bytes = sum(param_sizes.values())
    if total_param_bytes <= 0:
        logger.warning(
            f"[ac_split_skip] gm={gm_name} requested_subgraphs={num_checkpoints} "
            f"params={len(param_ph_list)} compute_nodes={len(compute_nodes)} "
            "reason=nonpositive_param_bytes"
        )
        return [stage_gm]

    sorted_params = sorted(param_ph_list, key=lambda pn: param_first_use[pn])
    running_bytes = 0
    next_param = 0

    candidate_cuts: list[tuple[int, int]] = []
    for cut_node in compute_nodes[:-1]:
        cut_idx = node_idx[cut_node]
        while next_param < len(sorted_params) and param_first_use[sorted_params[next_param]] <= cut_idx:
            running_bytes += param_sizes[sorted_params[next_param]]
            next_param += 1
        candidate_cuts.append((cut_idx, running_bytes))

    if len(candidate_cuts) < num_checkpoints - 1:
        logger.warning(
            f"[ac_split_reject] gm={gm_name} requested_subgraphs={num_checkpoints} "
            f"params={len(param_ph_list)} compute_nodes={len(compute_nodes)} "
            f"candidate_cuts={len(candidate_cuts)} required_cuts={num_checkpoints - 1} "
            "reason=insufficient_compute_cuts"
        )
        return [stage_gm]

    selected_cuts: list[int] = []
    start = 0
    for split_idx in range(1, num_checkpoints):
        target = total_param_bytes * split_idx / num_checkpoints
        remaining = (num_checkpoints - 1) - split_idx
        eligible = candidate_cuts[start: len(candidate_cuts) - remaining]
        if not eligible:
            logger.warning(
                f"[ac_split_reject] gm={gm_name} requested_subgraphs={num_checkpoints} "
                f"candidate_cuts={len(candidate_cuts)} chosen_cuts={len(selected_cuts)} "
                f"failed_pick={split_idx}/{num_checkpoints - 1} reason=no_eligible_cut"
            )
            return [stage_gm]
        best_rel = min(
            range(len(eligible)),
            key=lambda rel: abs(eligible[rel][1] - target),
        )
        chosen = eligible[best_rel]
        selected_cuts.append(chosen[0])
        start += best_rel + 1

    n_segs = num_checkpoints

    def _seg_of(idx: int) -> int:
        return sum(1 for cut in selected_cuts if cut < idx)

    node_seg: dict[fx.Node, int] = {}
    for nd in nodes:
        if nd.op == "output":
            node_seg[nd] = n_segs - 1
        elif nd.op == "placeholder":
            if nd in param_ph_set:
                uses = _use_indices(nd)
                node_seg[nd] = _seg_of(min(uses)) if uses else 0
            else:
                user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
                node_seg[nd] = _seg_of(min(user_idxs)) if user_idxs else 0
        elif nd.op == "get_attr":
            user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
            node_seg[nd] = _seg_of(min(user_idxs)) if user_idxs else 0
        else:
            node_seg[nd] = _seg_of(node_idx[nd])

    def _consumer_seg(user: fx.Node) -> int:
        if user.op == "output":
            return n_segs - 1
        return node_seg[user]

    seg_live_inputs: list[list[fx.Node]] = [[] for _ in range(n_segs)]
    for seg in range(1, n_segs):
        live_nodes: list[fx.Node] = []
        seen: set[fx.Node] = set()
        for nd in compute_nodes:
            if node_seg[nd] >= seg:
                continue
            if any(_consumer_seg(user) >= seg for user in nd.users):
                if nd not in seen:
                    seen.add(nd)
                    live_nodes.append(nd)
        seg_live_inputs[seg] = live_nodes

    seg_placeholder_inputs: list[list[fx.Node]] = [[] for _ in range(n_segs)]
    for nd in nodes:
        if nd.op != "placeholder" or nd in param_ph_set:
            continue
        for seg in range(n_segs):
            if any(node_seg[u] == seg for u in nd.users):
                seg_placeholder_inputs[seg].append(nd)

    results: list[fx.GraphModule] = []
    for seg in range(n_segs):
        sub_g = fx.Graph()
        remap: dict[fx.Node, fx.Node] = {}

        if seg > 0:
            for live_idx, boundary_nd in enumerate(seg_live_inputs[seg]):
                new_ph = sub_g.placeholder(f"_ac_in_{live_idx}")
                new_ph.type = boundary_nd.type
                remap[boundary_nd] = new_ph

        for nd in seg_placeholder_inputs[seg]:
            new_ph = sub_g.placeholder(nd.name)
            new_ph.type = nd.type
            remap[nd] = new_ph

        for nd in nodes:
            if nd.op == "placeholder" and nd in param_ph_set and node_seg[nd] == seg:
                new_ph = sub_g.placeholder(nd.name)
                new_ph.type = nd.type
                remap[nd] = new_ph

        for nd in nodes:
            if nd.op == "get_attr" and node_seg[nd] == seg:
                new_ga = sub_g.get_attr(nd.target)
                new_ga.type = nd.type
                remap[nd] = new_ga

        try:
            for nd in nodes:
                if nd.op in ("placeholder", "get_attr", "output"):
                    continue
                if node_seg[nd] != seg:
                    continue
                remap[nd] = sub_g.node_copy(nd, arg_transform=lambda x, r=remap: r[x])
        except KeyError as exc:
            logger.warning(
                f"[ac_split_reject] gm={gm_name} requested_subgraphs={num_checkpoints} "
                f"segment={seg} missing_node={exc} reason=non_local_tensor_during_lowering"
            )
            return [stage_gm]

        if seg == n_segs - 1:
            orig_out = next(nd for nd in nodes if nd.op == "output")
            sub_g.output(fx.map_arg(orig_out.args[0], lambda x: remap[x]))
        else:
            live_outputs = [remap[nd] for nd in seg_live_inputs[seg + 1]]
            if len(live_outputs) == 1:
                sub_g.output(live_outputs[0])
            else:
                sub_g.output(tuple(live_outputs))

        sub_g.lint()
        results.append(fx.GraphModule(stage_gm, sub_g))

    logger.debug(
        f"[ac_split_accept] gm={gm_name} requested_subgraphs={num_checkpoints} "
        f"actual_subgraphs={len(results)} candidate_cuts={len(candidate_cuts)} "
        f"selected_cuts={selected_cuts} "
        f"live_counts={[len(seg_live_inputs[i]) for i in range(1, n_segs)]}"
    )
    return results


# ---------------------------------------------------------------------------
# split_by_a2a: split an FX GraphModule at all-to-all annotation boundaries
# ---------------------------------------------------------------------------

def split_by_a2a(
    stage_gm: fx.GraphModule,
    graphargs: list,
    input_idxs: list[int],
    param_idxs: list[int],
) -> tuple[list[tuple], list[dict]]:
    """Split a stage GraphModule at all-to-all annotation boundaries.

    Each all-to-all annotation (``node.meta['custom']['collective'] ==
    'all_to_all_single'``) marks a tensor that will be communicated via an
    expert-parallel all-to-all operation.  The annotations must come in pairs;
    the graph is cut at each boundary so that the A2A operation itself is
    handled at the DAG level (FWD_A2A / BWD_A2A task types) rather than being
    embedded in the FX graph.

    Args:
        stage_gm: The stage ``fx.GraphModule`` to split.
        graphargs: Flat arg list for ``stage_gm.forward``.
        input_idxs: Positions of activation-input placeholders.
        param_idxs: Positions of parameter placeholders.

    Returns:
        ``(segments, boundary_infos)`` where:

        * *segments* is a list of ``(gm, input_idxs, param_idxs, graphargs)``
          tuples in execution order.  If no A2A annotations are found the
          original stage is returned as a single-element list with an empty
          *boundary_infos*.

        * *boundary_infos* is a list of dicts, one per A2A boundary (length ==
          number of annotation blocks).  Each dict has keys:

          - ``"tensor_idx"`` (int): index in the output tuple of the preceding
            segment of the tensor to communicate.
          - ``"reshape_input"`` (tuple | None): reshape the tensor to this
            shape before the A2A send.
          - ``"reshape_output"`` (tuple | None): reshape the A2A result to
            this shape before feeding the next segment.

    Raises:
        ValueError: If the number of A2A annotation blocks is not even
            (annotations must come in pairs).
    """
    nodes = list(stage_gm.graph.nodes)
    node_idx: dict[fx.Node, int] = {nd: i for i, nd in enumerate(nodes)}
    ph_nodes = [nd for nd in nodes if nd.op == "placeholder"]
    param_ph_set = {ph_nodes[i] for i in param_idxs if i < len(ph_nodes)}
    input_ph_set = {ph_nodes[i] for i in input_idxs if i < len(ph_nodes)}

    # ------------------------------------------------------------------
    # 1. Find all A2A annotation blocks (same logic as _insert_a2a_ops)
    # ------------------------------------------------------------------
    annotated_nodes = []
    for idx, node in enumerate(nodes):
        # Skip non-compute nodes: get_attr (parameters) and placeholders may
        # inherit annotations from the enclosing `with annotate(...)` block
        # but are not A2A compute boundaries.
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        custom = node.meta.get("custom")
        if isinstance(custom, dict) and custom.get("collective") == "all_to_all_single":
            reshape = custom.get("reshape")
            annotation_key = tuple(sorted(custom.items()))
            annotated_nodes.append((idx, node, annotation_key, reshape))

    if not annotated_nodes:
        return [(stage_gm, list(input_idxs), list(param_idxs), list(graphargs))], []

    logger.log(VERBOSE,
        f"split_by_a2a: found {len(annotated_nodes)} annotated nodes:\n"
        + "\n".join(
            f"  [{i}] pos={idx} name={nd.name!r} "
            f"key={'...' + str(ak)[-60:]}"
            for i, (idx, nd, ak, _) in enumerate(annotated_nodes)
        )
    )

    # Group CONSECUTIVE annotated nodes with the same annotation_key into blocks,
    # but only if they are graph-adjacent (no other compute nodes between them).
    # Adjacency is required to handle models where all A2A annotations share the
    # same key (e.g. dispatch and gather both tagged "all_to_all_single"): without
    # the adjacency check they would all merge into one block.
    # The grouping still handles the case where a single A2A boundary spans
    # multiple truly-adjacent nodes (e.g. reshape + contiguous inside one
    # `with annotate(...)` block).
    annotated_idx_set = {idx for idx, _, _, _ in annotated_nodes}
    blocks: list = []
    for idx, node, annotation_key, reshape in annotated_nodes:
        if blocks and blocks[-1]["annotation_key"] == annotation_key:
            prev_last_idx = blocks[-1]["last_idx"]
            # Only merge if no non-annotated compute nodes exist in the gap.
            gap_has_compute = any(
                nodes[j].op not in ("placeholder", "get_attr", "output")
                and j not in annotated_idx_set
                for j in range(prev_last_idx + 1, idx)
            )
            if not gap_has_compute:
                blocks[-1]["nodes"].append((idx, node))
                blocks[-1]["last_idx"] = idx
                continue
        blocks.append({
            "nodes": [(idx, node)],
            "last_idx": idx,
            "reshape": reshape,
            "annotation_key": annotation_key,
        })
    n_boundaries = len(blocks)

    logger.log(VERBOSE,
        f"split_by_a2a: grouped into {n_boundaries} consecutive blocks:\n"
        + "\n".join(
            f"  block[{i}]: {len(b['nodes'])} nodes "
            f"[{', '.join(nd.name for _, nd in b['nodes'])}] "
            f"reshape={b['reshape']}"
            for i, b in enumerate(blocks)
        )
    )

    if n_boundaries % 2 != 0:
        raise ValueError(
            f"split_by_a2a: expected an even number of A2A annotation blocks "
            f"(got {n_boundaries}).  Annotations must come in pairs.\n"
            f"Blocks: " + ", ".join(
                f"[{' '.join(nd.name for _, nd in b['nodes'])}]" for b in blocks
            )
        )

    # ------------------------------------------------------------------
    # 2. For each block, find the "boundary node" — the block node that
    #    has users outside the block.  This becomes the A2A output tensor.
    # ------------------------------------------------------------------
    boundary_nodes: list[tuple[fx.Node, tuple | None, tuple | None]] = []
    for block in blocks:
        block_node_set = {nd for _, nd in block["nodes"]}
        reshape_hint = block["reshape"]
        reshape_input = None
        reshape_output = None
        if isinstance(reshape_hint, (list, tuple)) and len(reshape_hint) == 2:
            kind, shape = reshape_hint
            if kind == "input":
                reshape_input = tuple(shape)
            elif kind == "output":
                reshape_output = tuple(shape)

        # Find output node of the block
        output_node = None
        for _, nd in block["nodes"]:
            for user in nd.users:
                if user not in block_node_set:
                    output_node = nd
                    break
            if output_node is not None:
                break
        if output_node is None:
            # Fallback: last node in the block by graph position
            output_node = max((nd for _, nd in block["nodes"]),
                              key=lambda n: node_idx[n])
        boundary_nodes.append((output_node, reshape_input, reshape_output))

    # ------------------------------------------------------------------
    # 3. Assign each node to a segment.
    #    Segments are separated by boundary_nodes.  Node `n` goes into
    #    segment `s` where `s` is the number of boundary nodes that appear
    #    strictly before `n` in graph order (i.e. boundary node itself is in
    #    the segment BEFORE the cut).
    # ------------------------------------------------------------------
    boundary_positions = [node_idx[bn] for bn, _, _ in boundary_nodes]
    # boundary_positions[i] is the graph index of the i-th boundary node.
    # Segment i contains nodes with graph index in [prev_cut+1, boundary_pos_i].
    # Segment n_boundaries contains all nodes after the last boundary.
    n_segs = n_boundaries + 1

    compute_set: frozenset[int] = frozenset(
        i for i, nd in enumerate(nodes)
        if nd.op not in ("placeholder", "get_attr", "output")
    )

    def _seg_of_idx(idx: int) -> int:
        """Return the segment index for a graph-node at position idx."""
        s = 0
        for bp in boundary_positions:
            if idx > bp:
                s += 1
            else:
                break
        return s

    node_seg: dict[fx.Node, int] = {}
    for nd in nodes:
        if nd.op == "output":
            node_seg[nd] = n_segs - 1
        elif nd.op == "placeholder":
            if nd in param_ph_set:
                user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
                node_seg[nd] = _seg_of_idx(min(user_idxs)) if user_idxs else 0
            else:
                node_seg[nd] = 0
        elif nd.op == "get_attr":
            user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
            node_seg[nd] = _seg_of_idx(min(user_idxs)) if user_idxs else 0
        else:
            node_seg[nd] = _seg_of_idx(node_idx[nd])

    # ------------------------------------------------------------------
    # 4. Compute cross-segment inputs (same pattern as bucket_stage)
    # ------------------------------------------------------------------
    node_max_user_seg: dict[fx.Node, int] = {}
    for nd in nodes:
        if nd.op == "output":
            continue
        node_max_user_seg[nd] = (
            max(node_seg[u] for u in nd.users) if nd.users else node_seg[nd]
        )

    seg_cross_in: list[list[fx.Node]] = [[] for _ in range(n_segs)]
    for nd in nodes:
        if nd.op in ("get_attr", "output"):
            continue
        if nd.op == "placeholder" and nd in param_ph_set:
            continue
        s = node_seg[nd]
        mu = node_max_user_seg[nd]
        for seg in range(s + 1, mu + 1):
            seg_cross_in[seg].append(nd)

    # ------------------------------------------------------------------
    # 5. Build per-segment sub-graphs (same pattern as bucket_stage)
    # ------------------------------------------------------------------
    results: list[tuple[fx.GraphModule, list[int], list[int], list]] = []
    boundary_infos: list[dict] = []

    for seg in range(n_segs):
        sub_g = fx.Graph()
        remap: dict[fx.Node, fx.Node] = {}
        new_input_idxs: list[int] = []
        new_param_idxs: list[int] = []
        new_graphargs: list = []
        pos = 0

        def _meta_tensor_for(nd: fx.Node):
            ev = nd.meta.get("example_value")
            if ev is None:
                ev = nd.meta.get("val")
            if ev is not None and hasattr(ev, "shape"):
                return torch.empty(ev.shape, dtype=ev.dtype, device="meta",
                                   requires_grad=getattr(ev, "requires_grad", False))
            return None

        def _add_ph(nd: fx.Node, name: str, is_input: bool) -> None:
            nonlocal pos
            new_ph = sub_g.placeholder(name)
            new_ph.type = nd.type
            remap[nd] = new_ph
            if is_input:
                new_input_idxs.append(pos)
                new_graphargs.append(_meta_tensor_for(nd))
            else:
                new_param_idxs.append(pos)
                i_orig = ph_nodes.index(nd) if nd in ph_nodes else -1
                new_graphargs.append(graphargs[i_orig] if 0 <= i_orig < len(graphargs) else None)
            pos += 1

        if seg == 0:
            for nd in nodes:
                if nd.op == "placeholder" and nd not in param_ph_set:
                    _add_ph(nd, nd.name, is_input=(nd in input_ph_set))
        else:
            for orig in seg_cross_in[seg]:
                _add_ph(orig, f"_xseg_{orig.name}", is_input=True)

        for nd in nodes:
            if nd.op == "placeholder" and nd in param_ph_set and node_seg[nd] == seg:
                _add_ph(nd, nd.name, is_input=False)

        for nd in nodes:
            if nd.op == "get_attr" and node_seg[nd] == seg:
                new_ga = sub_g.get_attr(nd.target)
                new_ga.type = nd.type
                remap[nd] = new_ga

        for nd in nodes:
            if nd.op in ("placeholder", "get_attr", "output"):
                continue
            if node_seg[nd] != seg:
                continue
            remap[nd] = sub_g.node_copy(nd, arg_transform=lambda x, r=remap: r[x])

        if seg == n_segs - 1:
            orig_out = next(nd for nd in nodes if nd.op == "output")
            sub_g.output(fx.map_arg(orig_out.args[0], lambda x: remap[x]))
        else:
            # Output is all cross-segment values for the next segment.
            out_nodes = [remap[orig] for orig in seg_cross_in[seg + 1]]
            sub_g.output(tuple(out_nodes) if len(out_nodes) != 1 else out_nodes[0])

            # Record which output position holds the A2A boundary tensor.
            # boundary_nodes[seg] is the A2A output node for the boundary AFTER seg.
            boundary_nd, reshape_input, reshape_output = boundary_nodes[seg]
            cross_out_list = seg_cross_in[seg + 1]
            if boundary_nd in cross_out_list:
                tensor_idx = cross_out_list.index(boundary_nd)
            else:
                raise ValueError(
                    f"split_by_a2a: A2A boundary node '{boundary_nd.name}' "
                    f"(segment {seg}) is not in seg_cross_in[{seg + 1}]; "
                    f"seg_cross_in = {[n.name for n in cross_out_list]}"
                )
            boundary_infos.append({
                "tensor_idx": tensor_idx,
                "reshape_input": reshape_input,
                "reshape_output": reshape_output,
            })

        sub_g.lint()
        results.append((fx.GraphModule(stage_gm, sub_g), new_input_idxs, new_param_idxs, new_graphargs))

    return results, boundary_infos


# ---------------------------------------------------------------------------
# get_overlappable_tasks / overlap_chunks
# ---------------------------------------------------------------------------

def get_overlappable_tasks(schedule: PipelineSchedule) -> list[tuple[Chunk, Chunk]]:
    """Return one ``(FWD, BWD)`` pair for each ``FWD_BWD`` schedule cell."""
    result: list[tuple[Chunk, Chunk]] = []
    for row in schedule.grid:
        for chunk in row:
            if chunk.type != TaskType.FWD_BWD or len(chunk.batches) != 2:
                continue
            fwd_batch, bwd_batch = chunk.batches
            result.append((
                Chunk(pp_rank=chunk.pp_rank, batches=[fwd_batch], type=TaskType.FWD),
                Chunk(pp_rank=chunk.pp_rank, batches=[bwd_batch], type=TaskType.BWD),
            ))
    return result


def _task_matches_chunk(task: Task, chunk: Chunk) -> bool:
    return (
        task.task_type != TaskType.UPD
        and task.associated_chunk == chunk
    )


def _tasks_for_chunk(rank_dag: TaskDAG, chunk: Chunk) -> list[Task]:
    tasks = [node for node in rank_dag.nodes if _task_matches_chunk(node, chunk)]
    if not tasks:
        raise ValueError(f"No DAG tasks found for chunk descriptor {chunk}")
    return tasks


def _chunk_head_tasks(tasks: list[Task]) -> list[Task]:
    task_ids = {id(task) for task in tasks}
    return [
        task for task in tasks
        if not any(id(pred) in task_ids for pred in list(task.data_preds) + list(task.temporal_preds))
    ]


def _local_descendant_counts(tasks: list[Task]) -> dict[int, set[int]]:
    task_ids = {id(task) for task in tasks}
    descendants: dict[int, set[int]] = {}
    for task in tasks:
        seen: set[int] = set()
        stack = [
            succ for succ in list(task.data_succs) + list(task.temporal_succs)
            if id(succ) in task_ids
        ]
        while stack:
            cur = stack.pop()
            cur_id = id(cur)
            if cur_id in seen:
                continue
            seen.add(cur_id)
            stack.extend(
                succ for succ in list(cur.data_succs) + list(cur.temporal_succs)
                if id(succ) in task_ids
            )
        descendants[id(task)] = seen
    return descendants


def _overlap_ready_priority(task: Task) -> tuple[int, int, int]:
    """Tie-break ready tasks during overlap scheduling.

    Lower tuples win. SEND/RECV take priority over collective comm tasks when
    downstream counts tie, then fall back to stable task construction order.
    """
    op_priority = {
        TaskType.SEND: 0,
        TaskType.RECV: 0,
        TaskType.ALL_GATHER: 1,
        TaskType.REDUCE_SCATTER: 1,
        TaskType.ALL_REDUCE: 1,
    }.get(task.task_type, 2)
    return (op_priority, task.uid)


def overlap_chunks(rank_dag: TaskDAG, chunk_pairs: list[tuple[Chunk, Chunk]]) -> None:
    """Overlap the tasks belonging to each ``(FWD, BWD)`` chunk pair in-place."""
    for first_chunk, second_chunk in chunk_pairs:
        first_tasks = _tasks_for_chunk(rank_dag, first_chunk)
        second_tasks = _tasks_for_chunk(rank_dag, second_chunk)
        first_ids = {id(task) for task in first_tasks}
        second_ids = {id(task) for task in second_tasks}
        pair_ids = first_ids | second_ids

        cross_temporal = [
            (src, dst)
            for src in first_tasks
            for dst in src.temporal_succs
            if id(dst) in second_ids
        ]
        if len(cross_temporal) != 1:
            raise ValueError(
                f"Expected exactly one temporal edge between overlapped chunks "
                f"{first_chunk} and {second_chunk}, found {len(cross_temporal)}"
            )
        _remove_temporal_edge(*cross_temporal[0])

        all_tasks = first_tasks + second_tasks
        descendants = {
            0: _local_descendant_counts(first_tasks),
            1: _local_descendant_counts(second_tasks),
        }
        chunk_index = {id(task): 0 for task in first_tasks}
        chunk_index.update({id(task): 1 for task in second_tasks})
        assigned: set[int] = set()
        ready = _chunk_head_tasks(first_tasks) + _chunk_head_tasks(second_tasks)
        ready_ids = {id(task) for task in ready}
        resource_bins: dict[str, list[Task]] = defaultdict(list)
        next_chunk_idx = 0

        def _is_locally_ready(task: Task) -> bool:
            preds = [
                pred for pred in list(task.data_preds) + list(task.temporal_preds)
                if id(pred) in pair_ids
            ]
            return all(id(pred) in assigned for pred in preds)

        def _remaining_downstream(task: Task) -> int:
            task_chunk = chunk_index[id(task)]
            return sum(1 for desc_id in descendants[task_chunk][id(task)] if desc_id not in assigned)

        while len(assigned) < len(all_tasks):
            ready = [task for task in ready if id(task) not in assigned and _is_locally_ready(task)]
            if not ready:
                raise ValueError(
                    f"No ready tasks remain while overlapping chunks {first_chunk} and {second_chunk}"
                )

            best_remaining = max(_remaining_downstream(task) for task in ready)
            candidates = [task for task in ready if _remaining_downstream(task) == best_remaining]
            preferred = [task for task in candidates if chunk_index[id(task)] == next_chunk_idx]
            current = min(preferred or candidates, key=_overlap_ready_priority)

            assigned.add(id(current))
            ready_ids.discard(id(current))
            resource_bins[current.resource].append(current)
            next_chunk_idx = 1 - chunk_index[id(current)]

            for succ in list(current.data_succs) + list(current.temporal_succs):
                succ_id = id(succ)
                if succ_id not in pair_ids or succ_id in assigned or succ_id in ready_ids:
                    continue
                if _is_locally_ready(succ):
                    ready.append(succ)
                    ready_ids.add(succ_id)

        for resource_tasks in resource_bins.values():
            for src, dst in zip(resource_tasks, resource_tasks[1:]):
                _add_temporal_edge(src, dst)

        compute_tasks = resource_bins.get("compute_stream", [])
        if not compute_tasks:
            continue

        first_compute = compute_tasks[0]
        last_compute = compute_tasks[-1]

        for task in compute_tasks:
            if task is not first_compute:
                external_preds = [pred for pred in list(task.temporal_preds) if id(pred) not in pair_ids]
                for pred in external_preds:
                    _remove_temporal_edge(pred, task)
                    _add_temporal_edge(pred, first_compute)

            if task is not last_compute:
                external_succs = [succ for succ in list(task.temporal_succs) if id(succ) not in pair_ids]
                for succ in external_succs:
                    _remove_temporal_edge(task, succ)
                    _add_temporal_edge(last_compute, succ)

# ---------------------------------------------------------------------------
# Assign time_step from temporal chain position
# ---------------------------------------------------------------------------

def assign_time_steps(dag: TaskDAG) -> None:
    """Assign ``time_step`` to every node in two passes.

    **Pass 1** – Kahn's toposort over *temporal* edges, restricted to
    compute / A2A task types (F / B / B_I / B_W / FWD_BWD / F_A2A /
    B_A2A / UPD).  Non-compute nodes do not contribute to the temporal
    ordering and are skipped.

    **Pass 2** – assign time steps to the remaining nodes via their
    direct data-edge neighbours, using BFS propagation from the compute
    nodes so that chains of non-compute nodes (e.g. P+ → AG → compute
    or compute → RS → G- → P-) are resolved correctly:

    * RECV / P+ (ALLOC_FULL_PARAMS) / G+ (ALLOC_FULL_GRADS) /
      AG (ALL_GATHER) → ``direct_data_succ.time_step - 1``
    * SEND / RS (REDUCE_SCATTER) / AR (ALL_REDUCE) /
      G- (FREE_FULL_GRADS) / P- (FREE_FULL_PARAMS)
      → ``direct_data_pred.time_step + 1``

    This function is idempotent and safe to call both on the full DAG
    (before :func:`split_dag_by_rank`) and on individual per-rank DAGs
    after :func:`insert_zero_ops` has added the ZeRO collective nodes.
    """
    _COMPUTE_TYPES = frozenset({
        TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W,
        TaskType.FWD_BWD, TaskType.FWD_A2A, TaskType.BWD_A2A, TaskType.UPD,
    })
    # Non-compute nodes that sit *before* their data successor: get ts - 1.
    _DOWNSTREAM_MINUS_ONE = frozenset({
        TaskType.RECV,
        TaskType.ALLOC_FULL_PARAMS,
        TaskType.ALLOC_FULL_GRADS,
        TaskType.ALL_GATHER,
    })
    # Non-compute nodes that sit *after* their data predecessor: get ts + 1.
    _UPSTREAM_PLUS_ONE = frozenset({
        TaskType.SEND,
        TaskType.REDUCE_SCATTER,
        TaskType.ALL_REDUCE,
        TaskType.FREE_FULL_GRADS,
        TaskType.FREE_FULL_PARAMS,
    })

    # ---- Pass 1: toposort compute nodes via temporal edges ----
    compute_nodes = [n for n in dag.nodes if n.task_type in _COMPUTE_TYPES]

    in_degree: dict[int, int] = {
        id(n): sum(1 for p in n.temporal_preds if p.task_type in _COMPUTE_TYPES)
        for n in compute_nodes
    }
    ts_map: dict[int, int] = {}
    queue: deque[Task] = deque()
    for node in compute_nodes:
        if in_degree[id(node)] == 0:
            ts_map[id(node)] = 0
            queue.append(node)

    while queue:
        node = queue.popleft()
        node.time_step = ts_map[id(node)]
        for succ in node.temporal_succs:
            if succ.task_type not in _COMPUTE_TYPES:
                continue
            in_degree[id(succ)] -= 1
            ts_map[id(succ)] = max(ts_map.get(id(succ), 0), node.time_step + 1)
            if in_degree[id(succ)] == 0:
                queue.append(succ)

    # ---- Pass 2: BFS from compute nodes to assign non-compute nodes ----
    # Forward BFS: compute → _UPSTREAM_PLUS_ONE successors (time_step + 1 per hop).
    visited: set[int] = {id(n) for n in compute_nodes}
    bfs: deque[Task] = deque(compute_nodes)
    while bfs:
        node = bfs.popleft()
        for succ in node.data_succs:
            if succ.task_type not in _UPSTREAM_PLUS_ONE or id(succ) in visited:
                continue
            visited.add(id(succ))
            succ.time_step = node.time_step + 1
            bfs.append(succ)

    # Backward BFS: compute → _DOWNSTREAM_MINUS_ONE predecessors (time_step - 1 per hop).
    visited2: set[int] = {id(n) for n in compute_nodes}
    bfs2: deque[Task] = deque(compute_nodes)
    while bfs2:
        node = bfs2.popleft()
        for pred in node.data_preds:
            if pred.task_type not in _DOWNSTREAM_MINUS_ONE or id(pred) in visited2:
                continue
            visited2.add(id(pred))
            pred.time_step = node.time_step - 1
            bfs2.append(pred)

    # ---- Pass 3: push UPD past any AR/RS temporal predecessors ----
    # AR/RS nodes are in _UPSTREAM_PLUS_ONE (assigned in Pass 2, not Pass 1).
    # If _wire_sync_to_upd added AR/RS → UPD temporal edges, UPD must land
    # at least one step after the latest AR/RS.
    for node in dag.nodes:
        if node.task_type == TaskType.UPD:
            for pred in node.temporal_preds:
                if pred.task_type in _UPSTREAM_PLUS_ONE:
                    node.time_step = max(node.time_step, pred.time_step + 1)
