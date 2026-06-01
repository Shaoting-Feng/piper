import hashlib
import json
import torch
import torch.fx as fx
from collections import defaultdict
from pathlib import Path
import operator

from .piper_utils import create_logger, LOG_LEVEL

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

    # Step 3: Profile each compute node with eager memory freeing
    # Use no_grad to prevent autograd graph accumulation — we only need forward timing.
    node_times = {}  # node -> time in ms

    num_warmup = 3
    num_profile = 5
    for node in compute_nodes:
        args = _fetch_arg(node.args)
        kwargs = _fetch_arg(node.kwargs)

        # Check for None inputs from failed predecessor nodes
        has_none_input = False
        for inp in node.all_input_nodes:
            if inp.op not in ("placeholder", "get_attr") and env.get(inp.name) is None:
                has_none_input = True
                break

        if has_none_input:
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


def _meta_tensor_like(example, *, requires_grad: bool, as_parameter: bool):
    sym_int_type = getattr(torch, "SymInt", ())
    sym_float_type = getattr(torch, "SymFloat", ())
    sym_bool_type = getattr(torch, "SymBool", ())

    if isinstance(example, torch.Tensor):
        t = torch.empty(example.shape, dtype=example.dtype, device="meta")
        t.requires_grad_(requires_grad)
    elif isinstance(example, (sym_int_type, int)):
        # Symbolic/python integers do not have .shape/.dtype; represent them as
        # singleton meta tensors so downstream graph-arg handling remains uniform.
        t = torch.empty((1,), dtype=torch.int64, device="meta")
    elif isinstance(example, (sym_float_type, float)):
        t = torch.empty((1,), dtype=torch.float32, device="meta")
    elif isinstance(example, (sym_bool_type, bool)):
        t = torch.empty((1,), dtype=torch.bool, device="meta")
    else:
        # Fallback for unknown scalar-like values.
        t = torch.empty((1,), dtype=torch.float32, device="meta")

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
                # Only runtime tensor activations should be stage inputs.
                # Symbolic placeholders (e.g. SymInt shape env entries) are not
                # fed by run_dag and must stay out of input_idxs.
                if isinstance(ex, torch.Tensor) and ('self' not in placeholder.name):
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



# ---------------------------------------------------------------------------
# bucket_stage - stage-level parameter bucketing (placeholder params)
# ---------------------------------------------------------------------------


def bucket_stage(
    stage_gm: fx.GraphModule,
    graphargs: list,
    input_idxs: list[int],
    param_idxs: list[int],
    bucket_size_bytes: int = 25 * 1024 * 1024,
    debug_name: str | None = None,
) -> list[tuple[fx.GraphModule, list[int], list[int], list]]:
    """Split a stage GraphModule into per-parameter-bucket sub-modules.

    This function handles stages produced by :func:`_split_gm_by_stages`
    where trainable parameters are passed as **placeholder inputs** identified
    by *param_idxs*.

    Args:
        stage_gm: The stage ``fx.GraphModule`` to split.
        graphargs: Flat arg list for ``stage_gm.forward``; entries at
            *param_idxs* are meta tensors, entries at *input_idxs* are None.
        input_idxs: Positions of activation-input placeholders.
        param_idxs: Positions of parameter placeholders.
        bucket_size_bytes: Target bucket size (default 25 MB).
        debug_name: Optional label included in bucket planning logs.

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

    debug_name = debug_name or getattr(stage_gm, "__class__", type(stage_gm)).__name__
    param_ranges = {pn: _use_range(pn) for pn in param_ph_list}
    param_sizes = {pn: _size(pn) for pn in param_ph_list}

    # Greedy bucket assignment
    bucket_id: dict[fx.Node, int] = {}
    cur_b, cur_sz = 0, 0
    for pn in param_ph_list:
        sz = param_sizes[pn]
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

    def _bucket_bytes(members: list[fx.Node]) -> int:
        return sum(param_sizes[pn] for pn in members)

    def _format_members(members: list[fx.Node], *, limit: int = 8) -> str:
        def _one(pn: fx.Node) -> str:
            return f"{pn.name}:{param_sizes[pn]}"

        if len(members) <= limit:
            return "[" + ", ".join(_one(pn) for pn in members) + "]"
        head = ", ".join(_one(pn) for pn in members[:4])
        tail = ", ".join(_one(pn) for pn in members[-2:])
        return f"[{head}, ... ({len(members) - 6} more), {tail}]"

    def _log_bucket_plan(phase: str, seg_ranges: list[tuple[int, int, list[fx.Node]]]) -> None:
        ownership: dict[fx.Node, list[int]] = defaultdict(list)
        for seg_idx, (_f, _l, members) in enumerate(seg_ranges):
            for pn in members:
                ownership[pn].append(seg_idx)
        missing = [pn.name for pn in param_ph_list if pn not in ownership]
        duplicates = [
            (pn.name, buckets)
            for pn, buckets in ownership.items()
            if len(buckets) != 1
        ]
        total_bytes = sum(param_sizes[pn] for pn in param_ph_list)
        max_bucket_bytes = max((_bucket_bytes(ms) for _f, _l, ms in seg_ranges), default=0)
        logger.info(
            "bucket_stage plan phase=%s node=%s bucket_size_bytes=%d bucket_size_mib=%.6f "
            "params=%d total_param_bytes=%d initial_buckets=%d final_buckets=%d "
            "owned_params=%d unique_owned_params=%d missing_params=%d duplicate_params=%d max_bucket_bytes=%d",
            phase,
            debug_name,
            bucket_size_bytes,
            bucket_size_bytes / (1024 * 1024),
            len(param_ph_list),
            total_bytes,
            n_init,
            len(seg_ranges),
            sum(len(ms) for _f, _l, ms in seg_ranges),
            len(ownership),
            len(missing),
            len(duplicates),
            max_bucket_bytes,
        )
        if missing or duplicates:
            logger.warning(
                "bucket_stage ownership-invalid phase=%s node=%s missing=%s duplicates=%s",
                phase,
                debug_name,
                missing[:16],
                duplicates[:16],
            )

        merge_candidates: list[tuple[int, int, int]] = []
        for left_idx in range(len(seg_ranges) - 1):
            left_bytes = _bucket_bytes(seg_ranges[left_idx][2])
            right_bytes = _bucket_bytes(seg_ranges[left_idx + 1][2])
            combined = left_bytes + right_bytes
            if combined <= bucket_size_bytes:
                merge_candidates.append((left_idx, left_idx + 1, combined))

        for seg_idx, (first, last, members) in enumerate(seg_ranges):
            bytes_ = _bucket_bytes(members)
            if bytes_ <= bucket_size_bytes:
                status = "within_limit"
            elif len(members) == 1:
                status = "single_param_over_limit"
            else:
                status = "forced_multi_param_over_limit"
            logger.info(
                "bucket_stage bucket-plan phase=%s node=%s bucket=%d/%d param_bytes=%d "
                "params=%d range=%d:%d status=%s members=%s",
                phase,
                debug_name,
                seg_idx,
                len(seg_ranges),
                bytes_,
                len(members),
                first,
                last,
                status,
                _format_members(members),
            )
        if merge_candidates:
            logger.warning(
                "bucket_stage fullness-warning phase=%s node=%s mergeable_adjacent_pairs=%s "
                "bucket_size_bytes=%d",
                phase,
                debug_name,
                merge_candidates[:16],
                bucket_size_bytes,
            )
        else:
            logger.info(
                "bucket_stage fullness-ok phase=%s node=%s adjacent_pairs=%d bucket_size_bytes=%d",
                phase,
                debug_name,
                max(0, len(seg_ranges) - 1),
                bucket_size_bytes,
            )

    _log_bucket_plan("range_merge", merged)

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
            if nd.op == "output":
                continue
            s = node_seg[nd]
            mu = node_max_user_seg[nd]
            # Only values that cross a segment boundary must be forwarded.
            if mu <= s:
                continue
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

    _log_bucket_plan("final", merged)

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
            boundary_custom = boundary_nd.meta.get("custom") if hasattr(boundary_nd, "meta") else None
            boundary_name = (
                boundary_custom.get("name")
                if isinstance(boundary_custom, dict)
                else None
            )
            boundary_infos.append({
                "tensor_idx": tensor_idx,
                "reshape_input": reshape_input,
                "reshape_output": reshape_output,
                "name": boundary_name,
            })

        sub_g.lint()
        results.append((fx.GraphModule(stage_gm, sub_g), new_input_idxs, new_param_idxs, new_graphargs))

    return results, boundary_infos
