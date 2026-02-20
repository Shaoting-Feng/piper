import ray
import torch
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union, Callable
import gc
from torch._guards import CompileId
from torch.nn import Parameter
from torch.autograd.graph import GradientEdge, Node
import torch.distributed as dist
from collections import defaultdict
from piper.utils import deserialize_graphmodule, piper_metadata, create_logger
from piper.backward_utils import get_param_groups, construct_reverse_graph, _get_grad_fn_or_grad_acc

torch.set_float32_matmul_precision('high')
torch._dynamo.config.compiled_autograd = False

CLEANUP_MEMORY = False

@ray.remote
class PiperActor:
    def __init__(self, actor_id, world_size, dp_rank=0, dp_degree=1, pp_degree=1, optim_fn=None):
        self.logger = create_logger("piper_actor", "DEBUG")
        
        start = time.perf_counter()

        self.actor_id = actor_id
        self.optim_fn = optim_fn
        
        # Data parallel attributes
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.world_size = world_size
        self.dp_group = None
        self.pp_degree = pp_degree

        if pp_degree == 1:
            self.global_rank = dp_rank
        elif dp_degree == 1:
            self.global_rank = actor_id
        else:
            self.global_rank = dp_rank * dp_degree + actor_id

        self.logger.info(f"Initializing Ray actor {actor_id} global rank {self.global_rank} with PID: {os.getpid()}")

        self.input = None
        self.truth = None
        self.fwd_objs = {}
        self.bwd_objs = {}

        # ordered list of frame ids for ordering the fx.Graphs on this actor
        self.frame_ids = []
        # map stage id -> compiled fx.Graph function
        self.compiled_fns = dict()
        # map stage id -> original GraphModule (for hook registration)
        self.graph_modules = dict()
        # map stage id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.parameters = dict()
        # map stage id -> indices of the input tensors (as opposed to model parameters) used by the fx.Graph
        self.input_idxs = dict()
        # map stage id -> optimizer for the fx.Graph
        self.optims = dict()
        # map stage -> mb_idx -> previous activation (if this stage is not first)
        self.prev_activation = defaultdict(dict)
        # map stage id -> mb_idx -> current activation
        self.activation = defaultdict(dict)
        # accumuate loss for each microbatch
        self.loss = []
        # stage_id -> mb_idx -> param_groups
        self._bw_param_groups = defaultdict(dict)

        # Timing infrastructure
        self.tracing = False  # Toggle for timing and memory tracing
        self.trace_data = {'update': {'total': [], 'peak_memory_delta': [], 'peak_memory': []}}

        end = time.perf_counter()
        self.logger.debug(f"__init__ took {(end-start)*1000:.2f}ms")

        self.logger.debug(f"Initialized actor {self.actor_id} for global rank {self.global_rank}")

    def id(self):
        return self.actor_id

    def send_input(self, tensor):
        self.input = tensor.to('cuda')
        return "done"
    
    def send_truth(self, tensor):
        self.truth = tensor.to('cuda')
        return "done"

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()
        return "done"

    def get_peak_memory(self):
        return torch.cuda.max_memory_allocated() / (1024**3)

    # @ray.method
    def join_process_groups(self):
        master_addr = os.environ.get('PIPER_MASTER_ADDR', "127.0.0.1")
        master_port = os.environ.get('PIPER_MASTER_PORT', "10000")
        init_method = f"tcp://{master_addr}:{master_port}"

        dist.init_process_group("nccl", init_method=init_method, rank=self.global_rank, world_size=self.world_size)
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} has GPU {os.environ['CUDA_VISIBLE_DEVICES']}, joined the global process group")

        if self.dp_degree > 1:
            self.join_dp_process_group()
    
    def join_dp_process_group(self):
        # Every process needs to participate in every subgroup creation
        num_dp_groups = self.world_size // self.dp_degree
        for dp_group_id in range(num_dp_groups):
            group_ranks = [(dp_group_id + num_dp_groups * i) for i in range(self.dp_degree)]
            process_group = dist.new_group(ranks=group_ranks, backend='nccl')
            if self.global_rank % num_dp_groups == dp_group_id:
                self.dp_group = process_group
                self.logger.info(f"Global rank {self.global_rank} joined its dp group {dp_group_id} along with ranks {group_ranks}")

    def compile_graph(self, stage_id, gm_data, compiler_fn, graphargs, input_idxs):
        self.logger.debug(f"Compiling graph on actor {self.actor_id} for stage id: {stage_id} with inputs: {len(graphargs)}")
        start = time.perf_counter()

        # set up tracing data structure
        self.trace_data[stage_id] = {
            'forward': {
                'forward': [],
                'total': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
            'backward': {
                'backward': [],
                'total': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
            'backward_input': {
                'time': [],
                'graph_pruning': [],
            },
            'backward_weight': {
                'time': [],
            },
        }

        # compile the graph with the given graphargs
        gm = deserialize_graphmodule(gm_data)
        
        # Store GraphModule reference
        self.graph_modules[stage_id] = gm
        
        compiled_fn = compiler_fn(gm, graphargs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fns[stage_id] = compiled_fn

        # discard the graphargs that correspond to input tensors
        for i in input_idxs:
            graphargs[i] = None
        self.input_idxs[stage_id] = input_idxs

        # save the graphargs that correspond to model parameters
        self.parameters[stage_id] = graphargs

        # initialize the optimizer for this stage
        ## Added this manual_seed for debugging
        torch.manual_seed(0)
        if [param for param in self.parameters[stage_id] if param is not None]:
            self.optims[stage_id] = self.optim_fn([param for param in self.parameters[stage_id] if param is not None])

        del gm_data

        end = time.perf_counter()
        self.logger.debug(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    @ray.method(tensor_transport="nccl")
    def forward(self, stage_id: int, mb_idx: int, *args):
        self.logger.debug(f"Calling forward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        if self.tracing:
            beginning_event = torch.cuda.Event(enable_timing=True)
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            beginning_event.record()
        
        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        def unwrap_arg(arg):
            if isinstance(arg, list) or isinstance(arg, tuple):
                assert len(arg) == 1
                return arg[0]
            else:
                return arg
        args = list(map(unwrap_arg, args))

        def detach_arg(arg):
            return arg.detach().clone()
        args = list(map(detach_arg, args))

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
            return x[0] if isinstance(x, list) or isinstance(x, tuple) else x
        args = list(map(unwrap, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input as previous activation
        if stage_id != 0:
            if not args[0].requires_grad:
                args[0].requires_grad_()
            self.prev_activation[stage_id][mb_idx] = args[0]


        # Record start event for forward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()
            forward_start_event.record()

        # Call compiled function
        out = self.compiled_fns[stage_id](*self.parameters[stage_id])

        # Record end event and calculate forward timing
        if self.tracing:
            forward_end_event.record()
            torch.cuda.synchronize()
            forward_time = forward_start_event.elapsed_time(forward_end_event)
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # save first output as activation BEFORE clearing input tensors
        # This ensures the activation doesn't share memory with parameters
        # that might be modified when processing other stages on the same actor
        activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.activation[stage_id][mb_idx] = activation_tensor

        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = forward_start_event.elapsed_time(end_event)
            # Store in trace_data (all microbatches stored sequentially)
            self.trace_data[stage_id]['forward']['forward'].append(forward_time)
            self.trace_data[stage_id]['forward']['total'].append(forward_time)
            self.trace_data[stage_id]['forward']['peak_memory_delta'].append(forward_peak_memory_delta_gb)
            self.trace_data[stage_id]['forward']['peak_memory'].append(forward_peak_memory_gb)

            if isinstance(out, list) or isinstance(out, tuple):
                self.fwd_objs[stage_id] = [torch.ones_like(t) for t in out]
            else:
                self.fwd_objs[stage_id] = torch.ones_like(out)
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return out

    def forward_no_nccl(self, stage_id: int, mb_idx: int, *args):
        self.logger.debug(f"Calling cpu forward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
                return x[0]
            else:
                return x
        args = list(map(unwrap, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input as previous activation
        if stage_id != 0:
            # args[0].requires_grad_()
            assert args[0].requires_grad 
            self.prev_activation[stage_id][mb_idx] = args[0]

        out = self.compiled_fns[stage_id](*self.parameters[stage_id])
        
        # save first output as activation BEFORE clearing input tensors
        # This ensures the activation doesn't share memory with parameters
        # that might be modified when processing other stages on the same actor
        activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.activation[stage_id][mb_idx] = activation_tensor
        
        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return out

    @ray.method(tensor_transport="nccl")
    def backward(self, stage_id: int, mb_idx: int, inp, loss_fn=None):
        self.logger.debug(f"Calling backward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        if self.tracing:
            beginning_event = torch.cuda.Event(enable_timing=True)
            backward_start_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            beginning_event.record()

        # get the activation for the current stage
        assert mb_idx in self.activation[stage_id], f"mb_idx {mb_idx} not in activation[stage_id {stage_id}]"
        activation = self.activation[stage_id][mb_idx]
        
        # Record start event for backward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            backward_start_memory = torch.cuda.memory_allocated()
            backward_start_event.record()
        
        # compute loss in the last stage. use the saved activation rather
        # than inp because the saved activation remembers the computation graph
        if loss_fn is not None:
            if self.truth is not None:
                labels = self.truth
            else:
                labels = inp[0]
            loss = loss_fn(activation, labels)
            loss.backward()
            self.loss.append(loss.item())
            # Clean up loss tensor after extracting item
            del loss
        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            assert inp is not None
            assert activation.shape == inp.shape
            activation.backward(gradient=inp)

        # Record end event and calculate backward timing
        if self.tracing:
            backward_end_event.record()
            torch.cuda.synchronize()
            
            # Calculate total backward time
            backward_time = backward_start_event.elapsed_time(backward_end_event)
            
            backward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - backward_start_memory) / (1024**3)
            backward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        del self.activation[stage_id][mb_idx]
        del activation

        # propagate the gradient backwards if not the first stage
        if stage_id != 0:
            ret = [self.prev_activation[stage_id][mb_idx]]
        else:
            ret = ["done"]
        
        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = beginning_event.elapsed_time(end_event)
            # Store in trace_data (all microbatches stored sequentially)
            self.trace_data[stage_id]['backward']['backward'].append(backward_time)
            self.trace_data[stage_id]['backward']['total'].append(total_time)
            self.trace_data[stage_id]['backward']['peak_memory_delta'].append(backward_peak_memory_delta_gb)
            self.trace_data[stage_id]['backward']['peak_memory'].append(backward_peak_memory_gb)
            if stage_id != 0:
                self.bwd_objs[stage_id] = torch.ones_like(ret[0])
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return ret + ret

    def synchronize_gradients(self):
        """Synchronize gradients across all DP ranks for this stage using all-reduce."""     
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} synchronizing gradients")
        
        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.parameters.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.dp_group)

    def update(self, *done_mbs):
        self.logger.debug(f"Calling update on actor {self.actor_id} global rank {self.global_rank}")
        
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.reset_peak_memory_stats()
            update_start_memory = torch.cuda.memory_allocated()

            start_event.record()
        
        # if dp degree > 1, synchronize the gradients
        if self.dp_degree > 1:
            self.synchronize_gradients()
        
        # step the optimizer for each stage
        assert self.optim_fn
        for _, optim in self.optims.items():
            optim.step()
            optim.zero_grad()
        losses = self.loss
        self.loss.clear()

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            update_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - update_start_memory) / (1024**3)
            update_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data['update']['total'].append(total_time)
            self.trace_data['update']['peak_memory_delta'].append(update_peak_memory_delta_gb)
            self.trace_data['update']['peak_memory'].append(update_peak_memory_gb)

        return losses

    def verify_weights(self, msg="first parameter"):
        for stage_id, parameters in self.parameters.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    if len(param.shape) == 2:
                        self.logger.info(f"Actor {self.actor_id} global rank {self.global_rank} stage {stage_id} {msg}: {param.shape, param[0][0], param.grad[0][0]}")
                    elif len(param.shape) == 1:
                        self.logger.info(f"Actor {self.actor_id} global rank {self.global_rank} stage {stage_id} {msg}: {param.shape, param[0], param.grad[0]}")
                    else:
                        assert False, f"Unsupported parameter shape: {param.shape}"
                    return "done"

    def clear_trace_data(self) -> None:
        """
        Clear all collected timing data.
        """
        for stage_id in self.trace_data:
            self.trace_data[stage_id] = {
                'forward': {
                    'forward': [],
                    'total': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
                'backward': {
                    'backward': [],
                    'total': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
                'backward_input': {
                    'time': [],
                    'graph_pruning': [],
                },
                'backward_weight': {
                    'time': [],
                },
            }
        self.trace_data['update'] = {
            'total': [],
            'peak_memory_delta': [],
            'peak_memory': []
        }
    
    def get_trace_data(self) -> dict:
        return self.trace_data

    def set_tracing(self, enabled: bool) -> None:
        """
        Enable or disable timing and memory tracing.
        
        Args:
            enabled (bool): True to enable tracing, False to disable.
        """
        self.tracing = enabled
        self.logger.info(f"Actor {self.actor_id}: Tracing {'enabled' if enabled else 'disabled'}")

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()
        return "done"
    
    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(f"actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        self.logger.info(f"Saved memory snapshot to actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return "done"
    
    @ray.method(tensor_transport="nccl")
    def get_object(self, is_fwd, stage_id):
        if is_fwd:
            return self.fwd_objs[stage_id]
        else:
            return self.bwd_objs[stage_id]

    def time_object_retrieval(self, obj) -> dict:
        objs = []
        if isinstance(obj, tuple) or isinstance(obj, list):
            objs = [t + 1 for t in obj]
        else:
            objs = [obj + 1]
        self.objs = objs
        return "done"
    
    @ray.method(tensor_transport="nccl")
    def backward_input(self, stage_id: int, mb_idx: int, inp, loss_fn=None):
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        is_last_stage = loss_fn is not None
        activation = self.activation[stage_id][mb_idx]

        # Use loss instead of activation if this is the backward_input call for the last stage
        if is_last_stage:
            labels = self.truth if self.truth is not None else inp[0]
            loss = loss_fn(activation, labels)
            self.loss.append(loss)

            activation_or_loss = loss
            upstream_grads = torch.ones_like(loss)

        else:
            activation_or_loss = activation
            upstream_grads = inp

        # In first stage, we treat the backward_input call like a NOOP
        if stage_id == 0:
            gx = torch.zeros_like(activation) # May get slight performance boost if we don't send the full ones tensor
            if self.tracing:
                end_event.record()
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event)
                self.trace_data[stage_id]['backward_input']['time'].append(total_time)
            return [gx, upstream_grads]

        stage_input = self.prev_activation[stage_id][mb_idx]
        stage_params = [p for p in self.parameters[stage_id] if p is not None and p.requires_grad]

        # Lists of autograd nodes for the layer outputs, inputs, and parameters
        # Tracing for graph pruning computation time
        if self.tracing:
            graph_pruning_start = torch.cuda.Event(enable_timing=True)
            graph_pruning_end = torch.cuda.Event(enable_timing=True)
            graph_pruning_start.record()
        
        output_nodes = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [activation_or_loss]) if n is not None]
        input_nodes  = [n for n in (_get_grad_fn_or_grad_acc(t) for t in [stage_input]) if n is not None]
        param_nodes   = [n for n in (_get_grad_fn_or_grad_acc(p) for p in stage_params) if n is not None]

        # Use the autograd graph with edges reversed to compute parameter groups, which are groups 
        # of parameters that share the same intermediate nodes. Intermediate nodes are the nodes that
        # lie on both (1) a backward path from the output node(s) to the stage input nodes and 
        # (2) in a path from the output node(s) a parameter node/gradient accumulator
        reverse_edges = construct_reverse_graph(output_nodes)
        param_groups = get_param_groups(input_nodes, param_nodes, reverse_edges)

        pgs = param_groups
        with_int = [pg for pg in pgs if pg.get("intermediates")]
        self.logger.info(
            f"stage {stage_id} mb {mb_idx}: groups={len(pgs)} with_intermediates={len(with_int)} "
            f"intermediates_sizes={[len(pg.get('intermediates', [])) for pg in with_int]}"
        )

        # Check if any intermediate Node object appears in >1 group (should be 0)
        seen = set()
        overlap = 0
        for pg in pgs:
            for n in (pg.get("intermediates", []) or []):
                k = id(n)
                overlap += int(k in seen)
                seen.add(k)
        self.logger.info(f"stage {stage_id} mb {mb_idx}: intermediate_overlaps={overlap}")

        def ancestors(start_nodes, reverse_edges):
            # reverse_edges: dict[node] -> list[node]
            seen = set()
            stack = list(start_nodes)
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                for p in reverse_edges.get(n, []):
                    stack.append(p)
            return seen

        # Build ancestor sets for a handful of groups (keep it cheap)
        anc_sets = []
        for pg in pgs[:10]:  # sample first 10 groups
            ints = pg.get("intermediates", [])
            anc_sets.append(ancestors(ints, reverse_edges))

        shared = 0
        for i in range(len(anc_sets)):
            for j in range(i + 1, len(anc_sets)):
                shared += int(len(anc_sets[i].intersection(anc_sets[j])) > 0)

        self.logger.info(f"stage {stage_id} mb {mb_idx}: sampled_pairs_with_shared_ancestors={shared}")

        self.logger.info(
            f"stage {stage_id} mb {mb_idx}: stage_params="
            f"{len(stage_params)} unique={len({id(p) for p in stage_params})} "
            f"leaf={sum(int(p.is_leaf) for p in stage_params)} "
            f"has_grad_fn={sum(int(p.grad_fn is not None) for p in stage_params)}"
        )

        # also log the grad-acc node ids you group on
        nodes = [ _get_grad_fn_or_grad_acc(p) for p in stage_params ]
        self.logger.info(
            f"stage {stage_id} mb {mb_idx}: param_nodes="
            f"{sum(n is not None for n in nodes)} unique={len({id(n) for n in nodes if n is not None})}"
        )

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

        if self.tracing:
            graph_pruning_end.record()
            torch.cuda.synchronize()
            graph_pruning_time = graph_pruning_start.elapsed_time(graph_pruning_end)
            if 'graph_pruning' not in self.trace_data[stage_id]['backward_input']:
                self.trace_data[stage_id]['backward_input']['graph_pruning'] = []
            self.trace_data[stage_id]['backward_input']['graph_pruning'].append(graph_pruning_time)
        gx = torch.autograd.grad(
            outputs=activation_or_loss,
            inputs=stage_input,
            grad_outputs=upstream_grads,
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

        # Save parameter groups for use in backward_weight
        self._bw_param_groups[stage_id][mb_idx] = param_groups

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            self.trace_data[stage_id]['backward_input']['time'].append(total_time)

        return [gx, upstream_grads]

    @ray.method(tensor_transport="nccl")
    def backward_weight(self, stage_id: int, mb_idx: int, gx, upstream, loss_fn=None):
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        stage_params = [p for p in self.parameters[stage_id] if p is not None and p.requires_grad]

        # We only pass to enforce temporal dependency.
        # upstream is only required for the stage 0 case
        gx = gx
        upstream = upstream

        # Special case to handle stage 0 since backward_input is a NOOP, 
        # meaning no parameter groups are created
        if stage_id == 0:
            is_last_stage = loss_fn is not None
            activation = self.activation[stage_id][mb_idx]

            if is_last_stage:
                labels = self.truth if self.truth is not None else None
                loss = loss_fn(activation, labels)

                gparams = torch.autograd.grad(
                    outputs=loss,
                    inputs=stage_params,
                    retain_graph=False,
                )

            else:
                gparams = torch.autograd.grad(
                    outputs=activation,
                    inputs=stage_params,
                    grad_outputs=upstream,
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

            del self.activation[stage_id][mb_idx]

            if self.tracing:
                end_event.record()
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event)
                self.trace_data[stage_id]['backward_weight']['time'].append(total_time)

            return [None] + ["done"]

        # Create mapping from autograd nodes -> parameters
        grad_acc_to_weight: Dict[Node, Tuple[Parameter, int]] = {}
        for param in stage_params:
            node = _get_grad_fn_or_grad_acc(param)
            grad_acc_to_weight[node] = param

        param_groups = self._bw_param_groups[stage_id][mb_idx]

        self.logger.info(
            f"stage {stage_id} mb {mb_idx}: stage_params="
            f"{len(stage_params)} unique={len({id(p) for p in stage_params})} "
            f"leaf={sum(int(p.is_leaf) for p in stage_params)} "
            f"has_grad_fn={sum(int(p.grad_fn is not None) for p in stage_params)}"
        )

        # also log the grad-acc node ids you group on
        nodes = [ _get_grad_fn_or_grad_acc(p) for p in stage_params ]
        self.logger.info(
            f"stage {stage_id} mb {mb_idx}: param_nodes="
            f"{sum(n is not None for n in nodes)} unique={len({id(n) for n in nodes if n is not None})}"
        )

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

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            self.trace_data[stage_id]['backward_weight']['time'].append(total_time)

        return [None] + ["done"]
