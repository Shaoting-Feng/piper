import queue
import ray
import torch
import inspect
import logging
import json, importlib, operator
import torch.fx as fx
from collections import defaultdict
from dataclasses import dataclass
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from typing import Any, Optional

LOG_LEVEL = "DEBUG"

""" 
Print the backward graph of a tensor
"""

def print_backward_graph(printer, tensor, prefix=""):
    seen = set()
    def _print(t, indent=0):
        fn = t.grad_fn if hasattr(t, 'grad_fn') and t.grad_fn is not None else None
        if fn is None:
            printer(" " * indent + f"{prefix}Tensor: no grad_fn")
            return
        if fn in seen:
            printer(" " * indent + f"{prefix}{type(fn).__name__} (recursive/ref)")
            return
        seen.add(fn)
        printer(" " * indent + f"{prefix}{type(fn).__name__}")
        for next_fn, _ in fn.next_functions:
            if next_fn is not None and hasattr(next_fn, 'variable'):
                printer(" " * (indent + 2) + f"{prefix}Variable: {type(next_fn.variable).__name__}")
            elif next_fn is not None:
                _print(type('Dummy', (), {'grad_fn': next_fn})(), indent + 2)
            else:
                printer(" " * (indent + 2) + f"{prefix}None")
    _print(tensor, 0)


"""
Logger utility
"""

def create_logger(name: str, log_level: str):
    match log_level:
        case "DEBUG":
            log_level = logging.DEBUG
        case "INFO":
            log_level = logging.INFO
        case "WARNING":
            log_level = logging.WARNING
        case "ERROR":
            log_level = logging.ERROR
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


_CRITICAL_OPS     = frozenset({"fwd_p2p_recv", "fwd_p2p_send", "bwd_p2p_recv", "bwd_p2p_send", "fwd_a2a", "bwd_a2a"})
_NON_CRITICAL_OPS = frozenset({"grad_allreduce", "grad_allreduce_naive"})

_nccl_log = logging.getLogger("piper_nccl")


class _DepGate:
    """One-shot handoff of a CUDA end-event from a critical op to a non-critical op.

    signal() is called by the critical op's after_kernel.
    wait()   is called by the non-critical op's before_kernel.

    wait() is intentionally non-blocking: if the critical op hasn't fired yet
    (e.g. both are dispatched from the same CPU thread and the critical op comes
    later in the schedule), enforcement is skipped rather than deadlocking.
    Unmatched signals are drained by NcclOverlapDetector.reset_iteration().
    """
    __slots__ = ("_q",)

    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def signal(self, end_event: "torch.cuda.Event") -> None:
        self._q.put(end_event)

    def wait(self, stream: "torch.cuda.Stream") -> None:
        try:
            event = self._q.get_nowait()
            stream.wait_event(event)    # GPU-side dependency
        except queue.Empty:
            pass  # critical op not yet dispatched; skip enforcement for this call

    def drain(self) -> None:
        """Discard any unmatched signals left over from the previous iteration."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break


@dataclass
class _NcclKernelRecord:
    kernel_name: str
    stream_label: str
    start_event: "torch.cuda.Event"
    end_event: Optional["torch.cuda.Event"] = None
    # GPU-side timestamps (ms from reference event); populated by find_overlaps()
    t_start_ms: Optional[float] = None
    t_end_ms: Optional[float] = None
    # Position of this kernel in dispatch order on its stream; set by build_enforcement
    stream_instance_idx: Optional[int] = None


class NcclOverlapDetector:
    """
    Passively records GPU-side start/end events for every NCCL kernel while
    monitoring is enabled, then performs a single retrospective overlap analysis
    after the iteration completes.

    No callbacks, no host functions, no GPU stalls.

    Typical usage in a training loop:

        # After warmup, enable monitoring for one iteration:
        actor.enable_nccl_monitoring.remote()
        losses = piper.step(batch)
        actor.print_nccl_overlaps.remote()   # syncs GPU, analyses, prints, then disables

    At each NCCL launch site (AFTER any stream.wait_stream / wait_event):
        token = detector.before_kernel(cuda_stream, kernel_name, stream_label)
        # ... launch the NCCL kernel ...
        detector.after_kernel(cuda_stream, token)

    reset_iteration() is called automatically at the end of every training iteration.
    It clears records only when monitoring is disabled, so a monitored iteration's
    records survive until print_nccl_overlaps() consumes them.
    """

    def __init__(self):
        self._records: list[_NcclKernelRecord] = []
        self._ref_event: Optional["torch.cuda.Event"] = None
        self.enabled: bool = False
        # Enforcement — keyed by (stream_label, stream_instance_idx)
        self._signal_gates: dict[tuple[str, int], list[_DepGate]] = {}
        self._wait_gates:   dict[tuple[str, int], list[_DepGate]] = {}
        self._per_stream_counter: dict[str, int] = {}  # reset each iteration
        self._enforcement_active: bool = False

    def enable(self) -> None:
        """Enable monitoring and reset any previous records."""
        self._records.clear()
        self._ref_event = torch.cuda.Event(enable_timing=True)
        self._ref_event.record()
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def before_kernel(
        self,
        cuda_stream: "torch.cuda.Stream",
        kernel_name: str,
        stream_label: str,
    ) -> str:
        """Record start_event on cuda_stream and return an opaque token.

        Must be called AFTER any stream.wait_stream() / stream.wait_event() so the
        event is enqueued only once GPU-side dependencies have resolved, giving an
        accurate kernel-start timestamp.
        """
        # Enforcement: compute per-stream instance index and wait on any gates for this slot
        instance_idx = -1
        if self._enforcement_active:
            instance_idx = self._per_stream_counter.get(stream_label, 0)
            for gate in self._wait_gates.get((stream_label, instance_idx), []):
                gate.wait(cuda_stream)
            self._per_stream_counter[stream_label] = instance_idx + 1

        if not self.enabled:
            # Enforcement-only: encode (stream_label, instance_idx) for after_kernel
            return f"E:{stream_label}:{instance_idx}" if self._enforcement_active else ""

        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(cuda_stream)
        token = str(len(self._records))
        self._records.append(_NcclKernelRecord(
            kernel_name=kernel_name,
            stream_label=stream_label,
            start_event=start_event,
            stream_instance_idx=instance_idx if self._enforcement_active else None,
        ))
        return token

    def after_kernel(self, cuda_stream: "torch.cuda.Stream", token: str) -> None:
        """Record end_event on cuda_stream. Call after the NCCL kernel."""
        if not token:
            return

        if token.startswith("E:"):
            # Enforcement-only mode: parse "E:{stream_label}:{instance_idx}"
            _, stream_label, idx_str = token.split(":")
            instance_idx = int(idx_str)
            end_event = torch.cuda.Event(enable_timing=False)
            end_event.record(cuda_stream)
            for gate in self._signal_gates.get((stream_label, instance_idx), []):
                gate.signal(end_event)
            return

        # Monitoring mode (with or without enforcement): token is a record index
        record_idx = int(token)
        record = self._records[record_idx]

        end_event = None
        if self.enabled:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(cuda_stream)
            record.end_event = end_event

        if self._enforcement_active and record.stream_instance_idx is not None:
            if end_event is None:
                end_event = torch.cuda.Event(enable_timing=False)
                end_event.record(cuda_stream)
            for gate in self._signal_gates.get(
                (record.stream_label, record.stream_instance_idx), []
            ):
                gate.signal(end_event)

    def build_enforcement(self, overlaps: list, logger) -> None:
        """Build per-instance gate map from detected overlaps. Call after find_overlaps().

        Assigns a stable stream_instance_idx to every record (= position in dispatch
        order on that stream), then creates one _DepGate per detected overlap instance.
        Gates are permanent — they are not rebuilt each iteration; only
        _per_stream_counter is reset so the same indices are reproduced.
        """
        # Assign stream_instance_idx to every record in monitoring order
        stream_counts: dict[str, int] = {}
        for r in self._records:
            r.stream_instance_idx = stream_counts.get(r.stream_label, 0)
            stream_counts[r.stream_label] = r.stream_instance_idx + 1

        # Build one gate per detected overlap instance
        signal: dict[tuple[str, int], list[_DepGate]] = {}
        wait:   dict[tuple[str, int], list[_DepGate]] = {}
        n_pairs = 0
        for a, b, _ in overlaps:
            if a.kernel_name in _CRITICAL_OPS and b.kernel_name in _NON_CRITICAL_OPS:
                crit, noncrit = a, b
            elif b.kernel_name in _CRITICAL_OPS and a.kernel_name in _NON_CRITICAL_OPS:
                crit, noncrit = b, a
            else:
                continue
            logger.debug(f"Enforcing dependency between {crit} -> {noncrit} NCCL kernels")
            gate = _DepGate()
            signal.setdefault((crit.stream_label,    crit.stream_instance_idx),    []).append(gate)
            wait.setdefault(  (noncrit.stream_label, noncrit.stream_instance_idx), []).append(gate)
            n_pairs += 1

        self._signal_gates = signal
        self._wait_gates   = wait
        self._enforcement_active = bool(n_pairs)
        if self._enforcement_active:
            _nccl_log.info(
                "NcclOverlapDetector: enforcement active — %d instance-level gate(s)", n_pairs
            )

    def find_overlaps(self, logger) -> list[tuple["_NcclKernelRecord", "_NcclKernelRecord", float]]:
        """Compute GPU-time intervals and return all overlapping cross-stream pairs.

        Must be called after torch.cuda.synchronize() so all events are complete.
        Returns a list of (record_a, record_b, overlap_ms) tuples.
        """
        if not self._records or self._ref_event is None:
            return []

        ref = self._ref_event
        for r in self._records:
            if r.end_event is None or r.t_start_ms is not None:
                continue
            try:
                r.t_start_ms = ref.elapsed_time(r.start_event)
                r.t_end_ms   = ref.elapsed_time(r.end_event)
            except Exception:
                pass

        valid = [r for r in self._records if r.t_start_ms is not None and r.t_end_ms is not None]
        overlaps = []
        for i, a in enumerate(valid):
            # logger.info(f"NCCL kernel '{a.kernel_name}' on stream '{a.stream_label}': {a.t_start_ms:.3f}-{a.t_end_ms:.3f} ms")
            for b in valid[i + 1:]:
                if a.stream_label == b.stream_label:
                    continue
                if a.t_start_ms < b.t_end_ms and b.t_start_ms < a.t_end_ms:
                    overlap_ms = (min(a.t_end_ms, b.t_end_ms)
                                  - max(a.t_start_ms, b.t_start_ms))
                    overlaps.append((a, b, overlap_ms))
        return overlaps

    def reset_iteration(self) -> None:
        """Called at the end of every training iteration (after streams are synced).

        Clears records only when monitoring is disabled.  When monitoring is enabled,
        records are preserved so that print_nccl_overlaps() can analyse them after
        the iteration returns.  Gates are permanent and not rebuilt; only
        _per_stream_counter is reset so each iteration reproduces the same
        (stream_label, index) assignments as during monitoring.
        """
        if not self.enabled:
            self._records.clear()
            self._ref_event = None
        if self._enforcement_active:
            # Drain any unmatched signals (cases where the critical op fired after
            # the non-critical op's non-blocking wait already returned empty).
            for gates in self._signal_gates.values():
                for gate in gates:
                    gate.drain()
            self._per_stream_counter.clear()


"""
Piper thread local storage for tracking Piper actors, stages, and microbatches
"""

class PiperMetadata:
    actors = dict()
    dag = set()
    stage_to_device = dict()
    naive_gradient_sync = False
    use_activation_checkpointing = False
    bucketing = False  # Whether to split stages into per-param-bucket sub-modules
    schedule = None   # Schedule2D set by piper_setup; used by the piper backend
    task_dag = None   # TaskDAG built from schedule by the piper backend
    per_rank_dags = None  # Per-rank TaskDAGs built by the piper backend

piper_metadata = PiperMetadata()


"""
Serialize/deserialize an fx.GraphModule
"""

def _encode_arg(a):
    if isinstance(a, fx.Node):
        return {"__node__": a.name}
    if isinstance(a, torch.device):
        return {"__device__": str(a)}
    if isinstance(a, torch.dtype):
        return {"__dtype__": str(a).replace("torch.", "")}
    if isinstance(a, slice):
        return {"__slice__": True,
                "start": _encode_arg(a.start),
                "stop": _encode_arg(a.stop),
                "step": _encode_arg(a.step)}
    if a is Ellipsis:
        return {"__ellipsis__": True}
    if isinstance(a, tuple):  # <-- preserve tuples
        return {"__tuple__": [_encode_arg(x) for x in a]}
    if isinstance(a, list):
        return [_encode_arg(x) for x in a]
    if isinstance(a, dict):
        return {k: _encode_arg(v) for k, v in a.items()}
    return a

def _decode_arg(a, name_to_node):
    if isinstance(a, dict):
        if "__node__" in a:
            return name_to_node[a["__node__"]]
        if "__device__" in a:
            return torch.device(a["__device__"])
        if "__dtype__" in a:
            return getattr(torch, a["__dtype__"])
        if "__slice__" in a:
            return slice(
                _decode_arg(a["start"], name_to_node),
                _decode_arg(a["stop"], name_to_node),
                _decode_arg(a["step"], name_to_node),
            )
        if "__ellipsis__" in a:
            return Ellipsis
        if "__tuple__" in a:  # <-- reconstruct tuples
            return tuple(_decode_arg(x, name_to_node) for x in a["__tuple__"])
        # generic dict
        return {k: _decode_arg(v, name_to_node) for k, v in a.items()}
    if isinstance(a, list):
        return [_decode_arg(x, name_to_node) for x in a]
    return a

def _is_op_overload(obj):
    # Works across PyTorch versions without importing private types directly
    return obj.__class__.__module__.startswith("torch._ops") or obj.__class__.__name__.startswith("OpOverload")

def _serialize_target(t):
    # print("SERIALIZING", t, type(t))
    # call_method uses a string method name, pass through
    if isinstance(t, str):
        return {"kind": "string", "value": t}

    target_str = str(t)
    if target_str.startswith("torch.ops."):
            # Remove "torch.ops." prefix to get the path
            path = target_str[len("torch.ops."):]
            return {"kind": "torch_op", "path": path}

    # Handle _VariableFunctionsClass
    if getattr(t, "__module__", "") == "torch._VariableFunctionsClass":
        public_name = t.__name__
        if hasattr(torch, public_name):
            return {"kind": "py_func", "module": "torch", "qualname": public_name}
        else:
            raise ValueError(f"No public torch alias for {t}")

    # torch.ops.* (aten, prim, etc.)
    if _is_op_overload(t) or (getattr(t, "__module__", "").startswith("torch._ops")):
        return {"kind": "torch_op", "path": str(t)}  # e.g. "aten.add.Tensor" or "aten.add"

    # regular python function or built-in
    if inspect.isfunction(t) or inspect.isbuiltin(t):
        mod = inspect.getmodule(t)

        # TODO: This may not be working as expected, since we use another 
        # workaround during deserialization. 
        # Check if inspect.getmodule returned torch.ops.* namespace
        # These should be serialized as torch_op, not py_func
        if mod is not None:
            mod_name = mod.__name__
            if mod_name.startswith("torch.ops"):
                # This is a torch.ops function, serialize as torch_op
                target_str = str(t)
                if target_str.startswith("torch.ops."):
                    path = target_str[len("torch.ops."):]
                    return {"kind": "torch_op", "path": path}
                # If str(t) doesn't work, try to construct path from module and name
                # e.g., module="torch.ops.higher_order", name="triton_kernel_wrapper_mutation"
                # -> path="higher_order.triton_kernel_wrapper_mutation"
                module_parts = mod_name.split(".")
                if len(module_parts) >= 3:
                    namespace = module_parts[2]  # e.g., "higher_order"
                    path = f"{namespace}.{t.__name__}"
                    return {"kind": "torch_op", "path": path}
        # Special handling for torch.autograd.Function.apply methods
        # These are static methods that may have the wrong module in inspect.getmodule
        if t.__name__ == "apply":
            qualname = getattr(t, "__qualname__", "")
            mod = inspect.getmodule(t)
            
            # If qualname contains a class name (e.g., "AllToAllSingleFunction.apply")
            if "." in qualname:
                class_name = qualname.split(".")[0]
                # Try to find the class by searching common modules
                # First try the module where the function was found
                if mod and hasattr(mod, class_name):
                    func_class = getattr(mod, class_name)
                    if isinstance(func_class, type) and issubclass(func_class, torch.autograd.Function):
                        # Serialize as the Function class with .apply
                        return {"kind": "py_obj", "module": func_class.__module__, "qualname": func_class.__qualname__ + ".apply"}
                
                # If not found, search sys.modules for the class
                # This handles the case where inspect.getmodule returns torch.autograd.function
                # instead of the actual module where the Function class is defined
                import sys
                for module_name, module_obj in sys.modules.items():
                    if module_obj is not None and hasattr(module_obj, class_name):
                        try:
                            func_class = getattr(module_obj, class_name)
                            if isinstance(func_class, type) and issubclass(func_class, torch.autograd.Function):
                                # Serialize as the Function class with .apply
                                return {"kind": "py_obj", "module": func_class.__module__, "qualname": func_class.__qualname__ + ".apply"}
                        except (TypeError, AttributeError):
                            continue
                
                # If still not found and we have a qualname, try to import the module directly
                # based on common patterns (e.g., if class is AllToAllSingleFunction, try src.piper_graph_transform)
                if class_name.startswith("AllToAll"):
                    try:
                        from src import piper_graph_transform
                        if hasattr(piper_graph_transform, class_name):
                            func_class = getattr(piper_graph_transform, class_name)
                            if isinstance(func_class, type) and issubclass(func_class, torch.autograd.Function):
                                return {"kind": "py_obj", "module": func_class.__module__, "qualname": func_class.__qualname__ + ".apply"}
                    except (ImportError, AttributeError):
                        pass
        
        mod = inspect.getmodule(t)
        if mod is None:
            raise ValueError(f"Cannot serialize function without module: {t}")
        return {"kind": "py_func", "module": mod.__name__, "qualname": t.__name__}

    # classes or callables rarely appear as call_function targets, but support anyway
    if inspect.isclass(t):
        mod = t.__module__
        return {"kind": "py_obj", "module": mod, "qualname": t.__qualname__}

    # operator functions (already covered by py_func, but ensure resolvable)
    if t in operator.__dict__.values():
        return {"kind": "py_func", "module": "operator", "qualname": t.__name__}

    # last resort: try module+name
    mod = getattr(t, "__module__", None)
    name = getattr(t, "__name__", None)
    if mod and name:
        return {"kind": "py_func", "module": mod, "qualname": name}

    raise NotImplementedError(f"Unsupported target type: {t} ({type(t)})")

def _resolve_qualname(mod, qualname):
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj

def _deserialize_target(payload):
    # print("DESERIALIZING", payload, flush=True)
    kind = payload["kind"]

    if kind == "string":
        return payload["value"]

    if kind == "py_func":
        #TODO: this is a workaround to deserialize torch.ops.* functions
        # if module is torch.ops.*, redirect to torch_op deserialization
        # this needs to be dealt with during the serialization. 
        if payload["module"].startswith("torch.ops"):
            # Reconstruct the path from module and qualname
            # e.g., module="torch.ops.higher_order", qualname="triton_kernel_wrapper_mutation"
            # -> path="higher_order.triton_kernel_wrapper_mutation"
            module_parts = payload["module"].split(".")
            if len(module_parts) >= 3:
                namespace = module_parts[2]  # e.g., "higher_order"
                path = f"{namespace}.{payload['qualname']}"
                # Recursively call with torch_op payload
                torch_op_payload = {"kind": "torch_op", "path": path}
                return _deserialize_target(torch_op_payload)
        mod = importlib.import_module(payload["module"])
        return _resolve_qualname(mod, payload["qualname"])

    if kind == "py_obj":
        mod = importlib.import_module(payload["module"])
        try:
            return _resolve_qualname(mod, payload["qualname"])
        except AttributeError as e:
            # Provide better error message for debugging
            raise AttributeError(
                f"Failed to resolve qualname '{payload['qualname']}' in module '{payload['module']}': {e}"
            ) from e

    if kind == "torch_op":
        obj = torch.ops
        path = payload["path"]
        try:
            for part in payload["path"].split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
                # if operator doesn't exist, try importing implementation module to trigger registration
                if path.startswith("higher_order."):
                    op_name = path.split(".", 1)[1]
                    op_to_module = {
                        "triton_kernel_wrapper_mutation": "torch._higher_order_ops.triton_kernel_wrap",
                        "triton_kernel_wrapper_functional": "torch._higher_order_ops.triton_kernel_wrap",
                    }
                    if op_name in op_to_module:
                        try:
                            # Import to trigger registration
                            importlib.import_module(op_to_module[op_name])
                            # Try again
                            obj = torch.ops
                            for part in path.split("."):
                                obj = getattr(obj, part)
                            return obj
                        except (ImportError, AttributeError):
                            pass
                raise AttributeError(
                    f"Failed to resolve torch.ops path '{path}'. "
                    f"Available attributes: {[x for x in dir(obj) if not x.startswith('_')][:20]}"
                )

    raise NotImplementedError(f"Unknown target kind: {kind}")

def _serialize_graphmodule(gm: fx.GraphModule) -> str:
    nodes = []
    for n in gm.graph.nodes:
        nodes.append({
            "name": n.name,
            "op": n.op,
            "target": _serialize_target(n.target) if n.op in ("call_function", "call_method", "call_module", "get_attr") else None,
            "args": _encode_arg(n.args),
            "kwargs": _encode_arg(n.kwargs),
        })

    # Serialize all direct child modules that are GraphModules
    submodules = {}
    for name, module in gm.named_children():
        if isinstance(module, fx.GraphModule):
            # Recursively serialize GraphModule submodules
            submodules[name] = _serialize_graphmodule(module)

    data = {
        "nodes": nodes,
        "state_dict": {k: v.detach().cpu().tolist() for k, v in gm.state_dict().items()},
        # save which device parameters were on, optional:
        "param_devices": {k: str(v.device) for k, v in gm.state_dict().items()},
        "submodules": submodules,  # Add serialized submodules
    }
    serialized = json.dumps(data, ensure_ascii=False)

    del data
    del nodes

    return serialized

def _unwrap_output_arg(decoded):
    # FX stores output as (value,), where value may itself be a tuple.
    # Inductor expects the inner tuple directly.
    if isinstance(decoded, (tuple, list)) and len(decoded) == 1 and isinstance(decoded[0], (tuple, list)):
        return decoded[0]
    return decoded

def _deserialize_graphmodule(s: str) -> fx.GraphModule:
    data = json.loads(s)
    g = fx.Graph()
    name_to_node = {}

    for n in data["nodes"]:
        op = n["op"]
        if op == "placeholder":
            node = g.placeholder(n["name"])
        elif op == "output":
            decoded = _decode_arg(n["args"], name_to_node)
            node = g.output(_unwrap_output_arg(decoded))
        elif op == "call_function":
            target = _deserialize_target(n["target"])
            args = _decode_arg(n["args"], name_to_node)
            kwargs = _decode_arg(n["kwargs"], name_to_node)
            node = g.call_function(target, tuple(args), kwargs)
        elif op == "call_method":
            target = _deserialize_target(n["target"])
            args = _decode_arg(n["args"], name_to_node)
            kwargs = _decode_arg(n["kwargs"], name_to_node)
            node = g.call_method(target, tuple(args), kwargs)
        elif op == "call_module":
            target = _deserialize_target(n["target"])
            args = _decode_arg(n["args"], name_to_node)
            kwargs = _decode_arg(n["kwargs"], name_to_node)
            node = g.call_module(target, tuple(args), kwargs)
        elif op == "get_attr":
            target = _deserialize_target(n["target"])
            node = g.get_attr(target)
        else:
            raise NotImplementedError(f"op {op} not handled")
        name_to_node[n["name"]] = node

    # Create root module and add submodules before creating GraphModule
    root_module = torch.nn.Module()
    
    # Deserialize and add submodules
    submodules = data.get("submodules", {})
    for module_name, serialized_submodule in submodules.items():
        # Recursively deserialize submodule
        submodule_gm = _deserialize_graphmodule(serialized_submodule)
        # Add as direct child module (handles both simple names and nested paths)
        # For nested paths like "layer.expert_0", we need to create intermediate modules
        parts = module_name.split(".")
        if len(parts) == 1:
            # Simple name, add directly
            root_module.add_module(module_name, submodule_gm)
        else:
            # Nested path, create intermediate modules
            current = root_module
            for part in parts[:-1]:
                if not hasattr(current, part):
                    current.add_module(part, torch.nn.Module())
                current = getattr(current, part)
            current.add_module(parts[-1], submodule_gm)
    
    gm = fx.GraphModule(root_module, g)
    state = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
    gm.load_state_dict(state, strict=False)
    return gm
