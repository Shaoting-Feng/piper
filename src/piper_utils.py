import ray
import torch
import uuid
import inspect
import logging
import json, importlib, operator
import torch.fx as fx
import threading
from collections import defaultdict
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from typing import Any, Optional

LOG_LEVEL = "INFO"

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


"""
Piper thread local storage for tracking Piper actors, stages, and microbatches
"""

class PiperMetadata:
    actors = dict()
    dag = set()
    stage_to_device = dict()
    naive_gradient_sync = False

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
    kind = payload["kind"]

    if kind == "string":
        return payload["value"]

    if kind == "py_func":
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
        for part in payload["path"].split("."):
            obj = getattr(obj, part)
        return obj

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
