import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import ray
import torch.fx as fx
from torch._dynamo.backends.registry import register_backend

from .piper_graph_transform import _profile_and_split_gm, _split_gm_by_stages, split_by_a2a, bucket_stage
from .piper_exec import TaskType
from .piper_utils import LOG_LEVEL, create_logger, piper_metadata

logger = create_logger("piper_backend", LOG_LEVEL)
_ANY_TAG_INDEX = "__ANY_TAG_INDEX__"
_FORWARD_PASS = "F"
_FUSED_BWD_PASS = "B"
_BWD_INPUT_PASS = "BI"
_BWD_WEIGHT_PASS = "BW"
_DEFAULT_STREAM = "default_stream"
_VALID_ORDER_PASSES = frozenset({
    _FORWARD_PASS,
    _FUSED_BWD_PASS,
    _BWD_INPUT_PASS,
    _BWD_WEIGHT_PASS,
})


def _is_backward_activation_subkind(subkind: str | None) -> bool:
    return subkind in ("BWD", "BWD_I")


def _is_backward_weight_subkind(subkind: str | None) -> bool:
    return subkind in ("BWD", "BWD_W")


def _is_backward_compute_subkind(subkind: str | None) -> bool:
    return subkind in ("BWD", "BWD_I", "BWD_W")

def _collect_triton_constant_args(gm: fx.GraphModule) -> dict[int, Any]:
    try:
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
    except ImportError:
        return {}
    out: dict[int, Any] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target_str = getattr(node.target, "__name__", "") or str(node.target)
        if "triton_kernel_wrapper" not in target_str:
            continue
        idx = node.kwargs.get("constant_args_idx")
        if idx is not None and idx in kernel_side_table.constant_args:
            out[int(idx)] = kernel_side_table.constant_args[idx]
    return out


@dataclass(frozen=True)
class TrainingDAGEdge:
    src_uid: str
    dst_uid: str
    dep_kind: str  # "data" | "temporal"
    tensor_name: str | None = None


@dataclass
class TrainingDAGNode:
    uid: str
    node_kind: str  # "COMPUTE" | "SEND_COMM" | "RECV_COMM" | "REDUCE_COMM" | "ALL_GATHER_COMM" | "REDUCE_SCATTER_COMM" | "A2A_COMM" | "ORDER_DUMMY"
    compute_subkind: str | None  # "FWD" | "BWD" | "BWD_I" | "BWD_W" when node_kind == "COMPUTE"
    tag: dict[str, int | None]
    device: list[int] | None
    stream: str
    node_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDAG:
    nodes: dict[str, TrainingDAGNode] = field(default_factory=dict)
    edges: list[TrainingDAGEdge] = field(default_factory=list)
    succs: dict[str, set[str]] = field(default_factory=dict)
    preds: dict[str, set[str]] = field(default_factory=dict)
    edge_keys: set[tuple[str, str, str, str | None]] = field(default_factory=set)

    def add_node(self, node: TrainingDAGNode) -> None:
        if node.uid in self.nodes:
            raise ValueError(f"Duplicate node uid: {node.uid}")
        self.nodes[node.uid] = node
        self.succs[node.uid] = set()
        self.preds[node.uid] = set()

    def add_edge(self, edge: TrainingDAGEdge) -> None:
        if edge.src_uid not in self.nodes or edge.dst_uid not in self.nodes:
            raise ValueError(f"Edge references unknown node: {edge}")
        edge_key = (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
        if edge_key in self.edge_keys:
            return
        self.edge_keys.add(edge_key)
        self.edges.append(edge)
        self.succs[edge.src_uid].add(edge.dst_uid)
        self.preds[edge.dst_uid].add(edge.src_uid)



def _flatten_output_nodes(out_arg: Any) -> list[fx.Node]:
    out_nodes: list[fx.Node] = []

    def _walk(v: Any) -> None:
        if isinstance(v, fx.Node):
            out_nodes.append(v)
        elif isinstance(v, (tuple, list)):
            for x in v:
                _walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)

    _walk(out_arg)
    return out_nodes



def _node_io_names(seg_gm: fx.GraphModule) -> tuple[list[str], list[str]]:
    inputs = [n.name for n in seg_gm.graph.nodes if n.op == "placeholder"]
    out_node = next((n for n in seg_gm.graph.nodes if n.op == "output"), None)
    if out_node is None:
        return inputs, []
    outputs = [n.name for n in _flatten_output_nodes(out_node.args[0])]
    return inputs, outputs


def _infer_ep_idx(segment_idx: int) -> int | None:
    """Heuristic: odd A2A segments are expert regions, even segments are non-expert."""
    if segment_idx % 2 == 0:
        return None
    return segment_idx // 2


def _format_tag(tag: dict[str, int | None]) -> str:
    if not tag:
        return "(no-tags)"
    keys = sorted(tag.keys(), key=lambda k: (k != "PP", k != "EP", k))
    return ", ".join(f"{k}={tag[k]}" for k in keys)


def _with_pass_tag(tag: dict[str, Any], pass_value: str | None) -> dict[str, Any]:
    t = dict(tag)
    t["PASS"] = pass_value
    return t


def _format_device(device: list[int] | None) -> str:
    return "None" if device is None else str(list(device))


def _debug_render_training_dag(training_dag: TrainingDAG, output_path: str = "out/training_dag") -> None:
    """Render TrainingDAG with node labels as tags and data-dependency edges."""
    try:
        import graphviz
    except ImportError:
        logger.warning("graphviz Python package not installed; skipping TrainingDAG render")
        return

    dot = graphviz.Digraph("TrainingDAG", comment="Training DAG (tag-labelled)")
    dot.attr(rankdir="LR")
    topo_levels = _topological_levels(training_dag)

    def _format_node_meta_lines(node: TrainingDAGNode) -> str:
        keys = (
            "bucket_key",
            "zero_alloc_full_grads_before",
            "zero_free_full_params_after",
        )
        lines = [
            f"{key}={node.node_meta[key]!r}"
            for key in keys
            if key in node.node_meta
        ]
        return "\\n".join(lines)

    for uid, node in training_dag.nodes.items():
        topo_label = f"topo={topo_levels.get(uid, '?')}"
        meta_label = _format_node_meta_lines(node)
        meta_suffix = f"\\n{meta_label}" if meta_label else ""
        if node.node_kind in (
            "SEND_COMM",
            "RECV_COMM",
            "REDUCE_COMM",
            "ALL_GATHER_COMM",
            "REDUCE_SCATTER_COMM",
            "A2A_COMM",
        ):
            label = (
                f"{topo_label}\\n{node.node_kind}\\n{_format_tag(node.tag)}\\n"
                f"{_format_device(node.device)}\\nstream={node.stream}"
                f"{meta_suffix}"
            )
            shape = "box"
        else:
            subkind = node.compute_subkind or "COMPUTE"
            label = (
                f"{topo_label}\\n{subkind}\\n{_format_tag(node.tag)}\\n{_format_device(node.device)}\\nstream={node.stream}"
                f"{meta_suffix}"
            )
            shape = "ellipse"
        dot.node(uid, label=label, shape=shape)

    nodes_by_level: dict[int, list[str]] = {}
    for uid, level in topo_levels.items():
        nodes_by_level.setdefault(level, []).append(uid)
    for level, uids in sorted(nodes_by_level.items()):
        with dot.subgraph(name=f"topo_{level}") as sub:
            sub.attr(rank="same")
            for uid in sorted(uids):
                sub.node(uid)
    level_reps = [
        sorted(uids)[0]
        for _level, uids in sorted(nodes_by_level.items())
        if uids
    ]
    for src_uid, dst_uid in zip(level_reps, level_reps[1:]):
        dot.edge(src_uid, dst_uid, style="invis", weight="100")

    for edge in training_dag.edges:
        if edge.dep_kind == "data":
            dot.edge(edge.src_uid, edge.dst_uid)
        elif edge.dep_kind == "temporal":
            dot.edge(edge.src_uid, edge.dst_uid, style="dashed", color="blue")

    out = dot.render(output_path, format="png", cleanup=True)
    logger.info("TrainingDAG debug graph saved to %s", out)


def _log_training_dag_dependencies(training_dag: TrainingDAG) -> None:
    """Log predecessor/successor tags for each TrainingDAG node."""
    def _sort_key(item: tuple[str, TrainingDAGNode]) -> tuple[int, int]:
        _uid, node = item
        if node.node_kind == "COMPUTE":
            return (int(node.node_meta.get("stage_id", 10**9)), int(node.node_meta.get("segment_id", 10**9)))
        return (10**9, 10**9)

    for uid, node in sorted(training_dag.nodes.items(), key=_sort_key):
        pred_tags = [
            _format_tag(training_dag.nodes[p].tag)
            for p in sorted(training_dag.preds.get(uid, []), key=lambda x: x)
        ]
        succ_tags = [
            _format_tag(training_dag.nodes[s].tag)
            for s in sorted(training_dag.succs.get(uid, []), key=lambda x: x)
        ]
        logger.debug(
            "TrainingDAG node=%s kind=%s subkind=%s tag=(%s) device=%s stream=%s preds=[%s] succs=[%s]",
            uid,
            node.node_kind,
            node.compute_subkind,
            _format_tag(node.tag),
            _format_device(node.device),
            node.stream,
            ", ".join(pred_tags),
            ", ".join(succ_tags),
        )


def _training_dag_task_type(node: TrainingDAGNode) -> TaskType:
    if node.node_kind == "COMPUTE":
        if node.compute_subkind == "FWD":
            return TaskType.FWD
        if node.compute_subkind == "BWD":
            return TaskType.BWD
        if node.compute_subkind == "BWD_I":
            return TaskType.BWD_I
        if node.compute_subkind == "BWD_W":
            return TaskType.BWD_W
    if node.node_kind == "UPD":
        return TaskType.UPD
    if node.node_kind == "SEND_COMM":
        return TaskType.SEND
    if node.node_kind == "RECV_COMM":
        return TaskType.RECV
    if node.node_kind == "ALL_GATHER_COMM":
        return TaskType.ALL_GATHER
    if node.node_kind == "REDUCE_SCATTER_COMM":
        return TaskType.REDUCE_SCATTER
    if node.node_kind == "REDUCE_COMM":
        return TaskType.ALL_REDUCE
    if node.node_kind == "A2A_COMM":
        return TaskType.FWD_A2A if node.tag.get("PASS") == "F" else TaskType.BWD_A2A
    if node.node_kind == "ORDER_DUMMY":
        return TaskType.ORDER_DUMMY
    return TaskType.FWD


def print_training_dag_order(
    dag: TrainingDAG,
    label: str = "",
    rank: int = 0,
    out_dir: str = "out",
) -> None:
    """Write and log the actor dispatch order for a TrainingDAG."""
    topo_levels = _topological_levels(dag)
    topo = _serial_topological_order(dag, topo_levels)
    data_preds: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    data_succs: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    temporal_preds: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    temporal_succs: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    for e in dag.edges:
        if e.dep_kind == "data":
            data_preds[e.dst_uid].append(e.src_uid)
            data_succs[e.src_uid].append(e.dst_uid)
        else:
            temporal_preds[e.dst_uid].append(e.src_uid)
            temporal_succs[e.src_uid].append(e.dst_uid)

    header = f"--- TrainingDAG execution order{': ' + label if label else ''} ---"
    lines = [header]
    for uid in topo:
        node = dag.nodes[uid]
        task_type = _training_dag_task_type(node)
        meta = node.node_meta
        extra = []
        for key in ("bucket_key", "fwd_uid", "peer_pp_rank", "from_uid", "to_uid"):
            if key in meta and meta[key] is not None:
                extra.append(f"{key}={meta[key]}")
        extra_str = ("  " + " ".join(extra)) if extra else ""
        line = (
            f"  topo={topo_levels[uid]:<3d}  "
            f"{task_type.value:<14s}  kind={node.node_kind:<19s}  uid={node.uid:<24s}  "
            f"tag=({_format_tag(node.tag)})  stream={node.stream}{extra_str}  "
            f"data_preds={sorted(data_preds[node.uid])} data_succs={sorted(data_succs[node.uid])}  "
            f"temp_preds={sorted(temporal_preds[node.uid])} temp_succs={sorted(temporal_succs[node.uid])}"
        )
        lines.append(line)

    lines.append("-" * len(header))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"training_dag_order_rank{rank}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("TrainingDAG execution order saved to %s", out_path)


def build_training_dag(
    stage_submodules: list[tuple[int, fx.GraphModule, list[int], list[int], list, list]],
    *,
    enable_ep: bool,
    dp_degree: int,
    stage_tag_name: str = "PP",
) -> TrainingDAG:
    """Build a TrainingDAG where each node is one stage/a2a segment GraphModule."""
    dag = TrainingDAG()

    global_prev_uid: str | None = None

    for stage_id, stage_gm, input_idxs, param_idxs, graphargs, _placeholders in stage_submodules:
        if enable_ep and dp_degree > 1:
            segments, boundary_infos = split_by_a2a(stage_gm, graphargs, input_idxs, param_idxs)
        else:
            segments = [(stage_gm, input_idxs, param_idxs, graphargs)]
            boundary_infos = []

        for seg_id, (seg_gm, seg_in, seg_param, seg_args) in enumerate(segments):
            uid = f"s{stage_id}.seg{seg_id}"
            tag: dict[str, int | None] = {stage_tag_name: stage_id}
            if seg_id % 2 == 1:
                boundary_name = None
                if seg_id - 1 < len(boundary_infos):
                    boundary_name = boundary_infos[seg_id - 1].get("name")
                if boundary_name:
                    tag[boundary_name] = _infer_ep_idx(seg_id)
            tag = _with_pass_tag(tag, "F")
            in_names, out_names = _node_io_names(seg_gm)
            boundary_after = boundary_infos[seg_id] if seg_id < len(boundary_infos) else None
            node = TrainingDAGNode(
                uid=uid,
                node_kind="COMPUTE",
                compute_subkind="FWD",
                tag=tag,
                device=None,
                stream="default_stream",
                node_meta={
                    "bucket_key": uid,
                    "stage_id": stage_id,
                    "segment_id": seg_id,
                    "gm": seg_gm,
                    "input_idxs": list(seg_in),
                    "param_idxs": list(seg_param),
                    "graphargs": list(seg_args),
                    "input_names": in_names,
                    "output_names": out_names,
                    "a2a_boundary_after": boundary_after,
                    "triton_constant_args": _collect_triton_constant_args(seg_gm),
                },
            )
            dag.add_node(node)

            # Explicit activation-flow ordering in stage/segment order.
            if global_prev_uid is not None:
                dag.add_edge(TrainingDAGEdge(src_uid=global_prev_uid, dst_uid=uid, dep_kind="data"))
            global_prev_uid = uid

    # Build backward-pass nodes and dependencies by reversing forward data edges.
    fwd_compute_uids = [
        uid for uid, node in dag.nodes.items()
        if node.node_kind == "COMPUTE" and node.compute_subkind == "FWD"
    ]
    fwd_uid_to_bwd_uid: dict[str, str] = {}
    for fwd_uid in fwd_compute_uids:
        fwd_node = dag.nodes[fwd_uid]
        bwd_uid = f"{fwd_uid}.bwd"
        fwd_uid_to_bwd_uid[fwd_uid] = bwd_uid
        dag.add_node(
            TrainingDAGNode(
                uid=bwd_uid,
                node_kind="COMPUTE",
                compute_subkind="BWD",
                tag=_with_pass_tag(fwd_node.tag, "B"),
                device=fwd_node.device,
                stream="default_stream",
                node_meta={
                    "fwd_uid": fwd_uid,
                    "bucket_key": fwd_node.node_meta.get("bucket_key", fwd_uid),
                    "compute_loss": False,
                },
            )
        )

    fwd_edges = [
        e for e in dag.edges
        if e.dep_kind == "data"
        and dag.nodes[e.src_uid].node_kind == "COMPUTE"
        and dag.nodes[e.dst_uid].node_kind == "COMPUTE"
        and dag.nodes[e.src_uid].compute_subkind == "FWD"
        and dag.nodes[e.dst_uid].compute_subkind == "FWD"
    ]

    # Reverse every forward dependency for the backward pass:
    # FWD:  u -> v    ==>    BWD: v' -> u'
    for e in fwd_edges:
        bwd_src = fwd_uid_to_bwd_uid[e.dst_uid]
        bwd_dst = fwd_uid_to_bwd_uid[e.src_uid]
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=bwd_src,
                dst_uid=bwd_dst,
                dep_kind="data",
                tensor_name=e.tensor_name,
            )
        )

    # Bridge from forward phase into backward phase:
    # last_fwd -> first_bwd (where first_bwd is the BWD node paired with last_fwd).
    if fwd_compute_uids:
        def _fwd_sort_key(uid: str) -> tuple[int, int]:
            n = dag.nodes[uid]
            return (
                int(n.node_meta.get("stage_id", -1)),
                int(n.node_meta.get("segment_id", -1)),
            )

        last_fwd_uid = max(fwd_compute_uids, key=_fwd_sort_key)
        first_bwd_uid = fwd_uid_to_bwd_uid[last_fwd_uid]
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=last_fwd_uid,
                dst_uid=first_bwd_uid,
                dep_kind="data",
                tensor_name=None,
            )
        )

        # Mark loss computation point: first BWD node (earliest in BWD execution).
        if first_bwd_uid in dag.nodes:
            dag.nodes[first_bwd_uid].node_meta["compute_loss"] = True

    # Add explicit UPD node with temporal dependency from every BWD compute node.
    # UPD participates in the compute temporal chain and is not scheduled from
    # data edges.
    upd_uid = "upd.0"
    dag.add_node(
        TrainingDAGNode(
            uid=upd_uid,
            node_kind="UPD",
            compute_subkind=None,
            tag={"PASS": None},
            device=None,
            stream="default_stream",
            node_meta={},
        )
    )
    for uid, node in list(dag.nodes.items()):
        if node.node_kind == "COMPUTE" and node.compute_subkind == "BWD":
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=uid,
                    dst_uid=upd_uid,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )

    return dag


def _match_filter(tag: dict[str, int | None], flt: dict[str, Any]) -> bool:
    for k, v in flt.items():
        if k not in tag:
            return False
        if v == _ANY_TAG_INDEX:
            # "*" wildcard matches any concrete index for this tag, but not None.
            if tag[k] is None:
                return False
        elif tag[k] != v:
            return False
    return True


def _apply_split_backward_stencil(
    dag: TrainingDAG,
    split_base_keys: set[tuple[tuple[str, Any], ...]],
) -> None:
    if not split_base_keys:
        return
    split_filters = [dict(key) for key in split_base_keys]
    for node in list(dag.nodes.values()):
        if (
            node.node_kind != "COMPUTE"
            or node.compute_subkind != "BWD"
            or not any(_match_filter(node.tag, flt) for flt in split_filters)
        ):
            continue
        _split_fused_bwd_node(dag, node.uid)


def _split_fused_bwd_node(dag: TrainingDAG, bwd_uid: str) -> None:
    bwd_i = dag.nodes[bwd_uid]
    if bwd_i.node_kind != "COMPUTE" or bwd_i.compute_subkind != "BWD":
        raise ValueError(f"expected fused BWD compute node, got {bwd_uid}: {bwd_i}")
    bwd_w_uid = f"{bwd_uid}.bw"
    if bwd_w_uid in dag.nodes:
        raise ValueError(f"duplicate BWD_W uid while splitting {bwd_uid}: {bwd_w_uid}")

    bwd_i.compute_subkind = "BWD_I"
    bwd_i.tag = _with_pass_tag(bwd_i.tag, _BWD_INPUT_PASS)

    bwd_w_meta = dict(bwd_i.node_meta)
    bwd_w_meta["compute_loss"] = False
    dag.add_node(
        TrainingDAGNode(
            uid=bwd_w_uid,
            node_kind="COMPUTE",
            compute_subkind="BWD_W",
            tag=_with_pass_tag(bwd_i.tag, _BWD_WEIGHT_PASS),
            device=list(bwd_i.device) if bwd_i.device is not None else None,
            stream=bwd_i.stream,
            node_meta=bwd_w_meta,
        )
    )

    for edge in [
        e for e in list(dag.edges)
        if e.dep_kind == "temporal"
        and e.src_uid == bwd_uid
        and dag.nodes[e.dst_uid].node_kind == "UPD"
    ]:
        _remove_edge(dag, edge)
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=bwd_w_uid,
                dst_uid=edge.dst_uid,
                dep_kind="temporal",
                tensor_name=edge.tensor_name,
            )
        )

    dag.add_edge(
        TrainingDAGEdge(
            src_uid=bwd_uid,
            dst_uid=bwd_w_uid,
            dep_kind="data",
            tensor_name=None,
        )
    )


def _normalize_filter_devices_directive(
    directive: Any
) -> tuple[str, list[dict[str, Any]], list[int], str | None, str | None, str | None, bool, bool, int | None]:
    if isinstance(directive, dict):
        op = directive.get("op")
        if op not in ("place", "replicate", "shard"):
            raise ValueError(f"Unsupported directive op: {op}")
        if "filter" not in directive:
            raise ValueError(
                f"{op} directive requires 'filter' (singular), not 'filters': {directive}"
            )
        if "filters" in directive:
            raise ValueError(
                f"{op} directive uses deprecated 'filters'; use 'filter' (singular): {directive}"
            )
        filter_spec = directive.get("filter")
        devices = directive.get("devices", directive.get("device"))
        stream = directive.get("stream")
        gather_stream = directive.get("gather_stream")
        reduce_stream = directive.get("reduce_stream")
        if op == "replicate":
            if stream is not None:
                raise ValueError(
                    f"replicate directive does not accept 'stream'; use 'gather_stream' and/or 'reduce_stream': {directive}"
                )
        else:
            if gather_stream is not None or reduce_stream is not None:
                raise ValueError(
                    f"{op} directive does not accept 'gather_stream'/'reduce_stream'; use 'stream': {directive}"
                )
        shard_params = bool(directive.get("shard_params", False))
        shard_grads = bool(directive.get("shard_grads", False))
        bucket_size_raw = directive.get("bucket_size", None)
        bucket_size = None if bucket_size_raw is None else int(bucket_size_raw)
        if not isinstance(devices, list) or not devices:
            raise ValueError(f"place directive requires non-empty devices list: {directive}")
        n = _normalize_filter_spec(filter_spec, directive)
        return (
            op,
            [n],
            [int(d) for d in devices],
            (None if stream is None else str(stream)),
            (None if gather_stream is None else str(gather_stream)),
            (None if reduce_stream is None else str(reduce_stream)),
            shard_params,
            shard_grads,
            bucket_size,
        )

    raise ValueError(f"Schedule directives must be JSON objects, got: {type(directive)}")


def _normalize_filter_spec(filter_spec: Any, directive: Any) -> dict[str, Any]:
    """Normalize a single filter spec into {tag_name: value}."""
    def _norm_value(k: str, v: Any) -> Any:
        if k == "PASS":
            if not isinstance(v, str):
                raise ValueError(
                    f"'PASS' filter value must be a string in {{'F','B','BI','BW'}}: {directive}"
                )
            if v not in ("F", "B", "BI", "BW"):
                raise ValueError(
                    f"Invalid 'PASS' filter value '{v}'. Allowed: 'F', 'B', 'BI', 'BW': {directive}"
                )
            return v
        if v == "*":
            return _ANY_TAG_INDEX
        if v is None:
            return None
        if isinstance(v, int):
            return int(v)
        if isinstance(v, str):
            # Keep pass-like symbolic tags (e.g., "FWD"/"BWD") as strings.
            try:
                return int(v)
            except ValueError:
                return v
        return v

    if isinstance(filter_spec, dict):
        out: dict[str, Any] = {}
        for k, v in filter_spec.items():
            if not isinstance(k, str):
                raise ValueError(f"Filter keys must be strings: {directive}")
            out[k] = _norm_value(k, v)
        return out
    if isinstance(filter_spec, list) and all(
        isinstance(item, (list, tuple)) and len(item) == 2
        for item in filter_spec
    ):
        out: dict[str, Any] = {}
        for k, v in filter_spec:
            if not isinstance(k, str):
                raise ValueError(f"Filter tuple tag must be a string: {directive}")
            out[k] = _norm_value(k, v)
        return out
    raise ValueError(
        f"Unsupported filter format. Expected dict or list[tuple[str, value]]: {directive}"
    )


def _iter_filter_specs(spec: Any):
    if isinstance(spec, dict):
        yield spec
        return
    if isinstance(spec, list):
        if all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            for item in spec
        ):
            yield spec
            return
        for item in spec:
            yield from _iter_filter_specs(item)


def _directive_mentions_tag(directive: dict[str, Any], tag_name: str) -> bool:
    specs = []
    if "filter" in directive:
        specs.append(directive["filter"])
    if isinstance(directive.get("filters"), list):
        specs.append(directive["filters"])
    for spec in _iter_filter_specs(specs):
        if isinstance(spec, dict) and tag_name in spec:
            return True
        if isinstance(spec, list):
            for item in spec:
                if isinstance(item, tuple) and len(item) == 2 and item[0] == tag_name:
                    return True
                if isinstance(item, list) and len(item) == 2 and item[0] == tag_name:
                    return True
    return False


def _directives_enable_ep(directives: Any) -> bool:
    if not isinstance(directives, list):
        return False
    return any(
        isinstance(directive, dict)
        and (directive.get("op") == "shard" or _directive_mentions_tag(directive, "EP"))
        for directive in directives
    )


def _normalize_split_directive(directive: Any) -> tuple[dict[str, Any], str, int]:
    if not isinstance(directive, dict):
        raise ValueError(f"split directive must be a dict: {directive}")
    if directive.get("op") != "split":
        raise ValueError(f"Unsupported split directive op: {directive}")
    if "filter" not in directive:
        raise ValueError(f"split directive requires 'filter': {directive}")
    dim_name = directive.get("dim_name")
    if not isinstance(dim_name, str) or not dim_name:
        raise ValueError(f"split directive requires non-empty string dim_name: {directive}")
    num_microbatches = int(directive.get("num_microbatches", 0))
    if num_microbatches <= 0:
        raise ValueError(f"split directive requires num_microbatches > 0: {directive}")
    flt = _normalize_filter_spec(directive["filter"], directive)
    return flt, dim_name, num_microbatches


def _is_filter_spec(spec: Any) -> bool:
    if isinstance(spec, dict):
        return True
    if isinstance(spec, list):
        return all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            for item in spec
        )
    return False


def _normalize_order_directive(directive: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(directive, dict):
        raise ValueError(f"order directive must be a dict: {directive}")
    if directive.get("op") != "order":
        raise ValueError(f"Unsupported order directive op: {directive}")
    raw_filters = directive.get("filters")
    if not isinstance(raw_filters, list) or len(raw_filters) < 2:
        raise ValueError(
            f"order directive requires filters list with at least 2 groups: {directive}"
        )

    filter_groups: list[list[dict[str, Any]]] = []
    for group_idx, raw_group in enumerate(raw_filters):
        if _is_filter_spec(raw_group):
            raw_group_items = [raw_group]
        elif isinstance(raw_group, list) and raw_group:
            raw_group_items = raw_group
        else:
            raise ValueError(
                f"order directive filter group[{group_idx}] must be a non-empty list "
                f"of filters: {directive}"
            )

        group = []
        for raw_filter in raw_group_items:
            if not _is_filter_spec(raw_filter):
                raise ValueError(
                    f"order directive filter group[{group_idx}] contains invalid "
                    f"filter spec: {raw_filter}"
                )
            flt = _normalize_filter_spec(raw_filter, directive)
            pass_value = flt.get("PASS")
            if pass_value is not None and pass_value not in _VALID_ORDER_PASSES:
                raise ValueError(
                    f"order directive has unsupported pass={pass_value!r}; "
                    f"expected one of {sorted(_VALID_ORDER_PASSES)}"
                )
            group.append(flt)
        filter_groups.append(group)

    return filter_groups


def _filter_key_without(flt: dict[str, Any], ignored_keys: set[str]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in flt.items() if k not in ignored_keys))


def _validate_split_backward_order_stencil(
    directives: list[Any],
) -> set[tuple[tuple[str, Any], ...]]:
    """Validate BI/BW order entries and return base filter keys to split.

    This is intentionally narrow: it validates only the current split-backward
    stencil until the schedule JSON grows a full schema validation pass.
    """
    split_keys: set[tuple[tuple[str, Any], ...]] = set()
    split_entries: dict[tuple[tuple[str, Any], ...], list[tuple[int, tuple[int, int], str]]] = {}

    for directive_idx, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "order":
            continue
        filter_groups = _normalize_order_directive(raw)
        seen_in_row: dict[tuple[tuple[str, Any], ...], dict[str, tuple[int, int]]] = {}
        for group_idx, group in enumerate(filter_groups):
            for filter_idx, flt in enumerate(group):
                filter_pos = (group_idx, filter_idx)
                pass_value = flt.get("PASS")
                if pass_value is None:
                    continue
                if pass_value == _FUSED_BWD_PASS:
                    continue
                if pass_value not in (_BWD_INPUT_PASS, _BWD_WEIGHT_PASS):
                    continue
                key = _filter_key_without(flt, {"PASS"})
                by_pass = seen_in_row.setdefault(key, {})
                if pass_value in by_pass:
                    raise ValueError(
                        f"order directive[{directive_idx}] has duplicate split-backward "
                        f"{pass_value} entry for {dict(key)} at indices "
                        f"{by_pass[pass_value]} and {filter_pos}"
                    )
                by_pass[pass_value] = filter_pos
                split_keys.add(key)
                split_entries.setdefault(key, []).append((directive_idx, filter_pos, pass_value))

    for key, entries in sorted(split_entries.items()):
        bi = [entry for entry in entries if entry[2] == _BWD_INPUT_PASS]
        bw = [entry for entry in entries if entry[2] == _BWD_WEIGHT_PASS]
        if len(bi) != 1 or len(bw) != 1:
            raise ValueError(
                f"split backward order entries for {dict(key)} must contain exactly "
                f"one BI and one BW; got BI={len(bi)} BW={len(bw)}"
            )
        if bi[0][0] != bw[0][0]:
            raise ValueError(
                f"split backward order entries for {dict(key)} must appear in the "
                "same order directive row"
            )
        if bi[0][1] > bw[0][1]:
            raise ValueError(
                f"split backward order entries for {dict(key)} must place BI before BW"
            )

    return split_keys


def _iter_data_edges(dag: TrainingDAG) -> list[TrainingDAGEdge]:
    return [e for e in dag.edges if e.dep_kind == "data"]


def _remove_edge(dag: TrainingDAG, edge: TrainingDAGEdge) -> None:
    edge_key = (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
    if edge_key not in dag.edge_keys:
        return
    dag.edge_keys.remove(edge_key)
    dag.edges = [
        e for e in dag.edges
        if not (
            e.src_uid == edge.src_uid
            and e.dst_uid == edge.dst_uid
            and e.dep_kind == edge.dep_kind
            and e.tensor_name == edge.tensor_name
        )
    ]
    if edge.src_uid in dag.succs:
        dag.succs[edge.src_uid].discard(edge.dst_uid)
    if edge.dst_uid in dag.preds:
        dag.preds[edge.dst_uid].discard(edge.src_uid)


def _device_to_physical_pp_rank(dag: TrainingDAG) -> dict[tuple[int, ...], int]:
    device_keys = sorted({
        tuple(sorted(node.device))
        for node in dag.nodes.values()
        if node.device is not None
    })
    return {key: idx for idx, key in enumerate(device_keys)}


def _insert_send_recv_comm_nodes(dag: TrainingDAG, comm_stream: str | None = None) -> None:
    data_edges = _iter_data_edges(dag)
    device_to_pp_rank = _device_to_physical_pp_rank(dag)
    send_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "SEND_COMM")
    recv_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "RECV_COMM")
    for edge in data_edges:
        src = dag.nodes[edge.src_uid]
        dst = dag.nodes[edge.dst_uid]
        if src.node_kind != "COMPUTE" or dst.node_kind != "COMPUTE":
            continue
        if src.device is None or dst.device is None:
            raise ValueError(f"cannot insert send/recv for unplaced edge {src.uid} -> {dst.uid}")
        if src.device == dst.device:
            continue
        src_device_key = tuple(sorted(src.device))
        dst_device_key = tuple(sorted(dst.device))
        send_uid = f"send.{send_idx}"
        send_idx += 1
        recv_uid = f"recv.{recv_idx}"
        recv_idx += 1
        stream_name = comm_stream if comm_stream is not None else "pp_stream"
        send_node = TrainingDAGNode(
            uid=send_uid,
            node_kind="SEND_COMM",
            compute_subkind=None,
            tag=dict(src.tag),
            device=(None if src.device is None else list(src.device)),
            stream=stream_name,
            node_meta={
                "from_uid": src.uid,
                "to_uid": dst.uid,
                "peer_pp_rank": device_to_pp_rank[dst_device_key],
                "bucket_key": src.node_meta.get("bucket_key"),
            },
        )
        recv_node = TrainingDAGNode(
            uid=recv_uid,
            node_kind="RECV_COMM",
            compute_subkind=None,
            tag=dict(dst.tag),
            device=(None if dst.device is None else list(dst.device)),
            stream=stream_name,
            node_meta={
                "from_uid": src.uid,
                "to_uid": dst.uid,
                "peer_pp_rank": device_to_pp_rank[src_device_key],
                "bucket_key": dst.node_meta.get("bucket_key"),
            },
        )
        dag.add_node(send_node)
        dag.add_node(recv_node)
        _remove_edge(dag, edge)
        dag.add_edge(TrainingDAGEdge(src_uid=src.uid, dst_uid=send_uid, dep_kind="data", tensor_name=edge.tensor_name))
        dag.add_edge(TrainingDAGEdge(src_uid=recv_uid, dst_uid=dst.uid, dep_kind="data", tensor_name=edge.tensor_name))


def _replicate_update_nodes_by_device(dag: TrainingDAG) -> None:
    """Replicate UPD nodes per upstream device set so split components stay homogeneous."""
    upd_nodes = [n for n in list(dag.nodes.values()) if n.node_kind == "UPD" and ".rep" not in n.uid]
    for upd in upd_nodes:
        in_edges = [e for e in list(dag.edges) if e.dep_kind == "temporal" and e.dst_uid == upd.uid]
        by_dev: dict[tuple[int, ...], list[TrainingDAGEdge]] = {}
        for e in in_edges:
            src = dag.nodes[e.src_uid]
            if src.device is None:
                continue
            key = tuple(sorted(src.device))
            by_dev.setdefault(key, []).append(e)
        if len(by_dev) <= 1:
            if len(by_dev) == 1:
                upd.device = list(next(iter(by_dev.keys())))
            continue
        created_uids: list[str] = []
        for i_upd, (dev_key, dev_edges) in enumerate(sorted(by_dev.items(), key=lambda kv: kv[0])):
            nu = upd.uid if i_upd == 0 else f"{upd.uid}.rep{i_upd}"
            if i_upd == 0:
                upd.device = list(dev_key)
            else:
                dag.add_node(
                    TrainingDAGNode(
                        uid=nu,
                        node_kind="UPD",
                        compute_subkind=None,
                        tag=dict(upd.tag),
                        device=list(dev_key),
                        stream=upd.stream,
                        node_meta=dict(upd.node_meta),
                    )
                )
            created_uids.append(nu)
            for e in dev_edges:
                _remove_edge(dag, e)
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=e.src_uid,
                        dst_uid=nu,
                        dep_kind=e.dep_kind,
                        tensor_name=e.tensor_name,
                    )
                )
        # Duplicate outgoing edges of the original UPD to all replicas.
        out_edges = [e for e in list(dag.edges) if e.dep_kind == "temporal" and e.src_uid == upd.uid]
        if out_edges:
            for e in out_edges:
                for nu in created_uids[1:]:
                    dag.add_edge(
                        TrainingDAGEdge(
                            src_uid=nu,
                            dst_uid=e.dst_uid,
                            dep_kind=e.dep_kind,
                            tensor_name=e.tensor_name,
                        )
                    )


def _remove_node_and_incident_edges(dag: TrainingDAG, uid: str) -> None:
    incident = [e for e in dag.edges if e.src_uid == uid or e.dst_uid == uid]
    for e in incident:
        _remove_edge(dag, e)
    dag.nodes.pop(uid, None)
    dag.succs.pop(uid, None)
    dag.preds.pop(uid, None)
    for s in dag.succs.values():
        s.discard(uid)
    for p in dag.preds.values():
        p.discard(uid)


def _rewire_bwd_successors_through_sync(
    dag: TrainingDAG,
    bwd_uid: str,
    sync_uid: str,
) -> None:
    """Add update ordering through a gradient sync comm node.

    BWD data successors carry activation gradients, including P2P sends to an
    upstream pipeline stage.  Gradient synchronization is for parameter grads,
    so it must not be inserted into those data paths.  UPD dependencies are
    temporal; add sync -> UPD temporal edges so the
    update waits for gradient synchronization as well as the BWD compute.
    """
    upd_temporal_outs = [
        e for e in list(dag.edges)
        if e.dep_kind == "temporal"
        and e.src_uid == bwd_uid
        and dag.nodes[e.dst_uid].node_kind == "UPD"
    ]
    for e in upd_temporal_outs:
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=sync_uid,
                dst_uid=e.dst_uid,
                dep_kind="temporal",
                tensor_name=None,
            )
        )


def _bucket_matched_fwd_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    bucket_size_mb: int,
) -> None:
    if bucket_size_mb <= 0:
        raise ValueError(f"bucket_size must be > 0 MB, got {bucket_size_mb}")
    bucket_size_bytes = int(bucket_size_mb) * 1024 * 1024

    fwd_targets = [
        uid for uid, node in dag.nodes.items()
        if node.node_kind == "COMPUTE"
        and node.compute_subkind == "FWD"
        and any(_match_filter(node.tag, flt) for flt in filters)
    ]

    # Deterministic rewrite order.
    fwd_targets.sort(key=lambda uid: (
        int(dag.nodes[uid].node_meta.get("stage_id", 10**9)),
        int(dag.nodes[uid].node_meta.get("segment_id", 10**9)),
    ))

    for fwd_uid in fwd_targets:
        if fwd_uid not in dag.nodes:
            continue
        fwd_node = dag.nodes[fwd_uid]
        bwd_uid = f"{fwd_uid}.bwd"
        bwd_node = dag.nodes.get(bwd_uid)
        if bwd_node is None or bwd_node.node_kind != "COMPUTE" or bwd_node.compute_subkind != "BWD":
            raise ValueError(f"Expected corresponding BWD node for {fwd_uid}, missing {bwd_uid}")

        gm = fwd_node.node_meta.get("gm")
        graphargs = fwd_node.node_meta.get("graphargs")
        input_idxs = fwd_node.node_meta.get("input_idxs")
        param_idxs = fwd_node.node_meta.get("param_idxs")
        if gm is None or graphargs is None or input_idxs is None or param_idxs is None:
            continue

        def _param_bytes(gargs: list, pidxs: list[int]) -> int:
            total = 0
            for i in pidxs:
                if i < 0 or i >= len(gargs):
                    continue
                t = gargs[i]
                if t is not None and hasattr(t, "numel") and hasattr(t, "element_size"):
                    total += int(t.numel()) * int(t.element_size())
            return total

        def _placeholder_names(module: fx.GraphModule) -> list[str]:
            return [n.name for n in module.graph.nodes if n.op == "placeholder"]

        def _param_names(module: fx.GraphModule, pidxs: list[int]) -> list[str]:
            names = _placeholder_names(module)
            return [
                names[i] if 0 <= i < len(names) else f"<invalid-param-idx:{i}>"
                for i in pidxs
            ]

        pre_bytes = _param_bytes(graphargs, param_idxs)
        logger.info(
            "bucket_stage pre-size node=%s stage=%s seg=%s bucket_size_value=%d "
            "bucket_size_bytes_passed=%d bucket_size_mib_passed=%.6f param_bytes=%d",
            fwd_uid,
            fwd_node.node_meta.get("stage_id"),
            fwd_node.node_meta.get("segment_id"),
            int(bucket_size_mb),
            bucket_size_bytes,
            bucket_size_bytes / (1024 * 1024),
            pre_bytes,
        )

        try:
            buckets = bucket_stage(
                gm,
                graphargs,
                input_idxs,
                param_idxs,
                bucket_size_bytes=bucket_size_bytes,
                debug_name=fwd_uid,
            )
        except Exception as exc:
            logger.warning(
                "replicate bucketing skipped for node %s (stage=%s seg=%s): %s",
                fwd_uid,
                fwd_node.node_meta.get("stage_id"),
                fwd_node.node_meta.get("segment_id"),
                exc,
            )
            continue
        if len(buckets) <= 1:
            continue
        original_param_names = set(_param_names(gm, param_idxs))
        lowered_param_counts: Counter[str] = Counter()
        for bi, (_bgm, _bin, b_param, b_args) in enumerate(buckets):
            post_bytes = _param_bytes(b_args, b_param)
            bucket_param_names = _param_names(_bgm, b_param)
            lowered_param_counts.update(bucket_param_names)
            logger.info(
                "bucket_stage bucket-size node=%s bucket=%d/%d param_bytes=%d params=%d names=%s",
                fwd_uid,
                bi,
                len(buckets),
                post_bytes,
                len(bucket_param_names),
                bucket_param_names,
            )
        missing_param_names = sorted(original_param_names - set(lowered_param_counts))
        unknown_param_names = sorted(set(lowered_param_counts) - original_param_names)
        duplicate_param_names = sorted(
            name for name, count in lowered_param_counts.items() if count != 1
        )
        logger.info(
            "bucket_stage lowered-ownership node=%s original_params=%d lowered_param_refs=%d "
            "unique_lowered_params=%d missing_params=%d duplicate_params=%d unknown_params=%d",
            fwd_uid,
            len(original_param_names),
            sum(lowered_param_counts.values()),
            len(lowered_param_counts),
            len(missing_param_names),
            len(duplicate_param_names),
            len(unknown_param_names),
        )
        if missing_param_names or duplicate_param_names or unknown_param_names:
            logger.warning(
                "bucket_stage lowered-ownership-invalid node=%s missing=%s duplicates=%s unknown=%s",
                fwd_uid,
                missing_param_names[:16],
                duplicate_param_names[:16],
                unknown_param_names[:16],
            )

        # Snapshot old incident edges before rewrite.
        in_fwd = [e for e in dag.edges if e.dst_uid == fwd_uid]
        out_fwd = [e for e in dag.edges if e.src_uid == fwd_uid]
        in_bwd = [e for e in dag.edges if e.dst_uid == bwd_uid]
        out_bwd = [e for e in dag.edges if e.src_uid == bwd_uid]

        orig_stage = int(fwd_node.node_meta.get("stage_id", -1))
        orig_seg = int(fwd_node.node_meta.get("segment_id", -1))

        fwd_bucket_uids: list[str] = []
        bwd_bucket_uids: list[str] = []
        for bi, (bgm, b_in, b_param, b_args) in enumerate(buckets):
            f_uid = f"{fwd_uid}.bucket{bi}"
            b_uid = f"{f_uid}.bwd"
            fwd_bucket_uids.append(f_uid)
            bwd_bucket_uids.append(b_uid)
            dag.add_node(
                TrainingDAGNode(
                    uid=f_uid,
                    node_kind="COMPUTE",
                    compute_subkind="FWD",
                    tag=_with_pass_tag(fwd_node.tag, "F"),
                    device=list(fwd_node.device) if fwd_node.device is not None else None,
                    stream=fwd_node.stream,
                    node_meta={
                        "stage_id": orig_stage,
                        "segment_id": orig_seg * 1000 + bi,
                        "gm": bgm,
                        "input_idxs": list(b_in),
                        "param_idxs": list(b_param),
                        "graphargs": list(b_args),
                        "input_names": [n.name for n in bgm.graph.nodes if n.op == "placeholder"],
                        "output_names": [],
                        "a2a_boundary_after": None,
                        "bucket_idx": bi,
                        "num_buckets": len(buckets),
                        "bucket_key": f_uid,
                    },
                )
            )
            dag.add_node(
                TrainingDAGNode(
                    uid=b_uid,
                    node_kind="COMPUTE",
                    compute_subkind="BWD",
                    tag=_with_pass_tag(fwd_node.tag, "B"),
                    device=list(bwd_node.device) if bwd_node.device is not None else None,
                    stream=bwd_node.stream,
                    node_meta={
                        "fwd_uid": f_uid,
                        "bucket_idx": bi,
                        "num_buckets": len(buckets),
                        "bucket_key": f_uid,
                        "compute_loss": bool(bwd_node.node_meta.get("compute_loss", False)) and bi == (len(buckets) - 1),
                    },
                )
            )

        first_fwd = fwd_bucket_uids[0]
        last_fwd = fwd_bucket_uids[-1]
        # Mirror backward bucket chain in reverse order.
        first_bwd = bwd_bucket_uids[-1]
        last_bwd = bwd_bucket_uids[0]

        def _remap(uid: str) -> str:
            if uid == fwd_uid:
                return last_fwd
            if uid == bwd_uid:
                return first_bwd
            return uid

        # Remove originals and their incident edges.
        _remove_node_and_incident_edges(dag, fwd_uid)
        _remove_node_and_incident_edges(dag, bwd_uid)

        # Re-attach incoming/outgoing edges at bucket chain boundaries.
        for e in in_fwd:
            dag.add_edge(TrainingDAGEdge(src_uid=_remap(e.src_uid), dst_uid=first_fwd, dep_kind=e.dep_kind, tensor_name=e.tensor_name))
        for e in out_fwd:
            dst = first_bwd if e.dst_uid == bwd_uid else e.dst_uid
            dag.add_edge(TrainingDAGEdge(src_uid=last_fwd, dst_uid=dst, dep_kind=e.dep_kind, tensor_name=e.tensor_name))
        for e in in_bwd:
            dag.add_edge(TrainingDAGEdge(src_uid=_remap(e.src_uid), dst_uid=first_bwd, dep_kind=e.dep_kind, tensor_name=e.tensor_name))
        for e in out_bwd:
            dag.add_edge(TrainingDAGEdge(src_uid=last_bwd, dst_uid=_remap(e.dst_uid), dep_kind=e.dep_kind, tensor_name=e.tensor_name))

        # Internal chain edges.
        for i in range(len(fwd_bucket_uids) - 1):
            dag.add_edge(TrainingDAGEdge(src_uid=fwd_bucket_uids[i], dst_uid=fwd_bucket_uids[i + 1], dep_kind="data", tensor_name=None))
        for i in range(len(bwd_bucket_uids) - 1, 0, -1):
            dag.add_edge(TrainingDAGEdge(src_uid=bwd_bucket_uids[i], dst_uid=bwd_bucket_uids[i - 1], dep_kind="data", tensor_name=None))


def _insert_reduce_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    reduce_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "REDUCE_COMM")
    expected = sorted(int(d) for d in devices)
    for node in list(dag.nodes.values()):
        if node.node_kind != "COMPUTE" or not _is_backward_weight_subkind(node.compute_subkind):
            continue
        if not any(_match_filter(node.tag, flt) for flt in filters):
            continue
        if node.device is None:
            raise ValueError(
                f"replicate requires placed backward weight-gradient nodes, but node {node.uid} has device=None; "
                f"expected devices={devices}"
            )
        got = sorted(int(d) for d in node.device)
        if got != expected:
            raise ValueError(
                f"replicate device mismatch for node {node.uid}: "
                f"expected devices={devices}, node_devices={node.device}"
            )
        reduce_uid = f"reduce.{reduce_idx}"
        reduce_idx += 1
        reduce_node = TrainingDAGNode(
            uid=reduce_uid,
            node_kind="REDUCE_COMM",
            compute_subkind=None,
            tag=dict(node.tag),
            device=list(node.device),
            stream=comm_stream if comm_stream is not None else "default_stream",
            node_meta={
                "bwd_uid": node.uid,
                "bucket_key": node.node_meta.get("bucket_key"),
            },
        )
        dag.add_node(reduce_node)
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=node.uid,
                dst_uid=reduce_uid,
                dep_kind="data",
                tensor_name=None,
            )
        )
        _rewire_bwd_successors_through_sync(dag, node.uid, reduce_uid)


def _node_has_trainable_params(dag: TrainingDAG, node: TrainingDAGNode) -> bool:
    meta = node.node_meta
    if _is_backward_compute_subkind(node.compute_subkind):
        fwd_uid = meta.get("fwd_uid")
        if isinstance(fwd_uid, str) and fwd_uid in dag.nodes:
            meta = dag.nodes[fwd_uid].node_meta
    graphargs = meta.get("graphargs")
    param_idxs = meta.get("param_idxs")
    if graphargs is None or param_idxs is None:
        return True
    return any(
        0 <= int(i) < len(graphargs)
        and graphargs[int(i)] is not None
        and bool(getattr(graphargs[int(i)], "requires_grad", False))
        for i in param_idxs
    )


def _insert_all_gather_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    ag_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "ALL_GATHER_COMM")
    expected = sorted(int(d) for d in devices)
    for node in list(dag.nodes.values()):
        if (
            node.node_kind != "COMPUTE"
            or node.compute_subkind not in ("FWD", "BWD", "BWD_I", "BWD_W")
        ):
            continue
        if not any(_match_filter(node.tag, flt) for flt in filters):
            continue
        if not _node_has_trainable_params(dag, node):
            continue
        if node.device is None:
            raise ValueError(
                f"replicate(shard_params=True) requires placed nodes, but node {node.uid} has device=None; "
                f"expected devices={devices}"
            )
        got = sorted(int(d) for d in node.device)
        if got != expected:
            raise ValueError(
                f"replicate device mismatch for node {node.uid}: "
                f"expected devices={devices}, node_devices={node.device}"
            )
        ag_uid = f"all_gather.{ag_idx}"
        ag_idx += 1
        ag_tag = dict(node.tag)
        ag_node = TrainingDAGNode(
            uid=ag_uid,
            node_kind="ALL_GATHER_COMM",
            compute_subkind=None,
            tag=ag_tag,
            device=list(node.device),
            stream=comm_stream if comm_stream is not None else "default_stream",
            node_meta={
                "compute_uid": node.uid,
                "bucket_key": node.node_meta.get("bucket_key"),
            },
        )
        dag.add_node(ag_node)
        node.node_meta["zero_free_full_params_after"] = True
        if node.compute_subkind == "BWD_W":
            for pred_uid in sorted(dag.preds.get(node.uid, set())):
                pred = dag.nodes[pred_uid]
                if pred.node_kind == "COMPUTE" and pred.compute_subkind == "BWD_I":
                    dag.add_edge(
                        TrainingDAGEdge(
                            src_uid=pred.uid,
                            dst_uid=ag_uid,
                            dep_kind="data",
                            tensor_name=None,
                        )
                    )
        # ALL_GATHER is a pre-dependency of compute: AG -> COMPUTE
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=ag_uid,
                dst_uid=node.uid,
                dep_kind="data",
                tensor_name=None,
            )
        )


def _insert_reduce_scatter_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    rs_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "REDUCE_SCATTER_COMM")
    expected = sorted(int(d) for d in devices)
    for node in list(dag.nodes.values()):
        if node.node_kind != "COMPUTE" or not _is_backward_weight_subkind(node.compute_subkind):
            continue
        if not any(_match_filter(node.tag, flt) for flt in filters):
            continue
        if not _node_has_trainable_params(dag, node):
            continue
        if node.device is None:
            raise ValueError(
                f"replicate(shard_grads/shard_params) requires placed backward weight-gradient nodes, but node {node.uid} has device=None; "
                f"expected devices={devices}"
            )
        got = sorted(int(d) for d in node.device)
        if got != expected:
            raise ValueError(
                f"replicate device mismatch for node {node.uid}: "
                f"expected devices={devices}, node_devices={node.device}"
            )
        rs_uid = f"reduce_scatter.{rs_idx}"
        rs_idx += 1
        rs_node = TrainingDAGNode(
            uid=rs_uid,
            node_kind="REDUCE_SCATTER_COMM",
            compute_subkind=None,
            tag=dict(node.tag),
            device=list(node.device),
            stream=comm_stream if comm_stream is not None else "default_stream",
            node_meta={
                "bwd_uid": node.uid,
                "bucket_key": node.node_meta.get("bucket_key"),
            },
        )
        node.node_meta["zero_alloc_full_grads_before"] = True
        if node.node_meta.pop("zero_free_full_params_after", None) is not None:
            rs_node.node_meta["zero_free_full_params_after"] = True
        dag.add_node(rs_node)
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=node.uid,
                dst_uid=rs_uid,
                dep_kind="data",
                tensor_name=None,
            )
        )
        _rewire_bwd_successors_through_sync(dag, node.uid, rs_uid)


def _insert_shard_a2a_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    expected = sorted(int(d) for d in devices)
    a2a_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "A2A_COMM")

    matched_uids = [
        uid for uid, node in dag.nodes.items()
        if node.node_kind == "COMPUTE" and any(_match_filter(node.tag, flt) for flt in filters)
    ]

    def _a2a_boundary_for_edge(src_node: TrainingDAGNode, dst_node: TrainingDAGNode) -> dict[str, Any]:
        # Backward data edges are reversed from the forward graph:
        #     FWD: B -> A    becomes    BWD: A.bwd -> B.bwd
        # The A2A tensor position is defined by the original forward producer,
        # which is the fwd node paired with the backward edge destination.
        if (
            _is_backward_activation_subkind(src_node.compute_subkind)
            and _is_backward_activation_subkind(dst_node.compute_subkind)
        ):
            fwd_producer_uid = dst_node.node_meta.get("fwd_uid")
            if not isinstance(fwd_producer_uid, str) or fwd_producer_uid not in dag.nodes:
                raise ValueError(
                    f"A2A_COMM could not resolve forward producer for BWD edge "
                    f"{src_node.uid}->{dst_node.uid}: fwd_uid={fwd_producer_uid!r}"
                )
            producer = dag.nodes[fwd_producer_uid]
        else:
            producer = src_node

        binfo = producer.node_meta.get("a2a_boundary_after")
        if not isinstance(binfo, dict) or binfo.get("tensor_idx") is None:
            raise ValueError(
                f"A2A_COMM missing tensor_idx for edge {src_node.uid}->{dst_node.uid}; "
                f"producer={producer.uid} boundary_info={binfo!r}"
            )
        return binfo

    for uid in matched_uids:
        if uid not in dag.nodes:
            continue
        node = dag.nodes[uid]
        if node.device is None or sorted(int(d) for d in node.device) != expected:
            raise ValueError(
                f"shard requires matched node {uid} to have devices={devices}, got node_devices={node.device}"
            )

        if node.compute_subkind == "FWD":
            node.node_meta["apply_zero"] = False
        elif _is_backward_compute_subkind(node.compute_subkind):
            fwd_uid = node.node_meta.get("fwd_uid")
            if isinstance(fwd_uid, str) and fwd_uid in dag.nodes:
                dag.nodes[fwd_uid].node_meta["apply_zero"] = False

        # Shard replaces replicate-style grad/param sync comms around matched
        # compute nodes; remove existing gather/reduce comm nodes attached to
        # this compute node before inserting A2A edges.
        removable_sync_kinds = {"ALL_GATHER_COMM", "REDUCE_COMM", "REDUCE_SCATTER_COMM"}
        removable_comm_uids: set[str] = set()
        for e in list(dag.edges):
            if e.dep_kind != "data":
                continue
            if e.src_uid == uid and e.dst_uid in dag.nodes:
                other = dag.nodes[e.dst_uid]
                if other.node_kind in removable_sync_kinds:
                    removable_comm_uids.add(other.uid)
            elif e.dst_uid == uid and e.src_uid in dag.nodes:
                other = dag.nodes[e.src_uid]
                if other.node_kind in removable_sync_kinds:
                    removable_comm_uids.add(other.uid)
        for comm_uid in sorted(removable_comm_uids):
            if comm_uid not in dag.nodes:
                continue
            # Bypass removed sync-comm nodes so dataflow remains connected:
            # pred -> COMM -> succ  ==>  pred -> succ
            in_comm = [e for e in list(dag.edges) if e.dep_kind == "data" and e.dst_uid == comm_uid]
            out_comm = [e for e in list(dag.edges) if e.dep_kind == "data" and e.src_uid == comm_uid]
            for ie in in_comm:
                for oe in out_comm:
                    dag.add_edge(
                        TrainingDAGEdge(
                            src_uid=ie.src_uid,
                            dst_uid=oe.dst_uid,
                            dep_kind="data",
                            tensor_name=(oe.tensor_name if oe.tensor_name is not None else ie.tensor_name),
                        )
                    )
            _remove_node_and_incident_edges(dag, comm_uid)
        if removable_comm_uids:
            node.node_meta.pop("zero_free_full_params_after", None)
        if node.compute_subkind == "BWD_W":
            continue

        def _is_a2a_compute_edge(e: TrainingDAGEdge) -> bool:
            src = dag.nodes[e.src_uid]
            dst = dag.nodes[e.dst_uid]
            if src.node_kind != "COMPUTE" or dst.node_kind != "COMPUTE":
                return False
            if node.compute_subkind == "FWD":
                return src.compute_subkind == "FWD" and dst.compute_subkind == "FWD"
            if _is_backward_activation_subkind(node.compute_subkind):
                return (
                    _is_backward_activation_subkind(src.compute_subkind)
                    and _is_backward_activation_subkind(dst.compute_subkind)
                )
            return False

        incoming = [
            e for e in list(dag.edges)
            if e.dep_kind == "data"
            and e.dst_uid == uid
            and _is_a2a_compute_edge(e)
        ]
        outgoing = [
            e for e in list(dag.edges)
            if e.dep_kind == "data"
            and e.src_uid == uid
            and _is_a2a_compute_edge(e)
        ]

        for e in incoming:
            src = dag.nodes[e.src_uid]
            if src.device is None or sorted(int(d) for d in src.device) != expected:
                raise ValueError(
                    f"shard precondition failed for incoming edge {e.src_uid}->{uid}: "
                    f"upstream compute must be replicated on devices={devices}, got {src.device}"
                )
            if src.tag.get("PASS") != node.tag.get("PASS"):
                raise ValueError(
                    f"A2A_COMM requires matching compute pass tags; "
                    f"incoming edge {e.src_uid}->{uid} has src.PASS={src.tag.get('PASS')} "
                    f"dst.PASS={node.tag.get('PASS')}"
                )
        for e in outgoing:
            dst = dag.nodes[e.dst_uid]
            if dst.device is None or sorted(int(d) for d in dst.device) != expected:
                raise ValueError(
                    f"shard precondition failed for outgoing edge {uid}->{e.dst_uid}: "
                    f"downstream compute must be replicated on devices={devices}, got {dst.device}"
                )
            if dst.tag.get("PASS") != node.tag.get("PASS"):
                raise ValueError(
                    f"A2A_COMM requires matching compute pass tags; "
                    f"outgoing edge {uid}->{e.dst_uid} has src.PASS={node.tag.get('PASS')} "
                    f"dst.PASS={dst.tag.get('PASS')}"
                )

        for e in incoming:
            src_node = dag.nodes[e.src_uid]
            binfo = _a2a_boundary_for_edge(src_node, node)
            a2a_tensor_idx = binfo["tensor_idx"]
            comm_uid = f"a2a.{a2a_idx}"
            a2a_idx += 1
            comm_node = TrainingDAGNode(
                uid=comm_uid,
                node_kind="A2A_COMM",
                compute_subkind=None,
                tag=dict(node.tag),
                device=list(node.device) if node.device is not None else None,
                stream=comm_stream if comm_stream is not None else "default_stream",
                node_meta={
                    "direction": "incoming",
                    "target_uid": uid,
                    "a2a_tensor_idx": a2a_tensor_idx,
                    "bucket_key": node.node_meta.get("bucket_key", src_node.node_meta.get("bucket_key")),
                },
            )
            dag.add_node(comm_node)
            _remove_edge(dag, e)
            dag.add_edge(TrainingDAGEdge(src_uid=e.src_uid, dst_uid=comm_uid, dep_kind="data", tensor_name=e.tensor_name))
            dag.add_edge(TrainingDAGEdge(src_uid=comm_uid, dst_uid=uid, dep_kind="data", tensor_name=e.tensor_name))

        for e in outgoing:
            dst_node = dag.nodes[e.dst_uid]
            binfo = _a2a_boundary_for_edge(node, dst_node)
            a2a_tensor_idx = binfo["tensor_idx"]
            comm_uid = f"a2a.{a2a_idx}"
            a2a_idx += 1
            comm_node = TrainingDAGNode(
                uid=comm_uid,
                node_kind="A2A_COMM",
                compute_subkind=None,
                tag=dict(node.tag),
                device=list(node.device) if node.device is not None else None,
                stream=comm_stream if comm_stream is not None else "default_stream",
                node_meta={
                    "direction": "outgoing",
                    "source_uid": uid,
                    "a2a_tensor_idx": a2a_tensor_idx,
                    "bucket_key": node.node_meta.get("bucket_key", dst_node.node_meta.get("bucket_key")),
                },
            )
            dag.add_node(comm_node)
            _remove_edge(dag, e)
            dag.add_edge(TrainingDAGEdge(src_uid=uid, dst_uid=comm_uid, dep_kind="data", tensor_name=e.tensor_name))
            dag.add_edge(TrainingDAGEdge(src_uid=comm_uid, dst_uid=e.dst_uid, dep_kind="data", tensor_name=e.tensor_name))


def _topological_order(dag: TrainingDAG) -> list[str]:
    in_deg: dict[str, int] = {uid: 0 for uid in dag.nodes}
    out_edges_by_src: dict[str, list[TrainingDAGEdge]] = {uid: [] for uid in dag.nodes}
    for e in dag.edges:
        if e.dst_uid in in_deg:
            in_deg[e.dst_uid] += 1
        if e.src_uid in out_edges_by_src:
            out_edges_by_src[e.src_uid].append(e)
    q = sorted(uid for uid, d in in_deg.items() if d == 0)
    out: list[str] = []
    while q:
        cur_level = q
        q = []
        for u in cur_level:
            out.append(u)
            # Decrement per outgoing edge (not per unique successor), because we
            # can have parallel edges between the same pair (e.g., data + temporal).
            for e in out_edges_by_src.get(u, []):
                v = e.dst_uid
                if v not in in_deg:
                    continue
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
        q.sort()
    if len(out) != len(dag.nodes):
        remaining = [uid for uid, d in in_deg.items() if d > 0]
        raise ValueError(
            "TrainingDAG topological sort failed: unresolved incoming edges remain; "
            f"possible real cycle or malformed parallel-edge bookkeeping. Remaining nodes: {remaining[:8]}"
        )
    return out


def _topological_levels(dag: TrainingDAG) -> dict[str, int]:
    """Return each node's topological level.

    Source nodes are level 0. Every other node is one level after its latest
    predecessor, so independent successors can share the same level.
    """
    topo = _topological_order(dag)
    levels: dict[str, int] = {}
    for uid in topo:
        pred_levels = [
            levels[pred_uid]
            for pred_uid in dag.preds.get(uid, set())
            if pred_uid in levels
        ]
        levels[uid] = (max(pred_levels) + 1) if pred_levels else 0
    return levels


def _serial_topological_order(
    dag: TrainingDAG,
    topo_levels: dict[str, int] | None = None,
) -> list[str]:
    """Serialize topological levels into a deterministic dispatch order.

    Nodes with lower topological levels always come first. Within a level,
    priority order is: SEND > other comm > compute > RECV.
    """
    if topo_levels is None:
        topo_levels = _topological_levels(dag)

    base_order = _topological_order(dag)
    topo_idx = {uid: i for i, uid in enumerate(base_order)}

    _OTHER_COMM_KINDS = {
        "REDUCE_COMM",
        "ALL_GATHER_COMM",
        "REDUCE_SCATTER_COMM",
        "A2A_COMM",
    }

    def node_priority(uid: str) -> int:
        kind = dag.nodes[uid].node_kind
        if kind == "SEND_COMM":
            return 0
        if kind in _OTHER_COMM_KINDS:
            return 1
        if kind == "RECV_COMM":
            return 3
        return 2

    return sorted(
        dag.nodes,
        key=lambda uid: (topo_levels[uid], node_priority(uid), topo_idx[uid]),
    )


def _has_path(dag: TrainingDAG, src_uid: str, dst_uid: str) -> bool:
    seen: set[str] = set()
    stack = [src_uid]
    while stack:
        uid = stack.pop()
        if uid == dst_uid:
            return True
        if uid in seen:
            continue
        seen.add(uid)
        stack.extend(dag.succs.get(uid, set()))
    return False


def _resolve_default_stream_order(dag: TrainingDAG) -> None:
    """Create a total ordering over default-stream COMPUTE nodes.

    Whenever multiple default-stream compute nodes share a topological level,
    chain them with temporal edges in descending order of downstream
    dependencies (more downstream nodes -> earlier in the chain). Downstream
    count is the size of each compute node's transitive successor set, which in
    a well-formed training DAG terminates at UPD.

    The new edges shift topological levels, so after each chain insertion we
    recompute levels and rescan from the earliest level. The pass terminates
    when every default-stream compute node sits at a unique level.
    """
    def _is_default_compute(uid: str) -> bool:
        node = dag.nodes[uid]
        return node.stream == _DEFAULT_STREAM and node.node_kind == "COMPUTE"

    compute_uids = [uid for uid in dag.nodes if _is_default_compute(uid)]
    if len(compute_uids) < 2:
        return

    def _downstream_count(uid: str) -> int:
        seen: set[str] = set()
        stack = list(dag.succs.get(uid, set()))
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            stack.extend(dag.succs.get(v, set()))
        return len(seen)

    while True:
        topo_levels = _topological_levels(dag)
        by_level: dict[int, list[str]] = {}
        for uid in compute_uids:
            by_level.setdefault(topo_levels[uid], []).append(uid)

        conflict_level: int | None = None
        for level in sorted(by_level):
            if len(by_level[level]) > 1:
                conflict_level = level
                break
        if conflict_level is None:
            return

        # Same-level nodes have no path between them (otherwise their levels
        # would differ), so chaining them in any direction is acyclic. Order by
        # downstream count desc; break ties by uid for determinism.
        group = by_level[conflict_level]
        ordered = sorted(group, key=lambda u: (-_downstream_count(u), u))
        for src, dst in zip(ordered, ordered[1:]):
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=src,
                    dst_uid=dst,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )


def resolve_total_order_per_stream(dag: TrainingDAG) -> None:
    """Serialize the default stream, then add temporal edges so each non-default
    logical stream is a strict chain.

    The default-stream pass runs first so the per-non-default-stream anchors
    below see the post-serialization topological order of default-stream nodes.

    Non-default stream nodes are ordered by the topological order of the
    default-stream nodes they directly depend on or are depended on by.
    """
    _resolve_default_stream_order(dag)

    topo = _topological_order(dag)
    topo_idx = {uid: i for i, uid in enumerate(topo)}
    topo_levels = _topological_levels(dag)
    default_uids = [
        uid for uid in topo
        if dag.nodes[uid].stream == _DEFAULT_STREAM
    ]
    streams = sorted({
        n.stream for n in dag.nodes.values()
        if n.stream != _DEFAULT_STREAM
    })
    if not default_uids or not streams:
        return

    for stream in streams:
        associated_by_default: dict[str, list[str]] = {uid: [] for uid in default_uids}
        default_anchors_by_stream_uid: dict[str, list[str]] = {}
        for edge in dag.edges:
            src = dag.nodes[edge.src_uid]
            dst = dag.nodes[edge.dst_uid]
            if src.stream == _DEFAULT_STREAM and dst.stream == stream:
                default_anchors_by_stream_uid.setdefault(edge.dst_uid, []).append(edge.src_uid)
            elif src.stream == stream and dst.stream == _DEFAULT_STREAM:
                default_anchors_by_stream_uid.setdefault(edge.src_uid, []).append(edge.dst_uid)

        for stream_uid in {
            uid for uid, node in dag.nodes.items()
            if node.stream == stream
        }:
            default_anchors = default_anchors_by_stream_uid.get(stream_uid, [])
            if not default_anchors:
                continue
            anchor_uid = min(default_anchors, key=lambda uid: (topo_levels[uid], topo_idx[uid]))
            associated_by_default.setdefault(anchor_uid, []).append(stream_uid)

        current_uid: str | None = None
        for default_uid in default_uids:
            stream_uids = sorted(
                set(associated_by_default.get(default_uid, [])),
                key=lambda u: (topo_levels[u], topo_idx[u]),
            )
            for next_uid in stream_uids:
                if current_uid is not None and current_uid != next_uid:
                    if _has_path(dag, next_uid, current_uid):
                        raise ValueError(
                            "resolve_total_order_per_stream would create a cycle while ordering "
                            f"stream={stream}: {current_uid} -> {next_uid}"
                        )
                    if not _has_path(dag, current_uid, next_uid):
                        dag.add_edge(
                            TrainingDAGEdge(
                                src_uid=current_uid,
                                dst_uid=next_uid,
                                dep_kind="temporal",
                                tensor_name=None,
                            )
                        )
                current_uid = next_uid


def _prune_zero_lifetime_metadata(dag: TrainingDAG) -> list[list[str]]:
    """Prune redundant ZeRO param/grad lifetimes across same-bucket compute chains.

    Returns the union of gradient and parameter chains used for pruning, one
    list per chain ordered by topo. The caller passes these to
    ``_add_inter_chain_temporal_edges`` *after*
    ``resolve_total_order_per_stream`` runs, so the inter-chain check sees the
    post-stream-serialization topo levels for the gather_stream and
    reduce_stream nodes.
    """
    topo_levels = _topological_levels(dag)
    topo = _serial_topological_order(dag, topo_levels)

    def _is_compute(uid: str) -> bool:
        node = dag.nodes[uid]
        return (
            node.node_kind == "COMPUTE"
            and node.compute_subkind in ("FWD", "BWD", "BWD_I", "BWD_W")
        )

    def _chain_bucket(uid: str) -> Any:
        return dag.nodes[uid].node_meta.get("bucket_key")

    def _compute_uids_with(topo_order: list[str], predicate) -> list[str]:
        return [
            uid for uid in topo_order
            if uid in dag.nodes and _is_compute(uid) and predicate(uid)
        ]

    def _has_direct_compute_dependency(src_uid: str, dst_uid: str) -> bool:
        return any(
            e.src_uid == src_uid
            and e.dst_uid == dst_uid
            and e.dep_kind in ("data", "temporal")
            and e.src_uid in dag.nodes
            and e.dst_uid in dag.nodes
            and _is_compute(e.src_uid)
            and _is_compute(e.dst_uid)
            for e in dag.edges
        )

    def _build_chains(candidates: list[str], topo_order: list[str]) -> list[list[str]]:
        chains: list[list[str]] = []
        by_bucket: dict[Any, list[str]] = {}
        for uid in candidates:
            by_bucket.setdefault(_chain_bucket(uid), []).append(uid)
        topo_idx = {uid: i for i, uid in enumerate(topo_order)}
        for bucket, uids in sorted(by_bucket.items(), key=lambda item: str(item[0])):
            uid_set = set(uids)
            links: dict[str, set[str]] = {uid: set() for uid in uids}
            for e in dag.edges:
                if (
                    e.dep_kind in ("data", "temporal")
                    and e.src_uid in uid_set
                    and e.dst_uid in uid_set
                    and _is_compute(e.src_uid)
                    and _is_compute(e.dst_uid)
                ):
                    links[e.src_uid].add(e.dst_uid)
                    links[e.dst_uid].add(e.src_uid)

            seen: set[str] = set()
            for uid in uids:
                if uid in seen:
                    continue
                component: list[str] = []
                stack = [uid]
                seen.add(uid)
                while stack:
                    cur = stack.pop()
                    component.append(cur)
                    for nxt in sorted(links[cur], key=lambda u: topo_idx[u]):
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                chains.append(sorted(component, key=lambda u: topo_idx[u]))
        return chains

    def _data_preds(uid: str) -> list[TrainingDAGEdge]:
        return [
            e for e in list(dag.edges)
            if e.dep_kind == "data" and e.dst_uid == uid
        ]

    def _data_succs(uid: str) -> list[TrainingDAGEdge]:
        return [
            e for e in list(dag.edges)
            if e.dep_kind == "data" and e.src_uid == uid
        ]

    def _all_gather_preds(uid: str) -> list[str]:
        return sorted(
            e.src_uid
            for e in dag.edges
            if e.dep_kind == "data"
            and e.dst_uid == uid
            and e.src_uid in dag.nodes
            and dag.nodes[e.src_uid].node_kind == "ALL_GATHER_COMM"
        )

    def _reduce_scatter_succs(uid: str) -> list[str]:
        return sorted(
            e.dst_uid
            for e in dag.edges
            if e.dep_kind == "data"
            and e.src_uid == uid
            and e.dst_uid in dag.nodes
            and dag.nodes[e.dst_uid].node_kind == "REDUCE_SCATTER_COMM"
        )

    def _remove_all_gather(ag_uid: str, compute_uid: str, bucket: Any) -> None:
        if ag_uid not in dag.nodes:
            return
        in_ag = _data_preds(ag_uid)
        out_ag = _data_succs(ag_uid)
        for ie in in_ag:
            for oe in out_ag:
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=ie.src_uid,
                        dst_uid=oe.dst_uid,
                        dep_kind="data",
                        tensor_name=(oe.tensor_name if oe.tensor_name is not None else ie.tensor_name),
                    )
                )
        _remove_node_and_incident_edges(dag, ag_uid)

    def _remove_reduce_scatter(rs_uid: str, compute_uid: str, bucket: Any) -> None:
        if rs_uid not in dag.nodes:
            return
        _remove_node_and_incident_edges(dag, rs_uid)

    has_grad_sync = any(
        n.node_kind in ("REDUCE_COMM", "REDUCE_SCATTER_COMM")
        for n in dag.nodes.values()
    )
    has_all_gathers = any(n.node_kind == "ALL_GATHER_COMM" for n in dag.nodes.values())

    grad_chains: list[list[str]] = []
    param_chains: list[list[str]] = []

    # Gradient lifetimes: keep one full-grad allocation and one reduce-scatter
    # per directly-connected same-bucket gradient chain.
    if has_grad_sync:
        grad_candidates = _compute_uids_with(
            topo,
            lambda uid: bool(dag.nodes[uid].node_meta.get("zero_alloc_full_grads_before"))
        )
        if grad_candidates:
            grad_chains = _build_chains(grad_candidates, topo)
            covered = {uid for chain in grad_chains for uid in chain}
            assert covered == set(grad_candidates), (
                "zero_lifetime_prune: gradient mode did not assign every grad-allocation "
                f"compute node to a chain; missing={sorted(set(grad_candidates) - covered)}"
            )

            for chain in grad_chains:
                bucket = _chain_bucket(chain[0])
                assert all(_chain_bucket(uid) == bucket for uid in chain), (
                    f"zero_lifetime_prune: gradient chain has mixed buckets: "
                    f"{[(uid, _chain_bucket(uid)) for uid in chain]}"
                )
                logger.info(
                    "zero_lifetime_prune_grad_chain bucket=%s nodes=%s keep_alloc_for=%s keep_reduce_scatter_for=%s",
                    bucket,
                    chain,
                    chain[0],
                    chain[-1],
                )

                for uid in chain:
                    rs_uids = _reduce_scatter_succs(uid)
                    assert rs_uids, (
                        f"zero_lifetime_prune: gradient chain node {uid} has "
                        "zero_alloc_full_grads_before but no outgoing reduce-scatter"
                    )

                for uid in chain[1:]:
                    dag.nodes[uid].node_meta.pop("zero_alloc_full_grads_before", None)
                for uid in chain[:-1]:
                    for rs_uid in _reduce_scatter_succs(uid):
                        _remove_reduce_scatter(rs_uid, uid, bucket)

                assert dag.nodes[chain[0]].node_meta.get("zero_alloc_full_grads_before"), (
                    f"zero_lifetime_prune: gradient chain root {chain[0]} lost grad allocation"
                )
                for prev_uid, uid in zip(chain, chain[1:]):
                    assert _has_direct_compute_dependency(prev_uid, uid), (
                        f"zero_lifetime_prune: gradient chain is not directly linked: "
                        f"{prev_uid} does not have a direct compute dependency to {uid}; chain={chain}"
                    )
                    assert not dag.nodes[uid].node_meta.get("zero_alloc_full_grads_before"), (
                        f"zero_lifetime_prune: non-root gradient chain node {uid} still allocates grads"
                    )
                live_rs = [
                    (uid, rs_uid)
                    for uid in chain
                    for rs_uid in _reduce_scatter_succs(uid)
                ]
                assert live_rs, (
                    f"zero_lifetime_prune: gradient chain has no live reduce-scatter; chain={chain}"
                )
                assert all(uid == chain[-1] for uid, _rs_uid in live_rs), (
                    f"zero_lifetime_prune: only chain tail {chain[-1]} should keep reduce-scatter, "
                    f"but live reduce-scatters are {live_rs}"
                )

    # Parameter lifetimes: keep one all-gather and one full-param free per
    # directly-connected same-bucket parameter chain.  This pass intentionally
    # does not change grad allocation metadata or remove reduce-scatters.
    if has_all_gathers:
        topo = _serial_topological_order(dag, _topological_levels(dag))
        ag_by_compute = {uid: _all_gather_preds(uid) for uid in topo if uid in dag.nodes and _is_compute(uid)}
        ag_by_compute = {uid: ags for uid, ags in ag_by_compute.items() if ags}

        candidates = _compute_uids_with(topo, lambda uid: uid in ag_by_compute)
        param_chains = _build_chains(candidates, topo)

        covered = {uid for chain in param_chains for uid in chain}
        assert covered == set(candidates), (
            "zero_lifetime_prune: parameter mode did not assign every all-gathered "
            f"compute node to a chain; missing={sorted(set(candidates) - covered)}"
        )

        for chain in param_chains:
            bucket = _chain_bucket(chain[0])
            assert all(_chain_bucket(uid) == bucket for uid in chain), (
                f"zero_lifetime_prune: parameter chain has mixed buckets: "
                f"{[(uid, _chain_bucket(uid)) for uid in chain]}"
            )

            for uid in chain[1:]:
                for ag_uid in ag_by_compute[uid]:
                    _remove_all_gather(ag_uid, uid, bucket)

            tail = chain[-1]
            for uid in chain:
                dag.nodes[uid].node_meta.pop("zero_free_full_params_after", None)
                for rs_uid in _reduce_scatter_succs(uid):
                    dag.nodes[rs_uid].node_meta.pop("zero_free_full_params_after", None)
            dag.nodes[tail].node_meta["zero_free_full_params_after"] = True

            remaining_ag = {uid: _all_gather_preds(uid) for uid in chain}
            assert remaining_ag[chain[0]], (
                f"zero_lifetime_prune: parameter chain root {chain[0]} lost all-gather"
            )
            for prev_uid, uid in zip(chain, chain[1:]):
                assert _has_direct_compute_dependency(prev_uid, uid), (
                    f"zero_lifetime_prune: parameter chain is not directly linked: "
                    f"{prev_uid} does not have a direct compute dependency to {uid}; chain={chain}"
                )
                assert not remaining_ag[uid], (
                    f"zero_lifetime_prune: non-root parameter chain node {uid} still "
                    f"has all-gather predecessors {remaining_ag[uid]}; chain={chain}"
                )
            assert dag.nodes[tail].node_meta.get("zero_free_full_params_after"), (
                f"zero_lifetime_prune: parameter chain tail {tail} should free params"
            )
            for uid in chain[:-1]:
                assert not dag.nodes[uid].node_meta.get("zero_free_full_params_after"), (
                    f"zero_lifetime_prune: non-tail parameter chain node {uid} still frees params"
                )
            stale_rs_param_frees = [
                rs_uid
                for uid in chain
                for rs_uid in _reduce_scatter_succs(uid)
                if dag.nodes[rs_uid].node_meta.get("zero_free_full_params_after")
            ]
            assert not stale_rs_param_frees, (
                f"zero_lifetime_prune: reduce-scatters still own param free metadata: "
                f"{stale_rs_param_frees}; chain={chain}"
            )

    return grad_chains + param_chains


def _add_inter_chain_temporal_edges(
    dag: TrainingDAG,
    chains: list[list[str]],
) -> None:
    """For each consecutive same-bucket chain pair (parameter or gradient),
    force the next chain's bucket materialization to wait for the prior chain's
    bucket free.

    Parameter chains: prev_tail compute frees full params, next root's
    all-gather predecessor materializes them. Edge: prev_tail -> ag_uid.

    Gradient chains: prev_tail's reduce-scatter successor frees full grads,
    next root compute allocates them. Edge: rs_uid -> next_root.

    Without these edges ``_serial_topological_order``'s other-comm-before-
    compute priority can dispatch the next bucket op before the prior free
    runs. Must be called after ``resolve_total_order_per_stream`` so the
    gather_stream and reduce_stream chains have populated topo levels.
    """
    if not chains:
        return

    topo_levels = _topological_levels(dag)

    def _chain_bucket(uid: str) -> Any:
        return dag.nodes[uid].node_meta.get("bucket_key")

    def _all_gather_preds(uid: str) -> list[str]:
        return sorted(
            e.src_uid
            for e in dag.edges
            if e.dep_kind == "data"
            and e.dst_uid == uid
            and e.src_uid in dag.nodes
            and dag.nodes[e.src_uid].node_kind == "ALL_GATHER_COMM"
        )

    def _reduce_scatter_succs(uid: str) -> list[str]:
        return sorted(
            e.dst_uid
            for e in dag.edges
            if e.dep_kind == "data"
            and e.src_uid == uid
            and e.dst_uid in dag.nodes
            and dag.nodes[e.dst_uid].node_kind == "REDUCE_SCATTER_COMM"
        )

    def _maybe_add_edge(src_uid: str, dst_uid: str) -> None:
        if topo_levels.get(src_uid, 0) < topo_levels.get(dst_uid, 0):
            return
        if _has_path(dag, src_uid, dst_uid):
            return
        if _has_path(dag, dst_uid, src_uid):
            return
        dag.add_edge(TrainingDAGEdge(
            src_uid=src_uid,
            dst_uid=dst_uid,
            dep_kind="temporal",
            tensor_name=None,
        ))

    # Group chains by type (param vs grad) and bucket, then within each group
    # pair consecutive chains by root topo and add the type-specific edge.
    param_by_bucket: dict[Any, list[list[str]]] = {}
    grad_by_bucket: dict[Any, list[list[str]]] = {}
    for chain in chains:
        bucket = _chain_bucket(chain[0])
        if _all_gather_preds(chain[0]):
            param_by_bucket.setdefault(bucket, []).append(chain)
        if dag.nodes[chain[0]].node_meta.get("zero_alloc_full_grads_before"):
            grad_by_bucket.setdefault(bucket, []).append(chain)

    for bucket_chains in param_by_bucket.values():
        bucket_chains.sort(key=lambda c: topo_levels.get(c[0], 0))
        for prev_chain, next_chain in zip(bucket_chains, bucket_chains[1:]):
            for ag_uid in _all_gather_preds(next_chain[0]):
                _maybe_add_edge(prev_chain[-1], ag_uid)

    for bucket_chains in grad_by_bucket.values():
        bucket_chains.sort(key=lambda c: topo_levels.get(c[0], 0))
        for prev_chain, next_chain in zip(bucket_chains, bucket_chains[1:]):
            for rs_uid in _reduce_scatter_succs(prev_chain[-1]):
                _maybe_add_edge(rs_uid, next_chain[0])


def _apply_split_directive(
    dag: TrainingDAG,
    flt: dict[str, Any],
    dim_name: str,
    num_microbatches: int,
) -> None:
    matched = {uid for uid, n in dag.nodes.items() if _match_filter(n.tag, flt)}
    if not matched:
        logger.warning("split(filter=%s): no nodes matched", flt)
        return

    # UPD nodes are global reduction/update sinks and must not be duplicated
    # by microbatch split; duplicated predecessors should continue to target
    # the same UPD node(s).
    matched = {uid for uid in matched if dag.nodes[uid].node_kind != "UPD"}
    if not matched:
        logger.warning("split(filter=%s): only UPD nodes matched; nothing to split", flt)
        return

    topo = _topological_order(dag)
    idx = {u: i for i, u in enumerate(topo)}
    mpos = sorted(idx[u] for u in matched)
    lo, hi = mpos[0], mpos[-1]
    interleaved = [
        u for u in topo[lo:hi + 1]
        if u not in matched and dag.nodes[u].node_kind != "UPD"
    ]
    if interleaved:
        raise ValueError(
            f"split requires a contiguous sub-DAG; found interleaved non-matching nodes: {interleaved[:8]}"
        )

    incoming_boundary = [e for e in list(dag.edges) if e.dst_uid in matched and e.src_uid not in matched]
    outgoing_boundary = [e for e in list(dag.edges) if e.src_uid in matched and e.dst_uid not in matched]

    sources = {
        u for u in matched
        if not any(pred in matched for pred in dag.preds.get(u, set()))
    }
    sinks = {
        u for u in matched
        if not any(succ in matched for succ in dag.succs.get(u, set()))
    }

    # Ensure outside edges only touch source/sink boundaries.
    for e in incoming_boundary:
        if dag.nodes[e.src_uid].node_kind == "UPD" or dag.nodes[e.dst_uid].node_kind == "UPD":
            continue
        if e.dst_uid not in sources:
            raise ValueError(
                f"split boundary violation: incoming outside edge targets non-source node {e.dst_uid}"
            )
    for e in outgoing_boundary:
        if dag.nodes[e.src_uid].node_kind == "UPD" or dag.nodes[e.dst_uid].node_kind == "UPD":
            continue
        if e.src_uid not in sinks:
            raise ValueError(
                f"split boundary violation: outgoing outside edge starts at non-sink node {e.src_uid}"
            )

    inside_edges = [e for e in list(dag.edges) if e.src_uid in matched and e.dst_uid in matched]

    # Tag originals as microbatch 0.
    for uid in matched:
        dag.nodes[uid].tag[dim_name] = 0

    # Create copies for microbatches 1..N-1.
    copy_uid: dict[tuple[str, int], str] = {}
    for mb in range(1, num_microbatches):
        for uid in topo:
            if uid not in matched:
                continue
            nu = f"{uid}.split{dim_name}{mb}"
            copy_uid[(uid, mb)] = nu

    uid_meta_fields = (
        "fwd_uid",
        "compute_uid",
        "bwd_uid",
        "source_uid",
        "target_uid",
        "src_uid",
        "dst_uid",
        "from_uid",
        "to_uid",
    )
    for mb in range(1, num_microbatches):
        for uid in topo:
            if uid not in matched:
                continue
            n = dag.nodes[uid]
            nu = copy_uid[(uid, mb)]
            copied_meta = dict(n.node_meta)
            for k in uid_meta_fields:
                v = copied_meta.get(k)
                if isinstance(v, str) and v in matched:
                    copied_meta[k] = copy_uid[(v, mb)]
            dag.add_node(
                TrainingDAGNode(
                    uid=nu,
                    node_kind=n.node_kind,
                    compute_subkind=n.compute_subkind,
                    tag={**n.tag, dim_name: mb},
                    device=list(n.device) if n.device is not None else None,
                    stream=n.stream,
                    node_meta=copied_meta,
                )
            )

    # Duplicate internal sub-DAG edges for each copied microbatch.
    for mb in range(1, num_microbatches):
        for e in inside_edges:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=copy_uid[(e.src_uid, mb)],
                    dst_uid=copy_uid[(e.dst_uid, mb)],
                    dep_kind=e.dep_kind,
                    tensor_name=e.tensor_name,
                )
            )

    # Duplicate incoming edges onto copied sources.
    for mb in range(1, num_microbatches):
        for e in incoming_boundary:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=e.src_uid,
                    dst_uid=copy_uid[(e.dst_uid, mb)],
                    dep_kind=e.dep_kind,
                    tensor_name=e.tensor_name,
                )
            )

    # Duplicate outgoing edges from copied sinks.
    for mb in range(1, num_microbatches):
        for e in outgoing_boundary:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=copy_uid[(e.src_uid, mb)],
                    dst_uid=e.dst_uid,
                    dep_kind=e.dep_kind,
                    tensor_name=e.tensor_name,
                )
            )


def _match_set_contiguous_subdag(
    dag: TrainingDAG,
    matched: set[str],
    *,
    context: str,
) -> tuple[set[str], set[str], set[str], set[str]]:
    if not matched:
        raise ValueError(f"{context}: filter matched zero nodes")
    # Contiguity should be dependency-based, not tied to one arbitrary
    # topological ordering (which can interleave independent branches).
    # A non-matching node is considered interleaving iff it is on some path
    # between matched nodes: matched -> ... -> node -> ... -> matched.
    interleaved: list[str] = []
    for u in dag.nodes.keys():
        if u in matched:
            continue
        # Has matched ancestor?
        has_matched_ancestor = False
        seen = {u}
        stack = list(dag.preds.get(u, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in matched:
                has_matched_ancestor = True
                break
            stack.extend(dag.preds.get(cur, set()))
        if not has_matched_ancestor:
            continue

        # Has matched descendant?
        has_matched_descendant = False
        seen = {u}
        stack = list(dag.succs.get(u, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in matched:
                has_matched_descendant = True
                break
            stack.extend(dag.succs.get(cur, set()))
        if has_matched_descendant:
            interleaved.append(u)

    if interleaved:
        raise ValueError(
            f"{context}: matched nodes are not contiguous; interleaved non-matching nodes: {interleaved[:8]}"
        )

    sources = {
        u for u in matched
        if not any(pred in matched for pred in dag.preds.get(u, set()))
    }
    sinks = {
        u for u in matched
        if not any(succ in matched for succ in dag.succs.get(u, set()))
    }
    if not sources or not sinks:
        raise ValueError(f"{context}: failed to identify non-empty source/sink sets")
    compute_nodes = {u for u in matched if dag.nodes[u].node_kind == "COMPUTE"}

    def _has_upstream_compute(u: str) -> bool:
        seen = {u}
        stack = list(dag.preds.get(u, set()))
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in matched:
                continue
            seen.add(cur)
            if dag.nodes[cur].node_kind == "COMPUTE":
                return True
            stack.extend(dag.preds.get(cur, set()))
        return False

    def _has_downstream_compute(u: str) -> bool:
        seen = {u}
        stack = list(dag.succs.get(u, set()))
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in matched:
                continue
            seen.add(cur)
            if dag.nodes[cur].node_kind == "COMPUTE":
                return True
            stack.extend(dag.succs.get(cur, set()))
        return False

    compute_sources = {u for u in compute_nodes if not _has_upstream_compute(u)}
    compute_sinks = {u for u in compute_nodes if not _has_downstream_compute(u)}
    if not compute_sources or not compute_sinks:
        raise ValueError(f"{context}: failed to identify non-empty compute source/sink sets")

    return sources, sinks, compute_sources, compute_sinks


@dataclass(frozen=True)
class _OrderSegment:
    source_uids: set[str]
    sink_uids: set[str]


def _single_device_for_nodes(dag: TrainingDAG, uids: set[str]) -> list[int] | None:
    device_keys = {
        tuple(n.device)
        for uid in uids
        for n in [dag.nodes[uid]]
        if n.device is not None
    }
    if len(device_keys) == 1:
        return list(next(iter(device_keys)))
    return None


def _add_order_dummy_node(
    dag: TrainingDAG,
    *,
    uid: str,
    directive_idx: int,
    group_idx: int,
    role: str,
    grouped_uids: set[str],
) -> str:
    dag.add_node(
        TrainingDAGNode(
            uid=uid,
            node_kind="ORDER_DUMMY",
            compute_subkind=None,
            tag={"order": directive_idx, "group": group_idx},
            device=_single_device_for_nodes(dag, grouped_uids),
            stream=_DEFAULT_STREAM,
            node_meta={"role": role},
        )
    )
    return uid


def _add_temporal_order_edge(dag: TrainingDAG, src_uid: str, dst_uid: str) -> None:
    logger.debug("Adding temporal dependency edge for order directive: %s -> %s", src_uid, dst_uid)
    dag.add_edge(
        TrainingDAGEdge(
            src_uid=src_uid,
            dst_uid=dst_uid,
            dep_kind="temporal",
            tensor_name=None,
        )
    )


def _apply_order_directive(
    dag: TrainingDAG,
    filter_groups: list[list[dict[str, Any]]],
    *,
    directive_idx: int,
) -> None:
    segments: list[_OrderSegment] = []

    for group_idx, group in enumerate(filter_groups):
        group_sources: list[set[str]] = []
        group_sinks: list[set[str]] = []
        grouped_uids: set[str] = set()
        for filter_idx, flt in enumerate(group):
            matched = {
                uid for uid, n in dag.nodes.items()
                if n.node_kind != "ORDER_DUMMY" and _match_filter(n.tag, flt)
            }
            _srcs, _snks, csrcs, csnks = _match_set_contiguous_subdag(
                dag,
                matched,
                context=f"order filter[{group_idx}][{filter_idx}]={flt}",
            )
            group_sources.append(csrcs)
            group_sinks.append(csnks)
            grouped_uids.update(matched)

        if len(group) == 1:
            segments.append(_OrderSegment(source_uids=group_sources[0], sink_uids=group_sinks[0]))
            continue

        src_uid = _add_order_dummy_node(
            dag,
            uid=f"order.{directive_idx}.group{group_idx}.source",
            directive_idx=directive_idx,
            group_idx=group_idx,
            role="source",
            grouped_uids=grouped_uids,
        )
        sink_uid = _add_order_dummy_node(
            dag,
            uid=f"order.{directive_idx}.group{group_idx}.sink",
            directive_idx=directive_idx,
            group_idx=group_idx,
            role="sink",
            grouped_uids=grouped_uids,
        )
        for csrcs in group_sources:
            for v in csrcs:
                _add_temporal_order_edge(dag, src_uid, v)
        for csnks in group_sinks:
            for u in csnks:
                _add_temporal_order_edge(dag, u, sink_uid)
        segments.append(_OrderSegment(source_uids={src_uid}, sink_uids={sink_uid}))

    for i in range(len(segments) - 1):
        for u in segments[i].sink_uids:
            for v in segments[i + 1].source_uids:
                _add_temporal_order_edge(dag, u, v)


def apply_schedule_directives(training_dag: TrainingDAG, directives: list[Any] | None) -> None:
    if not directives:
        return
    split_backward_keys = _validate_split_backward_order_stencil(directives)

    # Microbatch expansion must run before split-backward lowering so schedules
    # can mix fused B and split BI/BW for the same bucket on different MBs.
    for i, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "split":
            continue
        flt, dim_name, num_microbatches = _normalize_split_directive(raw)
        logger.info(
            "Applying directive[%d]: split(filter=%s, dim_name=%s, num_microbatches=%d)",
            i, flt, dim_name, num_microbatches
        )
        _apply_split_directive(training_dag, flt, dim_name, num_microbatches)

    _apply_split_backward_stencil(training_dag, split_backward_keys)
    place_streams: set[str] = set()
    for i, raw in enumerate(directives):
        if isinstance(raw, dict) and raw.get("op") != "place":
            continue
        op, filters, devices, stream, _gather_stream, _reduce_stream, shard_params, shard_grads, bucket_size = _normalize_filter_devices_directive(raw)
        logger.info(
            "Applying directive[%d]: %s(filters=%s, devices=%s, stream=%s, shard_params=%s, shard_grads=%s, bucket_size=%s)",
            i, op, filters, devices, stream, shard_params, shard_grads, bucket_size
        )
        for node in training_dag.nodes.values():
            if node.node_kind not in ("COMPUTE", "UPD"):
                continue
            if any(_match_filter(node.tag, flt) for flt in filters):
                node.device = list(devices)
        if stream is not None:
            place_streams.add(stream)

    if place_streams:
        if len(place_streams) != 1:
            raise ValueError(f"place directives must agree on stream, got {sorted(place_streams)}")
        place_stream = next(iter(place_streams))
    else:
        place_stream = None
    _replicate_update_nodes_by_device(training_dag)
    _insert_send_recv_comm_nodes(training_dag, comm_stream=place_stream)

    for i, raw in enumerate(directives):
        if isinstance(raw, dict) and raw.get("op") in ("place", "split", "order"):
            continue
        op, filters, devices, stream, gather_stream, reduce_stream, shard_params, shard_grads, bucket_size = _normalize_filter_devices_directive(raw)
        logger.info(
            "Applying directive[%d]: %s(filters=%s, devices=%s, stream=%s, gather_stream=%s, reduce_stream=%s, shard_params=%s, shard_grads=%s, bucket_size=%s)",
            i, op, filters, devices, stream, gather_stream, reduce_stream, shard_params, shard_grads, bucket_size
        )
        if op == "replicate":
            if bucket_size is not None:
                _bucket_matched_fwd_nodes(training_dag, filters, int(bucket_size))
            if shard_params:
                _insert_all_gather_comm_nodes(training_dag, filters, devices, comm_stream=gather_stream)
                _insert_reduce_scatter_comm_nodes(training_dag, filters, devices, comm_stream=reduce_stream)
            elif shard_grads:
                _insert_reduce_scatter_comm_nodes(training_dag, filters, devices, comm_stream=reduce_stream)
            else:
                _insert_reduce_comm_nodes(training_dag, filters, devices, comm_stream=reduce_stream)
        elif op == "shard":
            _insert_shard_a2a_comm_nodes(training_dag, filters, devices, comm_stream=stream)
        else:
            raise ValueError(f"Unsupported directive op after normalization: {op}")

    # Final pass for order directives (add temporal dependencies across sub-DAGs).
    for i, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "order":
            continue
        filter_groups = _normalize_order_directive(raw)
        logger.info("Applying directive[%d]: order(filters=%s)", i, filter_groups)
        _apply_order_directive(training_dag, filter_groups, directive_idx=i)


def _split_global_training_dag_by_pp_rank(training_dag: TrainingDAG) -> list[TrainingDAG]:
    """Split the global DAG into per-device-set disconnected DAGs."""
    # SEND/RECV pairs intentionally have no edge between them, so cross-rank
    # placement dependencies separate into disconnected local components here.
    undirected: dict[str, set[str]] = {uid: set() for uid in training_dag.nodes}
    for e in training_dag.edges:
        if e.src_uid in undirected and e.dst_uid in undirected:
            undirected[e.src_uid].add(e.dst_uid)
            undirected[e.dst_uid].add(e.src_uid)

    components: list[set[str]] = []
    seen: set[str] = set()
    for uid in training_dag.nodes:
        if uid in seen:
            continue
        comp: set[str] = set()
        stack = [uid]
        seen.add(uid)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in undirected.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)

    # Validate each component is device-homogeneous and component device-sets are distinct.
    comp_device_keys: list[tuple[int, ...]] = []
    for ci, comp in enumerate(components):
        device_keys = {
            tuple(sorted(node.device)) for uid in comp for node in [training_dag.nodes[uid]] if node.device is not None
        }
        if not device_keys:
            raise ValueError(f"component[{ci}] has no device assignment after P2P split")
        if len(device_keys) != 1:
            raise ValueError(
                f"component[{ci}] is not device-homogeneous; device sets present: {sorted(device_keys)}"
            )
        comp_device_keys.append(next(iter(device_keys)))
    if len(set(comp_device_keys)) != len(comp_device_keys):
        raise ValueError(
            f"expected distinct device sets across split components, got {comp_device_keys}"
        )

    # Materialize each component as a standalone TrainingDAG.
    subdags: list[TrainingDAG] = []
    for comp in components:
        sub = TrainingDAG()
        for uid in comp:
            n = training_dag.nodes[uid]
            sub.add_node(
                TrainingDAGNode(
                    uid=n.uid,
                    node_kind=n.node_kind,
                    compute_subkind=n.compute_subkind,
                    tag=dict(n.tag),
                    device=(None if n.device is None else list(n.device)),
                    stream=n.stream,
                    node_meta=dict(n.node_meta),
                )
            )
        for e in training_dag.edges:
            if e.src_uid in comp and e.dst_uid in comp:
                sub.add_edge(
                    TrainingDAGEdge(
                        src_uid=e.src_uid,
                        dst_uid=e.dst_uid,
                        dep_kind=e.dep_kind,
                        tensor_name=e.tensor_name,
                    )
                )
        subdags.append(sub)

    def _dag_device_key(d: TrainingDAG) -> tuple[int, ...]:
        keys = {tuple(sorted(n.device)) for n in d.nodes.values() if n.device is not None}
        if len(keys) != 1:
            raise ValueError(f"sub-DAG should have exactly one device key, got {keys}")
        return next(iter(keys))

    subdags.sort(key=_dag_device_key)
    return subdags


@register_backend
def piper(gm, example_inputs, **kwargs):
    """TrainingDAG backend: split by stage/a2a and lower schedule directives."""
    del example_inputs, kwargs

    schedule_info = getattr(piper_metadata, "schedule_info", {}) or {}
    schedule_directives = getattr(piper_metadata, "schedule_directives", None)
    num_stages = int(schedule_info.get("num_stages", schedule_info["pp_degree"]))
    has_annotations = any(
        isinstance(node.meta.get("custom"), dict) and node.meta["custom"].get("stage") is not None
        for node in gm.graph.nodes
    )

    stage_tag_name = "PP"
    if has_annotations:
        for node in gm.graph.nodes:
            custom = node.meta.get("custom")
            if isinstance(custom, dict) and custom.get("stage") is not None:
                if isinstance(custom.get("name"), str) and custom.get("name"):
                    stage_tag_name = custom["name"]
                break
    if has_annotations:
        _top_level_gm, stage_submodules = _split_gm_by_stages(gm)
    else:
        logger.info(
            "No stage annotations found, profiling graph to split into %d stages",
            num_stages,
        )
        _top_level_gm, stage_submodules = _profile_and_split_gm(gm, num_stages)

    # Build and store the new directed DAG representation for later scheduling transforms.
    dp_degree = int(os.environ.get("PIPER_DP_DEGREE", "1"))
    training_dag = build_training_dag(
        stage_submodules,
        enable_ep=_directives_enable_ep(schedule_directives),
        dp_degree=dp_degree,
        stage_tag_name=stage_tag_name,
    )
    apply_schedule_directives(
        training_dag,
        schedule_directives,
    )
    piper_metadata.training_dag = training_dag
    per_pp_training_dags = _split_global_training_dag_by_pp_rank(training_dag)
    output_dir = getattr(piper_metadata, "output_dir", "out")
    for i, subdag in enumerate(per_pp_training_dags):
        zero_chains = _prune_zero_lifetime_metadata(subdag)
        resolve_total_order_per_stream(subdag)
        _add_inter_chain_temporal_edges(subdag, zero_chains)
        _log_training_dag_dependencies(subdag)
        print_training_dag_order(subdag, label=f"pp{i}", rank=i, out_dir=output_dir)
        _debug_render_training_dag(subdag, output_path=os.path.join(output_dir, f"training_dag_pp{i}"))
    piper_metadata.per_pp_training_dags = per_pp_training_dags

    logger.info(
        "piper: built TrainingDAG with %d nodes and %d edges, split into %d per-PP DAG(s)",
        len(training_dag.nodes),
        len(training_dag.edges),
        len(per_pp_training_dags),
    )

    def callback(*args, _gm=gm):
        logger.warning(
            "piper compiled callback invoked directly; running local graph execution"
        )
        return _gm(*args)

    return callback


def piper_exec_dag(loss_fn, log_stats: bool = False) -> list:
    """Execute one training step using the loaded per-rank TrainingDAG."""
    actors = piper_metadata.actors
    run_refs = [
        actor.run_dag.remote(loss_fn=loss_fn)
        for actor in actors.values()
    ]
    t0 = time.perf_counter()
    results = ray.get(run_refs)
    step_time = time.perf_counter() - t0

    if log_stats:
        _log_step_stats(step_time, log_stats, actors)

    losses = []
    for result in results:
        if isinstance(result, dict):
            losses.extend(result.get("losses", []))
        elif result:
            losses.extend(result)
    return losses


def _log_step_stats(step_time: float, log_memory: bool, actors: dict) -> None:
    """Log throughput, MFU, and optionally per-rank peak GPU memory."""
    stats = [f"step_time={step_time:.3f}s"]

    tokens = getattr(piper_metadata, "tokens_per_step", None)
    if tokens is not None:
        stats.append(f"throughput={tokens / step_time:.1f} tok/s")

    logger.info("  ".join(stats))
