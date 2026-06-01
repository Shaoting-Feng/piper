from src.piper import (
    TrainingDAG,
    TrainingDAGEdge,
    TrainingDAGNode,
    _apply_order_directive,
    _apply_split_backward_stencil,
    _apply_split_directive,
    _normalize_order_directive,
    _serial_topological_order,
    _validate_split_backward_order_stencil,
)


def _compute(uid: str, tag: dict) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag=tag,
        device=[0],
        stream="default_stream",
        node_meta={},
    )


def _bwd(uid: str, fwd_uid: str, tag: dict) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="BWD",
        tag=tag,
        device=[0],
        stream="default_stream",
        node_meta={"fwd_uid": fwd_uid, "bucket_key": fwd_uid},
    )


def test_normalize_order_directive_accepts_nested_filter_groups() -> None:
    directive = {
        "op": "order",
        "filters": [
            [
                [["PP", 0], ["MB", 0], ["PASS", "F"]],
                [["PP", 0], ["MB", 1], ["PASS", "F"]],
            ],
            [
                [["PP", 0], ["MB", 0], ["PASS", "B"]],
            ],
        ],
    }

    groups = _normalize_order_directive(directive)

    assert groups == [
        [
            {"PP": 0, "MB": 0, "PASS": "F"},
            {"PP": 0, "MB": 1, "PASS": "F"},
        ],
        [
            {"PP": 0, "MB": 0, "PASS": "B"},
        ],
    ]


def test_normalize_order_directive_wraps_legacy_flat_filters() -> None:
    directive = {
        "op": "order",
        "filters": [
            [["PP", 0], ["MB", 0], ["PASS", "F"]],
            [["PP", 0], ["MB", 0], ["PASS", "B"]],
        ],
    }

    groups = _normalize_order_directive(directive)

    assert groups == [
        [{"PP": 0, "MB": 0, "PASS": "F"}],
        [{"PP": 0, "MB": 0, "PASS": "B"}],
    ]


def test_split_backward_stencil_handles_nested_filter_groups() -> None:
    directive = {
        "op": "order",
        "filters": [
            [
                [["PP", 0], ["MB", 0], ["PASS", "BI"]],
                [["PP", 0], ["MB", 0], ["PASS", "BW"]],
            ],
            [
                [["PP", 0], ["MB", 1], ["PASS", "F"]],
            ],
        ],
    }

    split_keys = _validate_split_backward_order_stencil([directive])

    assert split_keys == {(("MB", 0), ("PP", 0))}


def test_split_backward_stencil_allows_mixed_fused_and_split_microbatches() -> None:
    directive = {
        "op": "order",
        "filters": [
            [[["PP", 0], ["MB", 0], ["PASS", "B"]]],
            [[["PP", 0], ["MB", 1], ["PASS", "BI"]]],
            [[["PP", 0], ["MB", 1], ["PASS", "BW"]]],
        ],
    }
    dag = TrainingDAG()
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))
    dag.add_node(_bwd("bwd", "fwd", {"PP": 0, "PASS": "B"}))
    dag.add_edge(TrainingDAGEdge("fwd", "bwd", "data"))

    split_keys = _validate_split_backward_order_stencil([directive])
    _apply_split_directive(dag, {}, "MB", 2)
    _apply_split_backward_stencil(dag, split_keys)

    assert dag.nodes["bwd"].compute_subkind == "BWD"
    assert dag.nodes["bwd"].tag["PASS"] == "B"
    split_bwd_uid = "bwd.splitMB1"
    assert dag.nodes[split_bwd_uid].compute_subkind == "BWD_I"
    assert dag.nodes[split_bwd_uid].tag["PASS"] == "BI"
    assert dag.nodes[f"{split_bwd_uid}.bw"].compute_subkind == "BWD_W"


def test_apply_order_directive_groups_nested_subdags_with_dummy_source_sink() -> None:
    dag = TrainingDAG()
    for node in [
        _compute("prev", {"slot": 0}),
        _compute("lane0.first", {"slot": 1, "lane": 0}),
        _compute("lane0.last", {"slot": 1, "lane": 0}),
        _compute("lane1", {"slot": 1, "lane": 1}),
        _compute("next", {"slot": 2}),
    ]:
        dag.add_node(node)
    dag.add_edge(TrainingDAGEdge("lane0.first", "lane0.last", "data"))

    _apply_order_directive(
        dag,
        [
            [{"slot": 0}],
            [{"slot": 1, "lane": 0}, {"slot": 1, "lane": 1}],
            [{"slot": 2}],
        ],
        directive_idx=7,
    )

    source_uid = "order.7.group1.source"
    sink_uid = "order.7.group1.sink"
    assert dag.nodes[source_uid].node_kind == "ORDER_DUMMY"
    assert dag.nodes[sink_uid].node_kind == "ORDER_DUMMY"

    temporal_edges = {
        (edge.src_uid, edge.dst_uid)
        for edge in dag.edges
        if edge.dep_kind == "temporal"
    }
    assert temporal_edges == {
        ("prev", source_uid),
        (source_uid, "lane0.first"),
        (source_uid, "lane1"),
        ("lane0.last", sink_uid),
        ("lane1", sink_uid),
        (sink_uid, "next"),
    }

    topo = _serial_topological_order(dag)
    assert topo.index("prev") < topo.index(source_uid)
    assert topo.index(source_uid) < topo.index("lane0.first")
    assert topo.index(source_uid) < topo.index("lane1")
    assert topo.index("lane0.last") < topo.index(sink_uid)
    assert topo.index("lane1") < topo.index(sink_uid)
    assert topo.index(sink_uid) < topo.index("next")
