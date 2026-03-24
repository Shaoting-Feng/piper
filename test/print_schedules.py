"""
Step-by-step test of the Schedule2D → TaskDAG → per-rank DAG pipeline.

Run with:
    python3 -m test.print_schedules
"""

from .schedule_helpers import INTERLEAVED_1F1B_PP2_MB4_SCHEDULE, build_1f1b_schedule, print_schedule
from src.piper_exec import TaskType
from src.piper_graph_transform import schedule_to_dag, insert_p2p_ops, visualize_schedule, visualize_dag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type_abbrev(tt: TaskType) -> str:
    return {
        TaskType.FWD:     "F",
        TaskType.BWD:     "B",
        TaskType.UPD:     "U",
        TaskType.BWD_I:   "Bi",
        TaskType.BWD_W:   "Bw",
        TaskType.FWD_BWD: "FB",
        TaskType.SEND:    "Snd",
        TaskType.RECV:    "Rcv",
    }.get(tt, "?")


def _node_str(node) -> str:
    batches = "+".join(f"s{b.stage_id}m{b.mb_idx}" for b in node.task.batches)
    peer = f" peer=r{node.peer_pp_rank}" if node.peer_pp_rank is not None else ""
    return f"[r{node.pp_rank} t={node.time_step:+d} {_type_abbrev(node.task.type)} {batches}{peer}]"


def print_dag(dag, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for node in sorted(dag.nodes, key=lambda n: (n.pp_rank, n.time_step)):
        data_preds  = [_node_str(p) for p in node.data_preds]
        data_succs  = [_node_str(s) for s in node.data_succs]
        temp_pred   = _node_str(node.temporal_pred)  if node.temporal_pred  else "—"
        temp_succ   = _node_str(node.temporal_succ)  if node.temporal_succ  else "—"
        print(f"  {_node_str(node)}")
        print(f"    data_preds : {data_preds or '—'}")
        print(f"    data_succs : {data_succs or '—'}")
        print(f"    temp_pred  : {temp_pred}")
        print(f"    temp_succ  : {temp_succ}")


# ---------------------------------------------------------------------------
# Step 1: Build Schedule2D
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("  STEP 1 — 1F1B Schedule2D  (PP=2, MB=2)")
print("="*60)
schedule = build_1f1b_schedule(n_mbs=2, n_stages=2)
schedule = INTERLEAVED_1F1B_PP2_MB4_SCHEDULE
print_schedule(schedule)

# ---------------------------------------------------------------------------
# Step 2: Convert to TaskDAG (with cross-rank data edges)
# ---------------------------------------------------------------------------

dag = schedule_to_dag(schedule)
print_dag(dag, "STEP 2 — TaskDAG (cross-rank data edges present)")

# Summarise edge counts
n_data  = sum(len(n.data_succs) for n in dag.nodes)
n_temp  = sum(1 for n in dag.nodes if n.temporal_succ is not None)
n_cross = sum(
    1 for n in dag.nodes for s in n.data_succs if n.pp_rank != s.pp_rank
)
print(f"\n  nodes={len(dag.nodes)}  data_edges={n_data}  "
      f"temporal_edges={n_temp}  cross_rank_data_edges={n_cross}")

# ---------------------------------------------------------------------------
# Step 3: Insert SEND / RECV nodes (insert_p2p_ops)
# ---------------------------------------------------------------------------

rank_dags = insert_p2p_ops(dag)

visualize_schedule(schedule, "schedule_1f1b")

for rank, rank_dag in enumerate(rank_dags):
    visualize_dag(rank_dag, f"rank{rank}_dag")
    print_dag(rank_dag, f"STEP 3 — Per-rank DAG for PP rank {rank}")
    n_data = sum(len(n.data_succs) for n in rank_dag.nodes)
    n_temp = sum(1 for n in rank_dag.nodes if n.temporal_succ is not None)
    print(f"\n  nodes={len(rank_dag.nodes)}  data_edges={n_data}  temporal_edges={n_temp}")

print()
