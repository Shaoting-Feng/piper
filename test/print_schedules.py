"""
Step-by-step test of the PipelineSchedule → TaskDAG → per-rank DAG pipeline.

Run with:
    python3 -m test.print_schedules
"""

from .schedule_helpers import build_1f1b_schedule, build_interleaved_1f1b_schedule, build_zerobubble_schedule, visualize_pipeline_schedule, build_dualpipev_schedule, build_interleaved_zero_bubble
from src.piper_exec import DAGEdge, TaskType, _validate_schedule
from src.piper_graph_transform import expand_chunks_to_dags, add_temporal_dependencies, split_dag_by_rank, assign_time_steps, visualize_dag


mbs = 16
pp = 8

schedule = build_1f1b_schedule(n_mbs=mbs, n_stages=pp)
_validate_schedule(schedule, dag_edges=(DAGEdge(0, 1), DAGEdge(1, 2), DAGEdge(2, 3)), num_mbs=mbs)
visualize_pipeline_schedule(schedule, "out/1f1b-template")

schedule = build_interleaved_1f1b_schedule(n_mbs=mbs, pp=pp)
_validate_schedule(schedule, dag_edges=(DAGEdge(0, 1), DAGEdge(1, 2), DAGEdge(2, 3), DAGEdge(3, 4), DAGEdge(4, 5), DAGEdge(5, 6), DAGEdge(6, 7)), num_mbs=mbs)
visualize_pipeline_schedule(schedule, "out/interleaved-1f1b-template")

schedule = build_zerobubble_schedule(n_mbs=mbs, pp=pp)
_validate_schedule(schedule, dag_edges=(DAGEdge(0, 1), DAGEdge(1, 2), DAGEdge(2, 3)), num_mbs=mbs)
visualize_pipeline_schedule(schedule, "out/zero-bubble-template")

schedule = build_interleaved_zero_bubble(n_mbs=mbs, pp=pp)
_validate_schedule(schedule, dag_edges=(DAGEdge(0, 1), DAGEdge(1, 2), DAGEdge(2, 3), DAGEdge(3, 4), DAGEdge(4, 5), DAGEdge(5, 6), DAGEdge(6, 7)), num_mbs=mbs)
visualize_pipeline_schedule(schedule, "out/interleaved-zero-bubble-template")

schedule = build_dualpipev_schedule(n_mbs=10, pp=4)
_validate_schedule(schedule, dag_edges=(DAGEdge(0, 1), DAGEdge(1, 2), DAGEdge(2, 3), DAGEdge(3, 4), DAGEdge(4, 5), DAGEdge(5, 6), DAGEdge(6, 7)), num_mbs=8)
visualize_pipeline_schedule(schedule, "out/dualpipev-template")

schedule = build_dualpipev_schedule(n_mbs=10, pp=4, seq=True)
_validate_schedule(schedule, dag_edges=(DAGEdge(0, 1), DAGEdge(1, 2), DAGEdge(2, 3), DAGEdge(3, 4), DAGEdge(4, 5), DAGEdge(5, 6), DAGEdge(6, 7)), num_mbs=8)
visualize_pipeline_schedule(schedule, "out/dualpipev-sequential-template")
