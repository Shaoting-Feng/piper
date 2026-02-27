from .schedule_helpers import print_schedule, build_1f1b_schedule, build_gpipe_schedule
from .schedule_helpers import DUALPIPEV_SCHEDULE, DUALPIPEV_NO_ZB_SCHEDULE
from src.piper_exec import _validate_schedule, DAGEdge

schedule = DUALPIPEV_SCHEDULE
dag_edges = [DAGEdge(from_stage=0, to_stage=1), DAGEdge(from_stage=1, to_stage=2), DAGEdge(from_stage=2, to_stage=3)]
num_mbs = 6
_validate_schedule(schedule.grid, dag_edges, num_mbs)
