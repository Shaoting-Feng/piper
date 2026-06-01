from enum import Enum


class TaskType(Enum):
    FWD = "forward"
    BWD = "backward"
    UPD = "update"
    BWD_I = "backward_input"
    BWD_W = "backward_weight"
    FWD_BWD = "forward_backward"
    SEND = "send"
    RECV = "recv"
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    FWD_A2A = "forward_a2a"
    BWD_A2A = "backward_a2a"
    ORDER_DUMMY = "order_dummy"
