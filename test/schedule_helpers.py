
from src.piper_exec import (
    Task,
    BatchMeta,
    CompType,
    Schedule2D,
)


def _task_label(task: Task) -> str:
    """Short label for a task: stage:mb:f/b/u, or first+second for dual batches."""
    c = (
        "u" if task.type == CompType.UPD
        else "f" if task.type == CompType.FWD
        else "b" if task.type == CompType.BWD
        else "fb" if task.type == CompType.FWD_BWD
        else "bi" if task.type == CompType.BWD_I
        else "bw" if task.type == CompType.BWD_W
        else "?"
    )
    if task.type == CompType.UPD:
        label = "   u    "
    else:
        first = task.batches[0]
        label = f"{first.stage_id}:{first.mb_idx}:{c}"
    if len(task.batches) > 1:
        second = task.batches[1]
        label += f"+{second.stage_id}:{second.mb_idx}"
    else:
        label += "    "
    return label


def print_schedule(schedule: Schedule2D, path: str = "schedule") -> None:
    """Print the 2D schedule grid to stdout. Rows = ranks, columns = time steps. None cells shown as ' --- '."""
    for row in schedule.grid:
        for task in row:
            if task is None:
                print(" ------ ", end="\t")
            else:
                print(_task_label(task), end="\t")
        print()

def build_gpipe_schedule(n_mbs: int, n_stages: int) -> Schedule2D:
    steps = n_mbs + n_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(n_stages)]
    for step in range(steps):
        for stage_id in range(n_stages):
            mb_idx = step - stage_id
            if mb_idx >= 0 and mb_idx < n_mbs:
                schedule[stage_id][step] = Task(
                    pp_rank=stage_id,
                    batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                    type=CompType.FWD,
                )

    for step in range(steps, steps * 2):
        for stage_id in reversed(range(n_stages)):
            mb_idx = (step - steps) - (n_stages - stage_id - 1)
            if mb_idx >= 0 and mb_idx < n_mbs:
                schedule[stage_id][step] = Task(
                    pp_rank=stage_id,
                    batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                    type=CompType.BWD,
                )

    for i, stage in enumerate(range(n_stages)):
        schedule[stage][-i-1] = Task(
            pp_rank=stage,
            batches=[BatchMeta(stage_id=stage, mb_idx=0)],
            type=CompType.UPD,
        )
    return Schedule2D(grid=schedule)


def build_1f1b_schedule(n_mbs: int, n_stages: int) -> Schedule2D:
    steps = n_mbs + n_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(n_stages)]
    stage_mb = [[0, 0] for _ in range(n_stages)]
    for step in range(n_stages):
        for stage_id in range(n_stages):
            if step >= stage_id:
                mb_idx = stage_mb[stage_id][0]
                if mb_idx >= 0 and mb_idx < n_mbs:
                    schedule[stage_id][step] = Task(
                        pp_rank=stage_id,
                        batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                        type=CompType.FWD,
                    )
                    stage_mb[stage_id][0] += 1
    for step in range(n_stages, 2 * steps):
        relative_step = step - n_stages
        for stage_id in range(n_stages):
            inv_stage = n_stages - stage_id - 1
            if relative_step >= inv_stage:
                fwd_or_bwd = 1 - (relative_step + inv_stage) % 2
                task_type = CompType.FWD if fwd_or_bwd == 0 else CompType.BWD
                mb_idx = stage_mb[stage_id][fwd_or_bwd]
                if mb_idx >= 0 and mb_idx < n_mbs:
                    schedule[stage_id][step] = Task(
                        pp_rank=stage_id,
                        batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                        type=task_type,
                    )
                    stage_mb[stage_id][fwd_or_bwd] += 1
    for i, stage in enumerate(range(n_stages)):
        schedule[stage][-i-1] = Task(
            pp_rank=stage,
            batches=[BatchMeta(stage_id=stage, mb_idx=n_mbs - 1)],
            type=CompType.UPD,
        )
    return Schedule2D(grid=schedule)


ZEROBUBBLE_MB4_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),          # t0:  F0
            None,                                                                                   # t1:   -
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),          # t2:  F1
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD_I),        # t3:  I0
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD_W),        # t4:  W0
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),          # t5:  F2
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD_I),        # t6:  I1
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD_W),        # t7:  W1
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),          # t8:  F3
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD_I),        # t9:  I2
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD_W),        # t10: W2
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD_I),        # t11: I3
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD_W),        # t12: W3
            Task(pp_rank=0, batches=[], type=CompType.UPD),                                         # t13: U0
        ],
        [
            None,                                                                                   # t0:  -
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),          # t1:  F0
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD_I),        # t2:  I0
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),          # t3:  F1
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD_I),        # t4:  I1
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD_W),        # t5:  W0
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),          # t6:  F2
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD_I),        # t7:  I2
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD_W),        # t8:  W1
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),          # t9:  F3
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD_I),        # t10: I3
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD_W),        # t11: W2
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD_W),        # t12: W3
            Task(pp_rank=1, batches=[], type=CompType.UPD),                                         # t13: U0
        ],
    ])

NO_PP_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.UPD),
        ]
    ],
)

DUALPIPEV_MB6_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD_I),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD_W),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD_I),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD_W),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=2), BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4), BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=3), BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5), BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4), BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5), BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD_BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD_I),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD_I),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD_W),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD_W),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[], type=CompType.UPD),
            None,
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2), BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD_BWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=2), BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD_BWD),
            None,
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3), BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=3), BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4), BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4), BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5), BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5), BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD_I),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD_W),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD_I),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD_I),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD_W),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD_W),
            Task(pp_rank=1, batches=[], type=CompType.UPD),
        ]
    ])

DUALPIPEV_NOZB_MB6_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=2), BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4), BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=3), BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5), BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4), BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5), BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2), BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD_BWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=2), BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3), BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=3), BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4), BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4), BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5), BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5), BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD_BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=1, batches=[], type=CompType.UPD),
            None,
        ]
    ])

INTERLEAVED_1F1B_PP2_MB6_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=1, batches=[], type=CompType.UPD),
            None,
        ],
    ])

INTERLEAVED_1F1B_PP2_MB4_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.UPD),
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.UPD),
            None,
        ],
    ])

INTERLEAVED_GPIPE_PP2_MB4_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            None,
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[], type=CompType.UPD),
            None,
        ],
    ])

INTERLEAVED_1F1B_PP4_MB8_SCHEDULE = Schedule2D(
    grid=[
        [
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.FWD),
            None,
            None,
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.FWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            None,
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.BWD),
            Task(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.BWD),
            Task(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.FWD),
            None,
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.FWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.BWD),
            None,
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.BWD),
            Task(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.BWD),
            Task(pp_rank=1, batches=[], type=CompType.UPD),
            None,
        ],
        [
            None,
            None,
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.FWD),
            None,
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.FWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.BWD),
            None,
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.BWD),
            Task(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.BWD),
            Task(pp_rank=2, batches=[], type=CompType.UPD),
            None,
            None,
        ],
        [
            None,
            None,
            None,
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.FWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.BWD),
            Task(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.BWD),
            Task(pp_rank=3, batches=[], type=CompType.UPD),
            None,
            None,
            None,
        ],
    ])
