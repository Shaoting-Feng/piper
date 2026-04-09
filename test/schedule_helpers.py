
from src.piper_exec import (
    Chunk,
    BatchMeta,
    TaskType,
    CompType,  # backwards-compatible alias for TaskType
    PipelineSchedule,
    Task,
    TaskDAG,
    logical_time,
)
from src.piper_graph_transform import visualize_dag


def _chunk_label(chunk: Chunk) -> str:
    """Short label for a chunk: stage:mb:f/b/u, or first+second for dual batches."""
    c = (
        "u" if chunk.type == CompType.UPD
        else "f" if chunk.type == CompType.FWD
        else "b" if chunk.type == CompType.BWD
        else "fb" if chunk.type == CompType.FWD_BWD
        else "bi" if chunk.type == CompType.BWD_I
        else "bw" if chunk.type == CompType.BWD_W
        else "?"
    )
    if chunk.type == CompType.UPD:
        label = "   u    "
    else:
        first = chunk.batches[0]
        label = f"{first.stage_id}:{first.mb_idx}:{c}"
    if len(chunk.batches) > 1:
        second = chunk.batches[1]
        label += f"+{second.stage_id}:{second.mb_idx}"
    else:
        label += "    "
    return label


def print_schedule(schedule: PipelineSchedule) -> None:
    """Print the schedule as a 2-D timed grid (rows=ranks, columns=logical time)."""
    start_times_per_rank = schedule._compute_start_times()
    max_time = max(
        start + logical_time(chunk)
        for row, starts in zip(schedule.grid, start_times_per_rank)
        for chunk, start in zip(row, starts)
    )
    for rank, (row, starts) in enumerate(zip(schedule.grid, start_times_per_rank)):
        # Build a sparse mapping from start_time -> chunk for this rank
        time_to_chunk = {start: chunk for chunk, start in zip(row, starts)}
        for t in range(max_time):
            chunk = time_to_chunk.get(t)
            if chunk is None:
                print(" ------ ", end="\t")
            else:
                print(_chunk_label(chunk), end="\t")
        print()


def build_gpipe_schedule(n_mbs: int, n_stages: int) -> PipelineSchedule:
    rows: list[list[Chunk]] = [[] for _ in range(n_stages)]
    for stage_id in range(n_stages):
        for mb_idx in range(n_mbs):
            rows[stage_id].append(Chunk(
                pp_rank=stage_id,
                batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                type=CompType.FWD,
            ))
    for stage_id in reversed(range(n_stages)):
        for mb_idx in range(n_mbs):
            rows[stage_id].append(Chunk(
                pp_rank=stage_id,
                batches=[BatchMeta(stage_id=stage_id, mb_idx=mb_idx)],
                type=CompType.BWD,
            ))
    return PipelineSchedule(grid=rows)


def build_1f1b_schedule(n_mbs: int, n_stages: int) -> PipelineSchedule:
    """Standard 1F1B schedule (no virtual stages, BWD = fused)."""
    rows: list[list[Chunk]] = [[] for _ in range(n_stages)]
    # Warmup: rank r does r+1 forward passes before first backward
    fwd_mb = [0] * n_stages
    bwd_mb = [0] * n_stages
    for stage_id in range(n_stages):
        warmup = n_stages - 1 - stage_id
        for _ in range(min(warmup, n_mbs)):
            rows[stage_id].append(Chunk(
                pp_rank=stage_id,
                batches=[BatchMeta(stage_id=stage_id, mb_idx=fwd_mb[stage_id])],
                type=CompType.FWD,
            ))
            fwd_mb[stage_id] += 1
    # 1F1B steady state + cooldown
    for stage_id in range(n_stages):
        while bwd_mb[stage_id] < n_mbs:
            if fwd_mb[stage_id] < n_mbs:
                rows[stage_id].append(Chunk(
                    pp_rank=stage_id,
                    batches=[BatchMeta(stage_id=stage_id, mb_idx=fwd_mb[stage_id])],
                    type=CompType.FWD,
                ))
                fwd_mb[stage_id] += 1
            rows[stage_id].append(Chunk(
                pp_rank=stage_id,
                batches=[BatchMeta(stage_id=stage_id, mb_idx=bwd_mb[stage_id])],
                type=CompType.BWD,
            ))
            bwd_mb[stage_id] += 1
    return PipelineSchedule(grid=rows)


def build_interleaved_1f1b_schedule(n_mbs: int, pp: int, v: int = 2) -> PipelineSchedule:
    """Interleaved 1F1B schedule (v virtual stages per rank, BWD = fused).

    Rank r owns stages r, r+pp, r+2*pp, ..., r+(v-1)*pp.
    Uses PyTorch's warmup formula: warmup(r) = min((v-1)*pp + 2*(pp-1-r), n_mbs*v).
    """
    from collections import defaultdict
    microbatches_per_round = pp

    def warmup_ops(rank):
        return min((v - 1) * pp + 2 * (pp - 1 - rank), n_mbs * v)

    def forward_stage_index(step, rank, warmup):
        local_index = (step // microbatches_per_round) % v
        return local_index * pp + rank

    def backward_stage_index(step, rank, warmup):
        local_index = (v - 1 - ((step - warmup) // microbatches_per_round) % v)
        return local_index * pp + rank

    rows: list[list[Chunk]] = []
    for rank in range(pp):
        wu = warmup_ops(rank)
        microbatch_ops = v * n_mbs
        fwd_bwd_ops = microbatch_ops - wu
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        fwd_mb: dict[int, int] = defaultdict(int)
        bwd_mb: dict[int, int] = defaultdict(int)
        chunks: list[Chunk] = []

        for op in range(wu + fwd_bwd_ops + cooldown_ops):
            if op < wu:
                s = forward_stage_index(op, rank, wu)
                m = fwd_mb[s]; fwd_mb[s] += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(s, m)], type=CompType.FWD))
            elif op < wu + fwd_bwd_ops:
                s = forward_stage_index(op, rank, wu)
                m = fwd_mb[s]; fwd_mb[s] += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(s, m)], type=CompType.FWD))
                s = backward_stage_index(op, rank, wu)
                m = bwd_mb[s]; bwd_mb[s] += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(s, m)], type=CompType.BWD))
            else:
                s = backward_stage_index(op, rank, wu)
                m = bwd_mb[s]; bwd_mb[s] += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(s, m)], type=CompType.BWD))

        rows.append(chunks)
    return PipelineSchedule(grid=rows)


def build_zerobubble_schedule(n_mbs: int, pp: int) -> PipelineSchedule:
    """ZeroBubble (ZB-1) schedule: one stage per rank, backward split into BWD_I + BWD_W.

    Uses PyTorch's ZB warmup formula: warmup(r) = min(pp - 1 - r, n_mbs).
    BWD_W for microbatch m on rank r is deferred by r slots relative to BWD_I.
    """
    from collections import defaultdict

    def warmup_ops(rank):
        return min(pp - 1 - rank, n_mbs)

    rows: list[list[Chunk]] = []
    for rank in range(pp):
        wu = warmup_ops(rank)
        microbatch_ops = n_mbs
        fwd_bwd_ops = microbatch_ops - wu
        cooldown_ops = microbatch_ops - fwd_bwd_ops
        num_1f1b_microbatches = rank  # W is deferred by this many steps

        fwd_mb = 0
        bwd_mb = 0
        w_mb = 0
        chunks: list[Chunk] = []
        backward_ops: list[int] = []  # track op indices where BWD_I was emitted
        weight_op_count = 0

        for op in range(wu + fwd_bwd_ops + cooldown_ops):
            stage = rank  # single stage per rank
            if op < wu:
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, fwd_mb)], type=CompType.FWD))
                fwd_mb += 1
            elif op < wu + fwd_bwd_ops:
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, fwd_mb)], type=CompType.FWD))
                fwd_mb += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, bwd_mb)], type=CompType.BWD_I))
                bwd_mb += 1
                backward_ops.append(op)
                if op - wu >= num_1f1b_microbatches:
                    chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, w_mb)], type=CompType.BWD_W))
                    w_mb += 1
                    weight_op_count += 1
            else:
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, bwd_mb)], type=CompType.BWD_I))
                bwd_mb += 1
                backward_ops.append(op)
                if op - wu >= num_1f1b_microbatches:
                    chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, w_mb)], type=CompType.BWD_W))
                    w_mb += 1
                    weight_op_count += 1

        while weight_op_count < len(backward_ops):
            chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(rank, w_mb)], type=CompType.BWD_W))
            w_mb += 1
            weight_op_count += 1

        rows.append(chunks)
    return PipelineSchedule(grid=rows)



def build_interleaved_zero_bubble(n_mbs: int, pp: int, v: int = 2) -> PipelineSchedule:
    """Interleaved ZeroBubble schedule: v virtual stages per rank, backward split into BWD_I + BWD_W.

    Matches torch.distributed.pipelining ScheduleInterleavedZeroBubble op-order exactly.

    Rank r owns stages r, r+pp, r+2*pp, ..., r+(v-1)*pp (same as interleaved 1F1B).
    Warmup formula (multiply_factor=1, matching torch):
        warmup(r) = min((v-1)*microbatches_per_round + 1*(pp-1-r), n_mbs*v)
    where microbatches_per_round = n_mbs // max(1, n_mbs // pp).

    Contrast with interleaved 1F1B which uses multiply_factor=2. The ZeroBubble variant
    uses multiply_factor=1 because W ops fill the bubble time that would otherwise require
    extra warmup F ops, so fewer warmup forwards are needed.

    BWD_W is deferred by `rank` 1F1B steps relative to BWD_I (emit W once bwdi_count > rank),
    matching torch's condition `op - warmup_ops >= rank`. A FIFO weight queue tracks
    outstanding (stage, mb) pairs whose BWD_I has completed but BWD_W has not yet been emitted.

    Bubble placement: torch inserts explicit None slots (rank pre-padding + pp-rank-1
    post-warmup Nones). Piper omits explicit Nones; _compute_start_times() recovers the
    same absolute timing via FWD/BWD dependency constraints.
    """
    from collections import defaultdict

    microbatches_per_round = pp  # = n_mbs // max(1, n_mbs // pp) for balanced configs

    def warmup_ops(rank: int) -> int:
        # multiply_factor=1 (ZeroBubble) vs 2 (1F1B); matches torch's ScheduleInterleavedZeroBubble
        return min((v - 1) * microbatches_per_round + 1 * (pp - 1 - rank), n_mbs * v)

    def forward_stage_index(step: int, rank: int) -> int:
        local_index = (step // microbatches_per_round) % v
        return local_index * pp + rank

    def backward_stage_index(step: int, rank: int, warmup: int) -> int:
        local_index = (v - 1 - ((step - warmup) // microbatches_per_round) % v)
        return local_index * pp + rank

    rows: list[list[Chunk]] = []
    for rank in range(pp):
        wu = warmup_ops(rank)
        microbatch_ops = v * n_mbs
        fwd_bwd_ops = microbatch_ops - wu
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        fwd_mb: dict[int, int] = defaultdict(int)   # stage_id -> next fwd microbatch index
        bwd_mb: dict[int, int] = defaultdict(int)   # stage_id -> next bwdi microbatch index
        chunks: list[Chunk] = []
        weight_queue: list[tuple[int, int]] = []    # FIFO of (stage_id, mb_idx) pending BWD_W
        bwdi_count = 0
        # Defer the first `rank` BWD_Ws — same as non-interleaved ZeroBubble.
        # This fills the cooldown-phase bubble produced by pipeline depth `rank`.
        num_deferred = rank

        def _maybe_emit_w() -> None:
            """Emit the oldest queued BWD_W if the deferral window has passed."""
            if bwdi_count > num_deferred and weight_queue:
                ws, wm = weight_queue.pop(0)
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(ws, wm)], type=CompType.BWD_W))

        for op in range(wu + fwd_bwd_ops + cooldown_ops):
            if op < wu:
                # Warmup: forward only
                s = forward_stage_index(op, rank)
                m = fwd_mb[s]; fwd_mb[s] += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(s, m)], type=CompType.FWD))
            elif op < wu + fwd_bwd_ops:
                # Steady state: FWD then BWD_I, then maybe BWD_W
                s_fwd = forward_stage_index(op, rank)
                m_fwd = fwd_mb[s_fwd]; fwd_mb[s_fwd] += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(s_fwd, m_fwd)], type=CompType.FWD))

                s_bwd = backward_stage_index(op, rank, wu)
                m_bwd = bwd_mb[s_bwd]; bwd_mb[s_bwd] += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(s_bwd, m_bwd)], type=CompType.BWD_I))
                weight_queue.append((s_bwd, m_bwd))
                bwdi_count += 1
                _maybe_emit_w()
            else:
                # Cooldown: BWD_I only, then maybe BWD_W
                s_bwd = backward_stage_index(op, rank, wu)
                m_bwd = bwd_mb[s_bwd]; bwd_mb[s_bwd] += 1
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(s_bwd, m_bwd)], type=CompType.BWD_I))
                weight_queue.append((s_bwd, m_bwd))
                bwdi_count += 1
                _maybe_emit_w()

        # Drain any remaining deferred BWD_Ws
        while weight_queue:
            ws, wm = weight_queue.pop(0)
            chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(ws, wm)], type=CompType.BWD_W))

        rows.append(chunks)
    return PipelineSchedule(grid=rows)


def build_dualpipev_schedule(n_mbs: int, pp: int, seq: bool = False) -> PipelineSchedule:
    """DualPipeV schedule: V-layout, 2 virtual stages per rank, zero-bubble optimization.

    Stage assignment (V-layout): rank r owns stage r (stage0) and stage 2*pp-1-r (stage1).
    Directly ports PyTorch's ScheduleDualPipeV._calculate_single_rank_operations.
    """
    rows: list[list[Chunk]] = [[] for _ in range(pp)]

    for rank in range(pp):
        s0 = rank               # first (outer) stage
        s1 = 2 * pp - 1 - rank  # second (inner) stage
        chunks = rows[rank]
        cnt: dict[tuple[int, str], int] = {}
        weight_queue: list[tuple[int, int]] = []

        def c(stage: int, key: str) -> int:
            return cnt.get((stage, key), 0)

        def inc(stage: int, key: str) -> None:
            cnt[(stage, key)] = c(stage, key) + 1

        def inc_bwd(stage: int) -> None:
            inc(stage, "i"); inc(stage, "w")

        def fwd(stage: int) -> None:
            m = c(stage, "f")
            chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, m)], type=CompType.FWD))
            inc(stage, "f")

        def bwdi(stage: int) -> None:
            m = c(stage, "i")
            chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, m)], type=CompType.BWD_I))
            weight_queue.append((stage, m))
            inc(stage, "i")

        def drain_w() -> None:
            if not weight_queue:
                return
            stage, m = weight_queue.pop(0)
            chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, m)], type=CompType.BWD_W))
            inc(stage, "w")

        def full_bwd(stage: int) -> None:
            m = c(stage, "i")
            chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(stage, m)], type=CompType.BWD))
            inc_bwd(stage)

        def overlap_fb(fwd_stage: int, bwd_stage: int) -> None:
            fm = c(fwd_stage, "f")
            bm = c(bwd_stage, "i")
            if seq:
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(fwd_stage, fm)], type=CompType.FWD))
                chunks.append(Chunk(pp_rank=rank, batches=[BatchMeta(bwd_stage, bm)], type=CompType.BWD))
            else:
                chunks.append(Chunk(
                    pp_rank=rank,
                    batches=[BatchMeta(fwd_stage, fm), BatchMeta(bwd_stage, bm)],
                    type=CompType.FWD_BWD,
                ))
            inc(fwd_stage, "f")
            inc_bwd(bwd_stage)

        # Phase 1: F0 warmup
        for _ in range((pp - rank - 1) * 2):
            fwd(s0)

        # Phase 2: F0F1
        for _ in range(rank + 1):
            fwd(s0); fwd(s1)

        # Phase 3: I1 W1 F1 (zero-bubble)
        for _ in range(pp - rank - 1):
            bwdi(s1); drain_w(); fwd(s1)

        # Phase 4: Main — F0B1 + F1B0 (overlapped)
        for i in range(n_mbs - pp * 2 + rank + 1):
            if i == 0 and rank == pp - 1:
                fwd(s0); full_bwd(s1)
            else:
                overlap_fb(s0, s1)
            overlap_fb(s1, s0)

        # Phase 5: B1 + F1B0
        for _ in range(pp - rank - 1):
            full_bwd(s1); overlap_fb(s1, s0)

        # Phase 6: B1B0 (second half switches to zero-bubble split)
        enable_zb = False
        for i in range(rank + 1):
            if i == (rank + 1) // 2 and rank % 2 == 1:
                enable_zb = True
            (bwdi if enable_zb else full_bwd)(s1)
            if i == (rank + 1) // 2 and rank % 2 == 0:
                enable_zb = True
            (bwdi if enable_zb else full_bwd)(s0)

        # Phase 7: W0 B0
        for _ in range(pp - rank - 1):
            drain_w()
            (bwdi if enable_zb else full_bwd)(s0)

        # Phase 8: W0
        for _ in range(rank + 1):
            drain_w()

    return PipelineSchedule(grid=rows)


def visualize_pipeline_schedule(
    schedule: PipelineSchedule,
    output_path: str = "pipeline_schedule",
    fmt: str = "png",
) -> None:
    """Render a PipelineSchedule as a 2-D colored grid.

    Rows = PP ranks.  Columns = logical time steps.
    Each chunk occupies ``logical_time(chunk)`` consecutive columns.
    Colors: FWD = yellow, BWD/BWD_I = green, BWD_W = blue, UPD = orange.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COLORS = {
        CompType.FWD:     "#FFD700",   # yellow
        CompType.BWD:     "#90EE90",   # green
        CompType.BWD_I:   "#90EE90",   # green
        CompType.BWD_W:   "#87CEEB",   # blue
        CompType.UPD:     "#FFA500",   # orange
        CompType.FWD_BWD: "#DDA0DD",   # plum
    }

    start_times_per_rank = schedule._compute_start_times()
    n_ranks = len(schedule.grid)
    max_time = max(
        start_times_per_rank[r][i] + logical_time(schedule.grid[r][i])
        for r in range(n_ranks)
        for i in range(len(schedule.grid[r]))
    )

    CELL_W = 1.0   # width per logical time unit
    CELL_H = 1.0   # height per rank
    PAD = 0.04     # gap between cells

    fig_w = max_time * CELL_W + 1.5
    fig_h = n_ranks * CELL_H + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, max_time * CELL_W)
    ax.set_ylim(0, n_ranks * CELL_H)
    ax.axis("off")

    # Per-rank sorted unique stages, to determine virtual stage index (0=black, 1=white)
    rank_stages: list[list[int]] = []
    for row in schedule.grid:
        seen: list[int] = []
        for chunk in row:
            for b in chunk.batches:
                if b.stage_id not in seen:
                    seen.append(b.stage_id)
        rank_stages.append(sorted(seen))

    for rank, (row, starts) in enumerate(zip(schedule.grid, start_times_per_rank)):
        # rank 0 at top → y increases downward, so flip
        y = (n_ranks - 1 - rank) * CELL_H
        stages = rank_stages[rank]
        for chunk, t in zip(row, starts):
            lt = logical_time(chunk)
            color = COLORS.get(chunk.type, "#FFFFFF")
            x = t * CELL_W

            rect = mpatches.FancyBboxPatch(
                (x + PAD, y + PAD),
                lt * CELL_W - 2 * PAD,
                CELL_H - 2 * PAD,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="#555555",
                linewidth=0.6,
            )
            ax.add_patch(rect)

            if chunk.batches:
                b = chunk.batches[0]
                label = f"{chunk.type.name}\ns{b.stage_id} m{b.mb_idx}"
                vstage_idx = stages.index(b.stage_id) if b.stage_id in stages else 0
            else:
                label = chunk.type.name
                vstage_idx = 0
            font_color = "white" if vstage_idx % 2 == 1 else "black"
            ax.text(
                x + lt * CELL_W / 2,
                y + CELL_H / 2,
                label,
                ha="center", va="center",
                fontsize=7,
                fontweight="bold",
                color=font_color,
            )

    # Rank labels on the left
    for rank in range(n_ranks):
        y = (n_ranks - 1 - rank) * CELL_H + CELL_H / 2
        ax.text(-0.15, y, f"rank {rank}", ha="right", va="center", fontsize=8)

    plt.tight_layout(pad=0.3)
    out_file = f"{output_path}.{fmt}"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved schedule visualization to {out_file}")


ZEROBUBBLE_MB4_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),          # t0:  F0
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),          # t2:  F1
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD_I),        # t3:  I0
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD_W),        # t4:  W0
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),          # t5:  F2
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD_I),        # t6:  I1
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD_W),        # t7:  W1
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),          # t8:  F3
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD_I),        # t9:  I2
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD_W),        # t10: W2
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD_I),        # t11: I3
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD_W),        # t12: W3
        ],
        [
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),          # t1:  F0
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD_I),        # t2:  I0
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),          # t3:  F1
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD_I),        # t4:  I1
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD_W),        # t5:  W0
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),          # t6:  F2
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD_I),        # t7:  I2
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD_W),        # t8:  W1
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),          # t9:  F3
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD_I),        # t10: I3
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD_W),        # t11: W2
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD_W),        # t12: W3
        ],
    ])

NO_PP_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
        ]
    ],
)


DUALPIPEV_MB6_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=2), BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4), BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=3), BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5), BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4), BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5), BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
        ],
        [
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2), BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=2), BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3), BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=3), BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4), BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4), BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5), BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5), BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD_W),
        ]
    ])

DUALPIPEV_SEQUENTIAL_MB6_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
        ],
        [
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD_W),
        ]
    ])

DUALPIPEV_NOZB_MB6_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=2), BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4), BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=3), BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5), BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4), BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5), BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD_BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2), BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=2), BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3), BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=3), BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4), BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4), BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5), BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5), BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD_BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[], type=CompType.UPD),
        ]
    ])

INTERLEAVED_1F1B_PP2_MB6_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[], type=CompType.UPD),
        ],
    ])

INTERLEAVED_1F1B_PP2_MB4_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
        ],
        [
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
        ],
    ])


INTERLEAVED_GPIPE_PP2_MB4_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[], type=CompType.UPD),
        ],
    ])

INTERLEAVED_1F1B_PP4_MB8_SCHEDULE = PipelineSchedule(
    grid=[
        [
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[], type=CompType.UPD),
        ],
        [
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[], type=CompType.UPD),
        ],
        [
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[], type=CompType.UPD),
        ],
        [
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[], type=CompType.UPD),
        ],
    ])



INTERLEAVED_1F1B_PP8_MB16_SCHEDULE = PipelineSchedule(
    grid=[
        [  # rank 0, stages 0 and 8
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=8, mb_idx=15)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=15)], type=CompType.BWD),
        ],
        [  # rank 1, stages 1 and 9
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=9, mb_idx=15)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=15)], type=CompType.BWD),
        ],
        [  # rank 2, stages 2 and 10
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=10, mb_idx=15)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=15)], type=CompType.BWD),
        ],
        [  # rank 3, stages 3 and 11
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=11, mb_idx=15)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=15)], type=CompType.BWD),
        ],
        [  # rank 4, stages 4 and 12
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=12, mb_idx=15)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=15)], type=CompType.BWD),
        ],
        [  # rank 5, stages 5 and 13
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=13, mb_idx=15)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=15)], type=CompType.BWD),
        ],
        [  # rank 6, stages 6 and 14
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=14, mb_idx=15)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=15)], type=CompType.BWD),
        ],
        [  # rank 7, stages 7 and 15
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=15, mb_idx=15)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=8)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=9)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=10)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=11)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=12)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=13)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=14)], type=CompType.BWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=15)], type=CompType.BWD),
        ],
    ])


ZEROBUBBLE_PP8_MB16_SCHEDULE = PipelineSchedule(
    grid=[
        [  # rank 0, stage 0   warmup=7 1f1b=9 cooldown=7
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=2)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=3)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=6)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=7)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=8)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=8)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=9)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=9)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=10)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=10)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=11)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=11)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=12)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=12)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=13)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=13)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=14)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=14)], type=CompType.BWD_W),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=15)], type=CompType.BWD_I),
            Chunk(pp_rank=0, batches=[BatchMeta(stage_id=0, mb_idx=15)], type=CompType.BWD_W),
        ],
        [  # rank 1, stage 1   warmup=6 1f1b=10 cooldown=6
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=2)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=3)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=6)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=8)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=7)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=9)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=8)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=10)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=9)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=11)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=10)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=12)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=11)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=13)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=12)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=14)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=13)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=15)], type=CompType.BWD_I),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=14)], type=CompType.BWD_W),
            Chunk(pp_rank=1, batches=[BatchMeta(stage_id=1, mb_idx=15)], type=CompType.BWD_W),
        ],
        [  # rank 2, stage 2   warmup=5 1f1b=11 cooldown=5
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=2)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=3)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=8)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=6)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=9)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=7)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=10)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=8)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=11)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=9)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=12)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=10)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=13)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=11)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=14)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=12)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=15)], type=CompType.BWD_I),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=13)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=14)], type=CompType.BWD_W),
            Chunk(pp_rank=2, batches=[BatchMeta(stage_id=2, mb_idx=15)], type=CompType.BWD_W),
        ],
        [  # rank 3, stage 3   warmup=4 1f1b=12 cooldown=4
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=2)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=3)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=8)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=9)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=6)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=10)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=7)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=11)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=8)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=12)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=9)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=13)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=10)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=14)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=11)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=15)], type=CompType.BWD_I),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=12)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=13)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=14)], type=CompType.BWD_W),
            Chunk(pp_rank=3, batches=[BatchMeta(stage_id=3, mb_idx=15)], type=CompType.BWD_W),
        ],
        [  # rank 4, stage 4   warmup=3 1f1b=13 cooldown=3
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=2)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=3)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=8)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=9)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=10)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=6)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=11)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=7)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=12)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=8)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=13)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=9)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=14)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=10)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=15)], type=CompType.BWD_I),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=11)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=12)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=13)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=14)], type=CompType.BWD_W),
            Chunk(pp_rank=4, batches=[BatchMeta(stage_id=4, mb_idx=15)], type=CompType.BWD_W),
        ],
        [  # rank 5, stage 5   warmup=2 1f1b=14 cooldown=2
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=2)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=8)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=3)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=9)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=10)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=11)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=6)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=12)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=7)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=13)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=8)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=14)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=9)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=15)], type=CompType.BWD_I),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=10)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=11)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=12)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=13)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=14)], type=CompType.BWD_W),
            Chunk(pp_rank=5, batches=[BatchMeta(stage_id=5, mb_idx=15)], type=CompType.BWD_W),
        ],
        [  # rank 6, stage 6   warmup=1 1f1b=15 cooldown=1
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=8)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=2)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=9)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=3)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=10)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=11)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=12)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=6)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=13)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=7)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=14)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=8)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=15)], type=CompType.BWD_I),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=9)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=10)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=11)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=12)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=13)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=14)], type=CompType.BWD_W),
            Chunk(pp_rank=6, batches=[BatchMeta(stage_id=6, mb_idx=15)], type=CompType.BWD_W),
        ],
        [  # rank 7, stage 7   warmup=0 1f1b=16 cooldown=0
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=0)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=8)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=8)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=1)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=9)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=9)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=2)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=10)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=10)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=3)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=11)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=11)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=4)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=12)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=12)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=5)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=13)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=13)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=6)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=14)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=14)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=7)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=15)], type=CompType.FWD),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=15)], type=CompType.BWD_I),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=8)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=9)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=10)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=11)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=12)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=13)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=14)], type=CompType.BWD_W),
            Chunk(pp_rank=7, batches=[BatchMeta(stage_id=7, mb_idx=15)], type=CompType.BWD_W),
        ],
    ])
