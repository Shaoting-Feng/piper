"""Build JSON order directives for common pipeline schedules."""

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from html import escape
from pathlib import Path
import subprocess
from typing import Literal

Pass = Literal["F", "B", "BI", "BW"]
_VALID_PASSES = frozenset({"F", "B", "BI", "BW"})


@dataclass(frozen=True)
class _Op:
    pp: int
    mb: int
    pass_: Pass


# A "slot" is one timestep in a rank's row. A slot holds one op normally; when
# multiple ops are nested in the same order-directive filter group they share a
# slot and visualize as a vertically-stacked cell, indicating that they run
# interleaved within the same scheduling block.
_Slot = list[_Op]


_PASS_COLOR: dict[str, str] = {
    "F": "#FFE08A",
    "B": "#A8E6A1",
    "BI": "#A9D6FF",
    "BW": "#C6B6FF",
}


def _filter(pp: int, mb: int, pass_: Pass) -> list[list[int | str]]:
    return [["PP", pp], ["MB", mb], ["PASS", pass_]]


def _order_directive(ops: list[_Op]) -> dict:
    return _order_directive_from_slots([[op] for op in ops])


def _order_directive_from_slots(slots: list[_Slot]) -> dict:
    return {
        "op": "order",
        "filters": [
            [_filter(op.pp, op.mb, op.pass_) for op in slot]
            for slot in slots
        ],
    }


def _op_width(op: _Op) -> int:
    if op.pass_ == "B":
        return 2
    return 1


def _slot_width(slot: _Slot) -> int:
    return sum(_op_width(op) for op in slot)


def _cleanup_visualizer_sidecars(
    output_base: Path,
    keep_suffix: str | None = None,
) -> None:
    for suffix in (".dot", ".txt"):
        if suffix == keep_suffix:
            continue
        with suppress(FileNotFoundError):
            output_base.with_suffix(suffix).unlink()


def visualize_order_directives(
    order_directives: list[dict],
    output_path: str | Path = "out/schedule",
    fmt: str = "png",
) -> str:
    """Render a 2-D schedule grid, one row per physical rank.

    The order directives do not carry explicit ``None`` slots. Columns are
    reconstructed as earliest feasible logical times from per-rank temporal
    order and FWD/BWD pipeline data dependencies.
    """
    rows = _rows_from_order_directives(order_directives)
    placements = _compute_columns(rows)
    max_col = max(
        (col + _slot_width(rows[rank][slot_idx]) - 1 for (rank, slot_idx), col in placements.items()),
        default=-1,
    )

    table: list[list[_Slot | None]] = [
        [None for _ in range(max_col + 1)]
        for _ in rows
    ]
    for key, col in placements.items():
        rank, slot_idx = key
        table[rank][col] = rows[rank][slot_idx]

    output = Path(output_path)
    if output.suffix:
        fmt = output.suffix.lstrip(".")
        output = output.with_suffix("")

    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_suffix = f".{fmt}"
    _cleanup_visualizer_sidecars(output, keep_suffix=rendered_suffix)
    try:
        import graphviz

        dot = graphviz.Digraph("PipelineSchedule")
        dot.attr(rankdir="LR")
        dot.node("schedule", label=_html_table(table), shape="plain")
        try:
            return dot.render(str(output), format=fmt, cleanup=True)
        finally:
            _cleanup_visualizer_sidecars(output, keep_suffix=rendered_suffix)
    except ImportError:
        dot_path = output.with_suffix(".dot")
        rendered_path = output.with_suffix(f".{fmt}")
        dot_path.write_text(_dot_source(table), encoding="utf-8")
        rendered_ok = False
        try:
            subprocess.run(
                ["dot", f"-T{fmt}", str(dot_path), "-o", str(rendered_path)],
                check=True,
            )
            rendered_ok = True
            return str(rendered_path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            fallback = output.with_suffix(".txt")
            fallback.write_text(_format_text_table(table), encoding="utf-8")
            return str(fallback)
        finally:
            if rendered_ok:
                _cleanup_visualizer_sidecars(output, keep_suffix=rendered_suffix)


def build_1f1b_schedule(n_ranks: int, n_mbs: int) -> list[dict]:
    """Return one 1F1B order directive per rank.

    This preserves the old PipelineSchedule builder's non-None per-rank op order:
    rank r warms up with ``n_ranks - 1 - r`` forwards, then alternates forward
    and fused backward until all microbatches have completed.
    """
    _validate_positive("n_ranks", n_ranks)
    _validate_positive("n_mbs", n_mbs)

    rows: list[list[_Op]] = [[] for _ in range(n_ranks)]
    fwd_mb = [0] * n_ranks
    bwd_mb = [0] * n_ranks

    for rank in range(n_ranks):
        warmup = n_ranks - 1 - rank
        for _ in range(min(warmup, n_mbs)):
            rows[rank].append(_Op(pp=rank, mb=fwd_mb[rank], pass_="F"))
            fwd_mb[rank] += 1

    for rank in range(n_ranks):
        while bwd_mb[rank] < n_mbs:
            if fwd_mb[rank] < n_mbs:
                rows[rank].append(_Op(pp=rank, mb=fwd_mb[rank], pass_="F"))
                fwd_mb[rank] += 1
            rows[rank].append(_Op(pp=rank, mb=bwd_mb[rank], pass_="B"))
            bwd_mb[rank] += 1

    return [_order_directive(row) for row in rows]


def build_zerobubble_schedule(n_ranks: int, n_mbs: int) -> list[dict]:
    """Return one ZeroBubble order directive per rank.

    This is the old ZB-1 schedule builder expressed as order directives. Each
    rank owns one pipeline stage, fused backward is split into BI/BW, and BW is
    deferred by ``rank`` backward-input steps.
    """
    _validate_positive("n_ranks", n_ranks)
    _validate_positive("n_mbs", n_mbs)

    rows: list[list[_Op]] = []
    for rank in range(n_ranks):
        warmup = min(n_ranks - 1 - rank, n_mbs)
        fwd_bwd_ops = n_mbs - warmup
        cooldown_ops = n_mbs - fwd_bwd_ops
        deferred_bw_count = rank

        fwd_mb = 0
        bwd_mb = 0
        bw_mb = 0
        bwdi_count = 0
        emitted_bw_count = 0
        ops: list[_Op] = []

        for op_idx in range(warmup + fwd_bwd_ops + cooldown_ops):
            if op_idx < warmup:
                ops.append(_Op(pp=rank, mb=fwd_mb, pass_="F"))
                fwd_mb += 1
                continue

            if op_idx < warmup + fwd_bwd_ops:
                ops.append(_Op(pp=rank, mb=fwd_mb, pass_="F"))
                fwd_mb += 1

            ops.append(_Op(pp=rank, mb=bwd_mb, pass_="BI"))
            bwd_mb += 1
            bwdi_count += 1

            if bwdi_count > deferred_bw_count:
                ops.append(_Op(pp=rank, mb=bw_mb, pass_="BW"))
                bw_mb += 1
                emitted_bw_count += 1

        while emitted_bw_count < bwdi_count:
            ops.append(_Op(pp=rank, mb=bw_mb, pass_="BW"))
            bw_mb += 1
            emitted_bw_count += 1

        rows.append(ops)

    return [_order_directive(row) for row in rows]


def build_interleaved_1f1b_schedule(
    n_ranks: int,
    n_mbs: int,
    n_virtual_stages: int,
) -> list[dict]:
    """Return one interleaved 1F1B order directive per physical rank.

    Rank ``r`` owns virtual pipeline stages ``r + k * n_ranks``. The op order is
    the old interleaved PipelineSchedule order with the placeholder ``None`` slots
    omitted.
    """
    _validate_positive("n_ranks", n_ranks)
    _validate_positive("n_mbs", n_mbs)
    _validate_positive("n_virtual_stages", n_virtual_stages)

    microbatches_per_round = n_ranks

    def warmup_ops(rank: int) -> int:
        return min(
            (n_virtual_stages - 1) * n_ranks + 2 * (n_ranks - 1 - rank),
            n_mbs * n_virtual_stages,
        )

    def forward_stage_index(step: int, rank: int) -> int:
        local_index = (step // microbatches_per_round) % n_virtual_stages
        return local_index * n_ranks + rank

    def backward_stage_index(step: int, rank: int, warmup: int) -> int:
        local_index = (
            n_virtual_stages
            - 1
            - ((step - warmup) // microbatches_per_round) % n_virtual_stages
        )
        return local_index * n_ranks + rank

    rows: list[list[_Op]] = []
    for rank in range(n_ranks):
        warmup = warmup_ops(rank)
        microbatch_ops = n_virtual_stages * n_mbs
        fwd_bwd_ops = microbatch_ops - warmup
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        fwd_mb: dict[int, int] = defaultdict(int)
        bwd_mb: dict[int, int] = defaultdict(int)
        ops: list[_Op] = []

        for op_idx in range(warmup + fwd_bwd_ops + cooldown_ops):
            if op_idx < warmup:
                stage = forward_stage_index(op_idx, rank)
                mb = fwd_mb[stage]
                fwd_mb[stage] += 1
                ops.append(_Op(pp=stage, mb=mb, pass_="F"))
            elif op_idx < warmup + fwd_bwd_ops:
                stage = forward_stage_index(op_idx, rank)
                mb = fwd_mb[stage]
                fwd_mb[stage] += 1
                ops.append(_Op(pp=stage, mb=mb, pass_="F"))

                stage = backward_stage_index(op_idx, rank, warmup)
                mb = bwd_mb[stage]
                bwd_mb[stage] += 1
                ops.append(_Op(pp=stage, mb=mb, pass_="B"))
            else:
                stage = backward_stage_index(op_idx, rank, warmup)
                mb = bwd_mb[stage]
                bwd_mb[stage] += 1
                ops.append(_Op(pp=stage, mb=mb, pass_="B"))

        rows.append(ops)

    return [_order_directive(row) for row in rows]


def build_dualpipev_schedule(n_ranks: int, n_mbs: int) -> list[dict]:
    """Return DualPipeV order directives with overlapped FWD/BWD slots.

    Rank ``r`` owns V-layout stages ``r`` and ``2 * n_ranks - 1 - r``. The
    old ``FWD_BWD`` chunk is represented as one nested order filter group
    containing the corresponding FWD filter and fused BWD filter.
    """
    _validate_positive("n_ranks", n_ranks)
    _validate_positive("n_mbs", n_mbs)
    if n_mbs < 2 * n_ranks:
        raise ValueError(
            f"dualpipev requires n_mbs >= 2 * n_ranks, got n_mbs={n_mbs}, "
            f"n_ranks={n_ranks}"
        )

    rows: list[list[_Slot]] = [[] for _ in range(n_ranks)]

    for rank in range(n_ranks):
        s0 = rank
        s1 = 2 * n_ranks - 1 - rank
        slots = rows[rank]
        counts: dict[tuple[int, str], int] = {}
        weight_queue: list[tuple[int, int]] = []

        def count(stage: int, key: str) -> int:
            return counts.get((stage, key), 0)

        def inc(stage: int, key: str) -> None:
            counts[(stage, key)] = count(stage, key) + 1

        def inc_bwd(stage: int) -> None:
            inc(stage, "i")
            inc(stage, "w")

        def append_op(stage: int, mb: int, pass_: Pass) -> None:
            slots.append([_Op(pp=stage, mb=mb, pass_=pass_)])

        def fwd(stage: int) -> None:
            mb = count(stage, "f")
            append_op(stage, mb, "F")
            inc(stage, "f")

        def bwdi(stage: int) -> None:
            mb = count(stage, "i")
            append_op(stage, mb, "BI")
            weight_queue.append((stage, mb))
            inc(stage, "i")

        def drain_w() -> None:
            if not weight_queue:
                return
            stage, mb = weight_queue.pop(0)
            append_op(stage, mb, "BW")
            inc(stage, "w")

        def full_bwd(stage: int) -> None:
            mb = count(stage, "i")
            append_op(stage, mb, "B")
            inc_bwd(stage)

        def overlap_fb(fwd_stage: int, bwd_stage: int) -> None:
            fwd_mb = count(fwd_stage, "f")
            bwd_mb = count(bwd_stage, "i")
            slots.append([
                _Op(pp=fwd_stage, mb=fwd_mb, pass_="F"),
                _Op(pp=bwd_stage, mb=bwd_mb, pass_="B"),
            ])
            inc(fwd_stage, "f")
            inc_bwd(bwd_stage)

        # Phase 1: F0 warmup.
        for _ in range((n_ranks - rank - 1) * 2):
            fwd(s0)

        # Phase 2: F0F1.
        for _ in range(rank + 1):
            fwd(s0)
            fwd(s1)

        # Phase 3: I1 W1 F1.
        for _ in range(n_ranks - rank - 1):
            bwdi(s1)
            drain_w()
            fwd(s1)

        # Phase 4: Main overlapped F0B1 + F1B0.
        for i in range(n_mbs - n_ranks * 2 + rank + 1):
            if i == 0 and rank == n_ranks - 1:
                fwd(s0)
                full_bwd(s1)
            else:
                overlap_fb(s0, s1)
            overlap_fb(s1, s0)

        # Phase 5: B1 + F1B0.
        for _ in range(n_ranks - rank - 1):
            full_bwd(s1)
            overlap_fb(s1, s0)

        # Phase 6: B1B0, switching the second half to split backward.
        enable_zb = False
        for i in range(rank + 1):
            if i == (rank + 1) // 2 and rank % 2 == 1:
                enable_zb = True
            (bwdi if enable_zb else full_bwd)(s1)
            if i == (rank + 1) // 2 and rank % 2 == 0:
                enable_zb = True
            (bwdi if enable_zb else full_bwd)(s0)

        # Phase 7: W0 B0.
        for _ in range(n_ranks - rank - 1):
            drain_w()
            (bwdi if enable_zb else full_bwd)(s0)

        # Phase 8: W0.
        for _ in range(rank + 1):
            drain_w()

    return [_order_directive_from_slots(row) for row in rows]


def _validate_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _rows_from_order_directives(order_directives: list[dict]) -> list[list[_Slot]]:
    rows = []
    for directive in order_directives:
        if directive.get("op") != "order":
            raise ValueError(f"expected order directive, got {directive}")
        filters = directive.get("filters")
        if not isinstance(filters, list):
            raise ValueError(f"order directive requires filters list: {directive}")
        row: list[_Slot] = []
        for filter_slot in _parse_filter_slots(filters):
            slot: _Slot = []
            for flt in filter_slot:
                spec = _filter_to_dict(flt)
                pass_value = spec["PASS"]
                if pass_value not in _VALID_PASSES:
                    raise ValueError(
                        f"unsupported PASS={pass_value!r}; expected one of {sorted(_VALID_PASSES)}"
                    )
                slot.append(_Op(pp=int(spec["PP"]), mb=int(spec["MB"]), pass_=pass_value))
            row.append(slot)
        rows.append(row)
    return rows


def _is_filter_spec(flt: object) -> bool:
    if isinstance(flt, dict):
        return True
    if isinstance(flt, list):
        return all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            for item in flt
        )
    return False


def _parse_filter_slots(filters: list) -> list[list]:
    """Group the directive's filter entries into slots.

    Each top-level entry in ``filters`` corresponds to one slot (one timestep
    on the owning rank). A bare filter spec becomes a single-op slot. A list of
    filter specs becomes a multi-op slot whose ops run interleaved at the same
    scheduling step and visualize stacked vertically in one block.
    """
    slots: list[list] = []
    for item in filters:
        if _is_filter_spec(item):
            slots.append([item])
        elif isinstance(item, list):
            slots.append(list(item))
        else:
            raise ValueError(f"invalid order filter group: {item}")
    return slots


def _filter_to_dict(flt: object) -> dict:
    if isinstance(flt, dict):
        return dict(flt)
    if isinstance(flt, list):
        out = {}
        for item in flt:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"invalid order filter item: {item}")
            out[item[0]] = item[1]
        return out
    raise ValueError(f"invalid order filter: {flt}")


def _compute_columns(rows: list[list[_Slot]]) -> dict[tuple[int, int], int]:
    # Each (rank, slot_idx) is one timeslot. Multiple ops within a slot share
    # that slot's column; the slot's width is max(op width) across its ops.
    key_by_op: dict[tuple[int, int, str], tuple[int, int]] = {}
    for rank, row in enumerate(rows):
        for slot_idx, slot in enumerate(row):
            for op in slot:
                op_key = (op.pp, op.mb, op.pass_)
                if op_key in key_by_op:
                    raise ValueError(f"duplicate scheduled op: {op_key}")
                key_by_op[op_key] = (rank, slot_idx)

    preds: dict[tuple[int, int], set[tuple[int, int]]] = {
        (rank, slot_idx): set()
        for rank, row in enumerate(rows)
        for slot_idx, _slot in enumerate(row)
    }
    succs: dict[tuple[int, int], set[tuple[int, int]]] = {
        key: set() for key in preds
    }

    def add_edge(src: tuple[int, int] | None, dst: tuple[int, int] | None) -> None:
        if src is None or dst is None or src == dst:
            return
        preds[dst].add(src)
        succs[src].add(dst)

    for rank, row in enumerate(rows):
        for slot_idx in range(1, len(row)):
            add_edge((rank, slot_idx - 1), (rank, slot_idx))

    for rank, row in enumerate(rows):
        for slot_idx, slot in enumerate(row):
            dst = (rank, slot_idx)
            for op in slot:
                if op.pass_ == "F":
                    add_edge(key_by_op.get((op.pp - 1, op.mb, "F")), dst)
                elif op.pass_ == "B":
                    add_edge(key_by_op.get((op.pp, op.mb, "F")), dst)
                    add_edge(key_by_op.get((op.pp + 1, op.mb, "B")), dst)
                elif op.pass_ == "BI":
                    add_edge(key_by_op.get((op.pp, op.mb, "F")), dst)
                    add_edge(key_by_op.get((op.pp + 1, op.mb, "BI")), dst)
                elif op.pass_ == "BW":
                    add_edge(key_by_op.get((op.pp, op.mb, "BI")), dst)
                else:
                    raise ValueError(f"unsupported pass={op.pass_!r}")

    columns = {key: 0 for key in preds}
    pending = {key: set(value) for key, value in preds.items()}
    ready = [key for key, value in pending.items() if not value]
    visited = 0

    while ready:
        key = ready.pop(0)
        visited += 1
        for succ in sorted(succs[key]):
            rank, slot_idx = key
            columns[succ] = max(columns[succ], columns[key] + _slot_width(rows[rank][slot_idx]))
            pending[succ].discard(key)
            if not pending[succ]:
                ready.append(succ)

    if visited != len(preds):
        blocked = [key for key, value in pending.items() if value]
        raise ValueError(f"schedule dependencies contain a cycle; blocked={blocked[:8]}")
    return columns


def _html_table(table: list[list[_Slot | None]]) -> str:
    max_cols = max((len(row) for row in table), default=0)
    lines = ['<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">']
    lines.append('<TR><TD BGCOLOR="#eeeeee">rank</TD>')
    for col in range(max_cols):
        lines.append(f'<TD BGCOLOR="#eeeeee">t{col}</TD>')
    lines.append("</TR>")
    for rank, row in enumerate(table):
        lines.append(f'<TR><TD BGCOLOR="#eeeeee">r{rank}</TD>')
        col = 0
        while col < len(row):
            slot = row[col]
            if slot is None:
                lines.append('<TD WIDTH="72" HEIGHT="36"></TD>')
                col += 1
            else:
                colspan = _slot_width(slot)
                colspan_attr = f' COLSPAN="{colspan}"' if colspan > 1 else ""
                width = 72 * colspan
                if len(slot) == 1:
                    op = slot[0]
                    color = _PASS_COLOR[op.pass_]
                    label = f"{escape(op.pass_)}<BR/>PP{op.pp} MB{op.mb}"
                    lines.append(
                        f'<TD BGCOLOR="{color}" WIDTH="{width}" HEIGHT="36"{colspan_attr}>{label}</TD>'
                    )
                else:
                    # Multi-op slot: ops run interleaved and stack vertically in
                    # one block. Use a nested table so each op gets its own
                    # colored row spanning the full slot width.
                    inner = ['<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">']
                    for op in slot:
                        color = _PASS_COLOR[op.pass_]
                        label = f"{escape(op.pass_)}<BR/>PP{op.pp} MB{op.mb}"
                        inner.append(
                            f'<TR><TD BGCOLOR="{color}" WIDTH="{width}" HEIGHT="36">{label}</TD></TR>'
                        )
                    inner.append("</TABLE>")
                    lines.append(
                        f'<TD WIDTH="{width}"{colspan_attr} CELLPADDING="0">{"".join(inner)}</TD>'
                    )
                col += colspan
        lines.append("</TR>")
    lines.append("</TABLE>>")
    return "".join(lines)


def _dot_source(table: list[list[_Op | None]]) -> str:
    return "\n".join([
        "digraph PipelineSchedule {",
        "  rankdir=LR;",
        f"  schedule [shape=plain label={_html_table(table)}];",
        "}",
        "",
    ])


def _format_text_table(table: list[list[_Slot | None]]) -> str:
    lines = []
    for rank, row in enumerate(table):
        cells = []
        col = 0
        while col < len(row):
            slot = row[col]
            if slot is None:
                cells.append("--------")
                col += 1
            else:
                width = _slot_width(slot)
                # Stacked ops in one slot join with "/" to flag interleaving.
                label = "/".join(f"{op.pass_}:s{op.pp}:m{op.mb}" for op in slot)
                cells.append(label.ljust(8 * width, "="))
                col += width
        lines.append(f"rank {rank}: " + " | ".join(cells))
    return "\n".join(lines) + "\n"
