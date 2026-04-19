#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TORCHTITAN_COLOR = "#4C72B0"
PIPER_COLOR = "#DD8452"
SWEEPS = ("scalability", "zero", "schedule")
ZERO_MAP = {"none": 0, "zero1": 1, "zero2": 2, "zero3": 3}


@dataclass(frozen=True)
class CommonRow:
    source: str
    sweep: str
    key: tuple[Any, ...]
    label: str
    pp: int
    dp: int
    zero_level: int
    schedule: str
    mb_size: int
    seq_len: int
    status: str
    iter_time_mean: float | None
    iter_time_std: float | None
    peak_memory_values: list[float]


@dataclass(frozen=True)
class PlotSeries:
    name: str
    color: str
    row: CommonRow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create unified torchtitan+piper iteration-time and memory plots for matched Qwen e2e sweeps."
    )
    parser.add_argument("--torchtitan-results", required=True, help="Path to torchtitan all_results.csv")
    parser.add_argument("--piper-results", required=True, help="Path to piper all_results.csv")
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=SWEEPS,
        default=list(SWEEPS),
        help="Subset of sweeps to compare. Default: scalability zero schedule",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for combined plots and matched CSVs",
    )
    return parser.parse_args()



def parse_torchtitan_peak_memory(summary: str) -> list[float]:
    summary = summary.strip()
    if not summary:
        return []
    values: list[float] = []
    for part in summary.split("/"):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values



def parse_piper_peak_memory(summary: str) -> list[float]:
    summary = summary.strip()
    if not summary:
        return []
    values: list[float] = []
    for part in summary.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        _, value_text = part.split(":", maxsplit=1)
        try:
            values.append(float(value_text))
        except ValueError:
            continue
    return values



def torchtitan_status(raw: str) -> str:
    if raw == "ok":
        return "success"
    return raw



def torchtitan_key(row: dict[str, str]) -> tuple[Any, ...]:
    sweep = row["sweep"]
    if sweep == "scalability":
        return (sweep, int(row["pp"]), int(row["dp"]))
    if sweep == "zero":
        return (sweep, ZERO_MAP[row["zero_level"]], int(row["mb_size"]))
    return (sweep, row["schedule"])



def torchtitan_label(row: dict[str, str]) -> str:
    sweep = row["sweep"]
    if sweep == "scalability":
        return f"(pp={row['pp']}, dp={row['dp']})"
    if sweep == "zero":
        return f"(zero={ZERO_MAP[row['zero_level']]}, mb={row['mb_size']})"
    return row["schedule"]



def load_torchtitan_rows(path: Path) -> dict[tuple[Any, ...], CommonRow]:
    rows: dict[tuple[Any, ...], CommonRow] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iter_mean_text = row.get("iter_time_mean", "").strip()
            iter_std_text = row.get("iter_time_stddev", "").strip()
            iter_time_mean = float(iter_mean_text) if iter_mean_text else None
            iter_time_std = float(iter_std_text) if iter_std_text else None
            common = CommonRow(
                source="torchtitan",
                sweep=row["sweep"],
                key=torchtitan_key(row),
                label=torchtitan_label(row),
                pp=int(row["pp"]),
                dp=int(row["dp"]),
                zero_level=ZERO_MAP[row["zero_level"]],
                schedule=row["schedule"],
                mb_size=int(row["mb_size"]),
                seq_len=int(row["seq_len"]),
                status=torchtitan_status(row["status"]),
                iter_time_mean=iter_time_mean,
                iter_time_std=iter_time_std,
                peak_memory_values=parse_torchtitan_peak_memory(row.get("peak_memory_gb_by_rank", "")),
            )
            rows[common.key] = common
    return rows



def piper_key(row: dict[str, str]) -> tuple[Any, ...]:
    sweep = row["sweep"]
    if sweep == "scalability":
        return (sweep, int(row["pp"]), int(row["dp"]))
    if sweep == "zero":
        return (sweep, int(row["zero_stage"]), int(row["batch_size"]), int(row["gradient_accumulation"]))
    return (sweep, row["schedule"])



def load_piper_rows(path: Path) -> dict[tuple[Any, ...], CommonRow]:
    rows: dict[tuple[Any, ...], CommonRow] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iter_time_mean = row.get("iter_time_mean_s", "").strip()
            iter_time_std = row.get("iter_time_std_s", "").strip()
            common = CommonRow(
                source="piper",
                sweep=row["sweep"],
                key=piper_key(row),
                label=row["label"],
                pp=int(row["pp"]),
                dp=int(row["dp"]),
                zero_level=int(row["zero_stage"]),
                schedule=row["schedule"],
                mb_size=int(row["batch_size"]),
                seq_len=int(row["seq_len"]),
                status=row["status"],
                iter_time_mean=float(iter_time_mean) if iter_time_mean else None,
                iter_time_std=float(iter_time_std) if iter_time_std else None,
                peak_memory_values=parse_piper_peak_memory(row.get("peak_memory_gb_by_rank", "")),
            )
            rows[common.key] = common
    return rows



def validate_pair_consistency(torchtitan_row: CommonRow, piper_row: CommonRow) -> None:
    mismatches: list[str] = []
    for field in ("pp", "dp", "zero_level", "schedule", "mb_size", "seq_len"):
        tt_value = getattr(torchtitan_row, field)
        piper_value = getattr(piper_row, field)
        if tt_value != piper_value:
            mismatches.append(f"{field}: torchtitan={tt_value!r} piper={piper_value!r}")
    if mismatches:
        raise ValueError(
            "Inconsistent matched experiment for "
            f"sweep={torchtitan_row.sweep} "
            f"torchtitan_key={torchtitan_row.key!r} "
            f"piper_key={piper_row.key!r}: "
            + ", ".join(mismatches)
        )



def save_grouped_iter_time_plot(title: str, grouped_rows: list[list[PlotSeries]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [group[0].row.label for group in grouped_rows]
    xs = list(range(len(grouped_rows)))
    width = 0.24 if max((len(group) for group in grouped_rows), default=0) >= 3 else 0.34

    success_values = []
    for group in grouped_rows:
        for series in group:
            row = series.row
            if row.status == "success" and row.iter_time_mean is not None:
                success_values.append(row.iter_time_mean)
    marker_y = max(success_values, default=0.0) * 0.05 if success_values else 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(grouped_rows) * 0.9), 6))
    legend_seen: set[str] = set()
    for index, group in enumerate(grouped_rows):
        offsets = centered_offsets(len(group), width)
        for offset, series in zip(offsets, group):
            row = series.row
            xpos = index + offset
            label = series.name if series.name not in legend_seen else None
            if row.status == "success" and row.iter_time_mean is not None:
                ax.bar(
                    xpos,
                    row.iter_time_mean,
                    yerr=row.iter_time_std or 0.0,
                    width=width * 0.9,
                    color=series.color,
                    ecolor="#2F2F2F",
                    capsize=4,
                    label=label,
                )
            else:
                ax.scatter(
                    xpos,
                    marker_y,
                    marker="x",
                    s=120,
                    color=series.color,
                    linewidths=2.5,
                    label=label,
                )
            legend_seen.add(series.name)

    ax.set_title(title)
    ax.set_ylabel("Iteration Time (s)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def save_grouped_memory_plot(title: str, grouped_rows: list[list[PlotSeries]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [group[0].row.label for group in grouped_rows]
    xs = list(range(len(grouped_rows)))
    spacing = 0.16 if max((len(group) for group in grouped_rows), default=0) >= 3 else 0.18

    success_values = [
        value
        for group in grouped_rows
        for series in group
        if series.row.status == "success"
        for value in series.row.peak_memory_values
    ]
    marker_y = max(success_values, default=0.0) * 0.05 if success_values else 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(grouped_rows) * 0.9), 6))
    legend_seen: set[str] = set()
    for index, group in enumerate(grouped_rows):
        offsets = centered_offsets(len(group), spacing)
        for shift, series in zip(offsets, group):
            row = series.row
            xpos = index + shift
            label = series.name if series.name not in legend_seen else None
            if row.status == "success" and row.peak_memory_values:
                ax.scatter(
                    [xpos] * len(row.peak_memory_values),
                    row.peak_memory_values,
                    color=series.color,
                    s=36,
                    alpha=0.9,
                    label=label,
                )
            else:
                ax.scatter(
                    xpos,
                    marker_y,
                    marker="x",
                    s=120,
                    color=series.color,
                    linewidths=2.5,
                    label=label,
                )
            legend_seen.add(series.name)

    ax.set_title(title)
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def centered_offsets(count: int, spacing: float) -> list[float]:
    midpoint = (count - 1) / 2
    return [(index - midpoint) * spacing for index in range(count)]



def build_plot_groups(
    sweep: str,
    sweep_pairs: list[tuple[CommonRow, CommonRow]],
    piper_rows: dict[tuple[Any, ...], CommonRow],
) -> list[list[PlotSeries]]:
    if sweep != "zero":
        return [
            [
                PlotSeries(name="torchtitan", color=TORCHTITAN_COLOR, row=tt),
                PlotSeries(name="piper", color=PIPER_COLOR, row=pp),
            ]
            for tt, pp in sweep_pairs
        ]

    groups: list[list[PlotSeries]] = []
    for tt, _ in sweep_pairs:
        zero_stage = tt.key[1]
        batch_size = tt.key[2]
        piper_ga_on = piper_rows[(sweep, zero_stage, batch_size, 1)]
        piper_ga_off = piper_rows[(sweep, zero_stage, batch_size, 0)]
        groups.append(
            [
                PlotSeries(name="torchtitan", color=TORCHTITAN_COLOR, row=tt),
                PlotSeries(name="piper ga=on", color=PIPER_COLOR, row=piper_ga_on),
                PlotSeries(name="piper ga=off", color="#55A868", row=piper_ga_off),
            ]
        )
    return groups



def collect_standard_sweep_pairs(
    sweep: str,
    torchtitan_rows: dict[tuple[Any, ...], CommonRow],
    piper_rows: dict[tuple[Any, ...], CommonRow],
) -> list[tuple[CommonRow, CommonRow]]:
    common_keys = sorted(set(torchtitan_rows) & set(piper_rows))
    pairs: list[tuple[CommonRow, CommonRow]] = []
    for key in common_keys:
        torchtitan_row = torchtitan_rows[key]
        piper_row = piper_rows[key]
        if torchtitan_row.sweep != sweep or piper_row.sweep != sweep:
            continue
        validate_pair_consistency(torchtitan_row, piper_row)
        pairs.append((torchtitan_row, piper_row))
    return pairs



def collect_zero_sweep_pairs(
    torchtitan_rows: dict[tuple[Any, ...], CommonRow],
    piper_rows: dict[tuple[Any, ...], CommonRow],
) -> list[tuple[CommonRow, CommonRow]]:
    pairs: list[tuple[CommonRow, CommonRow]] = []
    for key in sorted(torchtitan_rows):
        tt = torchtitan_rows[key]
        if tt.sweep != "zero":
            continue
        zero_stage = tt.key[1]
        batch_size = tt.key[2]
        piper_ga_on_key = ("zero", zero_stage, batch_size, 1)
        piper_ga_off_key = ("zero", zero_stage, batch_size, 0)
        if piper_ga_on_key not in piper_rows:
            logger.warning(
                "Missing piper zero GA=on result for torchtitan experiment: label=%s key=%r",
                tt.label,
                key,
            )
            continue
        if piper_ga_off_key not in piper_rows:
            logger.warning(
                "Missing piper zero GA=off result for torchtitan experiment: label=%s key=%r",
                tt.label,
                key,
            )
            continue
        piper_ga_on = piper_rows[piper_ga_on_key]
        piper_ga_off = piper_rows[piper_ga_off_key]
        validate_pair_consistency(tt, piper_ga_on)
        validate_pair_consistency(tt, piper_ga_off)
        pairs.append((tt, piper_ga_on))
    return pairs



def write_matched_csv(path: Path, rows: list[tuple[CommonRow, CommonRow]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sweep",
            "label",
            "key",
            "torchtitan_status",
            "torchtitan_iter_time_mean_s",
            "torchtitan_iter_time_std_s",
            "torchtitan_peak_memory_gb_by_rank",
            "piper_status",
            "piper_iter_time_mean_s",
            "piper_iter_time_std_s",
            "piper_peak_memory_gb_by_rank",
        ])
        for tt, pp in rows:
            writer.writerow([
                tt.sweep,
                tt.label,
                repr(tt.key),
                tt.status,
                tt.iter_time_mean,
                tt.iter_time_std,
                "/".join(str(v) for v in tt.peak_memory_values),
                pp.status,
                pp.iter_time_mean,
                pp.iter_time_std,
                "/".join(str(v) for v in pp.peak_memory_values),
            ])



def main() -> int:
    args = parse_args()
    torchtitan_rows = load_torchtitan_rows(Path(args.torchtitan_results))
    piper_rows = load_piper_rows(Path(args.piper_results))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sweep in args.sweeps:
        if sweep == "zero":
            sweep_pairs = collect_zero_sweep_pairs(torchtitan_rows, piper_rows)
        else:
            sweep_pairs = collect_standard_sweep_pairs(sweep, torchtitan_rows, piper_rows)
        if not sweep_pairs:
            continue
        plot_groups = build_plot_groups(sweep, sweep_pairs, piper_rows)
        write_matched_csv(out_dir / f"{sweep}_matched.csv", sweep_pairs)
        save_grouped_iter_time_plot(
            f"Qwen {sweep.title()} Iteration Time: Torchtitan vs Piper",
            plot_groups,
            out_dir / f"{sweep}_iter_time.png",
        )
        save_grouped_memory_plot(
            f"Qwen {sweep.title()} Memory: Torchtitan vs Piper",
            plot_groups,
            out_dir / f"{sweep}_memory.png",
        )
        print(f"wrote {sweep} plots and matched csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
