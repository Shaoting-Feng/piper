#!/usr/bin/env python3
"""Export Qwen sweep results from out/ec2 metrics files to CSV.

By default this targets the completed Piper sweep:
  --model 9B
  --schedules 1f1b
  --pp-mbs 4:8 8:16
  --dp-values 1 2 4
  --zero-stages 0 2 3
  --batch-sizes 8

Each matching experiment produces one CSV row aggregated across dp_ranks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_MODEL = "9B"
DEFAULT_SCHEDULES = ("1f1b",)
DEFAULT_PP_MBS = ("4:8", "8:16")
DEFAULT_DP_VALUES = (1, 2, 4)
DEFAULT_ZERO_STAGES = (0, 2, 3)
DEFAULT_BATCH_SIZES = (8,)


def _parse_pp_mbs(values: list[str]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for value in values:
        pp_str, mbs_str = value.split(":", maxsplit=1)
        result[int(pp_str)].add(int(mbs_str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aggregated Qwen sweep results from out/ec2 metrics files."
    )
    parser.add_argument("--metrics-dir", default="out/ec2", help="Directory containing qwen metrics files")
    parser.add_argument("--output-csv", default="out/ec2/qwen_sweep_results.csv", help="CSV output path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to include")
    parser.add_argument(
        "--schedules",
        nargs="+",
        default=list(DEFAULT_SCHEDULES),
        help="Schedules to include. Default: 1f1b",
    )
    parser.add_argument(
        "--pp-mbs",
        nargs="+",
        default=list(DEFAULT_PP_MBS),
        metavar="PP:MBS",
        help="PP/MBS pairs to include. Default: 4:8 8:16",
    )
    parser.add_argument(
        "--dp-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_DP_VALUES),
        help="DP values to include. Default: 1 2 4",
    )
    parser.add_argument(
        "--zero-stages",
        nargs="+",
        type=int,
        default=list(DEFAULT_ZERO_STAGES),
        help="ZeRO stages to include. Default: 0 2 3",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to include. Default: 8",
    )
    return parser.parse_args()


def _iter_metrics_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            marker = "metrics_json="
            if marker not in line:
                continue
            payload = line.split(marker, 1)[1].strip()
            try:
                records.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return records


_METRICS_NAME_RE = re.compile(
    r"^qwen(?P<model>.+)-pp(?P<pp>\d+)-dp(?P<dp>\d+)"
    r"-zero(?P<zero_stage>\d+)-bs(?P<batch_size>\d+)"
    r"-sl(?P<seq_len>\d+)-ga(?P<gradient_accumulation>[01])"
    r"-aras(?P<ar_a2a_same_stream>[01])"
    r"(?:-ozo(?P<overlap_zero_ops>[01]))?"
    r"-(?P<schedule>[^/]+)$"
)


def _parse_metrics_filename(path: Path) -> dict | None:
    match = _METRICS_NAME_RE.match(path.name)
    if match is None:
        return None
    data = match.groupdict()
    return {
        "model": data["model"],
        "pp": int(data["pp"]),
        "dp": int(data["dp"]),
        "schedule": data["schedule"],
        "zero_stage": int(data["zero_stage"]),
        "batch_size": int(data["batch_size"]),
        "seq_len": int(data["seq_len"]),
        "gradient_accumulation": bool(int(data["gradient_accumulation"])),
        "ar_a2a_same_stream": bool(int(data["ar_a2a_same_stream"])),
        "overlap_zero_ops": bool(int(data["overlap_zero_ops"] or "0")),
    }


def _peak_memory_summary(peak_memory_by_rank: dict[str, dict]) -> str:
    ordered = []
    for rank in sorted(peak_memory_by_rank, key=lambda item: int(item)):
        ordered.append(f"{peak_memory_by_rank[rank]['peak_memory_gb']:.3f}")
    return "/".join(ordered)


def main() -> int:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pp_to_mbs = _parse_pp_mbs(args.pp_mbs)
    schedules = set(args.schedules)
    dp_values = set(args.dp_values)
    zero_stages = set(args.zero_stages)
    batch_sizes = set(args.batch_sizes)

    candidate_files: dict[tuple, tuple[bool, Path]] = {}
    for path in sorted(metrics_dir.glob("qwen*")):
        meta = _parse_metrics_filename(path)
        if meta is None:
            continue
        records = _iter_metrics_records(path)
        if not records:
            continue
        first = records[0]
        model = first.get("model", meta["model"])
        schedule = first.get("schedule", meta["schedule"])
        pp = int(first.get("pp", meta["pp"]))
        dp = int(first.get("dp", meta["dp"]))
        mbs = int(first.get("mbs"))
        zero_stage = meta["zero_stage"]
        batch_size = meta["batch_size"]
        seq_len = int(first.get("seq_len", meta["seq_len"]))
        gradient_accumulation = bool(
            first.get("gradient_accumulation", meta["gradient_accumulation"])
        )
        ar_a2a_same_stream = bool(
            first.get("ar_a2a_same_stream", meta["ar_a2a_same_stream"])
        )
        overlap_zero_ops = bool(
            first.get("overlap_zero_ops", meta["overlap_zero_ops"])
        )

        if model != args.model:
            continue
        if schedule not in schedules:
            continue
        if pp not in pp_to_mbs or mbs not in pp_to_mbs[pp]:
            continue
        if dp not in dp_values:
            continue
        if zero_stage not in zero_stages:
            continue
        if batch_size not in batch_sizes:
            continue

        key = (
            model,
            schedule,
            pp,
            dp,
            mbs,
            zero_stage,
            batch_size,
            seq_len,
            gradient_accumulation,
            ar_a2a_same_stream,
            overlap_zero_ops,
        )
        if key not in candidate_files:
            candidate_files[key] = (True, path)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for (
        model,
        schedule,
        pp,
        dp,
        mbs,
        zero_stage,
        batch_size,
        seq_len,
        gradient_accumulation,
        ar_a2a_same_stream,
        overlap_zero_ops,
    ), (_explicit, path) in sorted(candidate_files.items()):
        records = _iter_metrics_records(path)
        for record in records:
            key = (
                model,
                schedule,
                pp,
                dp,
                mbs,
                zero_stage,
                batch_size,
                seq_len,
                gradient_accumulation,
                ar_a2a_same_stream,
                overlap_zero_ops,
            )
            enriched = dict(record)
            enriched["_path"] = str(path)
            enriched["_zero_stage"] = zero_stage
            grouped[key].append(enriched)

    fieldnames = [
        "schedule",
        "pp",
        "dp",
        "zero_stage",
        "gradient_accumulation",
        "ar_a2a_same_stream",
        "overlap_zero_ops",
        "iter_time_mean",
        "iter_time_stddev",
        "peak_memory_gb_by_rank",
    ]

    rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        latest_by_dp_rank: dict[int, dict] = {}
        for record in grouped[key]:
            latest_by_dp_rank[int(record["dp_rank"])] = record
        records = [latest_by_dp_rank[rank] for rank in sorted(latest_by_dp_rank)]
        (
            model,
            schedule,
            pp,
            dp,
            mbs,
            zero_stage,
            batch_size,
            seq_len,
            gradient_accumulation,
            ar_a2a_same_stream,
            overlap_zero_ops,
        ) = key
        dp_ranks = [int(record["dp_rank"]) for record in records]
        iter_means = [float(record["iter_time_mean_s"]) for record in records]
        iter_stds = [float(record["iter_time_std_s"]) for record in records]

        peak_memory_by_rank: dict[str, dict] = {}
        for record in records:
            peak_memory_by_rank.update(record.get("peak_memory_by_rank", {}))
        peak_values = [
            float(stats["peak_memory_gb"])
            for rank, stats in sorted(peak_memory_by_rank.items(), key=lambda item: int(item[0]))
        ]

        rows.append(
            {
                "schedule": schedule,
                "pp": pp,
                "dp": dp,
                "zero_stage": zero_stage,
                "gradient_accumulation": int(gradient_accumulation),
                "ar_a2a_same_stream": int(ar_a2a_same_stream),
                "overlap_zero_ops": int(overlap_zero_ops),
                "iter_time_mean": sum(iter_means) / len(iter_means),
                "iter_time_stddev": sum(iter_stds) / len(iter_stds),
                "peak_memory_gb_by_rank": _peak_memory_summary(peak_memory_by_rank),
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} experiments to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
