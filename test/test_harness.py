"""Build a schedule JSON and run a model test with it.

This is a small harness around ``test.test_qwen`` / ``test.test_llama``. It
consumes only schedule-building arguments; all remaining arguments are forwarded
to the selected test module unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from .schedules.build_order_directives import (
        build_1f1b_schedule,
        build_dualpipev_schedule,
        build_interleaved_1f1b_schedule,
        build_zerobubble_schedule,
        visualize_order_directives,
    )
except ImportError:
    from schedules.build_order_directives import (
        build_1f1b_schedule,
        build_dualpipev_schedule,
        build_interleaved_1f1b_schedule,
        build_zerobubble_schedule,
        visualize_order_directives,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a full schedule JSON from a base schedule and generated order "
            "directives, then run test_qwen or test_llama with it."
        )
    )
    parser.add_argument(
        "--test-file",
        "--test",
        required=True,
        help="Test module/file to run: test_qwen, test_llama, test.test_qwen, or a .py path.",
    )
    parser.add_argument(
        "--base-schedule",
        required=True,
        type=Path,
        help="Input JSON schedule without order directives.",
    )
    parser.add_argument(
        "--schedule",
        "--schedule-name",
        dest="schedule",
        required=True,
        choices=("1f1b", "interleaved_1f1b", "zerobubble", "dualpipev", "custom"),
        help=(
            "Order directive schedule variant to append. Use 'custom' to keep the "
            "split/order directives already in --base-schedule untouched."
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=None,
        help="Number of PP ranks. Required unless --schedule custom.",
    )
    parser.add_argument(
        "--mbs",
        type=int,
        default=None,
        help="Number of microbatches. Required unless --schedule custom.",
    )
    parser.add_argument(
        "--virtual-stages",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--generated-schedule-file",
        "--output",
        type=Path,
        default=None,
        help="Where to write the generated full schedule JSON.",
    )
    parser.add_argument(
        "--keep-existing-order",
        action="store_true",
        help="Keep any order directives already present in the base schedule.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the generated schedule and print the test command without running it.",
    )
    parser.add_argument(
        "--schedule-viz-file",
        type=Path,
        default=None,
        help="Where to render the generated schedule visualization (default: out/<schedule>.png).",
    )
    parser.add_argument(
        "--no-schedule-viz",
        action="store_true",
        help="Do not render a Graphviz schedule visualization.",
    )
    parser.add_argument(
        "--pytorch-profiler",
        action="store_true",
        help="Run extra iterations under torch.profiler and combine the per-actor "
             "chrome traces into one trace per dp-rank under out/.",
    )
    parser.add_argument(
        "--pytorch-profile-dir",
        type=Path,
        default=None,
        help="Shared directory where actors write per-actor chrome traces "
             "(default: <run_dir>/pytorch_profiles). Removed after combining.",
    )
    args, test_args = parser.parse_known_args()

    if args.schedule != "custom":
        missing = [
            flag for flag, value in (("--ranks", args.ranks), ("--mbs", args.mbs))
            if value is None
        ]
        if missing:
            parser.error(
                f"the following arguments are required unless --schedule custom: {', '.join(missing)}"
            )
        if args.virtual_stages is not None:
            parser.error(
                "--virtual-stages is internal; it is 2 for interleaved_1f1b/dualpipev "
                "and 1 otherwise"
            )
        args.virtual_stages = _virtual_stages_for_schedule(args.schedule)

    # Each run gets its own timestamped output directory under out/, holding its
    # results.csv (one row per DP rank), schedule viz, and combined profiles.
    run_dir = Path("out") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    generated_schedule, viz_path = build_schedule_file(args, run_dir)
    forwarded_args = _replace_schedule_arg(test_args, generated_schedule)
    forwarded_args = _replace_arg(forwarded_args, "--output-dir", str(run_dir))

    profile_args: list[str] = []
    profile_dir: Path | None = None
    if args.pytorch_profiler:
        # Absolute path on the shared filesystem so remote actors (possibly on
        # other nodes) write to the same place the harness later reads.
        profile_dir = (args.pytorch_profile_dir or (run_dir / "pytorch_profiles")).resolve()
        if profile_dir.exists():
            for stale in profile_dir.glob("dp*_pp*.json"):
                stale.unlink()
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_args = ["--pytorch-profiler", "--pytorch-profile-dir", str(profile_dir)]

    if args.dry_run:
        cmd = [sys.executable, "-m", _test_module(args.test_file), *forwarded_args, *profile_args]
        print(" ".join(cmd))
        print(f"generated_schedule={generated_schedule}")
        print(f"order_directives={generated_schedule.with_name(generated_schedule.stem + '_order_directives.json')}")
        if viz_path is not None:
            print(f"schedule_viz={viz_path}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        metrics_json = Path(tmp) / "metrics.json"
        cmd = [
            sys.executable,
            "-m",
            _test_module(args.test_file),
            *forwarded_args,
            *profile_args,
            "--metrics-json-out",
            str(metrics_json),
        ]
        returncode = subprocess.run(cmd, check=False).returncode
        if returncode == 0 and metrics_json.exists():
            with metrics_json.open(encoding="utf-8") as f:
                dp_metrics = json.load(f)
            if dp_metrics:
                rows = _metrics_rows(dp_metrics)
                results_csv = run_dir / "results.csv"
                _write_results_csv(results_csv, rows)
                print(f"metrics written to {results_csv}")
        if returncode == 0 and profile_dir is not None:
            _combine_pytorch_profiles(
                profile_dir,
                run_dir,
                schedule=args.schedule,
                pp=args.ranks if args.ranks is not None else "X",
                mbs=args.mbs if args.mbs is not None else "X",
            )
    raise SystemExit(returncode)


def _combine_pytorch_profiles(
    profile_dir: Path,
    out_dir: Path,
    schedule: str,
    pp: int | str,
    mbs: int | str,
) -> None:
    """Group per-actor chrome traces (dp{dp}_pp{pp}.json) by dp-rank and merge
    each group into one trace per dp-rank under out_dir. The combined filename
    encodes the run configuration: schedule, pp degree, dp degree, mbs, dp-rank.

    Each pp-rank's events keep their own pid namespace via a collision-free
    remap, and process_name metadata is prefixed with the pp rank so the merged
    timeline groups work by stage. The intermediate per-actor traces and the
    (now-empty) profile_dir are removed once combining succeeds.
    """
    name_re = re.compile(r"^dp(\d+)_pp(\d+)\.json$")
    groups: dict[int, dict[int, Path]] = {}
    for path in sorted(profile_dir.glob("dp*_pp*.json")):
        m = name_re.match(path.name)
        if not m:
            continue
        dp_rank, pp_rank = int(m.group(1)), int(m.group(2))
        groups.setdefault(dp_rank, {})[pp_rank] = path

    if not groups:
        print(f"pytorch profiler: no trace files found in {profile_dir}")
        return

    dp_degree = len(groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    consumed: list[Path] = []
    for dp_rank, pp_paths in sorted(groups.items()):
        combined_events: list = []
        next_pid = 0
        for pp_rank, path in sorted(pp_paths.items()):
            with path.open(encoding="utf-8") as f:
                trace = json.load(f)
            events = trace.get("traceEvents", [])
            pid_map: dict = {}
            for ev in events:
                opid = ev.get("pid")
                if opid is not None and opid not in pid_map:
                    pid_map[opid] = next_pid
                    next_pid += 1
            for ev in events:
                if "pid" in ev and ev["pid"] in pid_map:
                    ev["pid"] = pid_map[ev["pid"]]
                if ev.get("name") == "process_name":
                    pname = ev.get("args", {}).get("name", "")
                    ev["args"]["name"] = f"dp{dp_rank} pp{pp_rank}: {pname}"
                combined_events.append(ev)
            consumed.append(path)
        out_name = (
            f"pytorch_profile_{schedule}_pp{pp}_dp{dp_degree}"
            f"_mbs{mbs}_dprank{dp_rank}.json"
        )
        out_path = out_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"traceEvents": combined_events}, f)
        print(
            f"pytorch profiler: combined {len(pp_paths)} pp-rank trace(s) "
            f"-> {out_path}"
        )

    for path in consumed:
        path.unlink(missing_ok=True)
    try:
        profile_dir.rmdir()
    except OSError:
        pass


def _metrics_rows(dp_metrics: list[dict]) -> list[dict]:
    """Summarize the raw metrics into one CSV row per DP rank (no averaging
    across DP ranks).

    Each row's timing scalars come from that DP rank's own iter times; peak
    memory is taken from that DP rank's per-(global=pp)-rank values and emitted
    as one peak_memory_pp{i}_gb column per pp rank.
    """
    rows: list[dict] = []
    for m in sorted(dp_metrics, key=lambda d: int(d.get("dp_rank", 0))):
        iter_times = m.get("iter_times_s") or []
        mean_iter = statistics.fmean(iter_times) if iter_times else float("nan")
        std_iter = statistics.pstdev(iter_times) if len(iter_times) > 1 else 0.0
        tokens = (
            (m.get("batch_size") or 0)
            * (m.get("num_microbatches") or 1)
            * (m.get("seq_len") or 0)
        )
        throughput = tokens / mean_iter if iter_times and mean_iter else float("nan")

        row: dict = {
            "dp_rank": m.get("dp_rank"),
            "model": m.get("model"),
            "schedule": m.get("schedule"),
            "pp": m.get("pp"),
            "dp": m.get("dp"),
            "batch_size": m.get("batch_size"),
            "num_microbatches": m.get("num_microbatches"),
            "seq_len": m.get("seq_len"),
            "samples": len(iter_times),
            "iter_time_mean_s": mean_iter,
            "iter_time_std_s": std_iter,
            "throughput_tokens_per_s": throughput,
        }
        # peak_memory_by_rank is keyed by global rank; within a DP replica the
        # global ranks ascend with pp stage (both pp-inner and pp-outer layouts),
        # so enumerate them as local stage indices for columns that align across
        # DP-rank rows.
        peak = m.get("peak_memory_by_rank") or {}
        for stage, rank in enumerate(sorted(peak, key=lambda r: int(r))):
            row[f"peak_memory_pp{stage}_gb"] = float(peak[rank]) / (1024 ** 3)
        rows.append(row)
    return rows


def _write_results_csv(csv_path: Path, rows: list[dict]) -> None:
    """Write ``rows`` (one per DP rank) to a fresh per-run results CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def build_schedule_file(args: argparse.Namespace, run_dir: Path) -> tuple[Path, str | None]:
    base = _load_schedule(args.base_schedule)
    if args.schedule == "custom":
        # Pass the base schedule through untouched: trust its existing split and
        # order directives. Only used for visualization below.
        schedule = list(base)
        order_directives = [
            d for d in schedule
            if isinstance(d, dict) and d.get("op") == "order"
        ]
    else:
        if args.keep_existing_order:
            schedule = list(base)
        else:
            schedule = [
                directive for directive in base
                if not (isinstance(directive, dict) and directive.get("op") == "order")
            ]
        _set_num_microbatches(schedule, args.mbs)

        if args.schedule == "1f1b":
            order_directives = build_1f1b_schedule(args.ranks, args.mbs)
        elif args.schedule == "zerobubble":
            order_directives = build_zerobubble_schedule(args.ranks, args.mbs)
        elif args.schedule == "dualpipev":
            order_directives = build_dualpipev_schedule(args.ranks, args.mbs)
        else:
            order_directives = build_interleaved_1f1b_schedule(
                args.ranks,
                args.mbs,
                args.virtual_stages,
            )

        _validate_generated_order_placement(
            schedule,
            order_directives,
            schedule_name=args.schedule,
            ranks=args.ranks,
        )
        schedule.extend(order_directives)
    schedule_stem = _schedule_attr_stem(
        args.schedule,
        args.ranks,
        args.mbs,
        args.virtual_stages,
        base_schedule=args.base_schedule,
    )
    output = args.generated_schedule_file or _default_output_path(schedule_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)
        f.write("\n")
    order_output = output.with_name(f"{output.stem}_order_directives.json")
    with order_output.open("w", encoding="utf-8") as f:
        json.dump(order_directives, f, indent=2)
        f.write("\n")

    # Drop a copy of the complete schedule into the run directory so the exact
    # directives used (base + appended order) are visible alongside results.csv,
    # profile traces, and the schedule viz.
    run_schedule = run_dir / f"{schedule_stem}.json"
    if run_schedule.resolve() != output.resolve():
        with run_schedule.open("w", encoding="utf-8") as f:
            json.dump(schedule, f, indent=2)
            f.write("\n")

    viz_path = None
    if not args.no_schedule_viz:
        viz_output = args.schedule_viz_file or run_dir / f"{schedule_stem}.png"
        viz_path = visualize_order_directives(order_directives, viz_output)
    return output, viz_path


def _schedule_attr_stem(
    schedule_name: str,
    ranks: int | None,
    mbs: int | None,
    virtual_stages: int | None,
    base_schedule: Path | None = None,
) -> str:
    if schedule_name == "custom":
        # ranks/mbs/virtual_stages aren't required for custom; derive a stable
        # name from the base schedule filename instead.
        if base_schedule is not None:
            return f"custom_{Path(base_schedule).stem}"
        return "custom"
    suffix = f"{schedule_name}_pp{ranks}_mbs{mbs}"
    if schedule_name == "interleaved_1f1b":
        suffix += f"_v{virtual_stages}"
    return suffix


def _virtual_stages_for_schedule(schedule_name: str) -> int:
    if schedule_name in ("interleaved_1f1b", "dualpipev"):
        return 2
    return 1


def _default_output_path(schedule_stem: str) -> Path:
    return Path("test/schedules/generated") / f"{schedule_stem}.json"


def _load_schedule(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"schedule JSON must contain a list, got {type(data)}")
    for i, directive in enumerate(data):
        if not isinstance(directive, dict):
            raise ValueError(f"schedule directive[{i}] must be an object, got {type(directive)}")
    return data


def _set_num_microbatches(schedule: list[dict], n_mbs: int) -> None:
    """Ensure the schedule has exactly one split directive with num_microbatches=n_mbs.

    If a split directive already exists, its num_microbatches is overwritten.
    Otherwise a default MB-split directive is appended.
    """
    found = False
    for directive in schedule:
        if directive.get("op") == "split":
            directive["num_microbatches"] = n_mbs
            found = True
    if not found:
        schedule.append({
            "op": "split",
            "filter": [],
            "dim_name": "MB",
            "num_microbatches": n_mbs,
        })


def _validate_generated_order_placement(
    base_schedule: list[dict],
    order_directives: list[dict],
    *,
    schedule_name: str,
    ranks: int | None,
) -> None:
    """Generated order rows are per physical rank and must be device-local."""
    stage_devices = _stage_devices_from_place_directives(base_schedule)
    if not stage_devices:
        return

    errors: list[str] = []
    for row_idx, directive in enumerate(order_directives):
        row_devices: set[tuple[int, ...]] = set()
        for slot_idx, slot in enumerate(_order_filter_slots(directive.get("filters", []))):
            slot_devices: set[tuple[int, ...]] = set()
            slot_pps: list[int] = []
            for flt in slot:
                spec = _filter_to_dict(flt)
                pp = spec.get("PP")
                if pp is None or pp == "*":
                    continue
                pp = int(pp)
                slot_pps.append(pp)
                dev = stage_devices.get(pp)
                if dev is None:
                    errors.append(
                        f"row {row_idx} slot {slot_idx} references PP{pp}, "
                        "but no matching place directive exists"
                    )
                    continue
                slot_devices.add(dev)
            if len(slot_devices) > 1:
                errors.append(
                    f"row {row_idx} slot {slot_idx} groups stages {slot_pps} "
                    f"across devices {sorted(slot_devices)}"
                )
            row_devices.update(slot_devices)
        if len(row_devices) > 1:
            errors.append(
                f"row {row_idx} orders stages across devices {sorted(row_devices)}"
            )

    if not errors:
        return

    hint = ""
    if schedule_name == "dualpipev" and ranks is not None:
        expected = [
            (rank, 2 * ranks - 1 - rank)
            for rank in range(ranks)
        ]
        hint = (
            " DualPipeV uses V-layout virtual stages; each pair "
            f"{expected} must be placed on the same physical device set."
        )
    raise ValueError(
        "generated order directives are not compatible with the base placement: "
        + "; ".join(errors[:4])
        + hint
    )


def _stage_devices_from_place_directives(schedule: list[dict]) -> dict[int, tuple[int, ...]]:
    stage_devices: dict[int, tuple[int, ...]] = {}
    for directive in schedule:
        if directive.get("op") != "place":
            continue
        devices = directive.get("devices", directive.get("device"))
        if not isinstance(devices, list):
            continue
        spec = _filter_to_dict(directive.get("filter", []))
        pp = spec.get("PP")
        if pp is None or pp == "*":
            continue
        stage_devices[int(pp)] = tuple(sorted(int(d) for d in devices))
    return stage_devices


def _order_filter_slots(filters: object) -> list[list[object]]:
    if not isinstance(filters, list):
        raise ValueError(f"order directive requires filters list, got {type(filters)}")
    slots: list[list[object]] = []
    for item in filters:
        if _is_filter_spec(item):
            slots.append([item])
        elif isinstance(item, list):
            slots.append(list(item))
        else:
            raise ValueError(f"invalid order filter group: {item}")
    return slots


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


def _filter_to_dict(flt: object) -> dict:
    if isinstance(flt, dict):
        return dict(flt)
    if isinstance(flt, list):
        out = {}
        for item in flt:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"invalid filter item: {item}")
            out[item[0]] = item[1]
        return out
    raise ValueError(f"invalid filter: {flt}")


def _replace_schedule_arg(test_args: list[str], schedule_path: Path) -> list[str]:
    return _replace_arg(test_args, "--schedule-directives-file", str(schedule_path))


def _replace_arg(test_args: list[str], flag: str, value: str) -> list[str]:
    """Drop any existing occurrence of ``flag`` (and its value) from forwarded
    test args, then append ``flag value`` so the harness setting wins."""
    out: list[str] = []
    skip_next = False
    for arg in test_args:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        if arg.startswith(f"{flag}="):
            continue
        out.append(arg)
    out.extend([flag, value])
    return out


def _test_module(test_file: str) -> str:
    if test_file in ("test_qwen", "test_llama"):
        return f"test.{test_file}"
    if test_file.endswith(".py"):
        path = Path(test_file)
        if len(path.parts) == 1 and path.stem in ("test_qwen", "test_llama"):
            return f"test.{path.stem}"
        if path.is_absolute():
            try:
                path = path.relative_to(Path.cwd())
            except ValueError as exc:
                raise ValueError(
                    f"absolute test file must be under the current repo: {test_file}"
                ) from exc
        return ".".join(path.with_suffix("").parts)
    return test_file


if __name__ == "__main__":
    main()
