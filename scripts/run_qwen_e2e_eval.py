#!/usr/bin/env python3
"""Run three end-to-end Qwen EC2 eval sweeps and generate plots/CSVs.

This script wraps scripts/run-qwen-ec2.sh, stores per-experiment logs under
e2e-eval/, copies the matching metrics file out of the remote piper_ray
container, aggregates throughput statistics, and saves one CSV and one plot for
each requested sweep.

Defaults that were not fully specified in the request are made explicit here:
  - model: 9B
  - default batch size for non-ZeRO sweeps: 4
  - seq len: 512
  - warmup / iters: 3 / 10
  - scalability sweep PP->MBS mapping: 4->8, 8->16
  - base schedule for scalability and ZeRO sweeps: 1f1b
  - schedule sweep ZeRO stage: 0
  - schedule sweep MBS: 16
  - ZeRO sweep batch sizes: 4, 8, 16
  - ZeRO sweep num microbatches: 2 * pp
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable


OOM_PATTERNS = (
    "out of memory",
    "torch.cuda.outofmemoryerror",
    "cuda out of memory",
    "oom-killed",
    "std::bad_alloc",
    "failed to cuda calloc async",
)


@dataclass(frozen=True)
class Experiment:
    sweep: str
    schedule: str
    pp: int
    dp: int
    ep: bool
    zero_stage: int
    bucket_size_mb: float | None
    batch_size: int
    seq_len: int
    mbs: int
    gradient_accumulation: bool
    ar_a2a_same_stream: bool
    overlap_zero_ops: bool
    overlap_chunks: bool
    nsight: bool

    def slug(self) -> str:
        bucket_part = (
            f"-bucket{_format_bucket_size(self.bucket_size_mb)}"
            if self.bucket_size_mb is not None
            else ""
        )
        nsight_part = "-nsight1" if self.nsight else ""
        return (
            f"{self.sweep}-sched_{self.schedule}-pp{self.pp}-dp{self.dp}-"
            f"ep{int(self.ep)}-zero{self.zero_stage}{bucket_part}{nsight_part}-bs{self.batch_size}-sl{self.seq_len}-"
            f"mbs{self.mbs}-ga{int(self.gradient_accumulation)}-"
            f"aras{int(self.ar_a2a_same_stream)}-ozo{int(self.overlap_zero_ops)}-"
            f"och{int(self.overlap_chunks)}"
        )

    def label(self) -> str:
        if self.sweep == "scalability":
            return f"(pp={self.pp}, dp={self.dp})"
        if self.sweep == "zero":
            ga = "on" if self.gradient_accumulation else "off"
            return f"(zero={self.zero_stage}, ga={ga}, batch_size={self.batch_size})"
        return self.schedule

    def key(self) -> tuple[object, ...]:
        return (
            self.sweep,
            self.schedule,
            self.pp,
            self.dp,
            int(self.ep),
            self.zero_stage,
            self.bucket_size_mb,
            self.batch_size,
            self.seq_len,
            self.mbs,
            int(self.gradient_accumulation),
            int(self.ar_a2a_same_stream),
            int(self.overlap_zero_ops),
            int(self.overlap_chunks),
            int(self.nsight),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three EC2 Qwen end-to-end sweeps and save plots/CSVs."
    )
    parser.add_argument("--model", default="1B")
    parser.add_argument("--pp", type=int, default=8, help="Default PP degree for non-scalability sweeps.")
    parser.add_argument("--dp", type=int, default=1, help="Default DP degree for non-scalability sweeps.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--bucket-size",
        type=float,
        default=None,
        help="Bucket size in MB for scalability and ZeRO sweeps. Default: unset (no bucketing).",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--ray-port", type=int, default=6379)
    parser.add_argument(
        "--remote-output-dir",
        default="/tmp/piper/out",
        help="Directory inside the remote piper_ray container where test_qwen writes outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="e2e-eval",
        help="Base directory where timestamped eval run folders are created.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional run directory name under --out-dir. Defaults to a UTC timestamp.",
    )
    parser.add_argument("--runner", default="scripts/run-qwen-ec2.sh")
    parser.add_argument("--base-schedule", default="1f1b")
    parser.add_argument(
        "--scalability-pp-mbs",
        nargs="+",
        default=["4:8", "8:16"],
        metavar="PP:MBS",
        help="PP->MBS mapping for the scalability sweep.",
    )
    parser.add_argument(
        "--zero-stages",
        nargs="+",
        type=int,
        default=[1,2,3],
    )
    parser.add_argument(
        "--zero-batch-sizes",
        nargs="+",
        type=int,
        default=[16,32,64],
    )
    parser.add_argument(
        "--zero-grad-acc-settings",
        nargs="+",
        choices=["on", "off"],
        default=["on"],
        help="Gradient accumulation settings to include in the zero sweep. "
             "Examples: --zero-grad-acc-settings on | off | off on",
    )
    parser.add_argument(
        "--schedule-values",
        nargs="+",
        default=["1f1b", "interleaved-1f1b", "interleaved-zerobubble", "dualpipev"],
    )
    parser.add_argument("--schedule-mbs", type=int, default=16)
    parser.add_argument("--schedule-zero-stage", type=int, default=0)
    parser.add_argument(
        "--nsight",
        action="store_true",
        help="Run each experiment with Nsight enabled and fetch matching Nsight artifacts.",
    )
    parser.add_argument(
        "--use-inductor",
        dest="use_inductor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to pass --use-inductor through to test_qwen. "
             "By default this is disabled for schedule sweeps and enabled for zero/scalability sweeps.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        dest="gradient_accumulation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pass gradient accumulation through to test_qwen (default: true).",
    )
    parser.add_argument(
        "--overlap-chunks",
        dest="overlap_chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to pass overlap_chunks through to test_qwen (default: false).",
    )
    parser.add_argument(
        "--no-udse-inductor",
        dest="use_inductor",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse completed rows from an existing all_results.csv.",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=["scalability", "zero", "schedule"],
        default=["scalability", "zero", "schedule"],
        help="Subset of e2e sweeps to run. Default: scalability zero schedule",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _format_bucket_size(bucket_size_mb: float | None) -> str:
    if bucket_size_mb is None:
        return "none"
    return f"{bucket_size_mb:g}"


def _parse_pp_mbs(values: list[str]) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for value in values:
        pp_text, mbs_text = value.split(":", maxsplit=1)
        parsed.append((int(pp_text), int(mbs_text)))
    return parsed


def _metrics_name(model: str, exp: Experiment) -> str:
    return (
        f"qwen{model}-pp{exp.pp}-dp{exp.dp}-zero{exp.zero_stage}-"
        f"ep{int(exp.ep)}-"
        f"bs{exp.batch_size}-sl{exp.seq_len}-ga{int(exp.gradient_accumulation)}-"
        f"aras{int(exp.ar_a2a_same_stream)}-ozo{int(exp.overlap_zero_ops)}-"
        f"och{int(exp.overlap_chunks)}-"
        f"{exp.schedule}"
    )


def _remote_experiment_output_dir(args: argparse.Namespace, exp: Experiment) -> str:
    return f"{args.remote_output_dir.rstrip('/')}/{exp.slug()}"


def _metrics_remote_paths(
    model: str,
    exp: Experiment,
    remote_output_dir: str,
) -> list[str]:
    return [
        f"{remote_output_dir}/{_metrics_name(model, exp)}",
        (
            f"{remote_output_dir}/qwen{model}-pp{exp.pp}-dp{exp.dp}-zero{exp.zero_stage}-"
            f"bs{exp.batch_size}-sl{exp.seq_len}-ga{int(exp.gradient_accumulation)}-"
            f"aras{int(exp.ar_a2a_same_stream)}-ozo{int(exp.overlap_zero_ops)}-"
            f"och{int(exp.overlap_chunks)}-"
            f"{exp.schedule}"
        ),
    ]


def _runner_command(args: argparse.Namespace, exp: Experiment, fetch_dir: Path) -> list[str]:
    remote_output_dir = _remote_experiment_output_dir(args, exp)
    command = [
        str(Path(args.runner).resolve()),
        "--model", args.model,
        "--schedule", exp.schedule,
        "--pp", str(exp.pp),
        "--dp", str(exp.dp),
        "--zero-stage", str(exp.zero_stage),
        "--batch-size", str(exp.batch_size),
        "--seq-len", str(exp.seq_len),
        "--mbs", str(exp.mbs),
        "--gradient-accumulation" if exp.gradient_accumulation else "--no-gradient-accumulation",
        "--ar-a2a-same-stream" if exp.ar_a2a_same_stream else "--no-ar-a2a-same-stream",
        "--overlap-zero-ops" if exp.overlap_zero_ops else "--no-overlap-zero-ops",
        "--overlap-chunks" if exp.overlap_chunks else "--no-overlap-chunks",
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--ray-port", str(args.ray_port),
        "--out-dir", str(fetch_dir),
        "--remote-output-dir", remote_output_dir,
    ]
    if exp.bucket_size_mb is not None:
        command.extend(["--bucket-size", _format_bucket_size(exp.bucket_size_mb)])
    if exp.ep:
        command.append("--ep")
    if exp.nsight:
        command.append("--nsight")
    use_inductor = args.use_inductor
    if use_inductor is None:
        use_inductor = exp.sweep != "schedule"
    if use_inductor:
        command.append("--use-inductor")
    else:
        command.append("--no-use-inductor")
    return command


def _ssh_head_command(remote_command: str) -> list[str]:
    ssh_key = os.environ["SSH_KEY"]
    head_public_ip = os.environ["HEAD_PUBLIC_IP"]
    return [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
        f"ubuntu@{head_public_ip}",
        remote_command,
    ]


def _ssh_worker_command(worker_ip: str, remote_command: str) -> list[str]:
    ssh_key = os.environ["SSH_KEY"]
    head_public_ip = os.environ["HEAD_PUBLIC_IP"]
    return [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        (
            "ProxyCommand="
            f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -W %h:%p ubuntu@{head_public_ip}"
        ),
        f"ubuntu@{worker_ip}",
        remote_command,
    ]


def _fetch_remote_metrics(
    model: str,
    exp: Experiment,
    destination: Path,
    remote_output_dir: str,
) -> bool:
    for remote_path in _metrics_remote_paths(model, exp, remote_output_dir):
        remote_command = (
            "docker exec piper_ray bash -lc "
            + json.dumps(f"test -f {remote_path} && cat {remote_path}")
        )
        result = subprocess.run(
            _ssh_head_command(remote_command),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.stdout, encoding="utf-8")
        return True

    return False


def _fetch_remote_dag_order_logs(
    args: argparse.Namespace,
    exp: Experiment,
    exp_logs_dir: Path,
) -> list[Path]:
    copied_paths: list[Path] = []
    remote_base = _remote_experiment_output_dir(args, exp)
    for rank in range(exp.pp):
        remote_path = f"{remote_base}/dag_order_rank{rank}"
        result = subprocess.run(
            _ssh_head_command(
                "docker exec piper_ray bash -lc "
                + json.dumps(f"test -f {remote_path} && cat {remote_path}")
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout:
            continue

        destination = exp_logs_dir / f"dag-order-rank{rank}.log"
        destination.write_text(result.stdout, encoding="utf-8")
        copied_paths.append(destination)

    return copied_paths


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_head_private_ip(log_path: Path) -> str | None:
    for line in _read_text_lossy(log_path).splitlines():
        clean = _strip_ansi(line)
        match = re.search(r"Namespace\(.*address='([^']+)'", clean)
        if match:
            return match.group(1)
    return None


def _node_ip_map_from_env(log_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    head_private_ip = os.environ.get("HEAD_PRIVATE_IP") or _parse_head_private_ip(log_path)
    if head_private_ip:
        mapping[head_private_ip] = "head"
    for idx in range(1, 4):
        worker_ip = os.environ.get(f"WORKER{idx}_PRIVATE_IP")
        if worker_ip:
            mapping[worker_ip] = f"worker{idx}"
    return mapping


def _extract_experiment_pids_by_node(log_path: Path) -> dict[str, set[str]]:
    pid_pattern = re.compile(r"pid=(\d+)(?:, ip=([0-9.]+))?")
    ip_to_label = _node_ip_map_from_env(log_path)
    pids_by_node: dict[str, set[str]] = {}

    for line in _read_text_lossy(log_path).splitlines():
        clean = _strip_ansi(line)
        for pid, ip in pid_pattern.findall(clean):
            if ip:
                node_label = ip_to_label.get(ip, f"node-{ip}")
            else:
                node_label = "head"
            pids_by_node.setdefault(node_label, set()).add(pid)

    return pids_by_node


def _iter_remote_nodes(log_path: Path) -> list[tuple[str, str, str | None]]:
    nodes: list[tuple[str, str, str | None]] = [("head", "head", None)]
    ip_to_label = _node_ip_map_from_env(log_path)
    for ip, label in sorted(ip_to_label.items(), key=lambda item: item[1]):
        if label == "head":
            continue
        nodes.append((label, "worker", ip))
    return nodes


def _scp_from_remote(
    *,
    node_kind: str,
    remote_host: str | None,
    remote_path: str,
    destination: Path,
) -> bool:
    ssh_key = os.environ["SSH_KEY"]
    head_public_ip = os.environ["HEAD_PUBLIC_IP"]
    command = [
        "scp",
        "-r",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
    ]
    if node_kind == "head":
        command.extend([f"ubuntu@{head_public_ip}:{remote_path}", str(destination)])
    else:
        command.extend(
            [
                "-o",
                (
                    "ProxyCommand="
                    f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -W %h:%p ubuntu@{head_public_ip}"
                ),
                f"ubuntu@{remote_host}:{remote_path}",
                str(destination),
            ]
        )
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.returncode == 0


def _stage_remote_nsight_dir(
    *,
    node_kind: str,
    remote_host: str | None,
    remote_tmp_dir: str,
    destination: Path,
) -> bool:
    remote_command = (
        "rm -rf "
        + remote_tmp_dir
        + " && mkdir -p "
        + remote_tmp_dir
        + " && docker cp piper_ray:/tmp/piper/ray_tmp/session_latest/logs/nsight/. "
        + remote_tmp_dir
        + "/"
    )
    run_command = (
        _ssh_head_command(remote_command)
        if node_kind == "head"
        else _ssh_worker_command(str(remote_host), remote_command)
    )
    result = subprocess.run(run_command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    return _scp_from_remote(
        node_kind=node_kind,
        remote_host=remote_host,
        remote_path=f"{remote_tmp_dir}/.",
        destination=destination,
    )


def _matching_nsight_paths(paths: Iterable[str], pids: set[str]) -> list[str]:
    matches: list[str] = []
    for path in paths:
        basename = Path(path).name
        if any(pid in basename or pid in path for pid in pids):
            matches.append(path)
    return sorted(set(matches))


def _copy_experiment_nsight_profiles(
    exp: Experiment,
    log_path: Path,
    exp_logs_dir: Path,
) -> list[Path]:
    if not exp.nsight:
        return []

    pids_by_node = _extract_experiment_pids_by_node(log_path)
    if not pids_by_node:
        return []

    copied_paths: list[Path] = []
    exp_nsight_dir = exp_logs_dir / "nsight"
    stage_root = exp_nsight_dir / ".stage"
    shutil.rmtree(stage_root, ignore_errors=True)
    for node_label, node_kind, remote_host in _iter_remote_nodes(log_path):
        node_pids = pids_by_node.get(node_label, set())
        if not node_pids:
            continue
        local_stage_dir = stage_root / node_label
        remote_tmp_dir = f"/tmp/piper-nsight-{exp_logs_dir.name}-{node_label}"
        if not _stage_remote_nsight_dir(
            node_kind=node_kind,
            remote_host=remote_host,
            remote_tmp_dir=remote_tmp_dir,
            destination=local_stage_dir,
        ):
            continue
        staged_paths = [str(path.relative_to(local_stage_dir)) for path in local_stage_dir.rglob("*.nsys-rep")]
        for relative_path in _matching_nsight_paths(staged_paths, node_pids):
            source = local_stage_dir / relative_path
            destination = exp_nsight_dir / node_label / Path(relative_path).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_paths.append(destination)

    pid_manifest = {
        node_label: sorted(pids)
        for node_label, pids in sorted(pids_by_node.items())
        if pids
    }
    if pid_manifest:
        exp_nsight_dir.mkdir(parents=True, exist_ok=True)
        (exp_nsight_dir / "matched_pids.json").write_text(
            json.dumps(pid_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    shutil.rmtree(stage_root, ignore_errors=True)

    return copied_paths


def _recover_metrics_from_cluster_log(
    exp: Experiment,
    cluster_log_path: Path,
    destination: Path,
    expected_iters: int,
) -> bool:
    if not cluster_log_path.is_file():
        return False

    dp_rank_by_pid: dict[str, int] = {}
    iter_times_by_pid: dict[str, list[float]] = {}
    peak_memory_by_pid: dict[str, dict[str, dict[str, float | int]]] = {}

    for raw_line in cluster_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_ansi(raw_line)

        dp_rank_match = re.search(r"run_dp_rank pid=(\d+).*DP rank (\d+) done\.", line)
        if dp_rank_match:
            dp_rank_by_pid[dp_rank_match.group(1)] = int(dp_rank_match.group(2))

        step_match = re.search(r"run_dp_rank pid=(\d+).*step_time=([0-9.]+)s", line)
        if step_match:
            pid = step_match.group(1)
            iter_times_by_pid.setdefault(pid, []).append(float(step_match.group(2)))

            rank_memories = {
                rank: {
                    "peak_memory_gb": float(value),
                    "peak_memory_bytes": int(float(value) * (1024 ** 3)),
                }
                for rank, value in re.findall(r"rank(\d+)_peak_mem=([0-9.]+)GB", line)
            }
            if rank_memories:
                peak_memory_by_pid[pid] = rank_memories

    if not iter_times_by_pid:
        return False

    for index, pid in enumerate(sorted(iter_times_by_pid, key=int)):
        dp_rank_by_pid.setdefault(pid, index)

    records: list[dict[str, object]] = []
    for pid, iter_times in sorted(iter_times_by_pid.items(), key=lambda item: dp_rank_by_pid[item[0]]):
        timed_iter_times = iter_times[-expected_iters:]
        iter_time_mean = sum(timed_iter_times) / len(timed_iter_times)
        iter_time_variance = sum((value - iter_time_mean) ** 2 for value in timed_iter_times) / len(timed_iter_times)
        iter_time_std = math.sqrt(iter_time_variance)
        throughput_mean = tokens_per_iter = exp.batch_size * exp.mbs * exp.seq_len
        throughput_mean = tokens_per_iter / iter_time_mean
        records.append(
            {
                "dp_rank": dp_rank_by_pid[pid],
                "model": exp.batch_size,
                "schedule": exp.schedule,
                "pp": exp.pp,
                "dp": exp.dp,
                "ep": exp.ep,
                "batch_size": exp.batch_size,
                "mbs": exp.mbs,
                "seq_len": exp.seq_len,
                "gradient_accumulation": exp.gradient_accumulation,
                "ar_a2a_same_stream": exp.ar_a2a_same_stream,
                "overlap_zero_ops": exp.overlap_zero_ops,
                "overlap_chunks": exp.overlap_chunks,
                "samples": len(timed_iter_times),
                "iter_time_mean_s": iter_time_mean,
                "iter_time_std_s": iter_time_std,
                "throughput_tokens_per_s": throughput_mean,
                "iter_times_s": timed_iter_times,
                "peak_memory_by_rank": peak_memory_by_pid.get(pid, {}),
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"rank {record['dp_rank']} metrics_json= {json.dumps(record, sort_keys=True)}\n")
    return True


def _recover_metrics_from_fetch_logs(
    args: argparse.Namespace,
    exp: Experiment,
    fetch_dir: Path,
    destination: Path,
) -> bool:
    bucket_part = (
        f"-bucket{_format_bucket_size(exp.bucket_size_mb)}"
        if exp.bucket_size_mb is not None
        else ""
    )
    nsight_part = "-nsight1" if exp.nsight else ""
    pattern = (
        f"qwen{args.model}-sched_{exp.schedule}-pp{exp.pp}-dp{exp.dp}-"
        f"zero{exp.zero_stage}{bucket_part}{nsight_part}-bs{exp.batch_size}-sl{exp.seq_len}-"
        f"mbs{exp.mbs}-ga{int(exp.gradient_accumulation)}-"
        f"aras{int(exp.ar_a2a_same_stream)}-ozo{int(exp.overlap_zero_ops)}-"
        f"och{int(exp.overlap_chunks)}_*.cluster.log"
    )
    candidates = sorted(fetch_dir.glob(pattern))
    if not candidates:
        return False
    return _recover_metrics_from_cluster_log(exp, candidates[-1], destination, args.iters)


def _parse_metrics_records(metrics_path: Path) -> list[dict]:
    records: list[dict] = []
    if not metrics_path.is_file():
        return records

    marker = "metrics_json="
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, maxsplit=1)[1].strip()
        try:
            records.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return records


def _row_from_metrics(
    exp: Experiment,
    log_path: Path,
    metrics_path: Path,
) -> dict[str, object] | None:
    records = _parse_metrics_records(metrics_path)
    if not records:
        return None

    row: dict[str, object] = {
        "sweep": exp.sweep,
        "label": exp.label(),
        "schedule": exp.schedule,
        "pp": exp.pp,
        "dp": exp.dp,
        "ep": int(exp.ep),
        "zero_stage": exp.zero_stage,
        "bucket_size_mb": _format_bucket_size(exp.bucket_size_mb) if exp.bucket_size_mb is not None else "",
        "batch_size": exp.batch_size,
        "seq_len": exp.seq_len,
        "mbs": exp.mbs,
        "gradient_accumulation": int(exp.gradient_accumulation),
        "ar_a2a_same_stream": int(exp.ar_a2a_same_stream),
        "overlap_zero_ops": int(exp.overlap_zero_ops),
        "overlap_chunks": int(exp.overlap_chunks),
        "nsight": int(exp.nsight),
        "status": "success",
        "failure_reason": "",
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
    }
    row.update(_aggregate_metrics(records))
    return row


def _aggregate_metrics(records: list[dict]) -> dict[str, float | int | str]:
    latest_by_dp_rank: dict[int, dict] = {}
    for record in records:
        latest_by_dp_rank[int(record["dp_rank"])] = record

    selected = [latest_by_dp_rank[rank] for rank in sorted(latest_by_dp_rank)]
    weighted_samples = 0
    weighted_iter_time_mean = 0.0
    weighted_iter_time_std = 0.0
    weighted_throughput_mean = 0.0
    weighted_throughput_std = 0.0
    peak_memory_gb_max = 0.0
    peak_memory_by_rank_summary: list[str] = []

    for record in selected:
        batch_size = int(record["batch_size"])
        mbs = int(record["mbs"])
        seq_len = int(record["seq_len"])
        tokens_per_iter = batch_size * mbs * seq_len
        samples = int(record.get("samples", len(record.get("iter_times_s", []))))
        if samples <= 0:
            continue

        iter_time_mean = record.get("iter_time_mean_s")
        if iter_time_mean is None:
            iter_times = [float(iter_time) for iter_time in record.get("iter_times_s", [])]
            if not iter_times:
                continue
            iter_time_mean = sum(iter_times) / len(iter_times)
        iter_time_mean = float(iter_time_mean)

        iter_time_std = record.get("iter_time_std_s")
        if iter_time_std is None:
            iter_times = [float(iter_time) for iter_time in record.get("iter_times_s", [])]
            if iter_times:
                variance = sum((value - iter_time_mean) ** 2 for value in iter_times) / len(iter_times)
                iter_time_std = math.sqrt(variance)
            else:
                iter_time_std = 0.0
        iter_time_std = float(iter_time_std)

        throughput_mean = record.get("throughput_tokens_per_s")
        if throughput_mean is None:
            throughput_mean = tokens_per_iter / iter_time_mean
        throughput_mean = float(throughput_mean)

        # Use the reported iteration-time stddev from the metrics file rather than
        # recomputing throughput variance from raw samples.
        throughput_std = throughput_mean * (iter_time_std / iter_time_mean) if iter_time_mean > 0 else 0.0

        weighted_samples += samples
        weighted_iter_time_mean += samples * iter_time_mean
        weighted_iter_time_std += samples * iter_time_std
        weighted_throughput_mean += samples * throughput_mean
        weighted_throughput_std += samples * throughput_std

        peak_memory_by_rank = record.get("peak_memory_by_rank", {})
        for rank, stats in sorted(peak_memory_by_rank.items(), key=lambda item: int(item[0])):
            peak_gb = float(stats["peak_memory_gb"])
            peak_memory_gb_max = max(peak_memory_gb_max, peak_gb)
            peak_memory_by_rank_summary.append(f"{rank}:{peak_gb:.3f}")

    if weighted_samples == 0:
        return {
            "samples": 0,
            "iter_time_mean_s": 0.0,
            "iter_time_std_s": 0.0,
            "throughput_mean_tokens_per_s": 0.0,
            "throughput_std_tokens_per_s": 0.0,
            "peak_memory_gb_max": peak_memory_gb_max,
            "peak_memory_gb_by_rank": ";".join(peak_memory_by_rank_summary),
        }

    return {
        "samples": weighted_samples,
        "iter_time_mean_s": weighted_iter_time_mean / weighted_samples,
        "iter_time_std_s": weighted_iter_time_std / weighted_samples,
        "throughput_mean_tokens_per_s": weighted_throughput_mean / weighted_samples,
        "throughput_std_tokens_per_s": weighted_throughput_std / weighted_samples,
        "peak_memory_gb_max": peak_memory_gb_max,
        "peak_memory_gb_by_rank": ";".join(peak_memory_by_rank_summary),
    }


def _looks_like_oom(log_path: Path) -> bool:
    log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return any(pattern in log_text for pattern in OOM_PATTERNS)


def _run_experiment(
    args: argparse.Namespace,
    exp: Experiment,
    index: int,
    total: int,
    logs_dir: Path,
    metrics_dir: Path,
    fetch_dir: Path,
) -> dict[str, object]:
    exp_logs_dir = logs_dir / exp.slug()
    exp_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_logs_dir / "run.log"
    metrics_path = metrics_dir / f"{exp.slug()}.metrics"
    command = _runner_command(args, exp, fetch_dir)

    print(
        f"[{index}/{total}] RUNNING {exp.sweep}: schedule={exp.schedule} pp={exp.pp} "
        f"dp={exp.dp} zero={exp.zero_stage} bucket_size_mb={exp.bucket_size_mb} "
        f"batch_size={exp.batch_size} mbs={exp.mbs} "
        f"ga={'on' if exp.gradient_accumulation else 'off'} nsight={'on' if exp.nsight else 'off'}"
    )
    sys.stdout.flush()

    if args.dry_run:
        return {
            "sweep": exp.sweep,
            "label": exp.label(),
            "schedule": exp.schedule,
            "pp": exp.pp,
            "dp": exp.dp,
            "ep": int(exp.ep),
            "zero_stage": exp.zero_stage,
            "bucket_size_mb": _format_bucket_size(exp.bucket_size_mb) if exp.bucket_size_mb is not None else "",
            "batch_size": exp.batch_size,
            "seq_len": exp.seq_len,
            "mbs": exp.mbs,
            "gradient_accumulation": int(exp.gradient_accumulation),
            "ar_a2a_same_stream": int(exp.ar_a2a_same_stream),
            "overlap_zero_ops": int(exp.overlap_zero_ops),
            "overlap_chunks": int(exp.overlap_chunks),
            "nsight": int(exp.nsight),
            "status": "dry-run",
            "failure_reason": "",
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
        }

    with open(log_path, "w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    metrics_found = _fetch_remote_metrics(
        args.model,
        exp,
        metrics_path,
        _remote_experiment_output_dir(args, exp),
    )
    dag_order_paths = _fetch_remote_dag_order_logs(args, exp, exp_logs_dir)
    nsight_paths = _copy_experiment_nsight_profiles(exp, log_path, exp_logs_dir)
    if not metrics_found:
        metrics_found = _recover_metrics_from_fetch_logs(args, exp, fetch_dir, metrics_path)
    records = _parse_metrics_records(metrics_path) if metrics_found else []

    status = "success"
    failure_reason = ""
    aggregated: dict[str, object] = {}
    if result.returncode != 0:
        if _looks_like_oom(log_path):
            status = "oom"
            failure_reason = "oom"
        else:
            status = "failed"
            failure_reason = f"runner_exit_{result.returncode}"
    elif not records:
        status = "failed"
        failure_reason = "missing_metrics"
    else:
        aggregated = _aggregate_metrics(records)

    print(
        f"[{index}/{total}] {status.upper()} {exp.sweep}: {exp.label()} "
        f"log={log_path}"
        + (
            f" dag_logs={','.join(str(path.name) for path in dag_order_paths)}"
            if dag_order_paths
            else ""
        )
        + (
            f" nsight_dir={exp_logs_dir / 'nsight'}"
            if nsight_paths
            else ""
        )
    )
    sys.stdout.flush()

    row: dict[str, object] = {
        "sweep": exp.sweep,
        "label": exp.label(),
        "schedule": exp.schedule,
        "pp": exp.pp,
        "dp": exp.dp,
        "zero_stage": exp.zero_stage,
        "bucket_size_mb": _format_bucket_size(exp.bucket_size_mb) if exp.bucket_size_mb is not None else "",
        "batch_size": exp.batch_size,
        "seq_len": exp.seq_len,
        "mbs": exp.mbs,
        "gradient_accumulation": int(exp.gradient_accumulation),
        "ar_a2a_same_stream": int(exp.ar_a2a_same_stream),
        "overlap_zero_ops": int(exp.overlap_zero_ops),
        "overlap_chunks": int(exp.overlap_chunks),
        "nsight": int(exp.nsight),
        "status": status,
        "failure_reason": failure_reason,
        "log_path": str(log_path),
        "metrics_path": str(metrics_path) if metrics_found else "",
    }
    row.update(aggregated)
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sweep",
        "label",
        "schedule",
        "pp",
        "dp",
        "ep",
        "zero_stage",
        "bucket_size_mb",
        "batch_size",
        "seq_len",
        "mbs",
        "gradient_accumulation",
        "ar_a2a_same_stream",
        "overlap_zero_ops",
        "overlap_chunks",
        "nsight",
        "status",
        "failure_reason",
        "samples",
        "iter_time_mean_s",
        "iter_time_std_s",
        "throughput_mean_tokens_per_s",
        "throughput_std_tokens_per_s",
        "peak_memory_gb_max",
        "peak_memory_gb_by_rank",
        "log_path",
        "metrics_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["sweep"],
        row["schedule"],
        int(row["pp"]),
        int(row["dp"]),
        int(row.get("ep", 0)),
        int(row["zero_stage"]),
        float(row["bucket_size_mb"]) if str(row.get("bucket_size_mb", "")).strip() else None,
        int(row["batch_size"]),
        int(row["seq_len"]),
        int(row["mbs"]),
        int(row["gradient_accumulation"]),
        int(row["ar_a2a_same_stream"]),
        int(row["overlap_zero_ops"]),
        int(row.get("overlap_chunks", 0)),
        int(row.get("nsight", 0)),
    )


def _load_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _hydrate_existing_rows(existing_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    hydrated: list[dict[str, object]] = []
    for row in existing_rows:
        metrics_path_text = str(row.get("metrics_path", "")).strip()
        if row.get("status") in {"success", "oom"} or not metrics_path_text:
            hydrated.append(row)
            continue

        metrics_path = Path(metrics_path_text)
        records = _parse_metrics_records(metrics_path)
        if not records:
            hydrated.append(row)
            continue

        aggregated = _aggregate_metrics(records)
        updated = dict(row)
        updated["status"] = "success"
        updated["failure_reason"] = ""
        updated.update(aggregated)
        hydrated.append(updated)
    return hydrated


def _completed_keys_from_metrics(
    experiments: list[Experiment],
    metrics_dir: Path,
    existing_by_key: dict[tuple[object, ...], dict[str, object]],
) -> set[tuple[object, ...]]:
    completed: set[tuple[object, ...]] = set()
    for exp in experiments:
        existing_row = existing_by_key.get(exp.key())
        if existing_row is not None and existing_row.get("status") not in {"success", "oom", "dry-run", ""}:
            continue
        metrics_path = metrics_dir / f"{exp.slug()}.metrics"
        records = _parse_metrics_records(metrics_path)
        if records:
            completed.add(exp.key())
    return completed


def _save_bar_plot(
    title: str,
    xlabel: str,
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    allow_oom_markers: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in rows]
    xs = list(range(len(rows)))

    success_rows = [row for row in rows if row["status"] == "success"]
    success_max = max(
        (float(row["iter_time_mean_s"]) for row in success_rows),
        default=0.0,
    )
    oom_marker_y = success_max * 0.05 if success_max > 0 else 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 6))
    for index, row in enumerate(rows):
        status = row["status"]
        if status == "success":
            ax.bar(
                index,
                float(row["iter_time_mean_s"]),
                yerr=float(row["iter_time_std_s"]),
                color="#4C72B0",
                ecolor="#2F2F2F",
                capsize=4,
            )
        elif allow_oom_markers and status == "oom":
            ax.scatter(index, oom_marker_y, marker="x", s=120, color="red", linewidths=2.5)
        else:
            ax.scatter(index, oom_marker_y, marker="x", s=100, color="black", linewidths=2.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Iteration Time (s)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _parse_peak_memory_values(row: dict[str, object]) -> list[float]:
    summary = str(row.get("peak_memory_gb_by_rank", "")).strip()
    if not summary:
        return []

    values: list[float] = []
    for item in summary.split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        _rank, value_text = item.split(":", maxsplit=1)
        try:
            values.append(float(value_text))
        except ValueError:
            continue
    return values


def _save_memory_plot(
    title: str,
    xlabel: str,
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    allow_oom_markers: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in rows]
    xs = list(range(len(rows)))

    success_values = [
        value
        for row in rows
        if row["status"] == "success"
        for value in _parse_peak_memory_values(row)
    ]
    marker_y = max(success_values, default=0.0) * 0.05 if success_values else 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 6))
    for index, row in enumerate(rows):
        status = row["status"]
        if status == "success":
            values = _parse_peak_memory_values(row)
            if values:
                ax.scatter(
                    [index] * len(values),
                    values,
                    color="#DD8452",
                    s=36,
                    alpha=0.9,
                )
        elif allow_oom_markers and status == "oom":
            ax.scatter(index, marker_y, marker="x", s=120, color="red", linewidths=2.5)
        else:
            ax.scatter(index, marker_y, marker="x", s=100, color="black", linewidths=2.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    enabled_sweeps = set(args.sweeps)

    if "scalability" in enabled_sweeps:
        for pp, mbs in _parse_pp_mbs(args.scalability_pp_mbs):
            for dp in (1, 2):
                experiments.append(
                    Experiment(
                        sweep="scalability",
                        schedule=args.base_schedule,
                        pp=pp,
                        dp=dp,
                        ep=0,
                        zero_stage=0,
                        bucket_size_mb=args.bucket_size,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        mbs=mbs,
                        gradient_accumulation=bool(args.gradient_accumulation),
                        ar_a2a_same_stream=False,
                        overlap_zero_ops=True,
                        overlap_chunks=bool(args.overlap_chunks),
                        nsight=bool(args.nsight),
                    )
                )

    if "zero" in enabled_sweeps:
        zero_grad_acc_settings = [setting == "on" for setting in args.zero_grad_acc_settings]
        for zero_stage in args.zero_stages:
            pp = args.pp
            mbs = 2 * pp
            for batch_size in args.zero_batch_sizes:
                for grad_acc in zero_grad_acc_settings:
                    experiments.append(
                        Experiment(
                            sweep="zero",
                            schedule=args.base_schedule,
                            pp=pp,
                            dp=args.dp,
                            ep=0,
                            zero_stage=zero_stage,
                            bucket_size_mb=args.bucket_size,
                            batch_size=batch_size,
                            seq_len=args.seq_len,
                            mbs=mbs,
                            gradient_accumulation=grad_acc,
                            ar_a2a_same_stream=False,
                            overlap_zero_ops=True,
                            overlap_chunks=bool(args.overlap_chunks),
                            nsight=bool(args.nsight),
                        )
                )

    if "schedule" in enabled_sweeps:
        for schedule in args.schedule_values:
            experiments.append(
                Experiment(
                    sweep="schedule",
                    schedule=schedule,
                    pp=args.pp,
                    dp=args.dp,
                    ep=args.dp,
                    zero_stage=args.schedule_zero_stage,
                    bucket_size_mb=args.bucket_size,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    mbs=args.schedule_mbs,
                    gradient_accumulation=bool(args.gradient_accumulation),
                    ar_a2a_same_stream=False,
                    overlap_zero_ops=True,
                    overlap_chunks=bool(args.overlap_chunks),
                    nsight=bool(args.nsight),
                )
            )

    return experiments


def _require_env_vars() -> None:
    missing = [name for name in ("SSH_KEY", "HEAD_PUBLIC_IP") if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            "Missing required environment variables: " + ", ".join(missing)
        )


def _default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> int:
    args = parse_args()
    _require_env_vars()

    runner_path = Path(args.runner).resolve()
    if not runner_path.is_file():
        raise SystemExit(f"Runner not found: {runner_path}")

    out_root = Path(args.out_dir)
    run_name = args.run_name or _default_run_name()
    out_dir = out_root / run_name
    logs_dir = out_dir / "logs"
    metrics_dir = out_dir / "metrics"
    fetch_dir = out_dir / "runner-fetch"
    plots_dir = out_dir / "plots"
    all_results_path = out_dir / "all_results.csv"
    for directory in (logs_dir, metrics_dir, fetch_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    experiments = _build_experiments(args)
    total = len(experiments)
    existing_rows = [] if args.no_resume else _hydrate_existing_rows(_load_existing_rows(all_results_path))
    rows: list[dict[str, object]] = list(existing_rows)
    existing_by_key = {_row_key(row): row for row in existing_rows}
    completed_keys = {
        key
        for key, row in existing_by_key.items()
        if row.get("status") in {"success", "oom"}
    }
    completed_keys |= _completed_keys_from_metrics(experiments, metrics_dir, existing_by_key)

    for exp in experiments:
        existing_row = existing_by_key.get(exp.key())
        if existing_row is not None and existing_row.get("status") in {"success", "oom", "dry-run"}:
            continue
        metrics_path = metrics_dir / f"{exp.slug()}.metrics"
        if not metrics_path.is_file():
            recovered = _recover_metrics_from_fetch_logs(args, exp, fetch_dir, metrics_path)
            if not recovered:
                continue
        cached_row = _row_from_metrics(exp, logs_dir / exp.slug() / "run.log", metrics_path)
        if cached_row is None:
            continue
        existing_by_key[exp.key()] = cached_row

    rows = list(existing_by_key.values())
    completed_keys = {
        key
        for key, row in existing_by_key.items()
        if row.get("status") in {"success", "oom", "dry-run"}
    }

    print(f"Starting e2e eval with {total} experiments")
    print(f"Run directory: {out_dir}")
    use_inductor_summary = (
        "auto(schedule=0,zero/scalability=1)"
        if args.use_inductor is None
        else str(int(args.use_inductor))
    )

    print(
        "Config: "
        f"model={args.model} batch_size={args.batch_size} seq_len={args.seq_len} "
        f"warmup={args.warmup} iters={args.iters} base_schedule={args.base_schedule} "
        f"schedule_zero_stage={args.schedule_zero_stage} schedule_mbs={args.schedule_mbs} "
        f"zero_batch_sizes={args.zero_batch_sizes} zero_grad_acc_settings={args.zero_grad_acc_settings} "
        f"bucket_size_mb={_format_bucket_size(args.bucket_size)} "
        f"zero_mbs=2*pp nsight={int(args.nsight)} use_inductor={use_inductor_summary} "
        f"gradient_accumulation={int(args.gradient_accumulation)}"
    )
    if existing_rows and not args.no_resume:
        print(
            f"Resume: loaded {len(existing_rows)} existing rows, "
            f"skipping {len(completed_keys)} completed experiments"
        )
    sys.stdout.flush()

    failures = 0
    oom_failures = 0
    interrupted = False
    for index, exp in enumerate(experiments, start=1):
        if exp.key() in completed_keys:
            print(f"[{index}/{total}] SKIP completed {exp.sweep}: {exp.label()}")
            sys.stdout.flush()
            continue
        try:
            row = _run_experiment(args, exp, index, total, logs_dir, metrics_dir, fetch_dir)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\nInterrupted during {exp.sweep}: {exp.label()}")
            sys.stdout.flush()
            break

        existing_by_key[_row_key(row)] = row
        rows = list(existing_by_key.values())
        if row["status"] == "oom":
            oom_failures += 1
        elif row["status"] not in ("success", "dry-run"):
            failures += 1
        completed = len(
            [experiment for experiment in experiments if experiment.key() in {_row_key(item) for item in rows}]
        )
        print(
            f"Progress: completed={completed}/{total} "
            f"success={sum(1 for item in rows if item['status'] == 'success')} "
            f"oom={sum(1 for item in rows if item['status'] == 'oom')} "
            f"failed={sum(1 for item in rows if item['status'] not in ('success', 'oom', 'dry-run'))}"
        )
        sys.stdout.flush()
        _write_csv(all_results_path, rows)

    _write_csv(all_results_path, rows)

    for sweep in ("scalability", "zero", "schedule"):
        sweep_rows = [row for row in rows if row["sweep"] == sweep]
        _write_csv(out_dir / f"{sweep}_results.csv", sweep_rows)

    if args.dry_run:
        print(f"Dry run finished. Planned artifacts are rooted at {out_dir}")
        return 0

    _save_bar_plot(
        "Scalability Sweep",
        "(pp, dp)",
        [row for row in rows if row["sweep"] == "scalability"],
        plots_dir / "scalability.png",
    )
    _save_bar_plot(
        "ZeRO Sweep",
        "(zero, ga, batch_size)",
        [row for row in rows if row["sweep"] == "zero"],
        plots_dir / "zero.png",
        allow_oom_markers=True,
    )
    _save_bar_plot(
        "Schedule Sweep",
        "schedule",
        [row for row in rows if row["sweep"] == "schedule"],
        plots_dir / "schedule.png",
    )
    _save_memory_plot(
        "Scalability Sweep Memory",
        "(pp, dp)",
        [row for row in rows if row["sweep"] == "scalability"],
        plots_dir / "scalability_memory.png",
    )
    _save_memory_plot(
        "ZeRO Sweep Memory",
        "(zero, ga, batch_size)",
        [row for row in rows if row["sweep"] == "zero"],
        plots_dir / "zero_memory.png",
        allow_oom_markers=True,
    )
    _save_memory_plot(
        "Schedule Sweep Memory",
        "schedule",
        [row for row in rows if row["sweep"] == "schedule"],
        plots_dir / "schedule_memory.png",
    )

    if interrupted:
        print(f"Interrupted. Partial results were saved to {out_dir}")
        return 130

    print(f"Finished. Results are in {out_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
