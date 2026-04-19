#!/usr/bin/env python3
"""Run a configurable sweep over scripts/run-qwen-ec2.sh.

Default sweep:
  SCHEDULE in {1f1b}
  (PP, MBS) in {(4, 8), (8, 16)}
  DP in {2, 4}
  ZERO_STAGE in {0, 1, 2, 3}
  BATCH_SIZE in {8, 16, 32}
  SEQ_LEN in {512}
  GRAD_ACC in {on, off}
  AR_A2A_SAME_STREAM in {on, off}
  OVERLAP_ZERO_OPS in {on, off}
  OVERLAP_CHUNKS in {off}

Each dimension can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
import traceback
from pathlib import Path


DEFAULT_PP_MBS = ("4:8", "8:16")
DEFAULT_SCHEDULES = ("1f1b",)
DEFAULT_DP_VALUES = (1, 2, 4)
DEFAULT_ZERO_STAGES = (0, 1, 2, 3)
DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_SEQ_LENS = (512,)
DEFAULT_GRADIENT_ACCUMULATION_VALUES = ("on", "off")
DEFAULT_AR_A2A_SAME_STREAM_VALUES = ("on", "off")
DEFAULT_OVERLAP_ZERO_OPS_VALUES = ("on", "off")
DEFAULT_OVERLAP_CHUNKS_VALUES = ("off",)


def _parse_pp_mbs(values: list[str]) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for value in values:
        try:
            pp_str, mbs_str = value.split(":", maxsplit=1)
            parsed.append((int(pp_str), int(mbs_str)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --pp-mbs value '{value}'. Expected format PP:MBS, e.g. 4:8."
            ) from exc
    return parsed


def _build_command(
    script_path: Path,
    *,
    model: str,
    schedule: str,
    pp: int,
    dp: int,
    zero_stage: int,
    batch_size: int,
    seq_len: int,
    mbs: int,
    gradient_accumulation: bool,
    ar_a2a_same_stream: bool,
    overlap_zero_ops: bool,
    overlap_chunks: bool,
    warmup: int,
    iters: int,
    extra_args: list[str],
) -> list[str]:
    command = [
        str(script_path),
        "--model", model,
        "--schedule", schedule,
        "--pp", str(pp),
        "--dp", str(dp),
        "--zero-stage", str(zero_stage),
        "--batch-size", str(batch_size),
        "--seq-len", str(seq_len),
        "--mbs", str(mbs),
        "--gradient-accumulation" if gradient_accumulation else "--no-gradient-accumulation",
        "--ar-a2a-same-stream" if ar_a2a_same_stream else "--no-ar-a2a-same-stream",
        "--overlap-zero-ops" if overlap_zero_ops else "--no-overlap-zero-ops",
        "--overlap-chunks" if overlap_chunks else "--no-overlap-chunks",
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    command.extend(extra_args)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep scripts/run-qwen-ec2.sh over configurable dimension values."
    )
    parser.add_argument("--model", default="1B", help="Model passed to run-qwen-ec2.sh")
    parser.add_argument(
        "--schedules",
        nargs="+",
        default=list(DEFAULT_SCHEDULES),
        help="Pipeline schedules to sweep. Default: 1f1b",
    )
    parser.add_argument(
        "--pp-mbs",
        nargs="+",
        default=list(DEFAULT_PP_MBS),
        metavar="PP:MBS",
        help="Pairs of pipeline degree and microbatch count. Default: 4:8 8:16",
    )
    parser.add_argument(
        "--dp-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_DP_VALUES),
        help="DP values to sweep. Default: 2 4",
    )
    parser.add_argument(
        "--zero-stages",
        nargs="+",
        type=int,
        default=list(DEFAULT_ZERO_STAGES),
        help="ZeRO stages to sweep. Default: 0 1 2 3",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to sweep. Default: 8 16 32",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEQ_LENS),
        help="Sequence lengths to sweep. Default: 512",
    )
    parser.add_argument(
        "--gradient-accumulation-values",
        nargs="+",
        choices=["on", "off"],
        default=list(DEFAULT_GRADIENT_ACCUMULATION_VALUES),
        help="Gradient accumulation settings to sweep. Default: on off",
    )
    parser.add_argument(
        "--ar-a2a-same-stream-values",
        nargs="+",
        choices=["on", "off"],
        default=list(DEFAULT_AR_A2A_SAME_STREAM_VALUES),
        help="AR/A2A same-stream settings to sweep. Default: on off",
    )
    parser.add_argument(
        "--overlap-zero-ops-values",
        nargs="+",
        choices=["on", "off"],
        default=list(DEFAULT_OVERLAP_ZERO_OPS_VALUES),
        help="overlap_zero_ops settings to sweep. Default: on off",
    )
    parser.add_argument(
        "--overlap-chunks-values",
        nargs="+",
        choices=["on", "off"],
        default=list(DEFAULT_OVERLAP_CHUNKS_VALUES),
        help="overlap_chunks settings to sweep. Default: off",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations")
    parser.add_argument(
        "--script",
        default="scripts/run-qwen-ec2.sh",
        help="Path to the underlying runner script",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Deprecated: the sweep now always continues after a failed run",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to run-qwen-ec2.sh. Prefix with '--'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pp_mbs_values = _parse_pp_mbs(args.pp_mbs)
    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        print(f"Runner script not found: {script_path}", file=sys.stderr)
        return 1

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    combinations: list[tuple[str, tuple[int, int], int, int, int, int, bool, bool, bool, bool]] = []
    for schedule, (pp, mbs), dp, batch_size, seq_len, gradient_accumulation, ar_a2a_same_stream, overlap_zero_ops, overlap_chunks in itertools.product(
        args.schedules,
        pp_mbs_values,
        args.dp_values,
        args.batch_sizes,
        args.seq_lens,
        [value == "on" for value in args.gradient_accumulation_values],
        [value == "on" for value in args.ar_a2a_same_stream_values],
        [value == "on" for value in args.overlap_zero_ops_values],
        [value == "on" for value in args.overlap_chunks_values],
    ):
        zero_stages = args.zero_stages if dp > 1 else [0]
        for zero_stage in zero_stages:
            combinations.append(
                (
                    schedule,
                    (pp, mbs),
                    dp,
                    zero_stage,
                    batch_size,
                    seq_len,
                    gradient_accumulation,
                    ar_a2a_same_stream,
                    overlap_zero_ops,
                    overlap_chunks,
                )
            )

    total = len(combinations)
    failures: list[dict[str, object]] = []
    for index, (
        schedule,
        (pp, mbs),
        dp,
        zero_stage,
        batch_size,
        seq_len,
        gradient_accumulation,
        ar_a2a_same_stream,
        overlap_zero_ops,
        overlap_chunks,
    ) in enumerate(
        combinations, start=1
    ):
        command = _build_command(
            script_path,
            model=args.model,
            schedule=schedule,
            pp=pp,
            dp=dp,
            zero_stage=zero_stage,
            batch_size=batch_size,
            seq_len=seq_len,
            mbs=mbs,
            gradient_accumulation=gradient_accumulation,
            ar_a2a_same_stream=ar_a2a_same_stream,
            overlap_zero_ops=overlap_zero_ops,
            overlap_chunks=overlap_chunks,
            warmup=args.warmup,
            iters=args.iters,
            extra_args=extra_args,
        )
        print(
            f"[{index}/{total}] schedule={schedule} pp={pp} mbs={mbs} dp={dp} zero_stage={zero_stage} "
            f"batch_size={batch_size} seq_len={seq_len} "
            f"gradient_accumulation={'on' if gradient_accumulation else 'off'} "
            f"ar_a2a_same_stream={'on' if ar_a2a_same_stream else 'off'} "
            f"overlap_zero_ops={'on' if overlap_zero_ops else 'off'} "
            f"overlap_chunks={'on' if overlap_chunks else 'off'}"
        )
        print("  " + " ".join(shlex.quote(part) for part in command))

        if args.dry_run:
            continue

        try:
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                continue
            failures.append(
                {
                    "index": index,
                    "schedule": schedule,
                    "pp": pp,
                    "mbs": mbs,
                    "dp": dp,
                    "zero_stage": zero_stage,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "gradient_accumulation": gradient_accumulation,
                    "ar_a2a_same_stream": ar_a2a_same_stream,
                    "overlap_zero_ops": overlap_zero_ops,
                    "overlap_chunks": overlap_chunks,
                    "returncode": result.returncode,
                    "command": command,
                }
            )
            print(f"  command failed with exit code {result.returncode}, continuing", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nSweep interrupted by user.", file=sys.stderr)
            failures.append(
                {
                    "index": index,
                    "schedule": schedule,
                    "pp": pp,
                    "mbs": mbs,
                    "dp": dp,
                    "zero_stage": zero_stage,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "gradient_accumulation": gradient_accumulation,
                    "ar_a2a_same_stream": ar_a2a_same_stream,
                    "overlap_zero_ops": overlap_zero_ops,
                    "overlap_chunks": overlap_chunks,
                    "returncode": "interrupted",
                    "command": command,
                }
            )
            break
        except Exception as exc:
            failures.append(
                {
                    "index": index,
                    "schedule": schedule,
                    "pp": pp,
                    "mbs": mbs,
                    "dp": dp,
                    "zero_stage": zero_stage,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "gradient_accumulation": gradient_accumulation,
                    "ar_a2a_same_stream": ar_a2a_same_stream,
                    "overlap_zero_ops": overlap_zero_ops,
                    "overlap_chunks": overlap_chunks,
                    "returncode": "exception",
                    "command": command,
                    "error": repr(exc),
                }
            )
            print(f"  command raised {type(exc).__name__}: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    if failures:
        print("\nSweep completed with failures:", file=sys.stderr)
        for failure in failures:
            print(
                f"  [{failure['index']}/{total}] "
                f"schedule={failure['schedule']} pp={failure['pp']} mbs={failure['mbs']} dp={failure['dp']} "
                f"zero_stage={failure['zero_stage']} batch_size={failure['batch_size']} "
                f"seq_len={failure['seq_len']} "
                f"gradient_accumulation={'on' if failure['gradient_accumulation'] else 'off'} "
                f"ar_a2a_same_stream={'on' if failure['ar_a2a_same_stream'] else 'off'} "
                f"overlap_zero_ops={'on' if failure['overlap_zero_ops'] else 'off'} "
                f"overlap_chunks={'on' if failure['overlap_chunks'] else 'off'} "
                f"status={failure['returncode']}",
                file=sys.stderr,
            )
            print(
                "    " + " ".join(shlex.quote(part) for part in failure["command"]),
                file=sys.stderr,
            )
            if "error" in failure:
                print(f"    error={failure['error']}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
